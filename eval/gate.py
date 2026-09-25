"""Did this build get worse, or did it just come out differently?

The gate used a percentage threshold: a metric losing more than 10 points was a regression. Two
things are wrong with that. The denominators differ by a factor of fifteen — `tool` is scored over
106 cases and `forbid` over 7 — so ten points means ten cases on one metric and less than one case
on another, and a threshold that is about right for the first cannot detect anything at all on the
second. And a rate says nothing about WHICH cases moved: three cases breaking while three others
get fixed is noise, but it reads as three cases broken.

Both runs cover the same cases, and each case either passes or fails, which is exactly what
McNemar's test is for. It looks only at the cases that DISAGREE between the two runs:

    b = passed in the baseline, fails now        c = failed in the baseline, passes now

Cases that agree carry no information about a change and are ignored. Under the null hypothesis
that the build is no worse, each disagreement is a coin flip, so b follows Binomial(b + c, 0.5) and
a one-sided tail gives the probability of seeing this many regressions by chance. One-sided on
purpose: a build that only improves must not fail the gate.

The exact binomial is used rather than the chi-squared approximation, which needs more
disagreements than a suite this size produces. Below `min_discordant` the answer is UNDECIDABLE,
not PASS — with seven cases, `forbid` can rarely produce five disagreements, so no test can speak
for it, and saying so is the honest output. A green light there would be manufactured.

Reference: paired evaluation needs roughly half the samples of unpaired, which is why the same
cases are rerun rather than sampled afresh.
"""
from math import comb
from typing import Dict, List, Optional, Tuple

ALPHA = 0.05
MIN_DISCORDANT = 5


def binom_tail(b: int, n: int) -> float:
    """P(X >= b) for X ~ Binomial(n, 0.5) — the chance that this many of the disagreements went
    the wrong way if the build were no worse."""
    if n == 0:
        return 1.0
    return sum(comb(n, k) for k in range(b, n + 1)) / (2 ** n)


def mcnemar(pairs: List[Tuple[bool, bool]], alpha: float = ALPHA,
            min_discordant: int = MIN_DISCORDANT) -> Dict:
    """One metric, as (baseline_passed, current_passed) per case."""
    b = sum(1 for was, now in pairs if was and not now)      # regressed
    c = sum(1 for was, now in pairs if now and not was)      # fixed
    n = b + c
    p = binom_tail(b, n) if n else None
    if b == 0:
        # Nothing broke. The question a gate asks is "did this get worse", and the answer is no —
        # reporting UNDECIDABLE here would show yellow for a build that is perfectly stable.
        verdict = "OK"
    elif n < min_discordant:
        verdict = "UNDECIDABLE"
    elif p is not None and p < alpha:
        verdict = "REGRESSED"
    else:
        verdict = "OK"
    return {"verdict": verdict, "regressed": b, "fixed": c, "discordant": n,
            "p": p, "scored": len(pairs)}


def compare(baseline: Dict[str, Dict], current: Dict[str, Dict],
            metrics: Optional[List[str]] = None, **kw) -> Dict[str, Dict]:
    """Per-metric McNemar over the cases both runs scored.

    `baseline` and `current` are {case_id: {metric: value}}. A metric is paired for a case only
    when BOTH runs judged it — None means not applicable, and a case present in one run and not
    the other (a case added or deleted since) is not a change in behaviour.
    """
    shared = sorted(set(baseline) & set(current))
    names = metrics or sorted({m for cid in shared for m in baseline[cid]})
    out = {}
    for m in names:
        pairs = [(baseline[cid][m], current[cid][m]) for cid in shared
                 if isinstance(baseline[cid].get(m), bool)
                 and isinstance(current[cid].get(m), bool)]
        if pairs:
            out[m] = mcnemar(pairs, **kw)
    return out


def format_report(results: Dict[str, Dict]) -> str:
    lines, worst = [], []
    for m, r in sorted(results.items()):
        p = "    —" if r["p"] is None else f"{r['p']:5.3f}"
        note = {"REGRESSED": "  <-- REGRESSION",
                "UNDECIDABLE": f"  (needs {MIN_DISCORDANT} disagreements, has {r['discordant']})"
                }.get(r["verdict"], "")
        lines.append(f"  {m:16} {r['verdict']:12} broke {r['regressed']:2}  fixed {r['fixed']:2}"
                     f"  of {r['scored']:3} scored   p={p}{note}")
        if r["verdict"] == "REGRESSED":
            worst.append(m)
    if worst:
        lines.append(f"\n  {len(worst)} metric(s) regressed: {', '.join(worst)}")
    return "\n".join(lines)
