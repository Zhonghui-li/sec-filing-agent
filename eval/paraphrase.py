#!/usr/bin/env python3
"""Does the answer survive being asked differently?

An agent that gives one figure for "Apple's revenue in fiscal 2024" and another for "how much did
Apple sell in FY24" is not usable, whichever one is right. That is a different question from
correctness, so it gets a different harness: the unit of measurement here is a GROUP of phrasings,
not a case, and it cannot be scored one row at a time the way eval/score.py does.

It also stays out of eval/testset.jsonl on purpose. Four phrasings of one fact would enter the
other metrics' denominators four times over, weighting whatever fact happens to be paraphrased.

Two test kinds, after CheckList (Ribeiro et al., ACL 2020), which names both:

    INV  invariance — reword the question, the figure must NOT change
    DIR  directional — change the YEAR or the COMPANY, the figure MUST change

DIR is what makes INV mean anything. A model that ignores the question and always returns the same
number passes a pure invariance suite perfectly; the first version of this file had no DIR at all
and would have scored it 5/5. It is the same control that sits in `forbid` and `clarify` — a
metric measuring only one direction can be satisfied by refusing to vary.

Each INV variant differs from the baseline in EXACTLY ONE way — metric synonym, fiscal-year form,
entity form, padding, language — so a failure names the dimension that caused it. The earlier
version changed several at once and could only report that something broke.

    INV      PASS    some value appears in EVERY answer that produced one
             FAIL    answers produced values, but no value is common to all
             UNKNOWN fewer than two answers produced a value — a refusal or a clarifying question
                     is not an inconsistency, and counting it as one would reward answering at
                     any cost
    DIR      PASS    the figure matches THAT question's own gold
             FAIL    it does not — the baseline's figure came back, or the wrong one did

DIR is checked against gold rather than against "differs from the baseline", because differing is
not the same as being right and the weaker test fails both ways. PepsiCo's FY2023 and FY2024
revenue are 0.42% apart, inside the tolerance, so a CORRECT answer to the shifted year looked like
the baseline figure coming back.

"Some value in every answer", rather than comparing the first figure in each, because an answer
legitimately carries more than one — "revenue was $391,035,000,000, up from $383,285,000,000".
Comparing position would call that instability; comparing membership does not.

    DATABASE_URL=... OPENAI_API_KEY=... python -m eval.paraphrase

Exit status is 1 if any group is inconsistent, so it can gate a release.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GROUPS = ROOT / "eval" / "paraphrase.jsonl"
TOL = 0.005          # tighter than the answer-correctness tolerance: this asks whether the SAME
                     # figure came back, not whether it is close enough to gold


_NOISE_RX = re.compile(r"\d{10}-\d{2}-\d{6}|https?://\S+|/Archives/\S+")

# Chinese magnitude words. A phrasing set that deliberately includes Chinese — because users ask
# that way — is not tested at all if the scorer cannot read the reply. Both Chinese variants
# answered CORRECTLY on the first run ("976.9 亿美元" is $97.69B, exactly right) and both were
# recorded as "no figure", so two groups quietly scored 3/4 on the scorer's blind spot, not the
# agent's.
_CN_MULT = {"万亿": 1e12, "亿": 1e8, "万": 1e4, "千": 1e3}
_CN_RX = re.compile(r"([\d,]+(?:\.\d+)?)\s*(万亿|亿|万|千)")


def _figures(answer: str):
    """Money-sized numbers ASSERTED in an answer, unit-normalised.

    Citations are stripped first. A magnitude window alone does not exclude them: NVIDIA's
    accession 0001045810-24-000316 carries its CIK 1,045,810, which sits just inside the window, and
    four phrasings all quoting the same filing therefore "agreed" on a figure none of them meant —
    two of the five groups passed that way on the first run, on a number read out of a citation.
    """
    from eval.score import extract_numbers
    clean = _NOISE_RX.sub(" ", answer or "")
    vals = {v for v, pct in extract_numbers(clean) if not pct}
    for num, unit in _CN_RX.findall(clean):
        try:
            vals.add(float(num.replace(",", "")) * _CN_MULT[unit])
        except ValueError:
            pass
    return sorted({v for v in vals if 1e6 < abs(v) < 1e13})


def _common(per_answer):
    """Values present in every one of these figure lists, within tolerance."""
    lists = [f for f in per_answer if f]
    if not lists:
        return []
    return [v for v in lists[0]
            if all(any(abs(v - w) <= TOL * max(abs(v), abs(w)) for w in other)
                   for other in lists[1:])]


def main():
    from agents.sec_agent import build_agent, run_agent
    groups = [json.loads(l) for l in GROUPS.open()]
    agent = build_agent()
    inv_fail, dir_fail = [], []
    by_dim = {}                      # which rewrite dimension broke a group
    print(f"{len(groups)} facts · {len(groups) * 8} phrasings "
          f"(6 invariance + 2 directional each)\n")
    for g in groups:
        figs, ask = {}, lambda q: _figures(run_agent(q, agent=agent)["answer"])
        for dim, q in g["inv"].items():
            figs[dim] = ask(q)
        decided = [f for f in figs.values() if f]
        common = _common(list(figs.values()))
        inv = "UNKNOWN" if len(decided) < 2 else "PASS" if common else "FAIL"

        base = figs.get("baseline") or []
        dirs = {}
        for dim, q in g["dir"].items():
            got = ask(q)
            want = (g.get("dir_gold") or {}).get(dim)
            # Against THAT question's own gold, not merely "different from the baseline".
            # Differing is not the same as being right, and it fails in both directions:
            # PepsiCo's FY2023 and FY2024 revenue are 0.42% apart, inside the tolerance, so a
            # correct answer to the shifted year read as "returned the baseline figure".
            dirs[dim] = ("UNKNOWN" if not got or want is None else
                         "PASS" if any(abs(v - want) <= TOL * max(abs(v), abs(want))
                                       for v in got) else "FAIL")

        bad_dirs = [d for d, v in dirs.items() if v == "FAIL"]
        mark = "  " if inv == "PASS" and not bad_dirs else "<-"
        shared = f"  shared={common[0]:,.0f}" if common else ""
        print(f"{mark} {g['id']}  INV {inv:8} DIR {'/'.join(dirs.values()):16} "
              f"{g['fact']:24} {len(decided)}/6 answered{shared}")
        if inv == "FAIL":
            inv_fail.append(g["id"])
            # name the dimension: which single rewrite lost the value the others shared
            majority = _common([f for d, f in figs.items() if d != "baseline" and f])
            for d, f in figs.items():
                if f and (not majority or not _common([majority, f])):
                    by_dim[d] = by_dim.get(d, 0) + 1
                    print(f"       {d:10} diverged: {', '.join(f'{v:,.0f}' for v in f)}")
        for d in bad_dirs:
            dir_fail.append(f"{g['id']}/{d}")
            print(f"       {d:10} returned the baseline figure for a different question")

    print(f"\nINV: {len(groups) - len(inv_fail)}/{len(groups)} consistent"
          + (f"  failed={inv_fail}" if inv_fail else ""))
    print(f"DIR: {len(groups) * 2 - len(dir_fail)}/{len(groups) * 2} moved as they should"
          + (f"  failed={dir_fail}" if dir_fail else ""))
    if by_dim:
        print("\nrewrite dimensions that broke a group: "
              + " · ".join(f"{d} ×{n}" for d, n in sorted(by_dim.items(), key=lambda x: -x[1])))
    return 1 if (inv_fail or dir_fail) else 0


if __name__ == "__main__":
    sys.exit(main())
