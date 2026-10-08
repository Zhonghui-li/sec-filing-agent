"""Does a model still answer in the SHAPE the harness and the guardrails expect?

Correctness metrics ask whether the answer is right. They do not ask whether it arrived in the
form everything downstream was built around, and a model swap changes that form first. Seen
within one smoke test on 2026-10-08: o4-mini answers "$391,035,000,000", gpt-5.4-mini answers
"**$391.035 billion**" — same fact, rounded, and wrapped in markdown. Scoring that compares digits
can read the second as wrong, and a reader of the metric would conclude the model is worse.

Reports, never gates. The point is to see the shape change before deciding what it means.

Usage: python -m eval.format_stability eval/_migration_phase1/*/run1.jsonl
"""
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.abstain import ABSTAIN_REASONS

_ACCN = re.compile(r"\d{10}-\d{2}-\d{6}")
_MD = re.compile(r"\*\*|^#{1,6} |^\s*[-*] ", re.M)
_EXACT = re.compile(r"\d{1,3}(?:,\d{3}){2,}")          # 1,234,567,890 — full precision
_ROUNDED = re.compile(r"\d+(?:\.\d+)?\s*(?:billion|million|bn|m\b)", re.I)


def profile(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    ans = [r.get("answer") or "" for r in rows]
    tr = [r.get("trace") or [] for r in rows]

    # abstain: called as a tool, and with a reason the enum knows
    calls = [c for t in tr for c in t if c.get("tool") == "abstain"]
    reasons = Counter((c.get("args") or {}).get("reason") for c in calls)
    unknown = {k: v for k, v in reasons.items() if k not in ABSTAIN_REASONS}

    # citation: when a tool emitted an accession, did the answer carry one through
    carried = emitted = 0
    for a, t in zip(ans, tr):
        if any(_ACCN.search(str(c.get("output") or "")) for c in t):
            emitted += 1
            carried += bool(_ACCN.search(a))

    return {
        "n": len(rows),
        "answer_chars_median": int(statistics.median(len(a) for a in ans)) if ans else 0,
        "markdown_in_answer": sum(bool(_MD.search(a)) for a in ans),
        "exact_digit_figures": sum(bool(_EXACT.search(a)) for a in ans),
        "rounded_unit_figures": sum(bool(_ROUNDED.search(a)) for a in ans),
        "abstain_calls": len(calls),
        "abstain_reasons": dict(reasons),
        "abstain_unknown_reason": unknown,
        "accession_emitted_by_tool": emitted,
        "accession_carried_into_answer": carried,
        "tool_calls_total": sum(len(t) for t in tr),
        "guardrail_fires": sum(bool(r.get("guardrail_reason")) for r in rows),
    }


def main(paths):
    profs = {Path(p).parent.name: profile(p) for p in paths}
    keys = ["n", "answer_chars_median", "markdown_in_answer", "exact_digit_figures",
            "rounded_unit_figures", "abstain_calls", "accession_emitted_by_tool",
            "accession_carried_into_answer", "tool_calls_total", "guardrail_fires"]
    w = max(len(k) for k in keys) + 2
    names = list(profs)
    print(f"{'':{w}}" + "".join(f"{n:>16}" for n in names))
    for k in keys:
        print(f"{k:{w}}" + "".join(f"{profs[n][k]:>16}" for n in names))
    print()
    for n in names:
        if profs[n]["abstain_unknown_reason"]:
            print(f"  ⚠️ {n}: abstain reason outside the enum -> {profs[n]['abstain_unknown_reason']}")
        print(f"  {n}: abstain reasons {profs[n]['abstain_reasons']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
