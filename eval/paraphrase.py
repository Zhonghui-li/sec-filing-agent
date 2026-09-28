#!/usr/bin/env python3
"""Does the answer survive being asked differently?

An agent that gives one figure for "Apple's revenue in fiscal 2024" and another for "how much did
Apple sell in FY24" is not usable, whichever one is right. That is a different question from
correctness, so it gets a different harness: the unit of measurement here is a GROUP of phrasings,
not a case, and it cannot be scored one row at a time the way eval/score.py does.

It also stays out of eval/testset.jsonl on purpose. Four phrasings of one fact would enter the
other metrics' denominators four times over, weighting whatever fact happens to be paraphrased.

Scoring, per group (the contract in Obsidian note 33):

    PASS     some value appears in EVERY answer that produced one
    FAIL     answers produced values, but no value is common to all
    UNKNOWN  fewer than two answers produced a value — a refusal or a clarifying question is not
             an inconsistency, and counting it as one would reward answering at any cost

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
    failed = 0
    print(f"{len(groups)} facts, {sum(len(g['variants']) for g in groups)} phrasings\n")
    for g in groups:
        per, detail = [], []
        for q in g["variants"]:
            out = run_agent(q, agent=agent)
            figs = _figures(out["answer"])
            per.append(figs)
            detail.append((q, figs, out["tools_used"]))
        decided = [f for f in per if f]
        common = _common(per)
        verdict = ("UNKNOWN" if len(decided) < 2 else "PASS" if common else "FAIL")
        mark = {"PASS": "  ", "FAIL": "<-", "UNKNOWN": "? "}[verdict]
        shared = f"  shared={common[0]:,.0f}" if common else ""
        print(f"{mark} {g['id']}  {verdict:8} {g['fact']:26} "
              f"{len(decided)}/{len(per)} answered{shared}")
        if verdict != "PASS":
            for q, figs, tools in detail:
                shown = ", ".join(f"{v:,.0f}" for v in figs) or "(no figure)"
                print(f"       {q[:58]:60} {shown}")
        if verdict == "FAIL":
            failed += 1
    print(f"\n{len(groups) - failed} consistent, {failed} inconsistent")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
