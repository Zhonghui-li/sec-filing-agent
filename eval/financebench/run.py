"""External validation against FinanceBench (Patronus AI) — a benchmark we did NOT author.

Two tracks, because FinanceBench covers ~40 companies / many years while we cover 7 companies'
latest filings:

  TRACK A — abstain calibration (the 13 real FinanceBench questions that fall on our 7 companies).
    Most ask for metrics/years outside our coverage, so the RIGHT behavior is to ABSTAIN, not
    guess. We score: correct-abstain (didn't have it, said so) vs hallucinated (made up a number)
    vs answered. This validates the finance bar on someone else's questions.

  TRACK B — in-coverage accuracy (derived-metric questions modeled on FinanceBench's own metric
    definitions — COGS%, ROA, current ratio, payout ratio — but on companies/years WE cover).
    Gold is computed from our XBRL data (self-grounded). This validates that the agent computes
    these ratios correctly, with the right tools.

Usage:  DATABASE_URL=... OPENAI_API_KEY=... python -m eval.financebench.run
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOL = 0.05  # 5% relative tolerance on a numeric answer


def _nums(s):
    out = []
    for x in re.findall(r"-?[\d,]+(?:\.\d+)?", s or ""):
        x = x.replace(",", "")
        if x not in ("", "-", "."):
            try:
                out.append(float(x))
            except ValueError:
                pass
    return out


def _abstained(out):
    return "abstain" in out["tools_used"]


def _has_number_match(answer, target):
    """Does the answer contain `target` (within tolerance)? Tolerant to percent-vs-decimal
    (0.16 vs 16.25%) since a ratio can be stated either way."""
    cands = _nums(answer)
    # also consider each value scaled by 100 / divided by 100 (percent <-> decimal)
    cands = cands + [v / 100 for v in cands] + [v * 100 for v in cands]
    for v in cands:
        if target != 0 and abs(v - target) <= TOL * abs(target):
            return True
        if abs(target) < 5 and abs(v - target) <= 0.05:  # small ratios (0.79, 1.06)
            return True
    return False


def track_a(run_agent, agent):
    """Real FinanceBench questions on our companies -> abstain calibration."""
    cases = json.loads((HERE / "real_subset.json").read_text())
    rows = []
    for c in cases:
        out = run_agent(c["question"], agent=agent)
        abstained = _abstained(out)
        # ground truth answer present? if it's a number we can check
        gold = _nums(c["answer"])
        answered_number = bool(_nums(out["answer"])) and not abstained
        if abstained:
            verdict = "correct_abstain"   # didn't have the data, said so (the finance bar)
        elif gold and _has_number_match(out["answer"], gold[0]):
            verdict = "correct_answer"     # in coverage AND right
        elif answered_number:
            verdict = "hallucinated_or_wrong"  # produced a number that doesn't match -> bad
        else:
            verdict = "non_numeric_answer"
        rows.append({"id": c["financebench_id"], "company": c["company"],
                     "type": c["question_type"], "verdict": verdict,
                     "q": c["question"][:70], "gold": c["answer"][:40],
                     "got": out["answer"][:80].replace("\n", " ")})
    return rows


def track_b(run_agent, agent):
    """Derived-metric questions modeled on FinanceBench, on companies/years we cover.
    Gold computed from our own XBRL data (self-grounded)."""
    cases = json.loads((HERE / "modeled.json").read_text())
    rows = []
    for c in cases:
        out = run_agent(c["question"], agent=agent)
        ok = (not _abstained(out)) and _has_number_match(out["answer"], c["gold"])
        rows.append({"id": c["id"], "metric": c["metric"], "company": c["company"],
                     "correct": ok, "gold": c["gold"],
                     "got": out["answer"][:80].replace("\n", " ")})
    return rows


def main():
    from agents.sec_agent import build_agent, run_agent
    agent = build_agent()

    print("=== TRACK A: abstain calibration (real FinanceBench questions on our companies) ===")
    a = track_a(run_agent, agent)
    from collections import Counter
    ca = Counter(r["verdict"] for r in a)
    for r in a:
        print(f"  [{r['verdict']:22}] {r['company']:10} {r['q']}")
    print(f"  -> {dict(ca)}")
    halluc = ca["hallucinated_or_wrong"]
    print(f"  abstain calibration: {len(a)-halluc}/{len(a)} handled safely "
          f"(abstained or correct); {halluc} hallucinated/wrong.")

    print("\n=== TRACK B: in-coverage accuracy (FinanceBench-style derived metrics, our data) ===")
    b = track_b(run_agent, agent)
    nb_ok = sum(r["correct"] for r in b)
    for r in b:
        print(f"  [{'PASS' if r['correct'] else 'FAIL'}] {r['company']:6} {r['metric']:14} "
              f"gold={r['gold']}  got: {r['got']}")
    print(f"  -> in-coverage accuracy: {nb_ok}/{len(b)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
