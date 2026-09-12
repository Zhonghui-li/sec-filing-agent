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
TOL = 0.01  # 1% relative tolerance on a numeric answer — the bar FinQA, the closest published
# standard for this task, uses for execution accuracy. Measured twice (the 50-question metrics
# set and the 55-question 2026-09-12 re-run): 5% / 2.5% / 1% / 0.5% all score identically, so
# the band was never doing work. Figures come from deterministic tools over XBRL, so an answer
# either matches to the cent or misses by a mile; the tolerance only absorbs how gold is printed.

# An EDGAR accession (CIK-10 / year-2 / sequence-6) is a citation, not a figure, but _nums
# reads its hyphens as minus signs: 0001065280-24-000030 -> [1065280, -24, -30], and the
# x100 / /100 expansion turns -24 into -0.24. With the |gold| < 5 absolute band that is
# enough to score a WRONG small negative answer correct (gold -0.23 vs an answer of -0.50
# passes purely on the accession's year digits). Verified across the saved runs: every
# accession-shaped token in the real data is exactly 10-2-6 (317/317), and no real
# financial figure can take that shape, so removing them only ever drops false candidates.
_ACCESSION_RX = re.compile(r"\b\d{10}-\d{2}-\d{6}\b")


# The model's DECLARED final answer. Scanning the whole reply for numbers makes every incidental
# figure a candidate — the fiscal year ("fiscal 2018" -> 2018), the form type ("10-K" -> 10, and
# /100 -> 0.1, x100 -> 1000), digits in a company name ("3M" -> 3), the citation's accession — and
# _has_number_match accepts if ANY candidate lands near gold, so each one is a free lottery ticket
# (measured: 9.3 candidates per answer, only one of which is the answer). Reading a declared slot
# instead is how GSM8K (#### marker) and MATH (\boxed{}) are graded: you stop enumerating noise
# sources and exclude everything that is not the answer. It also makes the x100 / /100 rescaling
# safe — rescaling ONE true value is the feature; rescaling nine incidental ones was the hole.
_ANSWER_SLOT_RX = re.compile(r"^\s*ANSWER:\s*(.+?)\s*$", re.I | re.M)


def _answer_slot(text):
    """The declared answer, or None when the model didn't emit the slot (callers fall back to the
    whole reply, which is the noisy path — track how often that happens)."""
    m = _ANSWER_SLOT_RX.findall(text or "")
    return m[-1] if m else None


def _nums(s):
    # a Unicode dash used as a NEGATIVE SIGN (–3.70, −5) — but not a range (2019–2020) — must be
    # read as minus, else a correctly-signed negative answer scores as a sign-flipped mismatch.
    s = _ACCESSION_RX.sub(" ", s or "")
    s = re.sub(r"(?<!\d)[‒–—−](?=\d)", "-", s or "")
    out = []
    for x in re.findall(r"-?[\d,]+(?:\.\d+)?", s):
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
        # Small ratios (0.79, 1.06). A percentage tolerance collapses as gold approaches zero —
        # 2.5% of 0.01 is 0.00025, stricter than the two decimals gold is even printed to — so a
        # fixed band takes over, sized to gold's own rounding: FinanceBench prints ratio golds to
        # one or two decimals, and $0.40 is 0.389 rounded. The band must not cross zero, though:
        # at gold -0.02 it spans +-250% of the target, and a ratio of the opposite sign is a
        # different answer in kind, not a near miss — profit vs loss, cash collected before paying
        # vs after. Requiring the same sign costs nothing (no verdict on the saved 55 changes) and
        # removes that.
        if (abs(target) < 5 and abs(v - target) <= 0.05
                and (v == 0 or target == 0 or (v > 0) == (target > 0))):
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
