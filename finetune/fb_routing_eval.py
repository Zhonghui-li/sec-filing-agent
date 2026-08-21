"""Functional routing eval on REAL FinanceBench questions (the metrics-generated set).

We don't hand-label a "gold tool call". Instead, given a model's PREDICTED tool call for a real
FinanceBench question, we EXECUTE it against the real tools and check whether the result matches
FinanceBench's gold ANSWER. So it measures what we actually care about: does the model route to a
call that PRODUCES THE RIGHT ANSWER? Robust to a different-but-correct expression, and honest
(real human-written questions + real gold answers). Routing to abstain/search on a numeric question
yields no number -> counted wrong.

The model runs in the notebook (on the fine-tuned / base model); it saves predictions as
  [{"question": ..., "tool": {"name": ..., "arguments": {...}}}, ...]
This scorer (repo, free — SEC XBRL only, no OpenAI) executes + scores them against the gold.

Usage:
    python -m finetune.fb_routing_eval --self-test          # validate the engine on a few gold calls
    python -m finetune.fb_routing_eval preds.json           # score a model's predictions file
"""
import argparse
import json
import sys
from pathlib import Path

from agents import finance_tools
from eval.financebench.run import _nums

# FinanceBench golds are often in MILLIONS ($1577) while get_financials returns raw dollars
# ($1,577,000,000); a ratio may be a % (16.5) or a decimal (0.165). So match a gold against each
# output number under money/percent scale variants. CCC-style convention gaps still (correctly) miss.
_SCALES = [1, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9, 100, 0.01]


def _num_match(output, gold):
    for v in _nums(output):
        for s in _SCALES:
            x = v * s
            if gold != 0 and abs(x - gold) <= 0.025 * abs(gold):
                return True
            if abs(gold) < 5 and abs(x - gold) <= 0.05:
                return True
    return False

HERE = Path(__file__).resolve().parent
_FB_CACHE = HERE.parent / "eval" / "financebench" / "_open_cache.jsonl"

# only these produce a number; routing a numeric question to anything else -> no number -> wrong.
_TOOLS = {
    "get_financials": finance_tools.get_financials,
    "get_ratio": finance_tools.get_ratio,
    "get_growth": finance_tools.get_growth,
    "compute_formula": finance_tools.compute_formula,
}


def metrics_questions():
    """The 50 metrics-generated FinanceBench questions with their gold numeric answers."""
    rows = [json.loads(l) for l in _FB_CACHE.read_text().splitlines() if l.strip()]
    return [{"question": r["question"], "gold": str(r["answer"])}
            for r in rows if r.get("question_type") == "metrics-generated"]


def _execute(tool):
    name, args = tool.get("name"), tool.get("arguments", {})
    fn = _TOOLS.get(name)
    if not fn:
        return ""                          # abstain / search_filings -> no number on a numeric Q
    try:
        return str(fn(**args))
    except Exception as e:                 # bad args -> wrong routing
        return f"(error: {str(e)[:60]})"


def score(preds):
    """preds: [{question, tool:{name,arguments}, gold}]. Returns (accuracy, per-item rows)."""
    rows = []
    for p in preds:
        out = _execute(p["tool"])
        gold = _nums(p["gold"])
        ok = bool(gold) and _num_match(out, gold[0])
        rows.append({"question": p["question"][:70], "tool": p["tool"]["name"],
                     "gold": p["gold"], "output": out[:80].replace("\n", " "), "correct": ok})
    acc = sum(r["correct"] for r in rows) / max(len(rows), 1)
    return acc, rows


# A few hand-written CORRECT tool calls (question -> the right call), to validate the engine and
# confirm these real questions are answerable by our tools (the routing "ceiling").
_SELF_TEST = [
    ("FY2018 capital expenditure amount (in USD millions) for 3M",
     {"name": "get_financials", "arguments": {"ticker": "3M", "metric": "capex", "fiscal_year": 2018}}),
    ("Amazon's year-over-year change in revenue from FY2016 to FY2017",
     {"name": "get_growth", "arguments": {"metric": "revenue", "ticker": "AMZN", "fiscal_year": 2017}}),
    ("FY2022 unadjusted EBITDA % margin for PepsiCo",
     {"name": "get_ratio", "arguments": {"ratio": "ebitda_margin", "ticker": "PEP", "fiscal_year": 2022}}),
    ("FY2019 cash conversion cycle (CCC) for General Mills",
     {"name": "get_ratio", "arguments": {"ratio": "cash_conversion_cycle", "ticker": "GIS", "fiscal_year": 2019}}),
]


def self_test():
    qs = metrics_questions()
    preds = []
    for needle, tool in _SELF_TEST:
        gold = next((q["gold"] for q in qs if needle in q["question"]), None)
        if gold is None:
            print(f"  ! could not find question for: {needle}")
            continue
        preds.append({"question": needle, "tool": tool, "gold": gold})
    acc, rows = score(preds)
    print(f"self-test accuracy (gold calls should ~all pass): {acc:.0%}")
    for r in rows:
        print(f"  [{'OK ' if r['correct'] else 'XX '}] gold {r['gold']:>10} | {r['tool']} | {r['output']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="?", help="model predictions JSON")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test or not args.preds:
        self_test()
        return 0
    preds = json.loads(Path(args.preds).read_text())
    # attach gold by matching the question text
    qs = {q["question"]: q["gold"] for q in metrics_questions()}
    for p in preds:
        p["gold"] = qs.get(p["question"], p.get("gold", ""))
    acc, rows = score(preds)
    print(f"\nROUTING accuracy (executed call produces the gold answer): {acc:.0%} "
          f"({sum(r['correct'] for r in rows)}/{len(rows)})")
    for r in rows:
        if not r["correct"]:
            print(f"  MISS [{r['tool']}] gold {r['gold']} | got: {r['output']} | {r['question']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
