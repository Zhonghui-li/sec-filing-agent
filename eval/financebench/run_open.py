"""Full external validation against the FinanceBench open set (150 Q; Patronus AI, CC-BY-NC-4.0).

We did NOT author this benchmark. The set is a third of numeric metric questions and two thirds
domain/novel reasoning — many needing segment, quarterly, or narrative data our XBRL numbers path
doesn't cover. So the RIGHT behavior on those is to ABSTAIN, not guess. We score 3-way and report:

  - numeric accuracy: on questions whose gold answer is a number, split into
      * coverage accuracy (correct / all numeric)         — depressed by out-of-coverage abstains
      * answered accuracy (correct / numeric we attempted) — "when we answer, are we right"
  - hallucinations: we asserted a NUMBER that doesn't match gold instead of abstaining — the
    finance-bar metric (should be ~0)
  - gap list: numeric questions we abstained on, categorized (segment / quarterly / line-item),
    so it tells us what to add next.

Dataset is downloaded on first run (external data, not committed).

Usage: DATABASE_URL=... OPENAI_API_KEY=... GEN_LLM_MODEL=o4-mini python -m eval.financebench.run_open [--limit N]
"""
import argparse
import json
import os
import re
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from eval.financebench.run import _nums, _has_number_match, _abstained

HERE = Path(__file__).resolve().parent
_CACHE = HERE / "_open_cache.jsonl"   # gitignored; external CC-BY-NC data
_URL = ("https://raw.githubusercontent.com/patronus-ai/financebench/main/"
        "data/financebench_open_source.jsonl")


def _load():
    if not _CACHE.exists():
        print(f"Downloading FinanceBench open set -> {_CACHE.name} ...")
        req = urllib.request.Request(_URL, headers={"User-Agent": "sec-filing-agent research"})
        with urllib.request.urlopen(req, timeout=60) as r:
            _CACHE.write_bytes(r.read())
    return [json.loads(l) for l in _CACHE.read_text().splitlines() if l.strip()]


def _gold_is_numeric(answer):
    """The expected answer is a specific figure (starts with a number / $ / paren-negative /
    a leading %). 'No, the company ...' is a judgment, not numeric, even if numbers follow."""
    a = str(answer).strip().lstrip("~≈").strip()
    return bool(re.match(r"^[-(]?\$?-?\d", a))


# specific line items FinanceBench asks for that our METRICS may lack -> what to add next
_LINE_ITEMS = [
    ("capex / capital expenditure", ["capital expenditure", "capex"]),
    ("PP&E / fixed assets", ["fixed asset", "property, plant", "property plant", "pp&e", " ppe"]),
    ("depreciation & amortization", ["depreciation", "amortization", "d&a"]),
    ("free cash flow", ["free cash flow", "fcf"]),
    ("operating cash flow", ["operating cash flow", "cash from operation", "cash flow from operation"]),
    ("interest expense", ["interest expense"]),
    ("income tax", ["income tax", "tax expense", "effective tax", "tax rate"]),
    ("goodwill / intangibles", ["goodwill", "intangible"]),
    ("SG&A / operating expenses", ["sg&a", "selling, general", "operating expense"]),
    ("EBITDA", ["ebitda"]),
    ("shares / per-share", ["shares outstanding", "per share", "book value per share"]),
    ("retained earnings", ["retained earnings"]),
    ("receivables / DSO", ["receivable", "dso", "days sales"]),
]


def _gap_category(question):
    q = question.lower()
    if "segment" in q or "business unit" in q or "division" in q:
        return "STRUCTURAL: segment (dimensioned XBRL — hard)"
    if re.search(r"\bq[1-4]\b", q) or "quarter" in q:
        return "STRUCTURAL: quarterly (10-Q — extendable)"
    # fixable derived metrics — targets for A (add a tool or the expression evaluator)
    if "cash conversion cycle" in q or re.search(r"\bccc\b", q):
        return "derived: cash conversion cycle (add tool)"
    if "ebitda" in q:
        return "derived: EBITDA / EBITDA margin (add tool)"
    if "defined as" in q:
        return "derived: custom formula given (needs expression evaluator)"
    for label, kws in _LINE_ITEMS:
        if any(k in q for k in kws):
            return f"line-item: {label}"
    return "line-item: other/uncategorized"


def score(run_agent, agent, limit=None):
    cases = _load()
    if limit:
        cases = cases[:limit]
    rows = []
    for i, c in enumerate(cases, 1):
        out = run_agent(c["question"], agent=agent)
        abstained = _abstained(out)
        gold_numeric = _gold_is_numeric(c["answer"])
        gold = _nums(c["answer"])
        answer = out["answer"]
        agent_has_num = bool(_nums(answer)) and not abstained

        if abstained:
            verdict = "abstain"
        elif gold_numeric and gold and _has_number_match(answer, gold[0]):
            verdict = "correct"
        elif gold_numeric and agent_has_num:
            verdict = "hallucinated"      # asserted a wrong number instead of abstaining
        elif not gold_numeric and not agent_has_num:
            verdict = "narrative_reply"   # non-numeric Q, no fabricated number (not auto-graded)
        else:
            verdict = "other"
        rows.append({"id": c["financebench_id"], "company": c["company"],
                     "type": c["question_type"], "gold_numeric": gold_numeric,
                     "verdict": verdict, "q": c["question"], "gold": str(c["answer"])[:40],
                     "got": answer[:90].replace("\n", " ")})
        print(f"  [{i:>3}/{len(cases)}] {verdict:13} {c['company'][:14]:14} {c['question'][:60]}")
    return rows


def report(rows):
    print("\n" + "=" * 78 + "\n FINANCEBENCH OPEN-SET RESULTS\n" + "=" * 78)
    numeric = [r for r in rows if r["gold_numeric"]]
    correct = [r for r in numeric if r["verdict"] == "correct"]
    halluc = [r for r in rows if r["verdict"] == "hallucinated"]
    attempted = [r for r in numeric if r["verdict"] in ("correct", "hallucinated")]
    abstained_num = [r for r in numeric if r["verdict"] == "abstain"]

    # split abstains: LEGIT (structural — segment/quarterly data XBRL company-facts can't give,
    # so abstaining is correct) vs FIXABLE (a metric we could compute with the right tool/mapping —
    # our design gap). This separates "true deficiency" from "correctly declined".
    legit_abstain = [r for r in abstained_num if _gap_category(r["q"]).startswith("STRUCTURAL")]
    fixable_abstain = [r for r in abstained_num if r not in legit_abstain]

    n = len(numeric)
    addressable = n - len(legit_abstain)   # questions we SHOULD be able to answer
    print(f"\nTotal: {len(rows)} | numeric-gold: {n} | narrative-gold: {len(rows)-n}")
    print(f"\n-- Numeric questions ({n}) --")
    print(f"  correct        : {len(correct):>3} ({len(correct)/max(n,1):.0%})")
    print(f"  abstain-FIXABLE : {len(fixable_abstain):>3}   <- our design gap (could answer with the right tool)")
    print(f"  abstain-LEGIT   : {len(legit_abstain):>3}   <- correct: needs segment/quarterly data XBRL can't give")
    print(f"  wrong           : {len(halluc):>3}   <- hallucinations (finance bar)")
    # PRIMARY metric: accuracy on what we SHOULD be able to answer (excludes the legitimately-
    # impossible ones, so it isolates OUR deficiency; still counts fixable-abstains against us so
    # it can't be gamed by abstaining).
    print(f"\n  ADDRESSABLE coverage (correct / answerable) : {len(correct)}/{addressable} "
          f"= {len(correct)/max(addressable,1):.0%}   <- PRIMARY")
    print(f"  raw coverage         (correct / all numeric): {len(correct)}/{n} = {len(correct)/max(n,1):.0%}")
    print(f"  [conditional] when we answered: {len(correct)}/{len(attempted)} "
          f"= {len(correct)/max(len(attempted),1):.0%} (reads high because we abstain, not guess)")
    print(f"\n-- Finance bar: hallucinations (wrong number, didn't abstain): {len(halluc)}/{len(rows)} --")
    for r in halluc:
        print(f"      ! {r['company']} — gold {r['gold']} — got: {r['got']}")

    print(f"\n-- Verdicts by question_type --")
    for qt in ("metrics-generated", "domain-relevant", "novel-generated"):
        sub = [r for r in rows if r["type"] == qt]
        print(f"  {qt:18}: {dict(Counter(r['verdict'] for r in sub))}")

    print(f"\n-- Gap list: numeric questions we abstained on ({len(abstained_num)}) --")
    gaps = Counter(_gap_category(r["q"]) for r in abstained_num)
    for cat, n in gaps.most_common():
        print(f"  {n:>3}  {cat}")
    return {"numeric": len(numeric), "correct": len(correct), "hallucinated": len(halluc),
            "addressable": addressable,
            "addressable_coverage": round(len(correct) / max(addressable, 1), 4),
            "raw_coverage": round(len(correct) / max(len(numeric), 1), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    from agents.sec_agent import build_agent, run_agent
    agent = build_agent()
    rows = score(run_agent, agent, limit=args.limit)
    summary = report(rows)
    (HERE / "_open_results.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {HERE/'_open_results.json'}")
    # append this run to the history log -> the coverage progression is tracked and reproducible
    record = {"ts": datetime.now(timezone.utc).isoformat(),
              "model": os.environ.get("GEN_LLM_MODEL", "gpt-4o-mini"),
              "reasoning_effort": os.environ.get("REASONING_EFFORT"),
              "limit": args.limit, "db": bool(os.environ.get("DATABASE_URL")), **summary}
    with (HERE / "runs_log.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"logged run -> {HERE/'runs_log.jsonl'} (addressable coverage {summary['addressable_coverage']:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
