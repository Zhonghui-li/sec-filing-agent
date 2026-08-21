"""Build a tool-ROUTING SFT dataset: (finance question -> the correct tool call).

We KNOW the right tool for a templated question (that's the whole point of the deterministic tools),
so the gold is exact and free to generate — no LLM, no API. Each example is
  {"question": str, "tool": {"name": str, "arguments": {...}}, "category": str}
Output: finetune/routing_train.jsonl / routing_heldout.jsonl. A few companies are held out entirely
(unseen-company generalization) plus a random 10% split.

Usage: python -m finetune.build_dataset
"""
import json
import random
from pathlib import Path

random.seed(3407)
HERE = Path(__file__).resolve().parent

# --- pools -------------------------------------------------------------------------------------
TRAIN_COMPANIES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMZN": "Amazon", "GOOGL": "Alphabet",
    "META": "Meta", "TSLA": "Tesla", "KO": "Coca-Cola", "PEP": "PepsiCo", "WMT": "Walmart",
    "COST": "Costco", "F": "Ford", "GM": "General Motors", "DIS": "Disney", "NKE": "Nike",
    "MCD": "McDonald's", "SBUX": "Starbucks", "INTC": "Intel", "AMD": "AMD", "CSCO": "Cisco",
    "ORCL": "Oracle", "CRM": "Salesforce", "PFE": "Pfizer", "JNJ": "Johnson & Johnson",
    "PG": "Procter & Gamble", "XOM": "ExxonMobil", "CVX": "Chevron", "BA": "Boeing",
    "CAT": "Caterpillar", "HD": "Home Depot",
}
# held out ENTIRELY -> tests routing on companies never seen in training
HELDOUT_COMPANIES = {"NFLX": "Netflix", "ADBE": "Adobe", "QCOM": "Qualcomm", "T": "AT&T"}
BANKS = {"JPM": "JPMorgan", "BAC": "Bank of America", "WFC": "Wells Fargo", "C": "Citigroup",
         "GS": "Goldman Sachs"}
YEARS = [2020, 2021, 2022, 2023, 2024]

# (human phrasings, canonical metric) for get_financials
FINANCIALS = [
    (["revenue", "total revenue", "sales"], "revenue"),
    (["net income", "net profit", "bottom line"], "net_income"),
    (["operating income", "operating profit"], "operating_income"),
    (["gross profit"], "gross_profit"),
    (["total assets"], "total_assets"),
    (["stockholders equity", "shareholders equity"], "stockholders_equity"),
    (["cash and cash equivalents", "cash"], "cash"),
    (["long-term debt"], "long_term_debt"),
    (["inventory"], "inventory"),
    (["cost of revenue", "COGS"], "cost_of_revenue"),
    (["capital expenditures", "capex"], "capex"),
    (["R&D expense", "research and development expense"], "rd_expense"),
    (["diluted EPS", "diluted earnings per share"], "eps_diluted"),
    (["EBITDA"], "ebitda"),
    (["free cash flow", "FCF"], "free_cash_flow"),
    (["operating cash flow", "cash from operations"], "operating_cash_flow"),
]
# (phrasings, canonical ratio) for get_ratio
RATIOS = [
    (["gross margin"], "gross_margin"), (["operating margin"], "operating_margin"),
    (["net margin", "net profit margin"], "net_margin"), (["EBITDA margin"], "ebitda_margin"),
    (["ROE", "return on equity"], "roe"), (["ROA", "return on assets"], "roa"),
    (["current ratio"], "current_ratio"), (["quick ratio"], "quick_ratio"),
    (["debt-to-equity", "debt to equity ratio"], "debt_to_equity"),
    (["DPO", "days payable outstanding"], "dpo"), (["DSO", "days sales outstanding"], "dso"),
    (["DIO", "days inventory outstanding"], "dio"),
    (["cash conversion cycle", "CCC"], "cash_conversion_cycle"),
    (["asset turnover"], "asset_turnover"), (["inventory turnover"], "inventory_turnover"),
    (["interest coverage", "interest coverage ratio"], "interest_coverage"),
    (["effective tax rate"], "effective_tax_rate"), (["payout ratio"], "payout_ratio"),
]
# metrics that make sense for a YoY growth question
GROWTH_METRICS = [(["revenue", "sales"], "revenue"), (["net income"], "net_income"),
                  (["operating income"], "operating_income"), (["EPS"], "eps_diluted"),
                  (["total assets"], "total_assets"), (["gross profit"], "gross_profit")]

QUAL_TOPICS = [
    ("main risk factors", "risk factors"), ("competitive strategy", "competition strategy"),
    ("what management attributes revenue growth to", "revenue growth drivers"),
    ("supply chain risks", "supply chain risk"), ("regulatory risks", "regulatory risk"),
    ("its business segments", "business segments overview"),
]


def _call(name, **args):
    return {"name": name, "arguments": {k: v for k, v in args.items() if v is not None}}


def _sample(companies, n):
    items = list(companies.items())
    return [random.choice(items) for _ in range(n)]


# --- generators (each returns [{question, tool, category}]) ------------------------------------
def gen_financials(companies, n):
    out = []
    for tk, name in _sample(companies, n):
        phrases, metric = random.choice(FINANCIALS)
        m, y = random.choice(phrases), random.choice(YEARS)
        t = random.choice([
            f"What was {name}'s {m} in FY{y}?",
            f"How much {m} did {name} report in {y}?",
            f"{name}'s {m} for fiscal {y}?",
            f"Give me {name}'s {m} ({y}).",
        ])
        args = {"ticker": tk, "metric": metric}
        if "fiscal" in t or f"FY{y}" in t or f"({y})" in t or f"in {y}" in t:
            args["fiscal_year"] = y
        out.append({"question": t, "tool": _call("get_financials", **args), "category": "get_financials"})
    return out


def gen_ratio(companies, n):
    out = []
    for tk, name in _sample(companies, n):
        phrases, ratio = random.choice(RATIOS)
        r, y = random.choice(phrases), random.choice(YEARS)
        t = random.choice([
            f"What is {name}'s {r} for FY{y}?",
            f"Calculate {name}'s {r} in {y}.",
            f"{name} {r}, fiscal {y}?",
            f"What was the {r} of {name} in FY{y}?",
        ])
        out.append({"question": t, "tool": _call("get_ratio", ratio=ratio, ticker=tk, fiscal_year=y),
                    "category": "get_ratio"})
    return out


def gen_growth(companies, n):
    out = []
    for tk, name in _sample(companies, n):
        phrases, metric = random.choice(GROWTH_METRICS)
        m, y = random.choice(phrases), random.choice(YEARS)
        t = random.choice([
            f"How did {name}'s {m} change year over year into FY{y}?",
            f"What was {name}'s YoY {m} growth in {y}?",
            f"{name} {m} year-over-year change for {y}?",
        ])
        out.append({"question": t, "tool": _call("get_growth", metric=metric, ticker=tk, fiscal_year=y),
                    "category": "get_growth"})
    return out


def gen_compute_formula(companies, n):
    formulas = [
        ("its operating income plus D&A, divided by revenue", "(operating_income + depreciation_amortization) / revenue"),
        ("(net income + interest expense) divided by total assets", "(net_income + interest_expense) / total_assets"),
        ("cost of revenue as a fraction of total assets", "cost_of_revenue / total_assets"),
        ("the 2-year revenue CAGR ending", "(revenue / prev(revenue, 2)) ** (1/2) - 1"),
    ]
    out = []
    for tk, name in _sample(companies, n):
        desc, expr = random.choice(formulas)
        y = random.choice(YEARS)
        t = f"Compute {name}'s {desc} for FY{y}."
        out.append({"question": t, "tool": _call("compute_formula", expression=expr, ticker=tk, fiscal_year=y),
                    "category": "compute_formula"})
    return out


def gen_abstain(companies, n):
    out = []
    # banks have no gross profit / COGS
    for tk, name in _sample(BANKS, n // 3):
        y = random.choice(YEARS)
        metric = random.choice(["gross profit", "cost of goods sold"])
        out.append({"question": f"What was {name}'s {metric} in FY{y}?",
                    "tool": _call("abstain", reason="not_reported",
                                  detail=f"a bank has no COGS, so no {metric}"), "category": "abstain"})
    # out-of-scope: advice / forecast / real-time
    for tk, name in _sample(companies, n // 3):
        t, detail = random.choice([
            (f"Should I buy {name} stock?", "investment advice is out of scope"),
            (f"What will {name}'s revenue be next year?", "a forecast is out of scope"),
            (f"What is {name}'s stock price right now?", "real-time market data is out of scope"),
        ])
        out.append({"question": t, "tool": _call("abstain", reason="off_topic", detail=detail),
                    "category": "abstain"})
    # nonexistent / future fiscal year
    for tk, name in _sample(companies, n - 2 * (n // 3)):
        y = random.choice([2035, 2040, 1985])
        out.append({"question": f"What was {name}'s revenue in FY{y}?",
                    "tool": _call("abstain", reason="year_unavailable",
                                  detail=f"FY{y} is not available"), "category": "abstain"})
    return out


def gen_search(companies, n):
    out = []
    for tk, name in _sample(companies, n):
        topic, query = random.choice(QUAL_TOPICS)
        t = random.choice([
            f"What are {name}'s {topic}?",
            f"Describe {name}'s {topic}.",
            f"According to its filings, what are {name}'s {topic}?",
        ])
        out.append({"question": t, "tool": _call("search_filings", query=query, ticker=tk),
                    "category": "search_filings"})
    return out


def build():
    tr = (gen_financials(TRAIN_COMPANIES, 700) + gen_ratio(TRAIN_COMPANIES, 700)
          + gen_growth(TRAIN_COMPANIES, 400) + gen_compute_formula(TRAIN_COMPANIES, 200)
          + gen_abstain(TRAIN_COMPANIES, 400) + gen_search(TRAIN_COMPANIES, 300))
    # dedup by question
    seen, rows = set(), []
    for r in tr:
        if r["question"] not in seen:
            seen.add(r["question"])
            rows.append(r)
    random.shuffle(rows)
    # held-out: unseen companies across every category + a random 10% of the rest
    held = (gen_financials(HELDOUT_COMPANIES, 80) + gen_ratio(HELDOUT_COMPANIES, 80)
            + gen_growth(HELDOUT_COMPANIES, 40) + gen_abstain(HELDOUT_COMPANIES, 40)
            + gen_search(HELDOUT_COMPANIES, 30))
    held = [r for r in held if r["question"] not in seen]
    cut = int(len(rows) * 0.10)
    heldout = held + rows[:cut]
    train = rows[cut:]

    (HERE / "routing_train.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n")
    (HERE / "routing_heldout.jsonl").write_text("\n".join(json.dumps(r) for r in heldout) + "\n")

    from collections import Counter
    print(f"train {len(train)} | heldout {len(heldout)} "
          f"(unseen-company {len(held)} + random {cut})")
    print("train by category:", dict(Counter(r["category"] for r in train)))
    print("\n--- 4 samples ---")
    for r in random.sample(train, 4):
        print(f"Q: {r['question']}\n  -> {json.dumps(r['tool'])}")


if __name__ == "__main__":
    build()
