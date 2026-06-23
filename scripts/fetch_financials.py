"""Fetch exact annual financials from SEC EDGAR XBRL (companyfacts API).

This is the ground-truth numeric layer: every figure the agent reports must come
from here (a tool), never from the LLM reading a number out of retrieved text.
companyfacts returns each us-gaap fact with its period, unit, and source form, so
values are exact and auditable.

Companies use different us-gaap tags for the "same" metric (esp. banks like JPM),
so each canonical metric maps to a list of candidate tags; we take the first that
exists. Flow metrics (revenue, net income) are full-year durations; balance-sheet
metrics (assets, equity) are point-in-time instants.

Usage:  python scripts/fetch_financials.py   ->   data/financials.json
"""
import json
import time
from datetime import date
from pathlib import Path

import httpx

UA = {"User-Agent": "Zhonghui Li lizhonghui923@gmail.com"}  # SEC requires a contact
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "JPM", "TSLA", "KO"]

# canonical metric -> candidate us-gaap tags (first present wins) + kind
METRICS = {
    "revenue":            (["RevenueFromContractWithCustomerExcludingAssessedTax",
                            "Revenues", "RevenueFromContractWithCustomerIncludingAssessedTax",
                            "SalesRevenueNet"], "duration"),
    "net_income":         (["NetIncomeLoss"], "duration"),
    "operating_income":   (["OperatingIncomeLoss"], "duration"),
    "gross_profit":       (["GrossProfit"], "duration"),
    "rd_expense":         (["ResearchAndDevelopmentExpense"], "duration"),
    "eps_diluted":        (["EarningsPerShareDiluted"], "duration"),
    "total_assets":       (["Assets"], "instant"),
    "total_liabilities":  (["Liabilities"], "instant"),
    "stockholders_equity": (["StockholdersEquity",
                            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"], "instant"),
    "cash":               (["CashAndCashEquivalentsAtCarryingValue"], "instant"),
    # added for FinanceBench-style derived metrics (COGS%, DPO, payout ratio, ROA, current ratio):
    "cost_of_revenue":    (["CostOfRevenue", "CostOfGoodsAndServicesSold",
                            "CostOfGoodsSold"], "duration"),
    "dividends_paid":     (["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"], "duration"),
    "accounts_payable":   (["AccountsPayableCurrent",
                            "AccountsPayableTradeCurrent"], "instant"),
    "inventory":          (["InventoryNet"], "instant"),
    "current_assets":     (["AssetsCurrent"], "instant"),
    "current_liabilities": (["LiabilitiesCurrent"], "instant"),
    "long_term_debt":     (["LongTermDebtNoncurrent", "LongTermDebt"], "instant"),
}

OUT = Path(__file__).resolve().parent.parent / "data" / "financials.json"


def ticker_to_cik():
    r = httpx.get("https://www.sec.gov/files/company_tickers.json", headers=UA, timeout=30)
    return {v["ticker"]: str(v["cik_str"]).zfill(10) for v in r.json().values()}


def annual_values(units, kind):
    """Return {fiscal_year_end -> {val, unit, accn, start}} for 10-K annual facts."""
    out = {}
    for u in units:
        if u.get("form") != "10-K":
            continue
        end = u["end"]
        if kind == "duration":
            if "start" not in u:
                continue
            days = (date.fromisoformat(end) - date.fromisoformat(u["start"])).days
            if not (350 <= days <= 380):   # full year only (drop quarters/stubs)
                continue
        else:  # instant (balance sheet) — fact has no start
            if "start" in u:
                continue
        # keep the latest-filed value per fiscal-year-end (restatements -> newest accn)
        prev = out.get(end)
        if prev is None or u.get("accn", "") > prev["accn"]:
            out[end] = {"val": u["val"], "accn": u.get("accn", "")}
    return out


def main():
    cik_map = ticker_to_cik()
    rows = []
    for tk in TICKERS:
        cik = cik_map[tk]
        r = httpx.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                      headers=UA, timeout=90)
        gaap = r.json()["facts"].get("us-gaap", {})
        found = {}
        for metric, (tags, kind) in METRICS.items():
            present = [t for t in tags if t in gaap]   # candidate tags, in preference order
            if not present:
                continue
            # merge across tags: companies migrate the same metric between tags over
            # the years (e.g. NVDA revenue moved to "Revenues"); preferred tag wins
            # for a given period, lower-priority tags fill the gap years.
            merged = {}
            for t in present:
                unit = next(iter(gaap[t]["units"]))      # USD, USD/shares, ...
                for end, info in annual_values(gaap[t]["units"][unit], kind).items():
                    if end not in merged:
                        merged[end] = {"val": info["val"], "accn": info["accn"],
                                       "tag": t, "unit": unit}
            for end, info in merged.items():
                rows.append({"ticker": tk, "cik": cik, "metric": metric,
                             "us_gaap_tag": info["tag"], "period_end": end,
                             "fiscal_year": int(end[:4]), "value": info["val"],
                             "unit": info["unit"], "accession": info["accn"]})
            found[metric] = present
        missing = [m for m in METRICS if m not in found]
        print(f"{tk} (CIK {cik}): {len(found)}/{len(METRICS)} metrics"
              + (f"  MISSING: {missing}" if missing else ""))
        time.sleep(0.2)   # be polite to SEC

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {len(rows)} (ticker,metric,year) rows -> {OUT}")


if __name__ == "__main__":
    main()
