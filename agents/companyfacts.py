"""On-demand exact financials from SEC EDGAR XBRL (companyfacts) — the live analogue of the
offline data/financials.json build (scripts/fetch_financials.py). Same extraction, so a company
fetched dynamically matches the curated set:

  - candidate us-gaap tag mapping (first present wins, merged across tags for gap years),
  - duration vs instant period handling (a flow like revenue is a full-year duration; a
    balance-sheet item like assets is a point-in-time instant),
  - restatement dedup (latest-filed accession per fiscal-year-end).

Numbers only — this is the ground-truth numeric layer; the agent never reads a figure from text.
Results are cached on disk (per company) so a ticker is fetched from SEC at most once per TTL.
"""
import json
import os
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

# SEC requires a User-Agent that identifies the caller with a contact.
_UA = {"User-Agent": os.environ.get("SEC_USER_AGENT", "Zhonghui Li lizhonghui923@gmail.com")}

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "companyfacts"
_CACHE_TTL_DAYS = 7

# canonical metric -> (candidate us-gaap tags [first present wins], period kind).
# Companies use different tags for the "same" metric (esp. banks like JPM), and migrate tags
# across years, so each metric maps to a fallback list. Kept here as the single source of truth
# (scripts/fetch_financials.py imports it).
METRICS = {
    "revenue":             (["RevenueFromContractWithCustomerExcludingAssessedTax",
                             "Revenues", "RevenueFromContractWithCustomerIncludingAssessedTax",
                             "RevenuesNetOfInterestExpense", "SalesRevenueNet"], "duration"),
    "net_income":          (["NetIncomeLoss"], "duration"),
    "operating_income":    (["OperatingIncomeLoss"], "duration"),
    "gross_profit":        (["GrossProfit"], "duration"),
    "rd_expense":          (["ResearchAndDevelopmentExpense"], "duration"),
    "eps_diluted":         (["EarningsPerShareDiluted"], "duration"),
    "total_assets":        (["Assets"], "instant"),
    "total_liabilities":   (["Liabilities"], "instant"),
    "stockholders_equity": (["StockholdersEquity",
                             "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
                            "instant"),
    "cash":                (["CashAndCashEquivalentsAtCarryingValue"], "instant"),
    "cost_of_revenue":     (["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
                            "duration"),
    "dividends_paid":      (["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"], "duration"),
    "accounts_payable":    (["AccountsPayableCurrent", "AccountsPayableTradeCurrent"], "instant"),
    "inventory":           (["InventoryNet"], "instant"),
    "current_assets":      (["AssetsCurrent"], "instant"),
    "current_liabilities": (["LiabilitiesCurrent"], "instant"),
    "long_term_debt":      (["LongTermDebtNoncurrent", "LongTermDebt"], "instant"),
}


def _get_json(url, timeout=90):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# --- ticker -> CIK ---------------------------------------------------------------------------
_ticker_map = None


def ticker_to_cik_map():
    """{TICKER -> 10-digit zero-padded CIK} from SEC's company_tickers.json (cached in-process)."""
    global _ticker_map
    if _ticker_map is None:
        data = _get_json(_TICKERS_URL, timeout=30)
        _ticker_map = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}
    return _ticker_map


def cik_for(ticker):
    return ticker_to_cik_map().get(ticker.strip().upper())


# --- extraction (identical logic to the offline script) --------------------------------------
def annual_values(units, kind):
    """{fiscal_year_end -> {val, accn}} for 10-K annual facts. A duration fact must span a full
    year (350-380 days, dropping quarters/stubs); an instant fact has no start (balance sheet).
    Restatements resolved by keeping the latest-filed accession per period-end."""
    out = {}
    for u in units:
        if u.get("form") != "10-K":
            continue
        end = u["end"]
        if kind == "duration":
            if "start" not in u:
                continue
            days = (date.fromisoformat(end) - date.fromisoformat(u["start"])).days
            if not (350 <= days <= 380):
                continue
        else:  # instant (balance sheet) — fact has no start
            if "start" in u:
                continue
        prev = out.get(end)
        if prev is None or u.get("accn", "") > prev["accn"]:
            out[end] = {"val": u["val"], "accn": u.get("accn", "")}
    return out


def extract_rows(gaap, ticker, cik):
    """Turn a company's us-gaap facts dict into rows matching data/financials.json's schema."""
    rows = []
    for metric, (tags, kind) in METRICS.items():
        present = [t for t in tags if t in gaap]   # candidate tags, in preference order
        if not present:
            continue
        # merge across tags: preferred tag wins for a period; lower-priority tags fill gap years
        merged = {}
        for t in present:
            unit = next(iter(gaap[t]["units"]))     # USD, USD/shares, ...
            for end, info in annual_values(gaap[t]["units"][unit], kind).items():
                if end not in merged:
                    merged[end] = {"val": info["val"], "accn": info["accn"], "tag": t, "unit": unit}
        for end, info in merged.items():
            rows.append({"ticker": ticker, "cik": cik, "metric": metric,
                         "us_gaap_tag": info["tag"], "period_end": end,
                         "fiscal_year": int(end[:4]), "value": info["val"],
                         "unit": info["unit"], "accession": info["accn"]})
    return rows


def fetch_facts(cik):
    """The raw us-gaap facts dict for a CIK (used by the offline build too)."""
    return _get_json(_FACTS_URL.format(cik=cik))["facts"].get("us-gaap", {})


# --- on-demand, cached -----------------------------------------------------------------------
_rows_mem = {}   # TICKER -> rows (in-process)


def _disk_path(ticker):
    return _CACHE_DIR / f"{ticker}.json"


def _load_disk(ticker):
    p = _disk_path(ticker)
    if not p.exists():
        return None
    try:
        blob = json.loads(p.read_text())
        fetched = datetime.fromisoformat(blob["fetched"])
        if (datetime.now(timezone.utc) - fetched).days > _CACHE_TTL_DAYS:
            return None
        return blob["rows"]
    except Exception:
        return None


def _save_disk(ticker, rows):
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _disk_path(ticker).write_text(json.dumps(
        {"fetched": datetime.now(timezone.utc).isoformat(), "rows": rows}))


def company_rows(ticker):
    """Financials rows for ANY public company, fetched live from SEC XBRL and cached. Same schema
    as data/financials.json. Returns [] for an unknown ticker or a fetch failure (the caller then
    abstains) — never guesses."""
    tk = ticker.strip().upper()
    if tk in _rows_mem:
        return _rows_mem[tk]
    cached = _load_disk(tk)
    if cached is not None:
        _rows_mem[tk] = cached
        return cached
    cik = cik_for(tk)
    if not cik:
        _rows_mem[tk] = []
        return []
    try:
        gaap = fetch_facts(cik)
    except Exception:
        return []                      # transient failure — don't cache
    rows = extract_rows(gaap, tk, cik)
    if rows:
        _save_disk(tk, rows)
    _rows_mem[tk] = rows
    return rows
