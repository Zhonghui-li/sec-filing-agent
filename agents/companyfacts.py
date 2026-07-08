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
import re
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
    "inventory":           (["InventoryNet", "InventoryFinishedGoodsNetOfReserves"], "instant"),
    "current_assets":      (["AssetsCurrent"], "instant"),
    "current_liabilities": (["LiabilitiesCurrent"], "instant"),
    "long_term_debt":      (["LongTermDebtNoncurrent", "LongTermDebt"], "instant"),
    # added after FinanceBench validation flagged these as the top missing line items:
    "capex":               (["PaymentsToAcquirePropertyPlantAndEquipment",
                             "PaymentsToAcquireProductiveAssets",
                             "PaymentsForCapitalImprovements"], "duration"),
    "depreciation_amortization": (["DepreciationDepletionAndAmortization",
                             "DepreciationAmortizationAndAccretionNet",
                             "DepreciationAndAmortization"], "duration"),
    "ppe_net":             (["PropertyPlantAndEquipmentNet"], "instant"),
    "operating_cash_flow": (["NetCashProvidedByUsedInOperatingActivities",
                             "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
                            "duration"),
    "accounts_receivable": (["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"], "instant"),
    "interest_expense":    (["InterestExpense", "InterestExpenseDebt",
                             "InterestExpenseNonoperating"], "duration"),
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


# --- name -> CIK (delisted / renamed issuers) --------------------------------------------------
# company_tickers.json only lists CURRENT issuers, so a delisted ticker (Activision's ATVI) or a
# stale one (Square's SQ, now Block/XYZ) doesn't resolve. A CIK never changes across a rename or
# delisting, so we fall back to SEC's cik-lookup-data.txt, which maps every company NAME — including
# FORMER names — to its CIK. The agent passes the company name when the ticker isn't a live listing.
_CIK_LOOKUP_URL = "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"
_CIK_LOOKUP_TTL_DAYS = 30
_SUFFIX_RE = re.compile(r"\b(INC|CORP|CORPORATION|CO|COMPANY|LTD|LLC|LP|PLC|HOLDINGS|GROUP|THE)\b")


def _normalize_name(s):
    s = _SUFFIX_RE.sub(" ", re.sub(r"[.,]", " ", s.strip().upper()))
    return re.sub(r"\s+", " ", s).strip()


def _cik_lookup_file():
    p = _CACHE_DIR.parent / "cik-lookup-data.txt"
    fresh = p.exists() and (datetime.now(timezone.utc)
                            - datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
                            ).days <= _CIK_LOOKUP_TTL_DAYS
    if not fresh:
        p.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(_CIK_LOOKUP_URL, headers=_UA)
        p.write_bytes(urllib.request.urlopen(req, timeout=120).read())
    return p


def name_to_cik(name):
    """Resolve a company NAME (incl. former names of renamed/delisted issuers) to a 10-digit CIK
    via SEC's cik-lookup-data.txt. Returns the CIK for a UNIQUE normalized match, else None (an
    ambiguous or unknown name abstains rather than guessing)."""
    target = _normalize_name(name)
    if not target:
        return None
    key = target.split(" ")[0]                      # cheap prefilter (the file is uppercase)
    try:
        path = _cik_lookup_file()
    except Exception:
        return None
    found = set()
    with open(path, encoding="latin-1") as f:
        for line in f:
            if key not in line:
                continue
            parts = line.rstrip("\n").split(":")
            if len(parts) >= 2 and parts[1] and _normalize_name(parts[0]) == target:
                found.add(parts[1].zfill(10))
                if len(found) > 1:
                    return None                     # ambiguous -> abstain
    return next(iter(found)) if found else None


def cik_for(ticker):
    """Resolve to a CIK. Tries the current-issuer ticker map first, then a name lookup that also
    covers delisted/renamed issuers (former names) — so the caller may pass a ticker OR a name."""
    q = ticker.strip()
    return ticker_to_cik_map().get(q.upper()) or name_to_cik(q)


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


def _close(a, b, tol=0.01):
    """Two figures are the 'same' line item if they agree within a relative tolerance (allows
    minor restatement rounding; a component vs its total won't agree, so it's rejected)."""
    hi = max(abs(a), abs(b))
    return hi == 0 or abs(a - b) <= tol * hi


def extract_rows(gaap, ticker, cik):
    """Turn a company's us-gaap facts dict into rows matching data/financials.json's schema."""
    rows = []
    for metric, (tags, kind) in METRICS.items():
        present = [t for t in tags if t in gaap]   # candidate tags, in preference order
        if not present:
            continue
        # merge across tags: preferred tag wins for a period; lower-priority tags fill gap years.
        # Part-2 guard: a lower-priority tag may fill gaps only if, on period-ends it SHARES with
        # the already-merged higher-priority tags, its values agree — otherwise it's a different
        # line (a component, not the same total, e.g. a segment or finished-goods subtotal) and
        # must not fill years. A clean tag switch (no shared periods) passes through, vetted by a
        # human when the tag was added to METRICS.
        merged = {}
        for t in present:
            unit = next(iter(gaap[t]["units"]))     # USD, USD/shares, ...
            vals = annual_values(gaap[t]["units"][unit], kind)
            # A lower-priority tag that is materially SMALLER IN MAGNITUDE than the already-merged
            # (preferred) value on a shared period is a component/subtotal (e.g. finished-goods vs
            # total inventory, or continuing-ops vs total cash flow) — don't let it fill gap years.
            # Magnitude (not signed value) because accounts can be negative (a net loss, negative
            # operating cash flow): |component| <= |total| holds by the accounting identity, signed
            # "<" does not. Equal (same line), larger (a fuller/alternative total, e.g. Block's
            # Revenues vs a partial RevenueFromContract), or non-overlapping (a clean tag switch,
            # e.g. Nike's inventory) is allowed to fill.
            if any(end in merged and abs(info["val"]) < abs(merged[end]["val"])
                   and not _close(merged[end]["val"], info["val"])
                   for end, info in vals.items()):
                continue
            for end, info in vals.items():
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


# --- Part 3: demand-driven miss log ----------------------------------------------------------
# When a REQUESTED metric comes back empty, append it here. This queue (not the fixed FinanceBench
# set) is what drives data-driven METRICS expansion — the long tail is surfaced by real traffic.
_MISS_LOG = _CACHE_DIR.parent / "metric_misses.jsonl"


def log_miss(ticker, metric, fiscal_year=None, reason="metric_absent"):
    """Best-effort append of a 'requested metric unavailable' event; never breaks a request."""
    try:
        _MISS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _MISS_LOG.open("a") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(),
                                "ticker": ticker.strip().upper(), "metric": metric,
                                "fiscal_year": fiscal_year, "reason": reason}) + "\n")
    except Exception:
        pass


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
