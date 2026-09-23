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
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

# SEC requires a User-Agent that identifies the caller with a contact.
_UA = {"User-Agent": os.environ.get("SEC_USER_AGENT", "Zhonghui Li lizhonghui923@gmail.com")}

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

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
    "dividends_per_share": (["CommonStockDividendsPerShareDeclared",
                             "CommonStockDividendsPerShareCashPaid"], "duration"),
    "accounts_payable":    (["AccountsPayableCurrent", "AccountsPayableTradeCurrent"], "instant"),
    "inventory":           (["InventoryNet", "InventoryFinishedGoodsNetOfReserves"], "instant"),
    "current_assets":      (["AssetsCurrent"], "instant"),
    "current_liabilities": (["LiabilitiesCurrent"], "instant"),
    "long_term_debt":      (["LongTermDebtNoncurrent", "LongTermDebt"], "instant"),
    # added after FinanceBench validation flagged these as the top missing line items:
    "capex":               (["PaymentsToAcquirePropertyPlantAndEquipment",
                             "PaymentsToAcquireProductiveAssets",
                             "PaymentsToAcquireOtherProductiveAssets",
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
    # income-tax components (added to enable effective_tax_rate; also usable in compute_formula):
    "income_tax_expense":  (["IncomeTaxExpenseBenefit"], "duration"),
    "pretax_income":       (["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                             "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
                            "duration"),
}

# Per-share metrics where a later filing's RETROACTIVE STOCK-SPLIT adjustment is the standard,
# comparable basis (not a restatement to warn about). For these we use the current/split-adjusted
# value, not the as-originally-reported (pre-split) one. Everything else defaults to as-reported.
_SPLIT_ADJUSTED_METRICS = {"eps_diluted", "dividends_per_share"}


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


# Well-known RETIRED tickers (delisted via acquisition, or renamed) -> the issuer's SEC name.
# company_tickers.json is current-only, so these no longer resolve by ticker; SEC has no clean
# former-ticker file, so a small curated map covers the famous cases (the long tail still resolves
# if the caller passes the company NAME). The value is resolved through name_to_cik (which validates
# against SEC's data and returns None if ambiguous), so a stale/wrong entry can't mis-resolve.
_RETIRED_TICKERS = {
    # NOTE: only include tickers that are truly RETIRED — never one that has been REASSIGNED to a
    # live issuer (e.g. FB, once Facebook, now trades as First Banks; Facebook is META / CIK1326801).
    # A reassigned ticker resolves via the current ticker map to the WRONG company. Each entry below
    # is verified to resolve to the intended issuer, not a name collision.
    "ATVI": "Activision Blizzard",   # acquired by Microsoft (2023)
    "TWTR": "Twitter",               # taken private / renamed X (2022)
    "XLNX": "Xilinx",                # acquired by AMD (2022)
    "SQ": "Block",                   # renamed from Square (2021); ticker later XYZ
    "FISV": "Fiserv",                # ticker changed to FI (2023)
    "FL": "Foot Locker",             # acquired by DICK'S Sporting Goods (2025)
}


def cik_for(ticker):
    """Resolve to a CIK. Tries the current-issuer ticker map first, then a name lookup that also
    covers delisted/renamed issuers (former names), then a curated retired-ticker map — so the
    caller may pass a live ticker, a company NAME, or a well-known retired ticker (e.g. ATVI)."""
    q = ticker.strip()
    return (ticker_to_cik_map().get(q.upper()) or name_to_cik(q)
            or (name_to_cik(_RETIRED_TICKERS[q.upper()]) if q.upper() in _RETIRED_TICKERS else None))


_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_FOREIGN_ANNUAL_FORMS = {"20-F", "40-F"}
_HOLDCO_RE = re.compile(r"\b(HOLDINGS?|HOLDCO|GROUP|NEW)\b")


def _tight_name(name):
    """A name with corporate-form words and all spacing removed, so "ExxonMobil Holdings Corp" and
    "EXXON MOBIL CORP" collapse to the same key. Deliberately looser than _normalize_name, and used
    only on the predecessor path below — as a general name lookup it would be too loose."""
    return re.sub(r"\s+", "", _HOLDCO_RE.sub(" ", _normalize_name(name)))


def predecessor_cik(cik, name):
    """The registrant holding this one's annual history, or None when there isn't a clear one.

    A holdco reorganization or a reincorporation registers a NEW entity that takes over the ticker
    while the filing history stays behind: SEC's ticker file sends XOM to CIK 2115436 (ExxonMobil
    Holdings Corp, first filed 2026-07-01, zero 10-Ks) while every Exxon 10-K sits under CIK 34088.
    This is the mirror of the retired-ticker case — there a ticker resolved to nothing, here it
    resolves to a live registrant with no history — and both follow from one fact: a ticker is not
    a stable key for a filing history.

    SEC publishes no machine-readable link between the two. The successor's `formerNames` is empty,
    and both the EIN and the state of incorporation change (Exxon's went NJ -> TX). What survives
    the reorganization is the NAME, so candidates are found on a tightened form of it and the tie
    is settled on EVIDENCE rather than on similarity: the candidate that actually filed the annual
    reports. That evidence test is also why foreign private issuers need no special case here —
    their tickers yield no 10-K rows either, but a name search for Toyota or Alibaba turns up no
    other registrant at all.
    """
    key = _tight_name(name)
    if len(key) < 5:                       # too short to discriminate
        return None
    try:
        path = _cik_lookup_file()
    except Exception:
        return None
    prefix = key[:5]                       # cheap prefilter; the tight compare still decides
    found = set()
    with open(path, encoding="latin-1") as f:
        for line in f:
            if prefix not in line:
                continue
            parts = line.rstrip("\n").split(":")
            if len(parts) >= 2 and parts[1] and _tight_name(parts[0]) == key:
                found.add(parts[1].zfill(10))
    ranked = []
    for cand in sorted(found - {cik}):
        try:
            forms = _get_json(_SUBMISSIONS_URL.format(cik=cand),
                              timeout=30)["filings"]["recent"]["form"]
        except Exception:
            continue
        n = sum(1 for f in forms if f.startswith("10-K"))
        if n:
            ranked.append((n, cand))
    ranked.sort(reverse=True)
    if not ranked or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
        return None                        # nobody filed annually, or no clear holder -> abstain
    return ranked[0][1]


def foreign_filer_note(query):
    """A one-line explanation if `query` is a FOREIGN PRIVATE ISSUER (files a 20-F/40-F annual
    report, not a 10-K), else None. A ticker like TM (Toyota) resolves fine, but its annual figures
    aren't in the us-GAAP 10-K path — so a tool can say WHY instead of a misleading 'no company
    found'. Resolve via the ticker map (the current filer), not the name (which can hit an alternate
    registration)."""
    cik = ticker_to_cik_map().get(query.strip().upper())
    if not cik:
        return None
    try:
        forms = set(_get_json(_SUBMISSIONS_URL.format(cik=cik), timeout=30)
                    .get("filings", {}).get("recent", {}).get("form", []))
    except Exception:
        return None
    if "10-K" in forms or not (forms & _FOREIGN_ANNUAL_FORMS):
        return None
    form = "40-F" if "40-F" in forms else "20-F"
    name = fetch_company(cik)[0].rstrip("/") or query
    return (f"{name} is a foreign private issuer — it files a Form {form} annual report, not a "
            f"10-K, so its annual figures aren't in the us-GAAP 10-K XBRL data this tool uses.")


# --- extraction (identical logic to the offline script) --------------------------------------
def _year_end_only(facts, cal=None):
    """Drop balance-sheet instants that aren't the company's year end.

    Having no start date makes a fact an instant, not a year end. A 10-K also carries instants from
    inside the year, and neither `form` nor `fp` separates them — Target's $1,000,000,000 at
    2010-07-31 arrives on a 10-K tagged fp=FY, six months off. Taking the latest end per fiscal year
    then picks the mid-year figure over the real one.

    Two tests, covering different eras because the submissions feed only reaches back about ten
    years. Within the span the feed does cover, the company's own filed period ends are the direct
    evidence: an instant on a date the company never closed a period on is not a period end
    (Walmart's cash at 2012-12-31 sits one month off a January year end, close enough to survive
    any month-based rule). Before that span there is no such list, so fall back on the pattern: a
    company's year ends in the same month every year, give or take a week for 52/53-week
    calendars, so an instant in any other month isn't one. The fallback only applies with enough
    years to establish the pattern and a clear majority behind it; below that, keeping a stray
    beats dropping a real one on a two-point guess.
    """
    known = set(cal._by_end) if cal is not None and cal._by_end else set()
    if known:
        lo, hi = min(known), max(known)
        facts = {end: v for end, v in facts.items() if not (lo <= end <= hi) or end in known}
    if len(facts) < 4:
        return facts
    months = Counter(end[5:7] for end in facts)
    # Count the WINDOW, not the single month. A 52/53-week fiscal year lands in two adjacent months
    # from year to year — Target's ends fall in both January and February — so no single month can
    # hold a majority and a per-month test would decline to filter anything.
    def window(m):
        return {f"{(int(m) + d - 1) % 12 + 1:02d}" for d in (-1, 0, 1)}
    near, n = max(((window(m), sum(months[x] for x in window(m))) for m in months),
                  key=lambda t: t[1])
    if n <= len(facts) / 2:                       # nothing dominant — can't tell, keep everything
        return facts
    return {end: v for end, v in facts.items() if end[5:7] in near}


def annual_values(units, kind, cal=None):
    """{fiscal_year_end -> {val, accn, restated_val, restated_accn}} for 10-K annual facts. A
    duration fact must span a full year (350-380 days, dropping quarters/stubs); an instant fact
    has no start (balance sheet).

    `val` is the figure AS ORIGINALLY REPORTED — from the filing whose OWN fiscal year is that of
    the period (its FY10-K, or a 10-K/A for it: the latest accession among same-fiscal-year facts;
    earliest filing as a fallback). This matches the source filing and keeps a multi-year series on
    one basis (never mixing an original year with a restated one). If a LATER filing re-presented
    the period with a materially different value (a restatement / reclassification), that
    current-basis figure is surfaced as restated_val; otherwise restated_val is None."""
    facts = {}                       # end -> [(accn, val, filing_fy), ...]
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
        facts.setdefault(end, []).append((u.get("accn", ""), u["val"], u.get("fy")))
    if kind == "instant":
        facts = _year_end_only(facts, cal)
    out = {}
    for end, cands in facts.items():
        # The filing this period was FIRST reported in — the one it is the current period of.
        # Matching on `fy == calendar year of the end` instead assumed the company names its
        # fiscal year after the year it ends in; for Target (year ending 2025-02-01 = its FY2024)
        # that matched NEXT year's 10-K, i.e. the comparative column, so `rep` and `latest` became
        # the same filing and a restatement could no longer be detected.
        own = cal.original_accn(end) if cal is not None else None
        if own is not None:
            same = [c for c in cands if c[0] == own]
        else:
            year = int(end[:4])
            same = [c for c in cands if c[2] == year]      # filed by that period's own FY 10-K(/A)
        rep = max(same, key=lambda c: c[0]) if same else min(cands, key=lambda c: c[0])
        latest = max(cands, key=lambda c: c[0])            # most-recent re-presentation (current basis)
        info = {"val": rep[1], "accn": rep[0], "restated_val": None, "restated_accn": None}
        if latest[0] != rep[0] and not _close(latest[1], rep[1]):
            info["restated_val"], info["restated_accn"] = latest[1], latest[0]
        out[end] = info
    return out


def quarterly_values(units, kind):
    """{period_end -> {val, accn}} for DISCRETE quarterly facts from 10-Q filings. A duration flow
    (revenue, net income) is reported both as a 3-month figure and a year-to-date figure in the same
    10-Q; we keep only the discrete quarter (85-95 days), never the YTD. An instant (balance sheet)
    fact is the quarter-end balance. Q4 has no 10-Q — derive it as annual minus Q1+Q2+Q3 if needed.
    Latest accession wins per period-end (as-reported for the quarter)."""
    facts = {}                       # end -> [(accn, val), ...]
    for u in units:
        if u.get("form") != "10-Q":
            continue
        end = u["end"]
        if kind == "duration":
            if "start" not in u:
                continue
            days = (date.fromisoformat(end) - date.fromisoformat(u["start"])).days
            if not (85 <= days <= 95):          # discrete quarter only (drop YTD 6-/9-month spans)
                continue
        else:                                   # instant balance at quarter-end (no start)
            if "start" in u:
                continue
        facts.setdefault(end, []).append((u.get("accn", ""), u["val"]))
    return {end: {"val": max(cands, key=lambda c: c[0])[1],
                  "accn": max(cands, key=lambda c: c[0])[0]}
            for end, cands in facts.items()}


def _rounded_month(end):
    """Month of a period-end, treating a date on the 1st-5th as the prior month (fiscal quarters
    often end on a Saturday that spills a day or two into the next month, e.g. Apple's 2023-04-01
    is really the March/Q2 close)."""
    y, m, d = int(end[:4]), int(end[5:7]), int(end[8:10])
    if d <= 5:
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    return y, m


def _fiscal_period(end, fye_month):
    """(fiscal_year, 'Q1'|'Q2'|'Q3'|'Q4') for a period-end, from the company's fiscal-year-end month.
    Derived from the DATE, not the filing's fy/fp (those tag the FILING and mislabel comparatives)."""
    y, m = _rounded_month(end)
    fy = y + 1 if m > fye_month else y
    start = (fye_month % 12) + 1                     # fiscal year starts the month after year-end
    return fy, f"Q{((m - start) % 12) // 3 + 1}"


def quarterly_rows(gaap, ticker, cik):
    """Discrete quarterly rows (Q1-Q3, from 10-Q) matching extract_rows' schema plus a `quarter`.
    fiscal_year/quarter are derived from each period-end date and the company's fiscal-year-end
    (NOT the fact's fy/fp, which tag the filing and mislabel comparative periods)."""
    # fiscal-year-end month = the most common month of the ANNUAL (10-K) period ends
    ann_months = []
    for metric, (tags, kind) in METRICS.items():
        if kind != "duration":
            continue
        t = next((t for t in tags if t in gaap), None)
        if t:
            unit = next(iter(gaap[t]["units"]))
            ann_months += [_rounded_month(e)[1] for e in annual_values(gaap[t]["units"][unit], kind)]
    fye = Counter(ann_months).most_common(1)[0][0] if ann_months else 12
    cal = fiscal_calendar(cik, gaap)

    rows = []
    for metric, (tags, kind) in METRICS.items():
        present = [t for t in tags if t in gaap]
        if not present:
            continue
        merged = {}
        for t in present:
            unit = next(iter(gaap[t]["units"]))
            for end, info in quarterly_values(gaap[t]["units"][unit], kind).items():
                if end not in merged:
                    merged[end] = {**info, "tag": t, "unit": unit}
        for end, info in merged.items():
            # the 10-Q's own fy/fp when this period is that filing's current quarter (the company
            # naming itself); _fiscal_period only guesses, and its guess hard-codes naming by the
            # year the fiscal year ENDS in, which is wrong for Target/Ulta/Home Depot/Lowe's
            q = cal.period(end)
            if q in ("Q1", "Q2", "Q3", "Q4"):
                fy = cal.fiscal_year(end)
            else:
                fy, q = _fiscal_period(end, fye)
            if q == "Q4":                            # Q4 has no 10-Q; a stray Q4-dated 10-Q fact is noise
                continue
            rows.append({"ticker": ticker, "cik": cik, "metric": metric,
                         "us_gaap_tag": info["tag"], "period_end": end,
                         "fiscal_year": fy, "quarter": q,
                         "value": info["val"], "unit": info["unit"], "accession": info["accn"]})
    return rows


class _FiscalCalendar:
    """How a company names its own fiscal periods — read from its filings, never assumed.

    A fact's `fy` tags the FILING (it comes from the 10-K cover's DocumentFiscalYearFocus), so
    every comparative column in that filing carries it too, and the same period picks up a
    different `fy` in next year's 10-K. Worse, the calendar year of a period end is not the
    company's own name for it: Target's year ending 2025-02-01 is Target's FY2024, while
    Walmart's year ending 2025-01-31 is Walmart's FY2025. Same fiscal calendar, opposite
    convention — so no rule derived from the fiscal-year-end month can be right for both.

    The one place the company states its own name for a period is the filing where that period
    is CURRENT. The submissions feed gives each filing's own period end (`reportDate`), and the
    facts give that filing's `fy`/`fp`; joining them is a lookup table, not an inference. Periods
    older than the submissions feed reaches fall back to the company's modal offset.
    """

    def __init__(self, by_end, accn_by_end, offset, degraded=False):
        self._by_end = by_end              # period_end -> (fiscal_year, 'FY'|'Q1'..'Q4')
        self._accn_by_end = accn_by_end    # period_end -> accn of the filing it is CURRENT in
        self.offset = offset               # (fy - calendar year), for periods not in the map
        # True when nothing could be learned at all — the feed was unreachable. Then `offset` is
        # 0 by default, which silently reproduces the old, wrong behaviour for a company that
        # names its years by the start year, so a caller that cares should say it cannot tell
        # rather than answer. Distinct from "offset happens to be 0", which is a real finding.
        self.degraded = degraded

    def fiscal_year(self, end):
        hit = self._by_end.get(end)
        return hit[0] if hit else int(end[:4]) + self.offset

    def period(self, end):
        """'FY' / 'Q1'..'Q4' as the company labels it, or None when it isn't in the map."""
        hit = self._by_end.get(end)
        return hit[1] if hit else None

    def original_accn(self, end):
        """The filing this period is the CURRENT period of — i.e. where it was first reported,
        rather than re-presented as a comparative. None when unknown."""
        return self._accn_by_end.get(end)


_cal_mem = {}   # CIK -> _FiscalCalendar (in-process; the corrected years are baked into cached rows)


_MIN_ANNUALS = 5        # below this the recent window is too thin to trust; page back for more


def _report_dates(cik):
    """accn -> the filing's OWN period end, for 10-K/10-Q.

    `filings.recent` is capped at ~1000 entries, and the cap is on ALL forms, so how far back it
    reaches depends on how much a company files rather than on years. Apple and Target get a
    decade; JPMorgan, which files thousands of structured-note 8-Ks a year, gets 26,014 entries
    covering twelve months and four 10-K/10-Qs. Older filings live in `filings.files`, so when the
    recent window is thin we page back — otherwise a high-volume filer's whole history falls to a
    fallback derived from one or two data points."""
    forms = ("10-K", "10-K/A", "10-Q", "10-Q/A")
    feed = _get_json(_SUBMISSIONS_URL.format(cik=cik))["filings"]
    out, pages = {}, [feed["recent"]]
    annuals = sum(1 for f in feed["recent"]["form"] if f.startswith("10-K"))
    for extra in (feed.get("files") or []):
        if annuals >= _MIN_ANNUALS:
            break
        try:
            page = _get_json("https://data.sec.gov/submissions/" + extra["name"])
        except Exception:
            break
        pages.append(page)
        annuals += sum(1 for f in page["form"] if f.startswith("10-K"))
    for page in pages:
        for a, f, d in zip(page["accessionNumber"], page["form"], page["reportDate"]):
            if f in forms and d:
                out.setdefault(a, d)
    return out


def fiscal_calendar(cik, gaap):
    """The company's own period naming. Fails soft: with no submissions feed (offline, unknown
    CIK) it returns an empty calendar whose offset is 0, which reproduces the previous
    calendar-year behaviour rather than erroring — and says so via .degraded, so a caller can
    tell "this company names its years normally" from "we could not find out"."""
    if cik in _cal_mem:
        return _cal_mem[cik]
    by_end, accn_by_end, offset = {}, {}, 0
    try:
        report = _report_dates(cik)
        need, label = set(report), {}
        for tag in gaap.values():                      # accn -> (fy, fp), stop once all are found
            if not need:
                break
            for arr in tag.get("units", {}).values():
                if not need:
                    break
                for u in arr:
                    a = u.get("accn")
                    if a in need and u.get("fy") is not None and u.get("fp"):
                        label[a] = (u["fy"], u["fp"])
                        need.discard(a)
        by_accn = {}
        offs = {}
        for a, end in report.items():
            if a not in label:
                continue
            fy, fp = label[a]
            by_end.setdefault(end, (fy, fp))
            by_accn.setdefault(end, []).append(a)
            # ANNUAL filings only. A fiscal year spans two calendar years, so a 10-Q's offset is
            # not the 10-K's (Walmart: FY2026 Q1 ends 2025-04-30, offset +1, while its FY2026
            # 10-K ends 2026-01-31, offset 0) — and there are three times as many 10-Qs, so
            # mixing them lets the quarterly offset win the vote. This value is only ever the
            # fallback for an ANNUAL period too old to be in the feed; quarters fall back to
            # _fiscal_period instead.
            if fp == "FY":
                offs.setdefault(end, fy - int(end[:4]))   # one period, one vote
        # an amendment supersedes the original for "as reported for that period", matching the
        # previous max(accn) choice among a period's own filings
        accn_by_end = {end: max(accns) for end, accns in by_accn.items()}
        # The MODE, not the oldest. The oldest looks like the better evidence for periods that
        # predate the map, but early-XBRL filings are the least reliably tagged: Walmart's two
        # 2013-14 filings say FY2012/FY2013 for years Walmart itself calls fiscal 2013/2014,
        # against twelve later ones that agree with the company. Taking the oldest would have
        # adopted the mis-tagged convention for the whole fallback. A majority is robust to a
        # handful of bad tags in a way any single filing is not.
        #
        # A split is worth knowing about either way — Kroger reads -1 through FY2022, 0 for the
        # next two years, then -1 again, which is a tagging inconsistency rather than a company
        # changing its mind — so it is logged rather than silently resolved. The periods
        # themselves are unaffected: each one carries its own filing's label from the map, and
        # only periods older than all of them ever reach this value.
        if offs:
            counts = Counter(offs.values())
            offset = counts.most_common(1)[0][0]
            if len(counts) > 1:
                log_miss(str(cik), "fiscal_calendar",
                         reason=f"inconsistent_fy_tags:{dict(counts)}")
            # A filing's own label is adopted only where it AGREES with that convention. The two
            # Walmart filings above don't just skew the fallback offset — they are the labels for
            # their own periods, so 2013-01-31 and 2014-01-31 each landed a year early: fiscal
            # 2013 ended up with two period ends and fiscal 2014 vanished from the data entirely.
            # The same shape shows up in Kroger, Ulta and Salesforce. A label that contradicts the
            # company's dominant convention is not evidence about that period, so it is dropped
            # and the year is derived the way a period too old for the feed already is. Annual
            # labels only — a 10-Q's offset legitimately differs, as the note above explains.
            for end, off in offs.items():
                if off != offset:
                    by_end[end] = (int(end[:4]) + offset, "FY")
    except Exception as e:
        # the same miss log the tools use, so a company whose fiscal-year naming we could not
        # establish leaves a trace instead of quietly answering with calendar years. Skipped for a
        # placeholder CIK: the L1 tests call extract_rows with 0000000000, and forty entries per
        # test run would bury the real ones.
        if str(cik).strip("0"):
            log_miss(str(cik), "fiscal_calendar",
                     reason=f"submissions_unavailable:{type(e).__name__}")
    cal = _FiscalCalendar(by_end, accn_by_end, offset, degraded=not by_end)
    if by_end:                        # only cache a real one, so a transient fetch failure
        _cal_mem[cik] = cal           # doesn't pin the fallback calendar for the process
    return cal


def fiscal_calendar_for(cik):
    """The calendar for a CIK, fetching the facts it needs when nothing has built it yet. For
    callers (statements.py) that work off edgartools rather than the companyfacts dict."""
    if cik in _cal_mem:
        return _cal_mem[cik]
    try:
        return fiscal_calendar(cik, fetch_facts(cik))
    except Exception:
        return _FiscalCalendar({}, {}, 0)


def _close(a, b, tol=0.01):
    """Two figures are the 'same' line item if they agree within a relative tolerance (allows
    minor restatement rounding; a component vs its total won't agree, so it's rejected)."""
    hi = max(abs(a), abs(b))
    return hi == 0 or abs(a - b) <= tol * hi


def extract_rows(gaap, ticker, cik):
    """Turn a company's us-gaap facts dict into rows matching data/financials.json's schema."""
    cal = fiscal_calendar(cik, gaap)
    rows = []
    for metric, (tags, kind) in METRICS.items():
        present = [t for t in tags if t in gaap]   # candidate tags, in preference order
        if not present:
            continue
        # Merge candidate tags in preference order: the preferred tag wins each period; lower-priority
        # tags only FILL periods it doesn't cover. A company migrates tags across years (AMD's old
        # SalesRevenueNet covers years before RevenueFromContract began; Nike tags its balance-sheet
        # inventory total as InventoryFinishedGoodsNetOfReserves). No size comparison is needed: by
        # accounting rules a footnote sub-component only appears alongside the total line (which is
        # always tagged somewhere), so it can never be the sole source for a gap year in a valid
        # filing — plain gap-fill can't substitute a component for a missing total.
        merged = {}
        for t in present:
            unit = next(iter(gaap[t]["units"]))     # USD, USD/shares, ...
            for end, info in annual_values(gaap[t]["units"][unit], kind, cal).items():
                if end not in merged:
                    merged[end] = {"val": info["val"], "accn": info["accn"], "tag": t, "unit": unit,
                                   "restated_val": info["restated_val"],
                                   "restated_accn": info["restated_accn"]}
        for end, info in merged.items():
            val, accn = info["val"], info["accn"]
            restated_val, restated_accn = info.get("restated_val"), info.get("restated_accn")
            # per-share: the split-adjusted (later) figure is the standard basis, use it and don't
            # flag (a stock split isn't a restatement).
            if metric in _SPLIT_ADJUSTED_METRICS and restated_val is not None:
                val, accn, restated_val, restated_accn = restated_val, restated_accn, None, None
            row = {"ticker": ticker, "cik": cik, "metric": metric,
                   "us_gaap_tag": info["tag"], "period_end": end,
                   "fiscal_year": cal.fiscal_year(end), "value": val,
                   "unit": info["unit"], "accession": accn}
            if restated_val is not None:                   # only when a real restatement exists
                row["restated_value"] = restated_val
                row["restated_accession"] = restated_accn
            rows.append(row)
    return rows


def fetch_company(cik):
    """(entityName, us-gaap facts) for a CIK — the entityName is SEC's official company name,
    used to echo which company a ticker/name actually resolved to."""
    d = _get_json(_FACTS_URL.format(cik=cik))
    return d.get("entityName", ""), d["facts"].get("us-gaap", {})


def fetch_facts(cik):
    """The raw us-gaap facts dict for a CIK (used by the offline build too)."""
    return fetch_company(cik)[1]


# --- on-demand, cached -----------------------------------------------------------------------
_rows_mem = {}   # TICKER -> annual rows (in-process)
_qrows_mem = {}  # TICKER -> quarterly rows (in-process)


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
    """Best-effort record of a 'requested metric unavailable' event; never breaks a request.

    Written twice on purpose. The file is the local queue, read by `scripts/check_misses.py`. On
    Cloud Run the container filesystem is per-instance and goes away with the instance, so the file
    there records nothing that survives; stdout is what Cloud Logging keeps, and it is the only
    place a production miss can be read back from.
    """
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "ticker": ticker.strip().upper(),
           "metric": metric, "fiscal_year": fiscal_year, "reason": reason}
    print("METRIC_MISS " + json.dumps(rec), flush=True)
    try:
        _MISS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _MISS_LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")
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
        entity, gaap = fetch_company(cik)
    except Exception:
        return []                      # transient failure — don't cache
    rows = extract_rows(gaap, tk, cik)
    if not rows:
        # A live ticker that extracts nothing is usually a successor registrant holding the ticker
        # while the history sits with its predecessor (see predecessor_cik). Retried only here, on
        # the path where we were about to abstain anyway, so a normal query pays nothing for it —
        # and NOT inside cik_for, because the successor is still the right answer for "who files
        # as XOM now": its own new 10-Q and 8-Ks are the current filings.
        prev = predecessor_cik(cik, entity or tk)
        if prev:
            try:
                entity, gaap = fetch_company(prev)
                rows = extract_rows(gaap, tk, prev)
                cik = prev
            except Exception:
                return []              # transient failure — don't cache
        if not rows:
            log_miss(tk, "company_rows", reason=f"no_annual_rows:{cik}")
    for r in rows:                     # stamp the resolved company name so tools can echo it
        r["entity_name"] = entity
    if rows:
        _save_disk(tk, rows)
    _rows_mem[tk] = rows
    return rows


def company_quarterly_rows(ticker):
    """Discrete quarterly rows (Q1-Q3, from 10-Q XBRL) for ANY public company, fetched live and
    cached in-process. Same schema as company_rows plus a `quarter` field. [] on unknown/failure."""
    tk = ticker.strip().upper()
    if tk in _qrows_mem:
        return _qrows_mem[tk]
    cik = cik_for(tk)
    if not cik:
        _qrows_mem[tk] = []
        return []
    try:
        entity, gaap = fetch_company(cik)
    except Exception:
        return []                          # transient failure — don't cache
    rows = quarterly_rows(gaap, tk, cik)
    for r in rows:
        r["entity_name"] = entity
    _qrows_mem[tk] = rows
    return rows
