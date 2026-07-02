"""Numeric tools for the SEC agent — the ONLY source of financial figures.

get_financials returns exact values from the XBRL-derived financials.json (with the
source filing for citation); compute does deterministic math. The agent is told to
route every number through these and never recall or do arithmetic itself — this is
what makes financial answers exact and auditable.
"""
import json
import os
from pathlib import Path

_DATA = Path(__file__).resolve().parent.parent / "data" / "financials.json"
_ROWS = None

# what the LLM might say -> our canonical metric key
_ALIASES = {
    "revenue": "revenue", "sales": "revenue", "total revenue": "revenue",
    "net income": "net_income", "net earnings": "net_income", "profit": "net_income",
    "operating income": "operating_income", "operating profit": "operating_income",
    "gross profit": "gross_profit",
    "r&d": "rd_expense", "research and development": "rd_expense",
    "rd expense": "rd_expense", "research and development expense": "rd_expense",
    "eps": "eps_diluted", "earnings per share": "eps_diluted",
    "diluted eps": "eps_diluted", "diluted earnings per share": "eps_diluted",
    "total assets": "total_assets", "assets": "total_assets",
    "total liabilities": "total_liabilities", "liabilities": "total_liabilities",
    "stockholders equity": "stockholders_equity", "shareholders equity": "stockholders_equity",
    "equity": "stockholders_equity",
    "cash": "cash", "cash and cash equivalents": "cash",
    "cost of revenue": "cost_of_revenue", "cost of goods sold": "cost_of_revenue",
    "cogs": "cost_of_revenue", "cost of sales": "cost_of_revenue",
    "dividends paid": "dividends_paid", "dividends": "dividends_paid",
    "total dividends": "dividends_paid", "cash dividends": "dividends_paid",
    "accounts payable": "accounts_payable", "trade payables": "accounts_payable",
    "inventory": "inventory", "inventories": "inventory",
    "current assets": "current_assets", "total current assets": "current_assets",
    "current liabilities": "current_liabilities",
    "total current liabilities": "current_liabilities",
    "long term debt": "long_term_debt", "long-term debt": "long_term_debt",
    "debt": "long_term_debt",
    "capex": "capex", "capital expenditure": "capex", "capital expenditures": "capex",
    "capital spending": "capex",
    "depreciation and amortization": "depreciation_amortization", "d&a": "depreciation_amortization",
    "depreciation": "depreciation_amortization", "depreciation & amortization": "depreciation_amortization",
    "ppe": "ppe_net", "pp&e": "ppe_net", "property plant and equipment": "ppe_net",
    "property, plant and equipment": "ppe_net", "fixed assets": "ppe_net", "net ppe": "ppe_net",
    "operating cash flow": "operating_cash_flow", "cash from operations": "operating_cash_flow",
    "cash flow from operations": "operating_cash_flow", "cfo": "operating_cash_flow",
    "operating cash flows": "operating_cash_flow",
    "accounts receivable": "accounts_receivable", "receivables": "accounts_receivable",
    "trade receivables": "accounts_receivable", "net receivables": "accounts_receivable",
    "interest expense": "interest_expense",
}


def _rows():
    global _ROWS
    if _ROWS is None:
        _ROWS = json.loads(_DATA.read_text())
    return _ROWS


def _rows_for(ticker):
    """Rows for one ticker: the curated local snapshot if present, else fetched live from SEC
    XBRL (companyfacts, cached) via the same extraction. Lets get_financials / get_ratio /
    get_growth answer ANY U.S. public company, while the curated companies keep byte-identical
    behavior (and the eval baseline). Returns [] if unknown / unreachable -> the caller abstains."""
    tk = ticker.strip().upper()
    local = [r for r in _rows() if r["ticker"] == tk]
    if local:
        return local
    try:
        from agents.companyfacts import company_rows
        return company_rows(tk)
    except Exception:
        return []


def edgar_url(cik, accession):
    """Direct link to the filing's index page on SEC EDGAR (uses the company CIK, not
    the accession prefix, which can be a filing agent's)."""
    return (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{accession.replace('-', '')}/{accession}-index.htm")


def cik_for(ticker):
    hits = _rows_for(ticker)
    return hits[0]["cik"] if hits else None


def _canon(metric: str) -> str:
    m = metric.strip().lower().replace("_", " ")
    return _ALIASES.get(m, metric.strip().lower().replace(" ", "_"))


def get_financials(ticker: str, metric: str, fiscal_year: int = None) -> str:
    """Return an EXACT financial figure for a company from SEC XBRL data, with its
    source filing. Use this for ANY financial number (revenue, net income, total assets,
    gross profit, EPS, cash, equity, ...) — never recall or estimate a number yourself.
    Pass fiscal_year (e.g. 2024) for a specific year, or omit it for the latest. If the
    company doesn't report the metric, this says so (do not fabricate a value)."""
    tk = ticker.strip().upper()
    key = _canon(metric)
    rows = _rows_for(tk)
    hits = [r for r in rows if r["metric"] == key]
    if not hits:
        avail = sorted({r["metric"] for r in rows})
        return (f"{tk} does not report '{metric}' (canonical: {key}) in the available "
                f"XBRL data. Reported metrics for {tk}: {', '.join(avail) or 'none'}.")
    if fiscal_year is not None:
        hits = [r for r in hits if r["fiscal_year"] == int(fiscal_year)]
        if not hits:
            yrs = sorted({r["fiscal_year"] for r in rows if r["metric"] == key})
            return (f"No {key} for {tk} FY{fiscal_year}. Available years: "
                    f"{yrs[0]}–{yrs[-1]}.")
    r = max(hits, key=lambda x: x["period_end"])   # latest if year omitted
    unit = r["unit"]
    val = f"${r['value']:,}" if unit == "USD" else f"{r['value']:,} {unit}"
    return (f"{tk} {key} for FY{r['fiscal_year']} (period ending {r['period_end']}): "
            f"{val}. [source: 10-K accession {r['accession']}, "
            f"{edgar_url(r['cik'], r['accession'])}]")


def _value(ticker, metric, year=None):
    """Raw numeric value + the source row, or (None, None) if unavailable. Used internally
    by get_ratio (which needs the number, not the formatted string)."""
    tk, key = ticker.strip().upper(), _canon(metric)
    hits = [r for r in _rows_for(tk) if r["metric"] == key]
    if year is not None:
        hits = [r for r in hits if r["fiscal_year"] == int(year)]
    if not hits:
        return None, None
    r = max(hits, key=lambda x: x["period_end"])
    return r["value"], r


# Declarative ratio definitions: name -> (formula spec, output kind, one-line definition).
# Each operand is a base metric; "avg:METRIC" means the average of this year and the prior
# year (the standard denominator for return ratios). Formulas are fixed in code so the LLM
# never picks the wrong base metric (e.g. "debt" must be long_term_debt, not total liabilities).
RATIOS = {
    "gross_margin":     (("gross_profit", "/", "revenue"), "pct", "gross profit / revenue"),
    "operating_margin": (("operating_income", "/", "revenue"), "pct", "operating income / revenue"),
    "net_margin":       (("net_income", "/", "revenue"), "pct", "net income / revenue"),
    "cogs_pct":         (("cost_of_revenue", "/", "revenue"), "pct", "cost of revenue / revenue"),
    "roa":              (("net_income", "/", "avg:total_assets"), "pct",
                         "net income / average total assets (avg of this and prior year)"),
    "roe":              (("net_income", "/", "avg:stockholders_equity"), "pct",
                         "net income / average shareholders' equity"),
    "current_ratio":    (("current_assets", "/", "current_liabilities"), "ratio",
                         "current assets / current liabilities"),
    "quick_ratio":      (("current_assets-inventory", "/", "current_liabilities"), "ratio",
                         "(current assets - inventory) / current liabilities"),
    "payout_ratio":     (("dividends_paid", "/", "net_income"), "ratio",
                         "dividends paid / net income"),
    "debt_to_equity":   (("long_term_debt", "/", "stockholders_equity"), "ratio",
                         "long-term debt / shareholders' equity (debt = interest-bearing debt, "
                         "NOT total liabilities)"),
    # activity ratios added after FinanceBench validation (the ad-hoc versions got miscomputed by
    # the model — e.g. DPO 365÷ratio instead of ×ratio; baking them in makes them deterministic).
    # "days" kind returns 365 × (num/den); "turns" returns num/den as a multiple.
    "dpo":              (("avg:accounts_payable", "/", "cost_of_revenue"), "days",
                         "365 × average accounts payable / cost of revenue (days payable outstanding)"),
    "dso":              (("avg:accounts_receivable", "/", "revenue"), "days",
                         "365 × average accounts receivable / revenue (days sales outstanding)"),
    "dio":              (("avg:inventory", "/", "cost_of_revenue"), "days",
                         "365 × average inventory / cost of revenue (days inventory outstanding)"),
    "asset_turnover":   (("revenue", "/", "avg:total_assets"), "turns",
                         "revenue / average total assets"),
    "fixed_asset_turnover": (("revenue", "/", "avg:ppe_net"), "turns",
                         "revenue / average net PP&E"),
    "capex_pct_revenue": (("capex", "/", "revenue"), "pct", "capex / revenue"),
    "interest_coverage": (("operating_income", "/", "interest_expense"), "turns",
                         "operating income / interest expense (times interest earned)"),
}


def _fmt_ratio(raw, kind):
    """Format a ratio's raw num/den by output kind (shared by the public and uploaded-doc tools)."""
    if kind == "pct":
        return f"{raw * 100:.1f}%"
    if kind == "days":
        return f"{raw * 365:.2f} days"
    if kind == "turns":
        return f"{raw:.2f}x"
    return f"{raw:.2f}"

_RATIO_ALIASES = {
    "gross margin": "gross_margin", "gross profit margin": "gross_margin",
    "operating margin": "operating_margin",
    "net margin": "net_margin", "net profit margin": "net_margin", "profit margin": "net_margin",
    "cogs %": "cogs_pct", "cogs percentage": "cogs_pct", "cogs margin": "cogs_pct",
    "cost of revenue %": "cogs_pct",
    "roa": "roa", "return on assets": "roa",
    "roe": "roe", "return on equity": "roe",
    "current ratio": "current_ratio",
    "quick ratio": "quick_ratio", "acid test": "quick_ratio",
    "payout ratio": "payout_ratio", "dividend payout ratio": "payout_ratio",
    "debt to equity": "debt_to_equity", "debt-to-equity": "debt_to_equity",
    "d/e": "debt_to_equity",
    "dpo": "dpo", "days payable outstanding": "dpo", "days payable": "dpo",
    "dso": "dso", "days sales outstanding": "dso", "days sales": "dso",
    "dio": "dio", "days inventory outstanding": "dio", "days inventory": "dio",
    "days of inventory": "dio",
    "asset turnover": "asset_turnover", "total asset turnover": "asset_turnover",
    "fixed asset turnover": "fixed_asset_turnover", "ppe turnover": "fixed_asset_turnover",
    "capex % of revenue": "capex_pct_revenue", "capex as a % of revenue": "capex_pct_revenue",
    "capex to revenue": "capex_pct_revenue", "capex margin": "capex_pct_revenue",
    "interest coverage": "interest_coverage", "times interest earned": "interest_coverage",
}


def _resolve_operand(spec, year, getval):
    """Resolve one side of a ratio formula using a source-agnostic value getter
    getval(metric, year) -> (value, src_row). Supports a base metric, "avg:METRIC" (this year
    and prior averaged), and "A-B" (difference of two base metrics, for quick ratio). Shared by
    the public XBRL ratio tool and the uploaded-document one — same formulas, different source."""
    if spec.startswith("avg:"):
        m = spec[4:]
        cur, r = getval(m, year)
        if cur is None:
            return None, None
        prev, _ = getval(m, int(year) - 1) if year else (None, None)
        return ((cur + prev) / 2 if prev is not None else cur), r
    if "-" in spec:
        parts = spec.split("-")
        vals, src = [], None
        for p in parts:
            v, r = getval(p, year)
            if v is None:
                return None, None
            vals.append(v)
            src = src or r
        return vals[0] - sum(vals[1:]), src
    return getval(spec, year)


def _operand(ticker, spec, year):
    """Public-XBRL operand resolution (reads _rows() by ticker)."""
    return _resolve_operand(spec, year, lambda m, y: _value(ticker, m, y))


def get_ratio(ratio: str, ticker: str, fiscal_year: int = None) -> str:
    """Compute a standard financial RATIO deterministically (the formula is fixed in code, so
    the right base metrics and conventions are always used). Supports: gross_margin,
    operating_margin, net_margin, cogs_pct, roa, roe, current_ratio, quick_ratio, payout_ratio,
    debt_to_equity, dpo, dso, dio, asset_turnover, fixed_asset_turnover, capex_pct_revenue,
    interest_coverage. Use this for ANY ratio (incl. days-outstanding and turnover ratios) instead
    of fetching pieces and dividing yourself — it returns the exact value, the formula, and source."""
    name = _RATIO_ALIASES.get(ratio.strip().lower(), ratio.strip().lower().replace(" ", "_"))
    if name not in RATIOS:
        return (f"Unknown ratio '{ratio}'. Supported: {', '.join(sorted(RATIOS))}.")
    (num_spec, _op, den_spec), kind, definition = RATIOS[name]
    tk = ticker.strip().upper()
    num, src = _operand(tk, num_spec, fiscal_year)
    den, _ = _operand(tk, den_spec, fiscal_year)
    if num is None or den is None:
        return (f"Cannot compute {name} for {tk}"
                f"{(' FY' + str(fiscal_year)) if fiscal_year else ''}: a required figure "
                f"({num_spec if num is None else den_spec}) isn't in the data. Abstain.")
    if den == 0:
        return f"Cannot compute {name} for {tk}: denominator is 0."
    raw = num / den
    fy = src["fiscal_year"] if src else fiscal_year
    out = _fmt_ratio(raw, kind)
    return (f"{tk} {name} for FY{fy} = {out} ({definition}). "
            f"[source: 10-K accession {src['accession']}, {edgar_url(src['cik'], src['accession'])}]")


def get_growth(metric: str, ticker: str, fiscal_year: int = None) -> str:
    """Compute year-over-year (YoY) change of a metric DETERMINISTICALLY. The tool itself
    fetches the given fiscal year (or the latest) AND the immediately preceding fiscal year,
    then returns the percent change — so the two years compared are ALWAYS consecutive. Use
    this for ANY 'year over year' / 'YoY' / 'how did X change' question instead of fetching two
    years and dividing yourself (that lets a wrong baseline slip in, e.g. a multi-year span
    mislabeled as YoY). If the prior fiscal year isn't in the data, it abstains rather than
    using a non-adjacent year. Returns the % change, both values, both years, and source filings."""
    tk, key = ticker.strip().upper(), _canon(metric)
    cur, rc = _value(tk, key, fiscal_year)        # latest if year omitted
    if cur is None:
        return get_financials(ticker, metric, fiscal_year)   # reuse its "not reported / bad year" msg
    cur_year = rc["fiscal_year"]
    prev, rp = _value(tk, key, cur_year - 1)      # the immediately preceding fiscal year, in code
    if prev is None:
        return (f"Cannot compute YoY for {tk} {key} FY{cur_year}: the prior fiscal year "
                f"(FY{cur_year - 1}) isn't in the data, so there's no adjacent year to compare. "
                f"Abstain rather than use a non-consecutive year.")
    if prev == 0:
        return f"Cannot compute YoY for {tk} {key}: prior-year value is 0."
    pct = (cur - prev) / abs(prev) * 100
    f = lambda v, r: (f"${v:,}" if r["unit"] == "USD" else f"{v:,} {r['unit']}")
    return (f"{tk} {key} grew {pct:+.1f}% year over year, from {f(prev, rp)} (FY{cur_year - 1}) "
            f"to {f(cur, rc)} (FY{cur_year}). [sources: 10-K accessions {rp['accession']}, "
            f"{rc['accession']}, {edgar_url(rc['cik'], rc['accession'])}]")


def compute(op: str, a: float, b: float) -> str:
    """Deterministic math on EXACT figures obtained from get_financials. Never do
    arithmetic yourself — call this. op: 'yoy' = percent change from b (prior) to a
    (current); 'diff' = a - b; 'ratio' = a / b (e.g. net margin = net_income / revenue)."""
    a, b = float(a), float(b)
    if op == "yoy":
        if b == 0:
            return "Cannot compute YoY: prior value is 0."
        return f"YoY change: {(a - b) / abs(b) * 100:+.1f}% (from {b:,.0f} to {a:,.0f})"
    if op == "diff":
        return f"Difference (a - b): {a - b:,.2f}"
    if op == "ratio":
        if b == 0:
            return "Cannot compute ratio: denominator is 0."
        return f"Ratio (a / b): {a / b:.4f} ({a / b * 100:.2f}%)"
    return f"Unknown op '{op}'. Use 'yoy', 'diff', or 'ratio'."


if __name__ == "__main__":
    print(get_financials("AAPL", "revenue", 2024))
    print(get_financials("NVDA", "revenue"))                 # latest
    print(get_financials("AAPL", "net income", 2024))        # alias
    print(get_financials("JPM", "gross profit"))             # not reported -> abstain
    print(get_financials("AAPL", "revenue", 2099))           # bad year
    print(compute("yoy", 215_938_000_000, 130_497_000_000))  # NVDA revenue YoY
    print(compute("ratio", 112_010_000_000, 416_161_000_000))  # AAPL net margin
