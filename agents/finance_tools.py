"""Numeric tools for the SEC agent — the ONLY source of financial figures.

get_financials returns exact values from the XBRL-derived financials.json (with the
source filing for citation); compute does deterministic math. The agent is told to
route every number through these and never recall or do arithmetic itself — this is
what makes financial answers exact and auditable.
"""
import ast
import difflib
import json
import os
import re
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

# Auto-derive aliases from the us-gaap tags each metric is built from (companyfacts.METRICS), so a
# model that expands a shorthand to the real XBRL concept name — e.g. "PP&E" -> the tag
# "property_plant_and_equipment_net" — resolves to our slug "ppe_net" instead of missing. Bounded
# (a finite tag set we already maintain) and self-maintaining (a new metric's tags come free), not
# hand-typed variants. Skips any tag whose normalized form already maps to a different metric.
try:
    from agents.companyfacts import METRICS as _METRICS
    for _slug, _spec in _METRICS.items():
        for _tag in (_spec[0] if isinstance(_spec[0], (list, tuple)) else ()):
            _k = re.sub(r"(?<!^)(?=[A-Z])", " ", _tag).lower()   # CamelCase -> space-separated
            if _ALIASES.get(_k, _slug) == _slug:
                _ALIASES[_k] = _slug
except Exception:
    pass


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


def _quarterly_rows_for(ticker):
    """Discrete quarterly rows (Q1-Q3) for one ticker, fetched live from SEC 10-Q XBRL (cached).
    Quarterly isn't in the curated annual snapshot, so it always goes through the live fetch."""
    try:
        from agents.companyfacts import company_quarterly_rows
        return company_quarterly_rows(ticker.strip().upper())
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
    # normalize punctuation/spacing (commas, hyphens, underscores -> spaces) so surface-form
    # variants of one name collapse before the alias lookup — one general rule, not per-variant
    # aliases; keep & (pp&e, r&d, d&a). Names it still can't resolve fall to the tool's abstain/
    # self-correction, which is the real backstop.
    m = re.sub(r"[^a-z0-9& ]+", " ", metric.lower())
    m = re.sub(r"\s+", " ", m).strip()
    return _ALIASES.get(m, m.replace(" ", "_"))


def _log_miss(ticker, metric, fiscal_year, reason):
    """Part 3: record a requested-but-unavailable metric (drives data-driven METRICS expansion).
    Lazy import so the curated-only path has no hard dependency; never breaks a request."""
    try:
        from agents.companyfacts import log_miss
        log_miss(ticker, metric, fiscal_year, reason)
    except Exception:
        pass


def _quarterly_answer(tk, ticker, key, metric, fiscal_year, quarter):
    """Discrete quarterly figure (Q1-Q3) from 10-Q XBRL. Q4 isn't filed on its own -> point to the
    full year. Mirrors get_financials' abstain-not-fabricate behavior."""
    try:
        q = int(quarter)
    except (ValueError, TypeError):
        return f"quarter must be 1, 2, or 3 (got {quarter!r})."
    if q == 4:
        return (f"Q4 isn't filed on its own (there is no Q4 10-Q). Call get_financials WITHOUT "
                f"quarter for the full fiscal year, or ask for Q1-Q3.")
    if q not in (1, 2, 3):
        return "quarter must be 1, 2, or 3."
    rows = _quarterly_rows_for(tk)
    hits = [r for r in rows if r["metric"] == key and r["quarter"] == f"Q{q}"]
    if fiscal_year is not None:
        hits = [r for r in hits if r["fiscal_year"] == int(fiscal_year)]
    if not hits:
        yrs = sorted({r["fiscal_year"] for r in rows if r["metric"] == key})
        if not yrs:
            return (f"No quarterly {key} available for {tk}" +
                    ("" if rows else " (no quarterly XBRL data — the company may not file 10-Qs)."))
        return (f"No Q{q} {key} for {tk}" + (f" FY{fiscal_year}" if fiscal_year else "") +
                f". Quarterly years available: {yrs[0]}–{yrs[-1]}.")
    r = max(hits, key=lambda x: x["period_end"])
    unit = r["unit"]
    val = f"${r['value']:,}" if unit == "USD" else f"{r['value']:,} {unit}"
    return (f"{_entity(r, ticker)} {key} for {r['quarter']} FY{r['fiscal_year']} (quarter ending "
            f"{r['period_end']}): {val}. [source: 10-Q accession {r['accession']}, "
            f"{edgar_url(r['cik'], r['accession'])}]")


def get_financials(ticker: str, metric: str, fiscal_year: int = None, quarter: int = None) -> str:
    """Return an EXACT financial figure for a company from SEC XBRL data, with its
    source filing. Use this for ANY financial number (revenue, net income, total assets,
    gross profit, EPS, cash, equity, ...) — never recall or estimate a number yourself.
    Pass fiscal_year (e.g. 2024) for a specific year, or omit it for the latest ANNUAL (10-K)
    figure. For a QUARTERLY figure pass quarter=1/2/3 (from 10-Q; Q4 isn't filed alone — use the
    full year). For a delisted or renamed company whose ticker no longer trades (e.g. Activision,
    or Square/Block), pass the company NAME as `ticker` instead. If the company doesn't report the
    metric, this says so (do not fabricate a value)."""
    tk = ticker.strip().upper()
    key = _canon(metric)
    if quarter is not None:
        return _quarterly_answer(tk, ticker, key, metric, fiscal_year, quarter)
    rows = _rows_for(tk)
    if not rows:                                   # ticker didn't resolve to any filer at all
        return (f"No company found for '{ticker}'. If it is delisted or renamed (its ticker no "
                f"longer trades — e.g. Activision, or Square/Block), retry with the full COMPANY "
                f"NAME instead of a ticker.")
    hits = [r for r in rows if r["metric"] == key]
    if not hits:
        _log_miss(tk, key, fiscal_year, "metric_absent")
        avail = sorted({r["metric"] for r in rows})
        # an unrecognized metric NAME (not a known metric, not reported) -> suggest the closest,
        # so the model self-corrects (e.g. "accounts_receivable_net" -> accounts_receivable)
        if key not in set(_ALIASES.values()) and key not in avail:
            near = difflib.get_close_matches(key, set(_ALIASES.values()) | set(avail), n=1, cutoff=0.6)
            hint = f" Did you mean '{near[0]}'?" if near else ""
            return (f"'{metric}' isn't a recognized metric.{hint} "
                    f"Reported metrics for {tk}: {', '.join(avail) or 'none'}.")
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
    return (f"{_entity(r, ticker)} {key} for FY{r['fiscal_year']} (period ending {r['period_end']}): "
            f"{val}. [source: 10-K accession {r['accession']}, "
            f"{edgar_url(r['cik'], r['accession'])}]" + _restatement_note([r]))


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


def _entity(row, ticker):
    """The RESOLVED company name (SEC's entityName) to echo back, so a wrong-ticker guess is visible
    (e.g. "BBBY" → "BED BATH & BEYOND", not silently reported as Best Buy). Falls back to the
    ticker/name the caller passed (curated rows carry no entity_name; those tickers are unambiguous)."""
    return (row.get("entity_name") if row else None) or ticker.strip().upper()


def _restatement_note(rows):
    """A one-line caveat when any figure used in the answer was later restated. Values are kept on
    the AS-REPORTED basis (matches the source filing, and keeps a multi-year answer on one basis);
    this tells the reader the current-basis value exists — a finance reader may not know a figure
    was reclassified. Rows without a restatement (the common case) add nothing."""
    seen, parts = set(), []
    for r in rows:
        if not r or r.get("restated_value") is None:
            continue
        k = (r["metric"], r["fiscal_year"])
        if k in seen:
            continue
        seen.add(k)
        usd = r.get("unit") == "USD"
        orig = f"${r['value']:,}" if usd else f"{r['value']:,}"
        rest = f"${r['restated_value']:,}" if usd else f"{r['restated_value']:,}"
        parts.append(f"{r['metric']} FY{r['fiscal_year']} from {orig} to {rest}")
    if not parts:
        return ""
    return (" [NOTE: figures are AS ORIGINALLY REPORTED and computed on that basis; a later filing "
            "restated " + "; ".join(parts) + " (current basis). Ask for the restated basis if needed.]")


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
    "inventory_turnover": (("cost_of_revenue", "/", "avg:inventory"), "turns",
                         "cost of revenue / average inventory (avg of this and prior year)"),
    "capex_pct_revenue": (("capex", "/", "revenue"), "pct", "capex / revenue"),
    "interest_coverage": (("operating_income", "/", "interest_expense"), "turns",
                         "operating income / interest expense (times interest earned)"),
    "effective_tax_rate": (("income_tax_expense", "/", "pretax_income"), "pct",
                         "income tax expense / pre-tax income (income before income taxes)"),
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
    "inventory turnover": "inventory_turnover", "inventory turnover ratio": "inventory_turnover",
    "stock turnover": "inventory_turnover",
    "capex % of revenue": "capex_pct_revenue", "capex as a % of revenue": "capex_pct_revenue",
    "capex to revenue": "capex_pct_revenue", "capex margin": "capex_pct_revenue",
    "interest coverage": "interest_coverage", "times interest earned": "interest_coverage",
    "effective tax rate": "effective_tax_rate", "tax rate": "effective_tax_rate",
    "effective income tax rate": "effective_tax_rate",
}


def _spec_metrics(spec):
    """The base metric names in a ratio operand spec ("avg:ppe_net" -> {ppe_net}; "a-b" -> {a,b})."""
    return {p.strip() for p in spec.replace("avg:", "").split("-") if p.strip()}


# base-metric SET -> named ratio, ONLY for raw-scale ratios (turns/ratio, so a compute_formula's
# a/b is on the same scale as our value) and ONLY unambiguous sets — lets compute_formula cross-check
# a hand-written formula against our standard convention without false comparisons.
_RATIO_BY_METRICS, _amb = {}, set()
for _rn, ((_ns, _op2, _ds), _k, _d) in RATIOS.items():
    if _k not in ("turns", "ratio"):
        continue
    _key = frozenset(_spec_metrics(_ns) | _spec_metrics(_ds))
    if _key in _RATIO_BY_METRICS or _key in _amb:
        _amb.add(_key); _RATIO_BY_METRICS.pop(_key, None)
    else:
        _RATIO_BY_METRICS[_key] = _rn


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
    debt_to_equity, dpo, dso, dio, asset_turnover, fixed_asset_turnover, inventory_turnover,
    capex_pct_revenue, interest_coverage. Use this for ANY ratio (incl. days-outstanding and
    turnover ratios) instead
    of fetching pieces and dividing yourself — it returns the exact value, the formula, and source."""
    name = _RATIO_ALIASES.get(ratio.strip().lower(), ratio.strip().lower().replace(" ", "_"))
    if name not in RATIOS:
        return (f"Unknown ratio '{ratio}'. Supported: {', '.join(sorted(RATIOS))}.")
    (num_spec, _op, den_spec), kind, definition = RATIOS[name]
    tk = ticker.strip().upper()
    used = []                                        # every source row touched (incl. avg: prior year)

    def _getval(m, y):
        v, r = _value(tk, m, y)
        if r:
            used.append(r)
        return v, r
    num, src = _resolve_operand(num_spec, fiscal_year, _getval)
    den, _ = _resolve_operand(den_spec, fiscal_year, _getval)
    if num is None or den is None:
        if not _rows_for(tk):                      # ticker didn't resolve to any filer at all
            return (f"No company found for '{ticker}'. If it is delisted or renamed, retry with "
                    f"the full COMPANY NAME instead of a ticker.")
        missing = num_spec if num is None else den_spec
        _log_miss(tk, missing.replace("avg:", ""), fiscal_year, f"ratio_base_absent:{name}")
        return (f"Cannot compute {name} for {tk}"
                f"{(' FY' + str(fiscal_year)) if fiscal_year else ''}: a required figure "
                f"({missing}) isn't in the data. Abstain.")
    if den == 0:
        return f"Cannot compute {name} for {tk}: denominator is 0."
    raw = num / den
    fy = src["fiscal_year"] if src else fiscal_year
    out = _fmt_ratio(raw, kind)
    return (f"{_entity(src, ticker)} {name} for FY{fy} = {out} ({definition}). "
            f"[source: 10-K accession {src['accession']}, {edgar_url(src['cik'], src['accession'])}]"
            + _restatement_note(used))


# --- deterministic CSV export (no LLM in the loop; values straight from the XBRL tools) ---------
def _cell_value(name, ticker, year):
    """Excel-friendly value of a base metric OR a ratio for one fiscal year, with a unit label.
    Returns (value_or_None, unit)."""
    tk = ticker.strip().upper()
    rname = _RATIO_ALIASES.get(name.strip().lower(), name.strip().lower().replace(" ", "_"))
    if rname in RATIOS:
        (num_spec, _op, den_spec), kind, _def = RATIOS[rname]
        num, _ = _operand(tk, num_spec, year)
        den, _ = _operand(tk, den_spec, year)
        unit = {"pct": "%", "days": "days", "turns": "x"}.get(kind, "")
        if num is None or not den:
            return None, unit
        raw = num / den
        return round({"pct": raw * 100, "days": raw * 365, "turns": raw, "ratio": raw}[kind], 4), unit
    v, r = _value(tk, name, year)
    return (v, r["unit"]) if v is not None else (None, "")


def financial_table_csv(ticker, metrics, years):
    """Build a metrics × fiscal-years grid as CSV, computed deterministically from the XBRL tools
    (no model in the loop, so the export is exact). Each row is a base metric or a ratio (ratio
    units are noted in the row label); missing cells are blank."""
    tk = ticker.strip().upper()
    years = sorted({int(y) for y in years})
    lines = ["metric," + ",".join(f"FY{y}" for y in years)]
    for m in metrics:
        m = m.strip()
        if not m:
            continue
        cells = [_cell_value(m, tk, y) for y in years]
        unit = next((u for v, u in cells if v is not None), "")
        label = m if unit in ("", "USD") else f"{m} ({unit})"
        row = ",".join("" if v is None else str(v) for v, _ in cells)
        lines.append(f"{label},{row}")
    return "\n".join(lines) + "\n"


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
    return (f"{_entity(rc, ticker)} {key} grew {pct:+.1f}% year over year, from {f(prev, rp)} (FY{cur_year - 1}) "
            f"to {f(cur, rc)} (FY{cur_year}). [sources: 10-K accessions {rp['accession']}, "
            f"{rc['accession']}, {edgar_url(rc['cik'], rc['accession'])}]" + _restatement_note([rc, rp]))


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


# --- Single-expression formula evaluator (Program-of-Thought) ---------------------------------
# For a question that SPELLS OUT a formula (or a metric no ratio tool covers). The model writes
# the WHOLE formula ONCE with our metric names as variables + avg()/delta()/prev() helpers; this
# fetches every figure from XBRL itself and evaluates the whole expression deterministically. That
# removes the two failure modes of chained compute() (wrong operand order, sum-vs-average) AND
# keeps the finance bar (the model never transcribes a number). Only arithmetic + metric names +
# the whitelisted helpers are allowed (a safe AST walk, never eval()).
_FORMULA_FUNCS = {"avg", "delta", "prev", "abs"}


def _eval_node(node, ticker, year, srcs):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, ticker, year, srcs)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        raise ValueError(f"non-numeric constant {node.value!r}")
    if isinstance(node, ast.BinOp):
        a = _eval_node(node.left, ticker, year, srcs)
        b = _eval_node(node.right, ticker, year, srcs)
        for typ, fn in ((ast.Add, lambda: a + b), (ast.Sub, lambda: a - b),
                        (ast.Mult, lambda: a * b), (ast.Div, lambda: a / b),
                        (ast.Pow, lambda: a ** b), (ast.Mod, lambda: a % b)):
            if isinstance(node.op, typ):
                return fn()
        raise ValueError("operator not allowed")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval_node(node.operand, ticker, year, srcs)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.Name):
        return _metric_at(node.id, ticker, year, srcs)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = node.func.id
        if fn not in _FORMULA_FUNCS:
            raise ValueError(f"call '{fn}' not allowed")
        if fn == "abs":
            if len(node.args) != 1:
                raise ValueError("abs() takes one argument")
            return abs(_eval_node(node.args[0], ticker, year, srcs))
        # avg/delta/prev(metric [, n]): optional integer literal n = years back (for multi-year
        # spans like a 2-year CAGR or a 3-year average). Default preserves the 1-year behavior.
        if not node.args or not isinstance(node.args[0], ast.Name) or len(node.args) > 2:
            raise ValueError(f"{fn}() takes a metric name and an optional integer number of years")
        m = node.args[0].id
        n = None
        if len(node.args) == 2:
            arg = node.args[1]
            if not (isinstance(arg, ast.Constant) and isinstance(arg.value, int)
                    and not isinstance(arg.value, bool) and arg.value >= 1):
                raise ValueError(f"{fn}() years must be a positive integer literal")
            n = arg.value
        if fn == "prev":
            return _metric_at(m, ticker, year - (n or 1), srcs)
        if fn == "delta":
            return _metric_at(m, ticker, year, srcs) - _metric_at(m, ticker, year - (n or 1), srcs)
        # avg: mean of the last n years (default 2 = this and prior, the original behavior)
        k = n or 2
        return sum(_metric_at(m, ticker, year - i, srcs) for i in range(k)) / k
    raise ValueError(f"disallowed expression element: {type(node).__name__}")


def _metric_at(name, ticker, year, srcs):
    v, r = _value(ticker, name, year)
    if v is None:
        raise KeyError(f"{_canon(name)} for {ticker} FY{year} not reported")
    if r:
        srcs.append(r)
    return float(v)


def compute_formula(expression: str, ticker: str, fiscal_year: int = None) -> str:
    """Evaluate a custom financial FORMULA deterministically. Use this when the question SPELLS OUT
    a formula, or asks for a metric get_ratio does not cover. Write the whole formula with our
    metric names as variables and the helpers avg(), delta() (this year minus prior), prev(), e.g.:
      DPO = "365 * avg(accounts_payable) / (cost_of_revenue + delta(inventory))"
      EBITDA margin = "(operating_income + depreciation_amortization) / revenue"
    For MULTI-YEAR spans, each helper takes an optional number of years back — prev(metric, n),
    avg(metric, n), delta(metric, n) — so use this (not get_growth, which is adjacent-year only):
      2-year revenue CAGR (FY2020->FY2022, pass fiscal_year=2022):
        "(revenue / prev(revenue, 2)) ** (1/2) - 1"
      3-year average capex % of revenue (FY2017-FY2019, pass fiscal_year=2019):
        "((capex/revenue) + (prev(capex,1)/prev(revenue,1)) + (prev(capex,2)/prev(revenue,2))) / 3"
    The tool fetches each figure from XBRL itself — you NEVER pass or transcribe numbers — and
    computes the whole expression, so there are no arithmetic mistakes. Pass ticker and fiscal_year.
    Returns the value and the figures used, or an error (then abstain)."""
    tk = ticker.strip().upper()
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return f"Invalid formula syntax: {expression!r}. Abstain."
    # catch a mistyped metric name and suggest the right one, so the model can self-correct
    # (e.g. "ppne" -> "ppe_net") instead of abstaining as if the figure were unavailable.
    known = set(_ALIASES.values())
    unknown = {n.id for n in ast.walk(tree)
               if isinstance(n, ast.Name) and n.id not in _FORMULA_FUNCS and _canon(n.id) not in known}
    if unknown:
        hints = []
        for u in sorted(unknown):
            near = difflib.get_close_matches(_canon(u), known, n=1, cutoff=0.5)
            hints.append(f"'{u}'" + (f" (did you mean '{near[0]}'?)" if near else ""))
        return (f"Unknown metric name(s): {', '.join(hints)}. Use exact metric names "
                f"(ppe_net, revenue, cost_of_revenue, ...). Retry with the correct name.")
    year = int(fiscal_year) if fiscal_year else None
    if year is None:
        for nm in [n.id for n in ast.walk(tree)
                   if isinstance(n, ast.Name) and n.id not in _FORMULA_FUNCS]:
            _, r = _value(tk, nm)
            if r:
                year = r["fiscal_year"]
                break
        if year is None:
            return "Provide a fiscal_year for the formula. Abstain."
    srcs = []
    try:
        val = _eval_node(tree, tk, year, srcs)
    except (KeyError, ValueError, ZeroDivisionError) as e:
        return (f"Cannot evaluate the formula for {tk} FY{year}: {e}. Abstain rather than guess.")
    accns = sorted({r["accession"] for r in srcs if r})
    src = srcs[0] if srcs else None
    cite = (f"[sources: 10-K accessions {', '.join(accns)}, "
            f"{edgar_url(src['cik'], accns[0])}]" if src and accns else "")
    # Keep precision for small results (ratios / margins / CAGR) — a fixed 2dp would show a
    # 0.4% CAGR or a 5.4% margin as "0.00" / "0.05" and lose the answer.
    # defense-in-depth cross-check: if the formula's metrics match a named RAW ratio and the value
    # diverges materially from our standard convention (same ballpark, >2%), note both and let the
    # reader judge — do NOT override, since a question may spell out a different convention (e.g.
    # ending vs average PP&E). Catches the model hand-computing a named ratio the wrong way.
    xcheck = ""
    names = frozenset(_canon(n.id) for n in ast.walk(tree)
                      if isinstance(n, ast.Name) and n.id not in _FORMULA_FUNCS)
    rn = _RATIO_BY_METRICS.get(names)
    if rn and year:
        (ns, _o, ds), _k, defn = RATIOS[rn]
        onum, _ = _operand(tk, ns, year)
        oden, _ = _operand(tk, ds, year)
        ours = onum / oden if (onum is not None and oden) else None
        if ours and val and (1 / 3) < abs(val) / abs(ours) < 3 and abs(val - ours) > 0.02 * abs(ours):
            xcheck = (f" [CHECK: this matches the '{rn}' ratio; on our standard convention ({defn}) "
                      f"it is {ours:,.2f}. If the question defines it differently, use that value.]")
    shown = f"{val:,.2f}" if abs(val) >= 1 else f"{val:.4g}"
    return (f"{_entity(src, ticker)} formula result for FY{year} = {shown}  (formula: {expression}). {cite}"
            + _restatement_note(srcs) + xcheck)


if __name__ == "__main__":
    print(get_financials("AAPL", "revenue", 2024))
    print(get_financials("NVDA", "revenue"))                 # latest
    print(get_financials("AAPL", "net income", 2024))        # alias
    print(get_financials("JPM", "gross profit"))             # not reported -> abstain
    print(get_financials("AAPL", "revenue", 2099))           # bad year
    print(compute("yoy", 215_938_000_000, 130_497_000_000))  # NVDA revenue YoY
    print(compute("ratio", 112_010_000_000, 416_161_000_000))  # AAPL net margin
