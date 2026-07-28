"""Report-level statement reconstruction. The full income statement / balance sheet / cash-flow line
items (labels + values) for any company-year, rebuilt from the filing's XBRL *presentation* linkbase
via edgartools — not the flat companyfacts tags. This SUPPLEMENTS the metric-level tools
(get_financials etc.), which stay the source of truth for tracked metrics; it fills what a fixed
metric set can't: company-specific line items (e.g. a bank's customer deposits) and statement
structure, so a question like "largest liability" is answerable.

Finance bar: values come straight from XBRL (never the model), each answer cites the filing, and
superlatives ("largest liability") are computed in code — the model never scans the numbers and picks.
"""
import os
import re

from agents.finance_tools import edgar_url, cik_for

# edgartools is imported lazily (inside _filing) so `import agents.statements` — and its pure-logic
# unit tests — don't require the heavy dep; matches how finance_tools defers edgar.

_STMT = {"balance_sheet": "balance_sheet", "income_statement": "income_statement",
         "cash_flow": "cash_flow_statement"}

# balance-sheet section -> a predicate on the row's parent XBRL concept (handles current/non-current
# splits too, e.g. us-gaap_LiabilitiesCurrent, while excluding the combined Liabilities+Equity total).
_SECTION = {
    "liabilities": lambda p: "Liabilit" in p and "StockholdersEquity" not in p and "AndStockholders" not in p,
    "assets":      lambda p: "Assets" in p and "AndStockholders" not in p,
    "equity":      lambda p: "StockholdersEquity" in p and "AndStockholders" not in p,
}


def _fmt(v):
    a = abs(v)
    if a >= 1e9:
        return f"${v / 1e9:.1f}B"
    if a >= 1e6:
        return f"${v / 1e6:.1f}M"
    return f"${v:,.0f}"


def _filing(cik, fiscal_year):
    """The 10-K for `fiscal_year` (±1 for non-December year-ends), or the latest 10-K if None."""
    from edgar import Company, set_identity
    set_identity("Zhonghui Li lizhonghui923@gmail.com")  # SEC requires a contact; idempotent
    cands = {}
    for f in Company(int(cik)).get_filings(form="10-K"):
        if f.form != "10-K":
            continue
        yr = str(getattr(f, "period_of_report", "") or "")[:4]
        if yr.isdigit() and (yr not in cands or f.accession_no > cands[yr].accession_no):
            cands[yr] = f
    if not cands:
        return None
    if fiscal_year is None:
        return cands[max(cands)]
    fy = str(int(fiscal_year))
    return next((cands[y] for y in (fy, str(int(fy) + 1), str(int(fy) - 1)) if y in cands), None)


def _load(ticker, statement, fiscal_year):
    """Return (dataframe, value_column, fiscal_year, accession) for a reconstructed statement, or
    (None, ...) if the company/statement/year isn't available."""
    if statement not in _STMT:
        return None, None, None, None
    cik = cik_for(ticker)
    if not cik:
        return None, None, None, None
    f = _filing(cik, fiscal_year)
    if f is None:
        return None, None, None, None
    period = str(getattr(f, "period_of_report", "") or "")
    fy = int(period[:4]) if period[:4].isdigit() else None
    try:
        df = getattr(f.obj().financials, _STMT[statement])().to_dataframe()
    except Exception:
        return None, None, None, None
    col = next((c for c in df.columns if fy and c.startswith(str(fy))), None) \
        or next((c for c in df.columns if c[:4].isdigit()), None)
    return df, col, fy, f.accession_no


def _concrete(df, col):
    """Real line items only: drop abstract headers and dimensional breakdown rows, keep rows with a value."""
    m = (~df["abstract"].fillna(False)) & (~df["dimension"].fillna(False)) & df[col].notna()
    return df[m]


def get_statement(ticker: str, statement: str = "balance_sheet", fiscal_year: int = None) -> str:
    """Return the FULL line items of a company's financial statement for a fiscal year — for
    statement structure or a line item NOT covered by get_financials (e.g. a bank's customer deposits).
    `statement` is "balance_sheet", "income_statement", or "cash_flow". Values come from XBRL and the
    result cites the filing. For a *tracked* metric (revenue, net income, assets…) use get_financials
    instead; for a "largest/smallest line item" question use largest_line_item."""
    df, col, fy, accn = _load(ticker, statement, fiscal_year)
    if df is None or col is None:
        return (f"No {statement} available for {ticker}"
                + (f" FY{fiscal_year}" if fiscal_year else "") + ".")
    lines = []
    for _, r in _concrete(df, col).iterrows():
        indent = "  " * max(0, int(r["level"]) - 4)
        lines.append(f"{indent}{r['label']}: {_fmt(r[col])}")
    src = f"[{ticker.upper()} · FY{fy} · {statement} · {accn} · {edgar_url(cik_for(ticker), accn)}]"
    return src + "\n" + "\n".join(lines)


def _pick_line_item(df, col, section, smallest=False):
    """Pure selection (unit-testable, no network): the largest/smallest LEAF line item (label, value)
    in a balance-sheet section, excluding abstract headers, dimensional rows, and subtotals ("Total …").
    Section membership is by the row's parent XBRL concept. Returns None if the section has no items."""
    match = _SECTION[section]
    items = [(r["label"], r[col]) for _, r in _concrete(df, col).iterrows()
             if isinstance(r["parent_concept"], str) and match(r["parent_concept"])
             and not str(r["label"]).lower().startswith("total")]
    if not items:
        return None
    return min(items, key=lambda x: x[1]) if smallest else max(items, key=lambda x: x[1])


def largest_line_item(ticker: str, section: str = "liabilities", fiscal_year: int = None,
                      smallest: bool = False) -> str:
    """The single largest (or smallest) LINE ITEM in a balance-sheet section — the deterministic answer
    to "what is X's largest liability/asset?". `section` is "liabilities", "assets", or "equity". The
    figure is from XBRL and the max/min is computed here (never scanned by the model); the answer cites
    the filing. Subtotals ("Total …") are excluded so it returns an actual line item."""
    if section not in _SECTION:
        return f"Unknown section '{section}'. Use one of: {', '.join(_SECTION)}."
    df, col, fy, accn = _load(ticker, "balance_sheet", fiscal_year)
    if df is None or col is None:
        return f"No balance sheet available for {ticker}" + (f" FY{fiscal_year}" if fiscal_year else "") + "."
    pick = _pick_line_item(df, col, section, smallest)
    if pick is None:
        return f"Could not identify {section} line items for {ticker} FY{fy}."
    sup = "smallest" if smallest else "largest"
    noun = {"liabilities": "liability", "assets": "asset", "equity": "equity item"}[section]
    return (f"{ticker.upper()}'s {sup} {noun} in FY{fy} is {pick[0]} at {_fmt(pick[1])}. "
            f"[source: FY{fy} balance sheet · {accn} · {edgar_url(cik_for(ticker), accn)}]")


# --- Part B: segment / geographic breakdown (dimensional XBRL) ------------------------------------
_AXIS = {"segment": "BusinessSegmentsAxis", "geography": "GeographicalAxis"}
_METRIC_CONCEPT = {"revenue": "Revenue", "operating_income": "OperatingIncomeLoss"}


def _value_for_fy(values, fy):
    """From a {period_key -> value} dict (keys like 'duration_2021-12-26_2022-12-31'), the value whose
    period ENDS in fiscal year `fy`. Pure/testable."""
    if not isinstance(values, dict):
        return None
    for k, v in values.items():
        dates = re.findall(r"(\d{4})-\d{2}-\d{2}", str(k))
        if dates and int(dates[-1]) == int(fy) and isinstance(v, (int, float)):
            return float(v)
    return None


def _member_for_axis(dim_meta, axis):
    """From a row's `dimension_metadata` (list of {dimension, member_label, ...}), the member label for
    `axis` — but ONLY if the row is a PURE breakdown by that axis: not also crossed with a different
    breakdown dimension (geography/product/customer/timing — a multi-axis cross-tab would double-count),
    and not an intersegment/corporate reconciling item. Returns None otherwise. Pure/testable. Uses the
    structured metadata (not the string label, whose member names can contain commas)."""
    if not isinstance(dim_meta, (list, tuple)):
        return None
    member = None
    for e in dim_meta:
        dim = str((e or {}).get("dimension") or "")
        lbl = str((e or {}).get("member_label") or "")
        if axis in dim:
            member = lbl
        elif "ConsolidationItemsAxis" in dim:
            if "Operating Segment" not in lbl:       # intersegment eliminations / corporate -> skip
                return None
        elif "Axis" in dim:                          # a second breakdown axis -> cross-tab -> skip
            return None
    return member


def _breakdown_from_xbrl(xb, axis, want_concept, fy):
    """Walk the Segment Reporting disclosures and collect {member -> value} for rows that are a PURE
    breakdown by `axis` (BusinessSegments / Geographical) whose concept matches `want_concept`, at
    fiscal year `fy`. Robust to per-company role names (matches by axis + concept, not a hardcoded role)
    and to multi-axis filers (rejects cross-tab rows via _member_for_axis, so segments don't double-count)."""
    import pandas as pd
    out = {}
    for s in xb.get_all_statements():
        if "Segment" not in (s.get("role_name") or "") or s.get("category") != "disclosure":
            continue
        try:
            df = pd.DataFrame(xb.get_statement(s["role"]))
        except Exception:
            continue
        if "dimension_metadata" not in df.columns or "values" not in df.columns:
            continue
        for _, r in df.iterrows():
            if not r.get("has_values") or want_concept not in str(r.get("concept") or ""):
                continue
            m = _member_for_axis(r.get("dimension_metadata"), axis)
            v = _value_for_fy(r.get("values"), fy)
            if m and v is not None and m not in out:
                out[m] = v
    return out


def get_segment_breakdown(ticker: str, dimension: str = "segment", metric: str = "revenue",
                          fiscal_year: int = None) -> str:
    """Revenue (or operating income) broken down BY BUSINESS SEGMENT or BY GEOGRAPHY for a fiscal year —
    dimensional XBRL data the flat metric tools (get_financials) can't reach. `dimension` is "segment"
    or "geography"; `metric` is "revenue" or "operating_income". Values come from XBRL and the answer
    cites the filing. Use this for "revenue by segment/region", "which segment is largest", etc."""
    if dimension not in _AXIS:
        return f"Unknown dimension '{dimension}'. Use 'segment' or 'geography'."
    if metric not in _METRIC_CONCEPT:
        return f"Unknown metric '{metric}'. Use 'revenue' or 'operating_income'."
    cik = cik_for(ticker)
    if not cik:
        return f"No data for {ticker}."
    try:
        f = _filing(cik, fiscal_year)
        if f is None:
            return f"No 10-K for {ticker}" + (f" FY{fiscal_year}" if fiscal_year else "") + "."
        period = str(getattr(f, "period_of_report", "") or "")
        fy = int(period[:4]) if period[:4].isdigit() else fiscal_year
        out = _breakdown_from_xbrl(f.xbrl(), _AXIS[dimension], _METRIC_CONCEPT[metric], fy)
    except Exception as e:
        return f"Could not extract {dimension} {metric} for {ticker}: {type(e).__name__}."
    if not out:
        return f"No {dimension} {metric} breakdown found for {ticker} FY{fy}."
    mname = "revenue" if metric == "revenue" else "operating income"
    lines = "\n".join(f"  {m}: {_fmt(v)}" for m, v in sorted(out.items(), key=lambda x: -x[1]))
    return (f"{ticker.upper()} FY{fy} {mname} by {dimension} (from the XBRL segment disclosure):\n{lines}\n"
            f"[source: FY{fy} 10-K · {f.accession_no} · {edgar_url(cik, f.accession_no)}]")


# --- Part B (P3): year-over-year growth by segment / geography ------------------------------------
def _growth_pct(this, prior):
    """(this - prior) / prior, or None if prior is 0/None. Pure/testable."""
    if prior in (None, 0) or this is None:
        return None
    return (this - prior) / abs(prior)


def _segment_growth(xb, axis, want_concept, fy):
    """{member -> (this_year, prior_year)} from the filing's OWN recast prior-year comparatives — so it
    survives a segment restructuring (the prior year is recast to the current structure IN THE SAME
    filing), unlike comparing two separate filings. Pure of superlatives; the caller computes growth."""
    import pandas as pd
    out = {}
    for s in xb.get_all_statements():
        if "Segment" not in (s.get("role_name") or "") or s.get("category") != "disclosure":
            continue
        try:
            df = pd.DataFrame(xb.get_statement(s["role"]))
        except Exception:
            continue
        if "dimension_metadata" not in df.columns or "values" not in df.columns:
            continue
        for _, r in df.iterrows():
            if not r.get("has_values") or want_concept not in str(r.get("concept") or ""):
                continue
            m = _member_for_axis(r.get("dimension_metadata"), axis)
            if not m or m in out:
                continue
            this, prior = _value_for_fy(r.get("values"), fy), _value_for_fy(r.get("values"), fy - 1)
            if this is not None and prior is not None:
                out[m] = (this, prior)
    return out


def get_segment_growth(ticker: str, dimension: str = "segment", metric: str = "revenue",
                       fiscal_year: int = None) -> str:
    """Year-over-year GROWTH of revenue (or operating income) BY BUSINESS SEGMENT or BY GEOGRAPHY — the
    deterministic answer to "which segment grew fastest / dragged down growth". Uses the filing's own
    recast prior-year figures (consistent even after a segment restructuring); growth is computed in
    code (never scanned by the model) and cited. NOTE: this is XBRL as-reported growth — it does NOT
    isolate organic / ex-M&A growth (that's a non-GAAP figure only in the narrative)."""
    if dimension not in _AXIS:
        return f"Unknown dimension '{dimension}'. Use 'segment' or 'geography'."
    if metric not in _METRIC_CONCEPT:
        return f"Unknown metric '{metric}'. Use 'revenue' or 'operating_income'."
    cik = cik_for(ticker)
    if not cik:
        return f"No data for {ticker}."
    try:
        f = _filing(cik, fiscal_year)
        if f is None:
            return f"No 10-K for {ticker}" + (f" FY{fiscal_year}" if fiscal_year else "") + "."
        period = str(getattr(f, "period_of_report", "") or "")
        fy = int(period[:4]) if period[:4].isdigit() else fiscal_year
        g = _segment_growth(f.xbrl(), _AXIS[dimension], _METRIC_CONCEPT[metric], fy)
    except Exception as e:
        return f"Could not compute {dimension} {metric} growth for {ticker}: {type(e).__name__}."
    rows = [(m, t, p, _growth_pct(t, p)) for m, (t, p) in g.items()]
    rows = sorted([r for r in rows if r[3] is not None], key=lambda x: -x[3])
    if not rows:
        return f"No {dimension} {metric} growth (with a comparable prior year) found for {ticker} FY{fy}."
    lines = "\n".join(f"  {m}: {_fmt(p)} -> {_fmt(t)}  ({gr:+.1%})" for m, t, p, gr in rows)
    return (f"{ticker.upper()} FY{fy} {metric} YoY by {dimension} (vs FY{fy - 1}, recast, as-reported):\n{lines}\n"
            f"  fastest: {rows[0][0]} ({rows[0][3]:+.1%}); most-declining: {rows[-1][0]} ({rows[-1][3]:+.1%})\n"
            f"[source: FY{fy} 10-K · {f.accession_no} · {edgar_url(cik, f.accession_no)}]")
