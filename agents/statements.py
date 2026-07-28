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
