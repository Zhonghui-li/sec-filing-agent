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
}


def _rows():
    global _ROWS
    if _ROWS is None:
        _ROWS = json.loads(_DATA.read_text())
    return _ROWS


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
    hits = [r for r in _rows() if r["ticker"] == tk and r["metric"] == key]
    if not hits:
        avail = sorted({r["metric"] for r in _rows() if r["ticker"] == tk})
        return (f"{tk} does not report '{metric}' (canonical: {key}) in the available "
                f"XBRL data. Reported metrics for {tk}: {', '.join(avail) or 'none'}.")
    if fiscal_year is not None:
        hits = [r for r in hits if r["fiscal_year"] == int(fiscal_year)]
        if not hits:
            yrs = sorted({r["fiscal_year"] for r in _rows()
                          if r["ticker"] == tk and r["metric"] == key})
            return (f"No {key} for {tk} FY{fiscal_year}. Available years: "
                    f"{yrs[0]}–{yrs[-1]}.")
    r = max(hits, key=lambda x: x["period_end"])   # latest if year omitted
    unit = r["unit"]
    val = f"${r['value']:,}" if unit == "USD" else f"{r['value']:,} {unit}"
    return (f"{tk} {key} for FY{r['fiscal_year']} (period ending {r['period_end']}): "
            f"{val}. [source: 10-K, accession {r['accession']}]")


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
