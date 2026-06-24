"""L1 deterministic tests for the numeric tools — no LLM, no DB, no secrets.

These gate every PR: they verify the exact-number / abstain / citation-URL logic that
the agent depends on, using only the committed data (data/financials.json).
"""
from agents.finance_tools import get_financials, compute, get_ratio, edgar_url, cik_for


def test_exact_revenue():
    out = get_financials("AAPL", "revenue", 2024)
    assert "391,035,000,000" in out
    assert "accession" in out and "sec.gov/Archives/edgar/data/320193" in out  # clickable cite


def test_metric_alias():
    # "net income" -> net_income
    assert "net_income" in get_financials("AAPL", "net income", 2024)


def test_abstain_metric_not_reported():
    out = get_financials("JPM", "gross profit")
    assert "does not report" in out.lower()
    assert "net_income" in out  # lists what IS reported


def test_abstain_year_unavailable():
    out = get_financials("AAPL", "revenue", 2099)
    assert "available years" in out.lower()


def test_unknown_company():
    assert "does not report" in get_financials("ZZZZ", "revenue").lower()


def test_compute_yoy():
    assert "+100.0%" in compute("yoy", 200, 100)


def test_compute_ratio():
    assert "25.00%" in compute("ratio", 1, 4)


def test_compute_diff():
    assert "7" in compute("diff", 10, 3)


def test_compute_guards():
    assert "Cannot" in compute("yoy", 5, 0)
    assert "Unknown" in compute("nope", 1, 2)


def test_edgar_url_uses_company_cik():
    # company CIK (320193), not the accession prefix; standard EDGAR index path
    url = edgar_url("0000320193", "0000320193-25-000079")
    assert url == ("https://www.sec.gov/Archives/edgar/data/320193/"
                   "000032019325000079/0000320193-25-000079-index.htm")


def test_cik_lookup():
    assert int(cik_for("AAPL")) == 320193
    assert cik_for("ZZZZ") is None


def test_get_ratio_gross_margin():
    # gross_profit / revenue, as a percent, with a citation
    out = get_ratio("gross_margin", "AAPL", 2024)
    assert "gross_margin" in out and "%" in out and "accession" in out


def test_get_ratio_alias_and_convention():
    # alias resolves; ROA states the AVERAGE-assets convention (not period-end)
    out = get_ratio("return on assets", "AAPL", 2024)
    assert "roa" in out and "average" in out.lower()


def test_get_ratio_debt_is_not_total_liabilities():
    # debt_to_equity uses long-term debt; the definition makes the convention explicit
    out = get_ratio("debt to equity", "NVDA", 2024)
    assert "long-term debt" in out.lower() and "total liabilities" in out.lower()


def test_get_ratio_abstains_when_base_missing():
    # a bank has no current_assets -> can't compute current_ratio, must say so (not fabricate)
    out = get_ratio("current_ratio", "JPM", 2024)
    assert "Cannot compute" in out


def test_get_ratio_unknown():
    assert "Unknown ratio" in get_ratio("made_up_ratio", "AAPL", 2024)
