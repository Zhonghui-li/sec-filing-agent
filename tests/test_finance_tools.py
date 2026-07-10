"""L1 deterministic tests for the numeric tools — no LLM, no DB, no secrets.

These gate every PR: they verify the exact-number / abstain / citation-URL logic that
the agent depends on, using only the committed data (data/financials.json).
"""
from agents.finance_tools import (get_financials, compute, get_ratio, get_growth,
                                  compute_formula, edgar_url, cik_for)


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
    # an unresolved ticker nudges toward the company name (covers delisted/renamed), not a metric msg
    out = get_financials("ZZZZ", "revenue").lower()
    assert "no company found" in out and "company name" in out


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


def test_get_ratio_inventory_turnover():
    # cost_of_revenue / average inventory; alias "inventory turnover" resolves; turns format
    out = get_ratio("inventory turnover", "AAPL", 2024)
    assert "inventory_turnover" in out and "x" in out and "average inventory" in out.lower()
    assert "accession" in out


def test_get_ratio_abstains_when_base_missing():
    # a bank has no current_assets -> can't compute current_ratio, must say so (not fabricate)
    out = get_ratio("current_ratio", "JPM", 2024)
    assert "Cannot compute" in out


def test_get_ratio_unknown():
    assert "Unknown ratio" in get_ratio("made_up_ratio", "AAPL", 2024)


def test_get_growth_uses_consecutive_years():
    # YoY must compare a year with its IMMEDIATELY preceding year (the bug was a non-adjacent
    # baseline mislabeled as YoY). NVDA FY2024 = $60.922B, FY2023 = $26.974B -> +125.9%.
    out = get_growth("revenue", "NVDA", 2024)
    assert "FY2024" in out and "FY2023" in out and "+125.9%" in out


def test_get_growth_latest_when_year_omitted():
    out = get_growth("revenue", "NVDA")           # latest = FY2026 vs FY2025
    assert "FY2026" in out and "FY2025" in out and "year over year" in out


def test_get_growth_abstains_when_metric_not_reported():
    # JPMorgan does not report gross profit -> no YoY to compute, must say so (not fabricate)
    out = get_growth("gross profit", "JPM")
    assert "does not report" in out or "Available" in out


def test_compute_formula_dpo():
    # the exact FinanceBench DPO formula on Amazon FY2017 -> 93.86 (the ad-hoc version got 1419):
    # 365 * average AP / (COGS + change in inventory), evaluated deterministically in code
    out = compute_formula("365 * avg(accounts_payable) / (cost_of_revenue + delta(inventory))",
                          "AMZN", 2017)
    assert "93.86" in out and "accession" in out


def test_compute_formula_abstains_on_missing_metric():
    # a bank has no gross_profit -> the formula must abstain, not fabricate
    out = compute_formula("gross_profit / revenue", "JPM", 2023)
    assert "Abstain" in out


def test_compute_formula_rejects_unsafe():
    # only arithmetic + metric names + avg/delta/prev/abs — arbitrary calls / attributes are
    # rejected (caught either as an unknown name or by the AST walk), never evaluated
    for bad in ["__import__('os').system('x')", "revenue.__class__"]:
        assert "formula result" not in compute_formula(bad, "AAPL", 2024)


def test_compute_formula_multiyear_prev():
    # prev(metric, n) reaches n years back — enables a multi-year CAGR / average
    from agents.finance_tools import _value
    rev_2yr_back = _value("AAPL", "revenue", 2022)[0]
    out = compute_formula("prev(revenue, 2)", "AAPL", 2024)
    assert f"{rev_2yr_back:,.2f}" in out          # the value two years before FY2024


def test_compute_formula_small_result_keeps_precision():
    # a sub-1 result (margin / CAGR) must not be flattened to "0.00" by 2dp formatting
    import re
    out = compute_formula("net_income / prev(revenue, 2)", "AAPL", 2024)
    m = re.search(r"= ([0-9.]+)", out)
    assert m and 0 < float(m.group(1)) < 1 and m.group(1) != "0.00"


def test_financial_table_csv():
    from agents.finance_tools import financial_table_csv
    csv = financial_table_csv("AAPL", ["revenue", "gross_margin", "made_up_metric"], [2023, 2024])
    lines = csv.strip().splitlines()
    assert lines[0] == "metric,FY2023,FY2024"
    assert lines[1] == "revenue,383285000000,391035000000"   # base metric -> raw Excel-friendly USD
    assert lines[2].startswith("gross_margin (%),")           # ratio -> value with unit in the label
    assert lines[3] == "made_up_metric,,"                     # unknown metric -> blank cells, no crash


def test_xbrl_tag_name_resolves_to_slug():
    # a model that emits the real us-gaap tag name (not our shorthand) must still resolve
    from agents.finance_tools import _canon
    assert _canon("property_plant_and_equipment_net") == "ppe_net"   # PropertyPlantAndEquipmentNet
    assert _canon("net_income_loss") == "net_income"                 # NetIncomeLoss
    assert _canon("property, plant and equipment, net") == "ppe_net"  # punctuation variant collapses
    assert _canon("PP&E") == "ppe_net" and _canon("R&D") == "rd_expense"   # & preserved
    # and it flows through compute_formula instead of erroring as an unknown metric
    out = compute_formula("revenue / property_plant_and_equipment_net", "AAPL", 2024)
    assert "Unknown metric" not in out and "formula result" in out


def test_compute_formula_crosschecks_named_ratio_convention():
    # model hand-computes fixed-asset turnover with ENDING PP&E (wrong convention); the cross-check
    # flags our standard (average PP&E) value without overriding the model's result.
    out = compute_formula("revenue / ppe_net", "AAPL", 2024)
    assert "fixed_asset_turnover" in out and "our standard convention" in out
    # the correct-convention form (average) must NOT be flagged
    assert "[CHECK" not in compute_formula("revenue / avg(ppe_net)", "AAPL", 2024)
    # same metrics but a different-magnitude formula (inverse) is a different quantity -> no false flag
    assert "[CHECK" not in compute_formula("ppe_net / revenue", "AAPL", 2024)


def test_compute_formula_rejects_bad_periods():
    # years-back must be a positive integer literal, never a metric or 0/negative
    assert "Abstain" in compute_formula("prev(revenue, 0)", "AAPL", 2024)
    assert "Abstain" in compute_formula("prev(revenue, revenue)", "AAPL", 2024)
