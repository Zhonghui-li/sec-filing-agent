"""Unit tests for the finance-bar-critical part of report-level reconstruction: picking the
largest/smallest LINE ITEM in a balance-sheet section. Pure logic on a synthetic statement — no
network — so a wrong pick (a subtotal, an equity row, the wrong section) is caught deterministically."""
import pandas as pd

from agents.statements import _pick_line_item


def _bs():
    # label, 2022 value, abstract, dimension, parent_concept  (mirrors edgartools' to_dataframe)
    rows = [
        ("Assets", None, True, False, None),                                  # header, no value
        ("Cash", 30.0, False, False, "us-gaap_Assets"),
        ("Financing receivables, net", 167.0, False, False, "us-gaap_Assets"),
        ("Total assets", 228.0, False, False, None),                          # total (parent None)
        ("Customer deposits", 110.0, False, False, "us-gaap_Liabilities"),
        ("Long-term debt", 42.0, False, False, "us-gaap_Liabilities"),
        ("Other liabilities", 37.0, False, False, "us-gaap_Liabilities"),
        ("Total liabilities", 203.0, False, False, "us-gaap_LiabilitiesAndStockholdersEquity"),
        ("Retained earnings", 16.0, False, False, "us-gaap_StockholdersEquity"),
        ("Total shareholders' equity", 24.0, False, False, "us-gaap_LiabilitiesAndStockholdersEquity"),
        ("Segment breakdown", 999.0, False, True, "us-gaap_Liabilities"),     # dimensional row -> excluded
    ]
    return pd.DataFrame(rows, columns=["label", "2022", "abstract", "dimension", "parent_concept"])


def test_largest_liability_is_a_line_item_not_a_total_or_equity_or_dimension():
    assert _pick_line_item(_bs(), "2022", "liabilities") == ("Customer deposits", 110.0)


def test_smallest_liability():
    assert _pick_line_item(_bs(), "2022", "liabilities", smallest=True) == ("Other liabilities", 37.0)


def test_largest_asset_excludes_the_total():
    assert _pick_line_item(_bs(), "2022", "assets") == ("Financing receivables, net", 167.0)


def test_equity_section_isolated_from_liabilities():
    assert _pick_line_item(_bs(), "2022", "equity") == ("Retained earnings", 16.0)


def test_empty_section_returns_none():
    only_assets = _bs()[_bs()["parent_concept"].isin(["us-gaap_Assets", None])]
    assert _pick_line_item(only_assets, "2022", "liabilities") is None


# --- Part B: segment breakdown pure helpers (no network) -----------------------------------------

def test_value_for_fy_picks_the_period_ending_in_the_fiscal_year():
    from agents.statements import _value_for_fy
    values = {"duration_2021-12-26_2022-12-31": 6043000000.0,
              "duration_2020-12-27_2021-12-25": 3694000000.0}
    assert _value_for_fy(values, 2022) == 6043000000.0
    assert _value_for_fy(values, 2021) == 3694000000.0
    assert _value_for_fy(values, 2019) is None          # no period ends in 2019
    assert _value_for_fy("not a dict", 2022) is None


def test_member_for_axis_pure_vs_crosstab():
    from agents.statements import _member_for_axis
    seg = [{"dimension": "us-gaap:StatementBusinessSegmentsAxis", "member_label": "Datacenter"}]
    assert _member_for_axis(seg, "BusinessSegmentsAxis") == "Datacenter"
    # ConsolidationItems=Operating Segments is a scope qualifier, still a pure segment breakdown
    scoped = [{"dimension": "srt:ConsolidationItemsAxis", "member_label": "Operating Segments"},
              {"dimension": "us-gaap:StatementBusinessSegmentsAxis", "member_label": "Commercial"}]
    assert _member_for_axis(scoped, "BusinessSegmentsAxis") == "Commercial"
    # crossed with ANOTHER breakdown axis (geography) -> rejected, else it double-counts
    crosstab = [{"dimension": "us-gaap:StatementBusinessSegmentsAxis", "member_label": "Commercial"},
                {"dimension": "srt:StatementGeographicalAxis", "member_label": "US"}]
    assert _member_for_axis(crosstab, "BusinessSegmentsAxis") is None
    # intersegment / corporate reconciling item -> rejected
    elim = [{"dimension": "srt:ConsolidationItemsAxis", "member_label": "Intersegment revenues, eliminated"}]
    assert _member_for_axis(elim, "BusinessSegmentsAxis") is None
    assert _member_for_axis("not a list", "BusinessSegmentsAxis") is None
