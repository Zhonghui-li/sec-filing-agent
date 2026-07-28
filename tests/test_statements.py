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
