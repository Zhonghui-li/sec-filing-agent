"""L1 deterministic tests for the on-demand XBRL extraction — no network, no secrets.

Uses a synthetic us-gaap facts dict to pin the extraction rules that let a dynamically fetched
company match the curated data/financials.json: full-year duration filtering (drop quarters/stubs),
instant handling for balance-sheet items, restatement dedup (latest accession wins), and candidate
tag fallback/merge. A live test that hits SEC is included but skipped unless SEC_LIVE_TEST is set.
"""
import os

import pytest

from agents.companyfacts import annual_values, extract_rows, quarterly_values

# Synthetic facts: revenue (duration) with a full year, a quarter, a restated year, and a 10-Q;
# a lower-priority revenue tag filling an older year; assets (instant).
GAAP = {
    "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
        {"form": "10-K", "start": "2023-10-01", "end": "2024-09-28", "val": 391000, "accn": "acc-2024"},
        {"form": "10-K", "start": "2024-06-30", "end": "2024-09-28", "val": 90000, "accn": "acc-2024"},   # quarter
        {"form": "10-K", "start": "2022-10-01", "end": "2023-09-30", "val": 350000, "accn": "acc-2024", "fy": 2024},  # later re-presentation
        {"form": "10-K", "start": "2022-10-01", "end": "2023-09-30", "val": 382000, "accn": "acc-2023", "fy": 2023},  # as originally reported
        {"form": "10-Q", "start": "2023-10-01", "end": "2024-09-28", "val": 999, "accn": "q"},            # wrong form
    ]}},
    "Revenues": {"units": {"USD": [
        {"form": "10-K", "start": "2019-10-01", "end": "2020-09-26", "val": 274000, "accn": "acc-2020"},  # gap-fill
    ]}},
    "Assets": {"units": {"USD": [
        {"form": "10-K", "end": "2024-09-28", "val": 365000, "accn": "acc-2024"},                          # instant
        {"form": "10-K", "start": "2024-06-30", "end": "2024-09-28", "val": 111, "accn": "acc-2024"},      # has start
    ]}},
}


def _rows():
    return extract_rows(GAAP, "TEST", "0000000000")


def _val(rows, metric, fy):
    hits = [r for r in rows if r["metric"] == metric and r["fiscal_year"] == fy]
    return hits[0]["value"] if hits else None


def test_full_year_duration_kept_quarter_dropped():
    rows = _rows()
    assert _val(rows, "revenue", 2024) == 391000   # the full year, not the 90000 quarter


def test_wrong_form_dropped():
    # the 10-Q value (999) must never appear
    assert all(r["value"] != 999 for r in _rows())


def test_restatement_prefers_as_reported_and_flags():
    # two accessions for FY2023: the value comes from that year's own FY2023 10-K (as reported),
    # NOT the later FY2024 re-presentation; the restated figure is surfaced separately.
    rows = [r for r in _rows() if r["metric"] == "revenue" and r["fiscal_year"] == 2023]
    assert len(rows) == 1
    assert rows[0]["value"] == 382000 and rows[0]["accession"] == "acc-2023"
    assert rows[0]["restated_value"] == 350000 and rows[0]["restated_accession"] == "acc-2024"


def test_candidate_tag_gap_fill():
    # 2020 comes only from the lower-priority "Revenues" tag
    r = [r for r in _rows() if r["metric"] == "revenue" and r["fiscal_year"] == 2020][0]
    assert r["value"] == 274000 and r["us_gaap_tag"] == "Revenues"


def test_quarterly_keeps_discrete_drops_ytd_and_annual():
    # a duration flow: keep the discrete 3-month quarter, drop the YTD span and the 10-K annual
    units = [
        {"form": "10-Q", "start": "2024-04-01", "end": "2024-06-29", "val": 85800, "accn": "q3",
         "fy": 2024, "fp": "Q3"},                                                                # ~89d discrete
        {"form": "10-Q", "start": "2024-01-01", "end": "2024-06-29", "val": 176600, "accn": "q3",
         "fy": 2024, "fp": "Q3"},                                                                # YTD 6-month
        {"form": "10-K", "start": "2023-10-01", "end": "2024-09-28", "val": 391000, "accn": "fy"},  # annual
    ]
    assert quarterly_values(units, "duration") == {"2024-06-29": {"val": 85800, "accn": "q3"}}


def test_quarterly_instant_at_quarter_end():
    # a balance-sheet instant at quarter-end is kept; a same-end duration (has start) is not
    units = [
        {"form": "10-Q", "end": "2024-06-29", "val": 331000, "accn": "q3", "fy": 2024, "fp": "Q3"},
        {"form": "10-Q", "start": "2024-04-01", "end": "2024-06-29", "val": 5, "accn": "q3",
         "fy": 2024, "fp": "Q3"},
    ]
    assert quarterly_values(units, "instant") == {"2024-06-29": {"val": 331000, "accn": "q3"}}


def test_fiscal_period_derives_quarter_from_date():
    from agents.companyfacts import _fiscal_period
    # Apple (FYE month 9): fiscal quarters end Dec/Mar/Jun/Sep
    assert _fiscal_period("2023-12-30", 9) == (2024, "Q1")   # holiday quarter -> fiscal Q1 2024
    assert _fiscal_period("2024-03-30", 9) == (2024, "Q2")
    assert _fiscal_period("2023-04-01", 9) == (2023, "Q2")   # spilled to the 1st -> prior month/quarter
    assert _fiscal_period("2024-06-29", 9) == (2024, "Q3")
    assert _fiscal_period("2024-09-28", 9) == (2024, "Q4")
    # calendar-year company (FYE month 12)
    assert _fiscal_period("2024-03-31", 12) == (2024, "Q1")
    assert _fiscal_period("2024-12-31", 12) == (2024, "Q4")
    # Walmart-style (FYE month 1, ends late January)
    assert _fiscal_period("2023-07-31", 1) == (2024, "Q2")


def test_instant_no_start_only():
    # assets is an instant: keep the point-in-time fact, drop the one that has a start
    rows = [r for r in _rows() if r["metric"] == "total_assets" and r["fiscal_year"] == 2024]
    assert len(rows) == 1 and rows[0]["value"] == 365000


def test_annual_values_duration_rejects_short_span():
    units = GAAP["RevenueFromContractWithCustomerExcludingAssessedTax"]["units"]["USD"]
    got = annual_values(units, "duration")
    assert got["2024-09-28"]["val"] == 391000        # ~362 days kept
    assert all(v["val"] != 90000 for v in got.values())  # ~90 days dropped


# Candidate-tag merge: the preferred tag wins each period; a lower-priority tag only fills periods
# it doesn't cover. A clean tag switch (no shared years) must fill the new tag's years.
SWITCH_GAAP = {
    "InventoryNet": {"units": {"USD": [
        {"form": "10-K", "end": "2011-05-31", "val": 2715, "accn": "a11"},  # old tag, ends 2011
    ]}},
    "InventoryFinishedGoodsNetOfReserves": {"units": {"USD": [
        {"form": "10-K", "end": "2021-05-31", "val": 6854, "accn": "a21"},  # new tag, no overlap
    ]}},
}


def test_part2_clean_switch_fills():
    rows = extract_rows(SWITCH_GAAP, "TEST", "0")
    inv = {r["fiscal_year"]: r["value"] for r in rows if r["metric"] == "inventory"}
    assert inv.get(2011) == 2715 and inv.get(2021) == 6854   # no overlap -> new tag fills


# A lower-priority tag LARGER than the preferred on the shared year is a fuller/alternative total
# (Block's Revenues vs a partial RevenueFromContract), not a component -> it must fill gap years.
ALT_TOTAL_GAAP = {
    "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
        {"form": "10-K", "start": "2018-01-01", "end": "2018-12-31", "val": 3205, "accn": "a18"},
    ]}},
    "Revenues": {"units": {"USD": [
        {"form": "10-K", "start": "2018-01-01", "end": "2018-12-31", "val": 3298, "accn": "a18"},  # larger
        {"form": "10-K", "start": "2019-01-01", "end": "2019-12-31", "val": 4713, "accn": "a19"},  # gap
        {"form": "10-K", "start": "2020-01-01", "end": "2020-12-31", "val": 9497, "accn": "a20"},  # gap
    ]}},
}


def test_part2_larger_alternative_total_fills():
    rows = extract_rows(ALT_TOTAL_GAAP, "TEST", "0")
    rev = {r["fiscal_year"]: r["value"] for r in rows if r["metric"] == "revenue"}
    assert rev.get(2018) == 3205                       # preferred still wins the shared year
    assert rev.get(2019) == 4713 and rev.get(2020) == 9497   # the fuller total fills the gaps


# AMD-style: an OLD tag differs a little across an accounting-standard transition (smaller one
# shared year, larger the next) — not a component, so its unique EARLY year must survive.
MIXED_TRANSITION_GAAP = {
    "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
        {"form": "10-K", "start": "2016-01-01", "end": "2016-12-31", "val": 4319, "accn": "b16"},
        {"form": "10-K", "start": "2017-01-01", "end": "2017-12-31", "val": 5253, "accn": "b17"},
    ]}},
    "SalesRevenueNet": {"units": {"USD": [
        {"form": "10-K", "start": "2015-01-01", "end": "2015-12-31", "val": 3991, "accn": "a16"},  # unique
        {"form": "10-K", "start": "2016-01-01", "end": "2016-12-31", "val": 4272, "accn": "a16"},  # ~1% smaller
        {"form": "10-K", "start": "2017-01-01", "end": "2017-12-31", "val": 5329, "accn": "a17"},  # larger
    ]}},
}


def test_part2_old_tag_mixed_transition_keeps_unique_year():
    rows = extract_rows(MIXED_TRANSITION_GAAP, "TEST", "0")
    rev = {r["fiscal_year"]: r["value"] for r in rows if r["metric"] == "revenue"}
    assert rev.get(2015) == 3991                             # only the old tag has it -> kept
    assert rev.get(2016) == 4319 and rev.get(2017) == 5253   # shared years use the preferred value


# One-directional but SMALL: an old tag restated down a few % on EVERY overlap year is below the
# component threshold (10%) -> kept, so its unique early year survives (the case the margin adds).
SMALL_DIFF_GAAP = {
    "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
        {"form": "10-K", "start": "2016-01-01", "end": "2016-12-31", "val": 1000, "accn": "b16"},
        {"form": "10-K", "start": "2017-01-01", "end": "2017-12-31", "val": 1100, "accn": "b17"},
    ]}},
    "SalesRevenueNet": {"units": {"USD": [
        {"form": "10-K", "start": "2015-01-01", "end": "2015-12-31", "val": 900, "accn": "a16"},   # unique
        {"form": "10-K", "start": "2016-01-01", "end": "2016-12-31", "val": 980, "accn": "a16"},   # 2% smaller
        {"form": "10-K", "start": "2017-01-01", "end": "2017-12-31", "val": 1078, "accn": "a17"},  # 2% smaller
    ]}},
}


def test_part2_one_directional_small_restatement_kept():
    rows = extract_rows(SMALL_DIFF_GAAP, "TEST", "0")
    rev = {r["fiscal_year"]: r["value"] for r in rows if r["metric"] == "revenue"}
    assert rev.get(2015) == 900                              # small (<10%) diff -> not a component -> kept


# Restatement: a later filing re-presents a prior year with a different value. as-reported (from
# the period's own FY10-K) is the primary value; the restated (latest accession) is surfaced too.
RESTATE_GAAP = {
    "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
        {"form": "10-K", "start": "2020-01-01", "end": "2020-12-31", "val": 382, "accn": "0-21-1", "fy": 2020},
        {"form": "10-K", "start": "2020-01-01", "end": "2020-12-31", "val": 173, "accn": "0-22-9", "fy": 2021},
        {"form": "10-K", "start": "2021-01-01", "end": "2021-12-31", "val": 500, "accn": "0-22-9", "fy": 2021},
    ]}},
}


def test_annual_values_reports_original_and_flags_restatement():
    from agents.companyfacts import annual_values
    av = annual_values(RESTATE_GAAP["NetCashProvidedByUsedInOperatingActivities"]["units"]["USD"],
                       "duration")
    fy2020 = av["2020-12-31"]
    assert fy2020["val"] == 382                       # as originally reported (its own FY2020 10-K)
    assert fy2020["restated_val"] == 173              # later re-presentation surfaced
    assert av["2021-12-31"]["restated_val"] is None   # unrestated year carries no restatement


def test_extract_rows_restatement_fields_only_when_present():
    rows = extract_rows(RESTATE_GAAP, "TEST", "0")
    by_year = {r["fiscal_year"]: r for r in rows if r["metric"] == "operating_cash_flow"}
    assert by_year[2020]["value"] == 382 and by_year[2020]["restated_value"] == 173
    assert "restated_value" not in by_year[2021]      # lean schema: absent when no restatement


@pytest.mark.skipif(not os.environ.get("SEC_LIVE_TEST"), reason="hits the SEC network")
def test_name_to_cik_delisted_and_renamed():
    from agents.companyfacts import name_to_cik, cik_for
    assert name_to_cik("Activision Blizzard") == "0000718877"   # delisted (ATVI no longer trades)
    assert name_to_cik("Block") == "0001512673"                 # renamed from Square
    assert cik_for("Activision Blizzard") == "0000718877"       # name falls through ticker miss
    assert name_to_cik("Square") is None                        # ambiguous name -> abstain


@pytest.mark.skipif(not os.environ.get("SEC_LIVE_TEST"), reason="hits the SEC network")
def test_live_uncovered_company():
    from agents.companyfacts import company_rows
    rows = company_rows("GOOGL")
    rev = [r for r in rows if r["metric"] == "revenue" and r["fiscal_year"] == 2024]
    assert rev and rev[0]["value"] > 300e9        # Alphabet FY2024 revenue ~ $350B
    assert company_rows("ZZZZZ") == []             # unknown ticker -> abstain path


# Targeted dynamic-path assertions on real, NON-curated companies — each exercises one edge of the
# live fetch. Network-gated (SEC_LIVE_TEST) so the fast/offline suite stays deterministic; run these
# on demand or in a separate live job when touching the dynamic code.
@pytest.mark.skipif(not os.environ.get("SEC_LIVE_TEST"), reason="hits the SEC network")
def test_live_delisted_resolves_by_name():
    # a delisted issuer (Activision, ticker ATVI gone) must resolve via its NAME -> CIK
    from agents.finance_tools import get_financials
    assert "6,489" in get_financials("Activision Blizzard", "revenue", 2019)   # FY2019 revenue $6,489M


@pytest.mark.skipif(not os.environ.get("SEC_LIVE_TEST"), reason="hits the SEC network")
def test_live_bank_has_revenue_but_abstains_on_inventory():
    # a bank reports revenue but has no inventory -> must give the figure, then abstain (not fabricate)
    from agents.finance_tools import get_financials
    assert "$" in get_financials("WFC", "revenue", 2023)
    assert "does not report" in get_financials("WFC", "inventory", 2023).lower()


@pytest.mark.skipif(not os.environ.get("SEC_LIVE_TEST"), reason="hits the SEC network")
def test_live_off_calendar_fiscal_year():
    # Walmart's fiscal year ends late January -> FY2023 must map to a January-2023 period end
    from agents.finance_tools import get_financials
    assert "2023-01" in get_financials("WMT", "revenue", 2023)


@pytest.mark.skipif(not os.environ.get("SEC_LIVE_TEST"), reason="hits the SEC network")
def test_live_tool_echoes_resolved_company_name():
    # a wrong-ticker guess must be VISIBLE: "BBBY" resolves to Bed Bath & Beyond, not Best Buy,
    # so the tool echoes the RESOLVED name — the mismatch can be caught instead of silently wrong
    from agents.finance_tools import get_financials
    assert "BED BATH" in get_financials("BBBY", "net income", 2016).upper()
    assert "BEST BUY" in get_financials("Best Buy", "net income", 2016).upper()
