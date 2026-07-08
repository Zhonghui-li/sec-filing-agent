"""L1 deterministic tests for the on-demand XBRL extraction — no network, no secrets.

Uses a synthetic us-gaap facts dict to pin the extraction rules that let a dynamically fetched
company match the curated data/financials.json: full-year duration filtering (drop quarters/stubs),
instant handling for balance-sheet items, restatement dedup (latest accession wins), and candidate
tag fallback/merge. A live test that hits SEC is included but skipped unless SEC_LIVE_TEST is set.
"""
import os

import pytest

from agents.companyfacts import annual_values, extract_rows

# Synthetic facts: revenue (duration) with a full year, a quarter, a restated year, and a 10-Q;
# a lower-priority revenue tag filling an older year; assets (instant).
GAAP = {
    "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
        {"form": "10-K", "start": "2023-10-01", "end": "2024-09-28", "val": 391000, "accn": "acc-2024"},
        {"form": "10-K", "start": "2024-06-30", "end": "2024-09-28", "val": 90000, "accn": "acc-2024"},   # quarter
        {"form": "10-K", "start": "2022-10-01", "end": "2023-09-30", "val": 383000, "accn": "acc-2024"},  # restated
        {"form": "10-K", "start": "2022-10-01", "end": "2023-09-30", "val": 382000, "accn": "acc-2023"},  # superseded
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


def test_restatement_latest_accession_wins():
    rows = [r for r in _rows() if r["metric"] == "revenue" and r["fiscal_year"] == 2023]
    assert len(rows) == 1
    assert rows[0]["value"] == 383000 and rows[0]["accession"] == "acc-2024"


def test_candidate_tag_gap_fill():
    # 2020 comes only from the lower-priority "Revenues" tag
    r = [r for r in _rows() if r["metric"] == "revenue" and r["fiscal_year"] == 2020][0]
    assert r["value"] == 274000 and r["us_gaap_tag"] == "Revenues"


def test_instant_no_start_only():
    # assets is an instant: keep the point-in-time fact, drop the one that has a start
    rows = [r for r in _rows() if r["metric"] == "total_assets" and r["fiscal_year"] == 2024]
    assert len(rows) == 1 and rows[0]["value"] == 365000


def test_annual_values_duration_rejects_short_span():
    units = GAAP["RevenueFromContractWithCustomerExcludingAssessedTax"]["units"]["USD"]
    got = annual_values(units, "duration")
    assert got["2024-09-28"]["val"] == 391000        # ~362 days kept
    assert all(v["val"] != 90000 for v in got.values())  # ~90 days dropped


# Part-2 guard: a lower-priority tag that is a *component* (conflicts on shared years) must not
# fill gap years; a clean tag switch (no shared years) must fill them.
GUARD_GAAP = {
    "InventoryNet": {"units": {"USD": [                                   # total (preferred)
        {"form": "10-K", "end": "2023-12-31", "val": 1000, "accn": "a23"},
        {"form": "10-K", "end": "2024-12-31", "val": 1100, "accn": "a24"},
    ]}},
    "InventoryFinishedGoodsNetOfReserves": {"units": {"USD": [            # finished-goods component
        {"form": "10-K", "end": "2024-12-31", "val": 600, "accn": "a24"},  # overlaps 2024, disagrees
        {"form": "10-K", "end": "2025-12-31", "val": 650, "accn": "a25"},  # a gap year (2025)
    ]}},
}

SWITCH_GAAP = {
    "InventoryNet": {"units": {"USD": [
        {"form": "10-K", "end": "2011-05-31", "val": 2715, "accn": "a11"},  # old tag, ends 2011
    ]}},
    "InventoryFinishedGoodsNetOfReserves": {"units": {"USD": [
        {"form": "10-K", "end": "2021-05-31", "val": 6854, "accn": "a21"},  # new tag, no overlap
    ]}},
}


def test_part2_guard_rejects_conflicting_component():
    rows = extract_rows(GUARD_GAAP, "TEST", "0")
    inv = {r["fiscal_year"]: r["value"] for r in rows if r["metric"] == "inventory"}
    assert inv.get(2024) == 1100           # kept the total, not the 600 component
    assert 2025 not in inv                 # component must NOT fill the gap year (it conflicts)


def test_part2_clean_switch_fills():
    rows = extract_rows(SWITCH_GAAP, "TEST", "0")
    inv = {r["fiscal_year"]: r["value"] for r in rows if r["metric"] == "inventory"}
    assert inv.get(2011) == 2715 and inv.get(2021) == 6854   # no overlap -> new tag fills


@pytest.mark.skipif(not os.environ.get("SEC_LIVE_TEST"), reason="hits the SEC network")
def test_live_uncovered_company():
    from agents.companyfacts import company_rows
    rows = company_rows("GOOGL")
    rev = [r for r in rows if r["metric"] == "revenue" and r["fiscal_year"] == 2024]
    assert rev and rev[0]["value"] > 300e9        # Alphabet FY2024 revenue ~ $350B
    assert company_rows("ZZZZZ") == []             # unknown ticker -> abstain path
