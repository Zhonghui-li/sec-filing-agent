"""L1 deterministic tests for the scorer's number logic — no LLM, no DB, no secrets."""
from eval.score import extract_numbers, near, expected


def test_extract_billion():
    nums = extract_numbers("revenue was $391.035 billion")
    assert any(abs(v - 391_035_000_000) < 1 and not p for v, p in nums)


def test_extract_percent():
    nums = extract_numbers("a year-over-year increase of 65.5%")
    assert (65.5, True) in [(round(v, 1), p) for v, p in nums]


def test_extract_plain_dollars():
    nums = extract_numbers("$1,234,567")
    assert any(abs(v - 1_234_567) < 1 for v, _ in nums)


def test_near_within_tolerance():
    assert near([391_035_000_000], 391_000_000_000)        # 0.009% off -> within 2.5%
    assert not near([100], 200)
    assert not near([], 100)


def test_expected_lookup():
    val, is_pct, accns = expected(
        {"number": {"ticker": "AAPL", "metric": "revenue", "fiscal_year": 2024}})
    assert abs(val - 391_035_000_000) < 1 and not is_pct and accns


def test_expected_yoy_is_percent():
    val, is_pct, _ = expected(
        {"number": {"op": "yoy", "ticker": "NVDA", "metric": "revenue",
                    "year_a": 2026, "year_b": 2025}})
    assert is_pct and val > 0   # NVDA FY2026 grew vs FY2025


# --- the scorer must not read a citation as a figure ---
def test_accession_digits_are_not_numeric_candidates():
    """An EDGAR accession is a citation. _nums used to read its hyphens as minus signs, so
    0001065280-24-000030 contributed -24 -> (/100) -0.24, which the |gold| < 5 absolute band
    accepted as a match for a gold of -0.23 — scoring a WRONG answer (-0.50) correct."""
    from eval.financebench.run import _nums, _has_number_match
    assert _nums("[source: 10-K accession 0001065280-24-000030]") == [10.0]
    assert not _has_number_match(
        "The ratio was -0.50. [source: 10-K accession 0001065280-24-000030]", -0.23)


def test_a_real_figure_beside_a_citation_still_matches():
    from eval.financebench.run import _has_number_match
    assert _has_number_match(
        "3M's capex in fiscal 2018 was $1,577 million. [Accession 0000066740-19-000009]", 1577.0)


# --- the declared-answer slot ---
def test_answer_slot_is_read_instead_of_the_prose():
    from eval.financebench.run import _answer_slot, _nums, _has_number_match
    reply = ("3M's capital expenditures in fiscal 2018 were $1,577 million.\n"
             "(Source: 3M 10-K FY2018 [Accession 0000066740-19-000009])\n"
             "ANSWER: 1577")
    assert _answer_slot(reply) == "1577"
    # the whole reply offers 3M -> 3, fiscal 2018 -> 2018, 10-K -> 10 as candidates; the slot does not
    assert len(_nums(reply)) > len(_nums(_answer_slot(reply)))
    assert _nums(_answer_slot(reply)) == [1577.0]
    assert _has_number_match(_answer_slot(reply), 1577.0)


def test_answer_slot_absent_returns_none_so_callers_can_fall_back():
    from eval.financebench.run import _answer_slot
    assert _answer_slot("3M's capex was $1,577 million.") is None


def test_answer_slot_none_declares_no_number():
    from eval.financebench.run import _answer_slot, _nums
    assert _nums(_answer_slot("I can't find it.\nANSWER: none")) == []


def test_answer_slot_takes_the_last_one():
    from eval.financebench.run import _answer_slot
    assert _answer_slot("ANSWER: 1\nsecond thoughts\nANSWER: 2") == "2"
