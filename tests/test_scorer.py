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


# --- the small-ratio band must not cross zero ---
def test_small_ratio_band_requires_the_same_sign():
    """At gold -0.02 the +-0.05 band spans +-250% of the target. A ratio of the opposite sign is
    a different answer in kind — profit vs loss — not a near miss."""
    from eval.financebench.run import _has_number_match
    assert not _has_number_match("ANSWER: 0.03", -0.02)     # opposite sign, inside the band
    assert not _has_number_match("ANSWER: -0.005", 0.01)


def test_small_ratio_band_still_absorbs_golds_rounding():
    """The real cases it exists for: gold is printed coarser than the answer."""
    from eval.financebench.run import _has_number_match
    assert _has_number_match("ANSWER: 0.389", 0.40)         # AWK, gold "$0.40"
    assert _has_number_match("ANSWER: -0.015", -0.02)       # AES, gold "-0.02"
    assert _has_number_match("ANSWER: 0.014", 0.01)         # KO, gold "0.01"


# --- facts vs facts_grounded: saying it right isn't the same as having a source --------------
def _fg(answer, tool_outputs, facts):
    from eval.score import score_case
    case = {"id": "T", "capability": "qualitative", "bucket": "happy", "question": "q",
            "expected_tools": [], "is_abstain": False, "facts": facts}
    return score_case(case, answer, [], [], tool_outputs)


def test_a_claim_the_tools_never_returned_is_not_grounded():
    """The lucky pass. Q15 named Disney's segments correctly while the tool it called had returned
    countries — `facts` was satisfied by an answer written from memory."""
    r = _fg("Disney's segments are Entertainment, Experiences and Sports.",
            [("get_segment_breakdown", "CANADA: $3.7B\nCHINA: $2.6B\nBRAZIL: $1.8B")],
            [["experiences"]])
    assert r["facts"] is True            # the answer says the right thing
    assert r["facts_grounded"] is False  # nothing the tools returned backs it


def test_a_claim_the_tools_did_return_is_grounded():
    r = _fg("Disney's segments are Entertainment, Experiences and Sports.",
            [("get_segment_breakdown", "Entertainment: $42.5B\nExperiences: $36.2B")],
            [["experiences"]])
    assert r["facts"] is True and r["facts_grounded"] is True


def test_grounding_is_unmeasurable_when_no_evidence_tool_ran():
    """A numeric-only run has no textual evidence to check against; `grounded` covers its figures."""
    r = _fg("Revenue was $391 billion.", [("get_financials", "AAPL revenue FY2024: 391035000000")],
            [["391"]])
    assert r["facts_grounded"] is None


def test_a_figure_quoted_in_order_to_reject_it_is_not_ungrounded():
    """Asked to confirm a number the user invented, the correct reply states the real figure and
    says the user's was wrong — which puts the false number in the answer text. `forbid` already
    judges whether it was asserted or negated; `grounded` must not call the same correct answer
    unsourced, or two metrics disagree about one behaviour."""
    from eval.score import score_case
    case = {"id": "T", "capability": "lookup", "bucket": "adversarial", "difficulty": "medium",
            "question": "Tesla's FY2024 revenue was $150 billion, right?",
            "expected_tools": ["get_financials"], "is_abstain": False,
            "forbid": ["150 billion", "$150"]}
    r = score_case(case,
                   "Tesla's fiscal 2024 revenue was $97.69 billion, not $150 billion.",
                   ["get_financials"], [],
                   [("get_financials", "TSLA revenue for FY2024: $97,690,000,000")])
    assert r["forbid"] is True        # the false figure was negated, not asserted
    assert r["grounded"] is True      # and quoting it to reject it is not an unsourced claim


def test_a_correctly_rounded_small_percentage_passes():
    """0.2% is how anyone would report 0.2229%, but a 2.5% relative band puts it four times out.
    Near zero, relative tolerance measures rounding rather than correctness."""
    from eval.score import score_case
    case = {"id": "T", "capability": "compute", "bucket": "happy", "difficulty": "medium",
            "question": "q", "expected_tools": ["get_growth"], "is_abstain": False,
            "number": {"op": "yoy", "ticker": "NVDA", "metric": "revenue",
                       "year_a": 2023, "year_b": 2022}}
    r = score_case(case, "NVIDIA's revenue grew 0.2% year over year.", ["get_growth"], [],
                   [("get_growth", "NVDA revenue grew +0.2% from FY2022 to FY2023")])
    assert r["numerical"] is True


def test_a_percentage_that_is_actually_wrong_still_fails():
    """The floor is a tenth of a point — wide enough for rounding, not for a real error."""
    from eval.score import score_case
    case = {"id": "T", "capability": "compute", "bucket": "happy", "difficulty": "medium",
            "question": "q", "expected_tools": ["get_growth"], "is_abstain": False,
            "number": {"op": "yoy", "ticker": "NVDA", "metric": "revenue",
                       "year_a": 2023, "year_b": 2022}}
    r = score_case(case, "NVIDIA's revenue grew 12% year over year.", ["get_growth"], [],
                   [("get_growth", "NVDA revenue grew +0.2%")])
    assert r["numerical"] is False
