"""L1 tests for trajectory scoring — no LLM, no network. Fixed traces in, verdicts out.

The contract these pin is as much about what the scorer REFUSES to judge as what it judges: a
metric that guesses when it can't tell becomes the next silent regression, so undecidable cases
must come back None and stay out of the denominator.
"""
from eval.trajectory import TRAJECTORY_METRICS, score_trajectory

CASE = {
    "id": "T1",
    "trajectory": {
        "paths": [{
            "id": "verify_then_abstain",
            "steps": [
                {"sid": "s1", "tool": "get_financials",
                 "args": {"ticker": "TSLA", "metric": "revenue", "fiscal_year": 2005},
                 "produces": "lookup"},
                {"sid": "s2", "tool": "abstain",
                 "args": {"reason": ["year_unavailable", "not_reported"]},
                 "depends_on": ["s1"]},
            ],
        }],
        "forbidden_tools": [], "call_slack": 1,
    },
}


def _call(tool, args, output=""):
    return {"tool": tool, "args": args, "output": output}


GOOD = [
    _call("get_financials", {"ticker": "TSLA", "metric": "revenue", "fiscal_year": 2005},
          "No revenue for TSLA FY2005. Available years: 2008-2025."),
    _call("abstain", {"reason": "year_unavailable", "detail": "TSLA has no FY2005 filing"}),
]


def test_a_clean_verify_then_abstain_passes():
    r = score_trajectory(CASE, GOOD)
    assert r["tool_precision"] and r["arg_correct"] and r["order_ok"] and r["call_budget"]


def test_either_accepted_reason_is_allowed():
    """The case lists two acceptable categories; picking the other one is not a failure."""
    trace = [GOOD[0], _call("abstain", {"reason": "not_reported"})]
    assert score_trajectory(CASE, trace)["arg_correct"]


def test_abstaining_without_checking_first_fails_order():
    assert score_trajectory(CASE, [GOOD[1], GOOD[0]])["order_ok"] is False


def test_a_wrong_year_is_a_wrong_argument_not_a_missing_tool():
    """The right tool with the wrong year must surface as arg_correct, not as a phantom absence."""
    trace = [_call("get_financials", {"ticker": "TSLA", "metric": "revenue", "fiscal_year": 2015}),
             GOOD[1]]
    r = score_trajectory(CASE, trace)
    assert r["arg_correct"] is False
    assert r["tool_precision"] is True          # the call was accounted for, not counted as extra


def test_an_unasked_for_call_costs_precision_not_recall():
    trace = [GOOD[0], _call("search_filings", {"query": "tesla 2005"}), GOOD[1]]
    r = score_trajectory(CASE, trace)
    assert r["tool_precision"] is False
    assert r["arg_correct"] is True             # the declared steps were still called correctly


def test_repeating_a_lookup_blows_the_budget():
    trace = [GOOD[0], GOOD[0], GOOD[0], GOOD[1]]      # 4 calls, 2 steps + 1 slack
    assert score_trajectory(CASE, trace)["call_budget"] is False


# --- what the scorer declines to judge -----------------------------------------------------
def test_a_case_with_no_declared_path_scores_nothing():
    """Every case in the suite today. Also an ordinary refusal: calling no tool is correct there
    and there is no path to compare it against, so these must not become five new failures."""
    assert score_trajectory({"id": "X"}, []) == {m: None for m in TRAJECTORY_METRICS}


def test_a_dependency_carried_in_free_text_is_undecidable_not_wrong():
    """s2 follows s1 and its argument is prose with no literal overlap. The agent may well have
    used what s1 returned — we can't see it, so the verdict is None. Scoring this False would
    punish a correct run for being hard to observe."""
    case = {"trajectory": {"paths": [{"id": "p", "steps": [
        {"sid": "s1", "tool": "get_segment_breakdown", "args": {"ticker": "DIS"}},
        {"sid": "s2", "tool": "search_filings", "depends_on": ["s1"]},
    ]}], "call_slack": 1}}
    trace = [_call("get_segment_breakdown", {"ticker": "DIS"}, "Experiences Segment: $36.2B"),
             _call("search_filings", {"query": "theme park attendance drivers"})]
    assert score_trajectory(case, trace)["dependency_ok"] is None


def test_a_dependency_that_visibly_reuses_the_output_passes():
    case = {"trajectory": {"paths": [{"id": "p", "steps": [
        {"sid": "s1", "tool": "get_segment_breakdown", "args": {"ticker": "DIS"}},
        {"sid": "s2", "tool": "search_filings", "depends_on": ["s1"]},
    ]}], "call_slack": 1}}
    trace = [_call("get_segment_breakdown", {"ticker": "DIS"}, "Experiences Segment: $36.2B"),
             _call("search_filings", {"query": "Experiences segment outlook"})]
    assert score_trajectory(case, trace)["dependency_ok"] is True


def test_a_free_text_argument_cannot_fail_arg_correct():
    """search_filings takes a query; no wording is declared, so it has nothing to be wrong about."""
    case = {"trajectory": {"paths": [{"id": "p", "steps": [
        {"sid": "s1", "tool": "search_filings"},
    ]}], "call_slack": 1}}
    assert score_trajectory(case, [_call("search_filings", {"query": "anything at all"})])["arg_correct"] is None


# --- multiple accepted paths ----------------------------------------------------------------
TWO_PATHS = {"trajectory": {"paths": [
    {"id": "narrative", "steps": [{"sid": "n1", "tool": "search_filings"}]},
    {"id": "structured", "steps": [{"sid": "g1", "tool": "get_segment_breakdown",
                                    "args": {"dimension": "segment"}}]},
], "call_slack": 1}}


def test_either_declared_path_is_accepted():
    for trace in ([_call("search_filings", {"query": "segments"})],
                  [_call("get_segment_breakdown", {"ticker": "DIS", "dimension": "segment"})]):
        assert score_trajectory(TWO_PATHS, trace)["tool_precision"] is True


def test_a_run_matching_neither_path_loses_precision():
    assert score_trajectory(TWO_PATHS, [_call("get_financials", {"ticker": "DIS"})])["tool_precision"] is False


def test_a_metric_alias_is_not_a_wrong_argument():
    """The tools accept several spellings of one metric and normalise them. Comparing raw strings
    marked a correct call wrong on this scorer's first real run (L27: the agent said
    "research_and_development_expense", the annotation said something else, both resolve to
    rd_expense). The scorer reuses the tools' own canonicaliser rather than keeping a second copy
    that could drift from it."""
    case = {"trajectory": {"paths": [{"id": "p", "steps": [
        {"sid": "s1", "tool": "get_financials", "args": {"ticker": "KO", "metric": "rd_expense"}},
    ]}], "call_slack": 1}}
    for spelling in ("research_and_development_expense", "research and development expense", "R&D"):
        trace = [_call("get_financials", {"ticker": "KO", "metric": spelling})]
        assert score_trajectory(case, trace)["arg_correct"] is True, spelling
    wrong = [_call("get_financials", {"ticker": "KO", "metric": "revenue"})]
    assert score_trajectory(case, wrong)["arg_correct"] is False
