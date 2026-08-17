"""Deterministic, offline tests for the evidence-path replay (agents/replay.py). A fake tool
registry stands in for the live XBRL tools, so no network/keys are needed."""
from agents.replay import replay

_STORED = "PepsiCo, Inc. ebitda margin for FY2023 = 16.3% [sources: 0000077476-24-000008]"
_TRACE = [{"tool": "get_ratio",
           "args": {"ratio": "ebitda_margin", "ticker": "PEP", "fiscal_year": 2023},
           "output": _STORED}]


def _tools(output=_STORED, raises=False):
    def fn(**_):
        if raises:
            raise KeyError("ebitda_margin for PEP FY2023 not reported")
        return output
    return {"get_ratio": fn}


def test_verified_when_tool_reproduces_and_answer_grounded():
    rep = replay(_TRACE, "PepsiCo's EBITDA margin was 16.3%.", tools=_tools())
    assert rep["verdict"] == "VERIFIED"
    assert rep["numbers_grounded"] is True
    assert rep["checks"][0]["status"] == "reproduced"


def test_source_changed_when_fresh_value_differs():
    # the live figure moved (a later restatement) — a signal, not a fabrication
    fresh = "PepsiCo, Inc. ebitda margin for FY2023 = 18.9% [sources: 0000077476-24-000008]"
    rep = replay(_TRACE, "PepsiCo's EBITDA margin was 16.3%.", tools=_tools(fresh))
    assert rep["verdict"] == "SOURCE_CHANGED"
    assert rep["checks"][0]["status"] == "source_changed"


def test_ungrounded_when_answer_asserts_a_figure_no_tool_supports():
    # tool (stored and fresh) says 16.3%, but the answer claims 25% -> fabricated figure
    rep = replay(_TRACE, "PepsiCo's EBITDA margin was 25.0%.", tools=_tools())
    assert rep["verdict"] == "UNGROUNDED"
    assert 25.0 in rep["ungrounded"]


def test_irreproducible_when_a_tool_errors():
    rep = replay(_TRACE, "PepsiCo's EBITDA margin was 16.3%.", tools=_tools(raises=True))
    assert rep["verdict"] == "IRREPRODUCIBLE"
    assert rep["checks"][0]["status"] == "error"


def test_no_deterministic_evidence_for_a_narrative_only_answer():
    trace = [{"tool": "search_filings", "args": {"ticker": "NVDA"}, "output": "MD&A prose ..."}]
    rep = replay(trace, "Management attributes growth to data-center demand.", tools=_tools())
    assert rep["verdict"] == "NO_DETERMINISTIC_EVIDENCE"


def test_grounding_allows_unit_scale_change():
    # answer says "$6.93 billion"; the tool output is in whole dollars -> must still ground
    trace = [{"tool": "get_financials",
              "args": {"ticker": "NFLX", "metric": "free cash flow", "fiscal_year": 2023},
              "output": "NETFLIX INC free cash flow for FY2023 = $6,925,749,000"}]
    fresh = "NETFLIX INC free cash flow for FY2023 = $6,925,749,000"
    rep = replay(trace, "Netflix's free cash flow was $6.93 billion.",
                 tools={"get_financials": lambda **_: fresh})
    assert rep["verdict"] == "VERIFIED"
    assert rep["numbers_grounded"] is True
