"""Cross-turn fiscal_year carry — reproducible reliability check for the prompt rule that a follow-up
which omits the year reuses the conversation's fiscal_year (and does NOT when the user starts a new /
current-period / different-company request). Runs the real agent (SEC + OpenAI + DB), so it is gated
behind SEC_LIVE_TEST like the other live tests; skipped in the fast suite.

Run:  SEC_LIVE_TEST=1 <venv>/bin/python -m pytest tests/test_followup_year_live.py -q
"""
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("SEC_LIVE_TEST"),
                                reason="runs the real agent against SEC + OpenAI")

YEAR = 2022


def _run_followup(t1, t2):
    """Run Turn 1, then the follow-up Turn 2 with Turn 1 in history; return the fiscal_years the agent
    passed to search_filings on Turn 2."""
    import agents.sec_agent as sa
    calls = []
    orig = sa._search_filings

    def spy(query, ticker=None, k=5, fiscal_year=None):
        calls.append(fiscal_year)
        return orig(query, ticker, k, fiscal_year)

    sa._search_filings = spy
    try:
        r1 = sa.run_agent(t1)
        hist = [{"role": "user", "content": t1}, {"role": "assistant", "content": r1["answer"]}]
        calls.clear()
        sa.run_agent(t2, history=hist)
    finally:
        sa._search_filings = orig
    return calls


@pytest.mark.parametrize("t1,t2", [
    ("What drove 3M's operating margin change in FY2022?", "What major risk factors did management highlight?"),
    ("Summarize AMD's business and strategy for FY2022.", "And what were its main risk factors?"),
])
def test_same_company_followup_carries_the_year(t1, t2):
    assert YEAR in _run_followup(t1, t2)          # the follow-up reused FY2022


@pytest.mark.parametrize("t1,t2", [
    ("What drove 3M's FY2022 gross margin change?", "What are Apple's most recent risk factors?"),
    ("Summarize AMD's FY2022 strategy.", "What are AMD's most current risks, as of the latest filing?"),
])
def test_new_or_current_request_does_not_carry_the_year(t1, t2):
    assert YEAR not in _run_followup(t1, t2)       # new company / explicit latest -> not FY2022
