"""Which knob _chat_model sends, pinned to what the API actually accepts.

Sending the wrong one is a 400 at call time rather than at construction, so the failure shows up
as an eval full of errors, or a live service that answers nothing — never as something a unit test
would normally see. The table below is a transcription of real calls made on 2026-10-11 with
bind_tools(TOOLS) against both API paths; these tests cost nothing and spend nothing, they only
guarantee the routing still matches what was measured.

Re-measure before editing the table. The last time it was edited from memory rather than from a
call, the gpt-5.4 line was recorded backwards.
"""
import pytest

# L1 installs pytest and pandas only — deliberately, so it catches anything that depends on a
# package the minimal environment lacks. Importing agents.sec_agent pulls in langchain, so without
# this guard the module raises during COLLECTION, which does not skip one test: it interrupts the
# run and the other 203 never execute. test_filings_cache.py guards psycopg the same way.
#
# The consequence is that these tests run locally and skip in L1, so they catch a wrong knob at
# the moment the table is edited rather than in CI. Running the suite inside the built image,
# where langchain exists, is what would close that — see note 38's action list.
pytest.importorskip("langchain_openai")

from agents.sec_agent import _chat_model  # noqa: E402  (must follow the guard)

# model -> (knob on chat/completions, knob on /v1/responses), as measured 2026-10-11.
# "effort" = reasoning_effort accepted; "temp" = only temperature=0 accepted.
MEASURED = {
    "gpt-4o":       ("temp",   "temp"),
    "o4-mini":      ("effort", "effort"),
    "gpt-5-mini":   ("effort", "effort"),   # takes both; effort is the one we want
    "gpt-5-nano":   ("effort", "effort"),   # takes both
    "gpt-5.4":      ("temp",   "effort"),   # the only family that differs by path
    "gpt-5.4-mini": ("temp",   "effort"),
    "gpt-6-astra":  ("effort", "effort"),
}


def _knob_sent(model, responses, monkeypatch):
    monkeypatch.setenv("RESPONSES_API", "1" if responses else "0")
    llm = _chat_model(model)
    sent_effort = getattr(llm, "reasoning_effort", None) is not None
    return "effort" if sent_effort else "temp"


@pytest.mark.parametrize("model", sorted(MEASURED))
def test_chat_completions_knob_matches_what_the_api_accepts(model, monkeypatch):
    assert _knob_sent(model, False, monkeypatch) == MEASURED[model][0]


@pytest.mark.parametrize("model", sorted(MEASURED))
def test_responses_knob_matches_what_the_api_accepts(model, monkeypatch):
    assert _knob_sent(model, True, monkeypatch) == MEASURED[model][1]


def test_the_gpt_54_family_is_the_one_that_depends_on_the_path(monkeypatch):
    """The whole reason the pattern is not a constant. If this ever passes trivially — both paths
    agreeing for 5.4 — the branch has collapsed and one of the two paths is now sending a knob
    that family rejects."""
    assert _knob_sent("gpt-5.4-mini", False, monkeypatch) == "temp"
    assert _knob_sent("gpt-5.4-mini", True, monkeypatch) == "effort"


def test_responses_api_is_the_default_and_zero_is_the_escape_hatch(monkeypatch):
    monkeypatch.delenv("RESPONSES_API", raising=False)
    assert _chat_model("o4-mini").use_responses_api is True
    monkeypatch.setenv("RESPONSES_API", "0")
    assert _chat_model("o4-mini").use_responses_api in (None, False)
