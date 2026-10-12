"""A source we could not reach is not a fact about what the filing contains.

On 2026-10-12 the live service told a user "I could not find CVS Health's main business segments
in the available filings", labelled NOT IN THE FILINGS, when the FY2025 10-K states them plainly
and a single SEC fetch had failed. Two layers of `except Exception: return []` had turned a
network failure into an empty result, and an empty result reads as absence.

That is the project's own thesis failing from the other side. A fabricated number and a confident
claim about what a filing does NOT contain are the same error: both assert something the evidence
does not support.

These tests spend nothing and reach no network.
"""
import json
import urllib.error
import urllib.request

import pytest

from agents import companyfacts as cf
from agents import finance_tools as ft


# --- the retry, which is what should keep the distinction rare ---------------------------------

class _Resp:
    def __init__(self, payload): self._p = payload
    def read(self): return json.dumps(self._p).encode()
    def __enter__(self): return self
    def __exit__(self, *a): return False


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(cf.time, "sleep", lambda _: None)


def test_a_transient_failure_is_retried_not_surfaced(no_sleep, monkeypatch):
    calls = {"n": 0}

    def flaky(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("simulated transient")
        return _Resp({"ok": True})

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    assert cf._get_json("https://example.invalid/x") == {"ok": True}
    assert calls["n"] == 3


def test_a_404_is_an_answer_and_is_not_retried(no_sleep, monkeypatch):
    """Retrying a definitive response wastes SEC's rate limit and our latency for nothing."""
    calls = {"n": 0}

    def not_found(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", not_found)
    with pytest.raises(urllib.error.HTTPError):
        cf._get_json("https://example.invalid/y")
    assert calls["n"] == 1


def test_a_503_is_retried_to_the_limit(no_sleep, monkeypatch):
    calls = {"n": 0}

    def unavailable(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError("u", 503, "Service Unavailable", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", unavailable)
    with pytest.raises(urllib.error.HTTPError):
        cf._get_json("https://example.invalid/z")
    assert calls["n"] == 3


# --- and when the retries are exhausted, the failure must stay distinguishable -----------------

def _always_unavailable(ticker):
    raise cf.SourceUnavailable("simulated: SEC unreachable")


def test_an_unreachable_source_is_raised_not_flattened_to_no_rows(monkeypatch):
    """CVS is not in the curated snapshot, so there is nothing to fall back to. Returning [] here
    is what produced "No data for CVS."."""
    monkeypatch.setattr(cf, "company_rows", _always_unavailable)
    with pytest.raises(cf.SourceUnavailable):
        ft._rows_for("CVS")


def test_curated_rows_still_serve_when_the_live_source_is_down(monkeypatch):
    """Apple IS in the snapshot. Losing the live extras should degrade, not fail."""
    monkeypatch.setattr(cf, "company_rows", _always_unavailable)
    assert ft._rows_for("AAPL"), "curated rows should still be served"


def test_the_tool_says_retrieval_failure_and_forbids_concluding_absence(monkeypatch):
    """The wording matters more than the exception: this string is what the model reads, and its
    default is to treat an empty tool result as evidence of absence and abstain not_in_filings."""
    pytest.importorskip("langchain_openai")
    from agents.sec_agent import TOOLS

    monkeypatch.setattr(cf, "company_rows", _always_unavailable)
    get_financials = next(t for t in TOOLS if t.name == "get_financials")
    out = get_financials.invoke({"ticker": "CVS", "metric": "revenue"})

    assert "RETRIEVAL FAILURE" in out
    assert "No company found" not in out, "must not read as 'this company does not exist'"
    assert "Do NOT conclude" in out and "not_in_filings" in out
