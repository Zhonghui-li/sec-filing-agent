"""The abstain vocabulary, and the abstain the model writes instead of calling.

Its own module for the same reason agents/cold_starts.py is: nothing here needs a model, a
database or a network, so the L1 test lane — which installs pytest and pandas and nothing else —
can import it. agents/sec_agent.py pulls in langchain_core at module level, and a test that
reaches for this through there fails at collection, which fails the whole suite rather than one
file. That has now happened twice.
"""
import json
import re

# The categories a refusal may be filed under. Enforced in the tool: an unrecognised one comes
# back as an error rather than being recorded, so the counts mean something.
ABSTAIN_REASONS = {"out_of_scope", "not_reported", "not_in_filings",
                   "year_unavailable", "off_topic"}

_ABSTAIN_JSON_RX = re.compile(r'\{[^{}]*"reason"\s*:\s*"([a-z_]+)"[^{}]*\}', re.S)


def abstain_written_as_text(answer: str):
    """The abstain call the model wrote into the message CONTENT instead of calling, or None.

    Off-topic questions are where a refusal misses the tool, so the metric that counts refusals
    sees nothing — a refusal nobody can count. Three of the five off-topic cases fail that way, and
    what the user gets is the raw call:

        {"reason":"off_topic","detail":"Real-time market data such as current stock prices ..."}

    The arguments are right; only the channel is wrong. So this is a shipped output bug as much as
    an eval gap. Recognising a well-formed call is parsing, not persuading the model — the
    distinction note 31 draws when it says to record the behaviour rather than fight it: what it
    rules out is another prompt rule, not reading what the model actually produced.

    Deliberately strict. Only a JSON object whose `reason` is a declared category counts; prose
    that merely refuses is left alone, because inferring an abstention from wording is the
    keyword-matching this suite replaced with a structured signal in the first place.
    """
    m = _ABSTAIN_JSON_RX.search(answer or "")
    if not m or m.group(1) not in ABSTAIN_REASONS:
        return None
    try:
        payload = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict) or payload.get("reason") not in ABSTAIN_REASONS:
        return None
    return {"reason": payload["reason"], "detail": str(payload.get("detail", "")).strip()}
