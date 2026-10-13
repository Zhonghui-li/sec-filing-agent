"""The UI must be able to name every reason the agent can give.

Until 2026-10-12 the badge read "Withheld · not in the filings" for all five abstain reasons. A
refusal to forecast a stock price — reason `off_topic` — was therefore displayed as a claim about
what the filings contain: an assertion about the world that the UI made and the agent did not. It
is the same failure as a retrieval error presented as an absence of data, with the same cause,
which is two places holding the same list and nothing comparing them.
"""
import re
from pathlib import Path

from agents.abstain import ABSTAIN_REASONS

INDEX = Path(__file__).resolve().parents[1] / "service" / "static" / "index.html"


def _ui_labels():
    """The keys of ABSTAIN_LABEL in the frontend."""
    block = re.search(r"const ABSTAIN_LABEL = \{(.*?)\};", INDEX.read_text(), re.S)
    assert block, "ABSTAIN_LABEL not found in index.html — did the badge move?"
    return dict(re.findall(r"(\w+):\s*'([^']*)'", block.group(1)))


def test_every_abstain_reason_has_a_label():
    missing = sorted(set(ABSTAIN_REASONS) - set(_ui_labels()))
    assert not missing, (
        f"the agent can abstain with {missing} and the UI has no wording for it, so it would "
        f"fall back to a generic label — or, before this test existed, to the wrong one"
    )


def test_the_ui_invents_no_reason_the_agent_cannot_give():
    extra = sorted(set(_ui_labels()) - set(ABSTAIN_REASONS))
    assert not extra, f"the UI names reasons the agent cannot produce: {extra}"


def test_no_label_claims_something_about_the_filings_unless_it_is_that_reason():
    """The specific regression: only `not_in_filings` may say the filings do not contain it."""
    for reason, label in _ui_labels().items():
        if reason != "not_in_filings":
            assert "not in the filings" not in label, (
                f"{reason} is labelled '{label}', which states something about the filings that "
                f"this reason does not mean"
            )
