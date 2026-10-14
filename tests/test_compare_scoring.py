"""A comparison question is scored on the conclusion, not only on the figures.

`compare.winner` was declared by every comparison case and read by nothing. The scorer checked
that both figures appeared in the answer, so an answer that listed them and named the WRONG
company scored the same as a correct one. The conclusion was carried instead by per-case `facts`
substring groups, which failed a correct answer on 2026-10-13 because the case listed "had higher"
and the answer said "had THE higher".

The check is clause-scoped. "Microsoft had higher net income" is a claim about Microsoft; the same
answer listing "Amazon (AMZN) FY2025: $77.67B" earlier is not a claim about Amazon, and a window
that crossed sentences would read it as one.
"""
import pytest

pytest.importorskip("pandas")          # eval.score imports it

from eval.score import _names_the_larger


def test_the_winner_named_as_larger_passes():
    a = ("Apple had the higher operating income in its latest fiscal year. "
         "Apple (AAPL) FY2025: $133.05 billion. Microsoft (MSFT) FY2025: $128.528 billion.")
    assert _names_the_larger(a, "AAPL", "MSFT") is True


def test_the_loser_named_as_larger_fails():
    assert _names_the_larger("Microsoft had higher net income than Apple.", "AAPL", "MSFT") is False


def test_listing_both_figures_without_a_conclusion_fails():
    """The whole point. This used to score the same as a correct answer."""
    assert _names_the_larger("Apple: $133B. Microsoft: $128B.", "AAPL", "MSFT") is False


def test_the_loser_appearing_earlier_is_not_read_as_a_claim():
    """Clause scoping. The winner is named second here, which is how the agent often writes it."""
    a = ("Microsoft had higher net income in its latest fiscal year. "
         "Amazon (AMZN) FY2025 net income: $77.67B. Microsoft (MSFT) FY2025: $101.83B.")
    assert _names_the_larger(a, "MSFT", "AMZN") is True


def test_wording_the_phrase_groups_used_to_miss():
    for phrasing in ("Apple had the higher operating income",
                     "Apple reported the higher operating income",
                     "Apple, at $133.05 billion, had higher operating income",
                     "Apple posted higher operating income"):
        assert _names_the_larger(phrasing + ". Microsoft was lower.", "AAPL", "MSFT") is True, phrasing


def test_an_unmapped_company_is_inconclusive_not_failed():
    """Adding a comparison case for a company the scorer cannot name must not start failing
    correct answers. None means the check does not apply."""
    assert _names_the_larger("Foo had higher revenue than Bar.", "FOO", "BAR") is None
