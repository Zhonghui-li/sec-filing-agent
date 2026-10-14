"""The scorer has to read a number in whatever form the agent wrote it.

`$77.67B` parsed as 77.67 until 2026-10-13, so an answer that was right scored as a wrong number.
It surfaced while measuring comparison questions: one run wrote `$77.67B` where the others wrote
`$77.67 billion`, and only that run was marked wrong. The agent is free to choose either form, so
a scorer that reads one of them measures formatting rather than correctness, and a formatting
change looks exactly like a regression.
"""
import pytest

pytest.importorskip("pandas")          # eval.score imports it

from eval.score import extract_numbers


def _values(text):
    return [v for v, _ in extract_numbers(text)]


@pytest.mark.parametrize("written,expected", [
    ("$77.67 billion", 77_670_000_000),
    ("$77.67B", 77_670_000_000),
    ("$101.83bn", 101_830_000_000),
    ("$450 million", 450_000_000),
    ("$450M", 450_000_000),
    ("$450mm", 450_000_000),
    ("$1.2 trillion", 1_200_000_000_000),
    ("$1.2T", 1_200_000_000_000),
    ("$101,832,000,000", 101_832_000_000),
])
def test_every_form_of_the_same_figure_parses_the_same(written, expected):
    assert _values(written) == pytest.approx([expected], rel=1e-9)


def test_a_bare_k_is_not_a_magnitude():
    """"10k shares" is a count, not ten thousand dollars, and nothing in this domain reports money
    that way. Reading it as a magnitude would invent figures out of ordinary prose."""
    assert _values("10k shares outstanding") == []


def test_percentages_are_still_flagged_and_not_scaled():
    vals = extract_numbers("revenue grew 12% year over year")
    assert vals == [(12.0, True)]


def test_a_suffix_must_be_a_whole_token():
    """A letter glued to a number is only a magnitude when it is the whole suffix. Without this,
    any identifier ending in a digit followed by b, m or t would become a billion."""
    assert _values("FY2025") == [2025.0]
    assert _values("item 7a") == []
