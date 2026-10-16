"""The scale a financial statement states once, and never repeats.

A statement prints "(dollars in millions)" as a caption above the table; the cells then read
101,808. Reporting the cell alone is accurate about the number and wrong by a factor of a million
about the amount, which on a page of financial statements is the entire point of the figure.

The caption lives in the PAGE text, not in the column header -- the header for the filing this was
found on is a clean "Second Quarter.2026" with no unit in it -- so it is matched there and attached
to every fact from that page.
"""
import re

import pytest

_SRC = __import__("pathlib").Path(__file__).resolve().parents[1] / "ingest" / "ingest.py"


@pytest.fixture(scope="module")
def page_scale():
    """ingest.py imports docling at module level and the test venv deliberately has no docling
    (that is the whole point of the parser being a separate environment), so the function under
    test is taken from the source rather than by importing the module."""
    src = _SRC.read_text()
    ns = {"re": re}
    exec(src[src.index("_SCALE = re.compile"):src.index("def extract_table_facts")], ns)
    return ns["page_scale"]


@pytest.mark.parametrize("text,expected", [
    ("(dollars in millions except per share amounts)", "millions"),   # the one actually observed
    ("(in millions)", "millions"),
    ("($ in millions)", "millions"),
    ("(in $ millions)", "millions"),
    ("(in thousands)", "thousands"),
    ("(amounts in billions)", "billions"),
    ("(In Millions)", "millions"),                                    # case of the caption varies
])
def test_a_caption_states_the_scale(page_scale, text, expected):
    assert page_scale(text) == expected


@pytest.mark.parametrize("text", [
    "revenue grew in millions of transactions",     # prose, says nothing about the table
    "we serve customers in thousands of locations",
    "",
    None,
    "(no scale here)",
    "in millions",                                  # unparenthesised: not the convention
])
def test_prose_about_quantity_is_not_a_caption(page_scale, text):
    """Only a parenthesised caption counts. Publishers write the scale that way, so matching the
    convention is narrower -- and safer -- than matching the words anywhere on the page, where a
    sentence about millions of customers would relabel every figure in the document."""
    assert page_scale(text) is None


def test_the_caption_is_found_among_a_full_page_of_text(page_scale):
    page = ("BERKSHIRE HATHAWAY INC.\n"
            "Consolidated Statements of Earnings\n"
            "(dollars in millions except per share amounts)\n\n"
            "| | Second Quarter.2026 | Second Quarter.2025 |\n"
            "| Total revenues | 101,808 | 92,515 |\n")
    assert page_scale(page) == "millions"
