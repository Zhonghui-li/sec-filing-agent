"""The answer bubble renders the model's Markdown bold instead of printing asterisks.

195 of the 298 answers in the last two eval runs contain `**...**`; nothing in the prompt
tells the model to stop, so the page has to handle it. These tests read the shipped page,
the way tests/test_abstain_labels.py does, because that is where the bug was: the renderer
escaped and linkified the answer and then dropped it into the DOM with the asterisks intact.
"""
import re
from pathlib import Path

INDEX = Path(__file__).resolve().parents[1] / "service" / "static" / "index.html"


def _render_line():
    html = INDEX.read_text()
    line = re.search(r"html \+= '<span class=\"num\">' \+ (.*?) \+ '</span>';", html)
    assert line, "the answer render line moved — this test no longer guards anything"
    return line.group(1)


def _bold_pattern():
    """The actual regex literal the page ships, not a copy of it."""
    body = re.search(r"function bold\(s\)\{ return s\.replace\(/(.*?)/g, '(.*?)'\); \}",
                     INDEX.read_text())
    assert body, "bold() is gone or reshaped — the asterisk bug can come back silently"
    return re.compile(body.group(1)), body.group(2)


def _unmath_pattern():
    body = re.search(r"function unmath\(s\)\{ return s\.replace\(/(.*?)/g, ''\); \}",
                     INDEX.read_text())
    assert body, "unmath() is gone or reshaped — LaTeX delimiters can reach the page again"
    return re.compile(body.group(1))


def test_the_answer_goes_through_bold_and_is_escaped_first():
    call = _render_line()
    assert "bold(" in call, f"answer is rendered without bold(): {call}"
    assert "unmath(" in call, f"answer is rendered without unmath(): {call}"
    assert re.search(r"bold\((?:\w+\()*esc\(", call.replace(" ", "")), (
        f"esc() must run innermost, before any tag is inserted, or bold() hands the DOM "
        f"markup from the model's own text: {call}"
    )


def test_bold_pattern_renders_a_figure_and_stops_at_a_line_break():
    pattern, repl = _bold_pattern()
    out = pattern.sub(repl.replace("$1", r"\1"),
                      "margin was **16.3%** using **(op income + D&A) / revenue**.")
    assert "**" not in out and out.count("<strong>") == 2, out

    # a stray pair of asterisks must not swallow everything up to the next one a line later
    stray = "a ** b\nc ** d"
    assert pattern.sub(repl.replace("$1", r"\1"), stray) == stray


def test_latex_math_delimiters_do_not_reach_the_page():
    """Seen live: "16.3% \\((operating income + D&A) / revenue\\)" printed its backslashes."""
    out = _unmath_pattern().sub("", r"was 16.3% \((operating income + D&A) / revenue\).")
    assert out == "was 16.3% (operating income + D&A) / revenue.", out
