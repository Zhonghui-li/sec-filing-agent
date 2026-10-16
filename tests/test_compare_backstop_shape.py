"""Detecting a comparison by what the agent DID, not by how the question was worded.

The backstop that stops "I fetched both figures and then declined to say which is larger" had
three conditions, two of which read the trace and one of which read the question with a regex
wanting which/who/whose beside a comparative. So "Compare that with Apple." -- a question
containing the word compare -- was invisible to it, and that phrasing refused on roughly 15% of
draws with nothing to catch it.

The note that documented this backstop had already drawn the lesson: detect the SHAPE of the
failure, not its phrasing, because the model only has to say it differently for phrasing to miss.
The trigger was then left as phrasing. The shape is in the trace: one metric fetched for two
different tickers IS a comparison, whatever words produced it, because the agent's own calls say
how it read the question.
"""
import pytest

pytest.importorskip("langchain_openai")
from agents.sec_agent import _declined_with_figures_in_hand, _fetched_one_metric_for_two_companies

TWO_COMPANIES = [
    {"tool": "get_financials", "args": {"ticker": "MSFT", "metric": "net_income"}, "output": "101832000000"},
    {"tool": "get_financials", "args": {"ticker": "AAPL", "metric": "net_income"}, "output": "112010000000"},
]
ONE_COMPANY = [
    {"tool": "get_growth", "args": {"ticker": "AAPL", "metric": "revenue"}, "output": "12.5"},
]
REFUSAL = ("I can't give a reliable figure for this — it needs a computation I don't have a "
           "deterministic tool for.")
ANSWERED = "Apple's net income was $112.010 billion versus Microsoft's $101.832 billion."


def test_an_imperative_comparison_is_now_seen():
    """The case this exists for. It was measured at 8/10 and 9/10 on two builds, so the ~15%
    refusal had no backstop at all -- the question never matched the trigger."""
    assert _declined_with_figures_in_hand("Compare that with Apple.", TWO_COMPANIES, REFUSAL)


@pytest.mark.parametrize("question", [
    "Compare that with Apple.",
    "Apple vs Microsoft on net income?",
    "How does Apple stack up against Microsoft?",
    "",                                   # a follow-up carrying no words of its own
])
def test_the_wording_stops_mattering(question):
    """Each of these refused with both figures in hand; none matches the comparative regex. The
    trace is the same in all four, which is the point."""
    assert _declined_with_figures_in_hand(question, TWO_COMPANIES, REFUSAL)


def test_the_phrasings_that_already_worked_still_do():
    assert _declined_with_figures_in_hand(
        "Which had higher net income in its latest fiscal year, Amazon or Microsoft?",
        TWO_COMPANIES, REFUSAL)


def test_an_answer_carrying_figures_does_not_fire():
    """The agent did the right thing; a retry here costs latency and money for nothing."""
    assert not _declined_with_figures_in_hand("Compare that with Apple.", TWO_COMPANIES, ANSWERED)


def test_a_single_company_percentage_answer_does_not_fire():
    """A growth answer is "12.5%" with no dollar figure in it, so the no-$ condition alone would
    fire on every one of them. Requiring two tickers for ONE metric is what holds them out."""
    assert not _declined_with_figures_in_hand(
        "What was Apple's revenue growth?", ONE_COMPANY, "Revenue grew 12.5%.")


def test_two_companies_but_different_metrics_is_not_a_comparison():
    """Fetching Apple's revenue and Microsoft's headcount is two lookups, not a comparison, and
    the retry message -- "say which is larger" -- would be nonsense advice."""
    trace = [
        {"tool": "get_financials", "args": {"ticker": "AAPL", "metric": "revenue"}, "output": "391035000000"},
        {"tool": "get_financials", "args": {"ticker": "MSFT", "metric": "total_assets"}, "output": "512163000000"},
    ]
    assert not _fetched_one_metric_for_two_companies(trace)


def test_the_same_company_twice_is_not_a_comparison():
    """Two years of one company is a trend question; nothing here is being ranked."""
    trace = [
        {"tool": "get_financials", "args": {"ticker": "AAPL", "metric": "revenue", "fiscal_year": 2025}, "output": "391035000000"},
        {"tool": "get_financials", "args": {"ticker": "AAPL", "metric": "revenue", "fiscal_year": 2024}, "output": "383285000000"},
    ]
    assert not _fetched_one_metric_for_two_companies(trace)


def test_a_trace_without_figure_tools_is_not_a_comparison():
    assert not _fetched_one_metric_for_two_companies(
        [{"tool": "search_filings", "args": {"ticker": "AAPL", "metric": "x"}, "output": "..."}])
