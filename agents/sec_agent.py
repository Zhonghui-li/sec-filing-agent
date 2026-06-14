"""SEC Filing Agent — a tool-calling agent (LangGraph ReAct) over SEC 10-K filings.

The agent decides which tool to call for a financial question:
  - get_financials : exact figures from XBRL (any number must come from here)
  - compute        : deterministic YoY / ratios / differences (no mental math)
  - search_filings : qualitative narrative (risks, strategy, MD&A), with citations

Finance bar, enforced by the system prompt: numbers only from tools, every claim cited
to its source filing, and honest abstention when something isn't in the data.

Same create_react_agent orchestration as the Slug Advisor; only the tools and the
system prompt change (raised to the finance bar).
"""
import os
from typing import Dict, List

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError

from agents.finance_tools import get_financials as _get_financials, compute as _compute
from agents.filings_retrieval import search_filings as _search_filings

DEFAULT_RECURSION_LIMIT = 15
COMPANIES = {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon",
             "JPM": "JPMorgan Chase", "TSLA": "Tesla", "KO": "Coca-Cola"}
_COMPANY_LIST = ", ".join(f"{t} ({n})" for t, n in COMPANIES.items())

ABSTAIN_REASONS = {"out_of_scope", "not_reported", "not_in_filings",
                   "year_unavailable", "off_topic"}


def abstain(reason: str, detail: str = "") -> str:
    """Call this INSTEAD of answering whenever you cannot answer from the available data,
    so the refusal is explicit. `reason` MUST be one of:
      - out_of_scope: the company is not one of the covered companies
      - not_reported: the company does not report the requested metric (e.g. a bank's gross profit)
      - not_in_filings: the topic is not discussed in the filings
      - year_unavailable: the requested fiscal year is not available
      - off_topic: not a financial-analysis question answerable from the filings (writing
        tasks, opinions, investment advice, forecasts/predictions, real-time market data)
    Give a one-line `detail`. After calling this, briefly tell the user you can't answer and why."""
    r = reason.strip().lower()
    if r not in ABSTAIN_REASONS:
        return f"Invalid reason '{reason}'. Use one of: {', '.join(sorted(ABSTAIN_REASONS))}."
    return f"ABSTAIN[{r}] {detail}".strip()


TOOLS = [tool(_get_financials), tool(_compute), tool(_search_filings), tool(abstain)]

SYSTEM_PROMPT = f"""You are a financial-analysis assistant that answers questions about \
public companies' SEC 10-K filings. Covered companies (ticker = name): {_COMPANY_LIST}. \
Map any company name to its ticker before calling tools (e.g. "JPMorgan" -> JPM).

Use the tools; never rely on memory for facts or figures:
- get_financials: any exact financial number (revenue, net income, assets, EPS, ...).
- compute: any arithmetic — year-over-year change, ratios (e.g. net margin), differences.
- search_filings: qualitative content — risk factors, strategy, management's discussion.
- abstain: call this (instead of answering) whenever you cannot answer from the data.

HARD RULES (this is finance — wrong or unsupported numbers are unacceptable):
1. EVERY financial number must come from get_financials. NEVER state, estimate, or \
recall a figure yourself, and NEVER read a number out of search_filings text.
2. EVERY calculation must go through compute. Do NOT do arithmetic in your head.
3. For comparison/trend questions, call get_financials for each period, THEN compute.
4. CITE your sources: include the filing accession that the tools return for every \
factual claim (number or quote).
5. If you cannot answer from the data, call the `abstain` tool with the matching reason \
(not_reported / year_unavailable / not_in_filings), THEN briefly explain. But you MUST \
verify BEFORE abstaining: for not_reported or year_unavailable, FIRST call get_financials \
(the data may include years/metrics you don't expect — e.g. a fiscal year ending early next \
year may already be filed); for not_in_filings, FIRST call search_filings. Only abstain if \
the tool confirms it. Never fabricate a number, ratio, rate, or fact, and NEVER substitute \
a different metric as a "proxy" for the one asked (e.g. do NOT report revenue when asked \
for gross profit) — abstain instead.
6. Only the companies above are covered. If asked about another company, call `abstain` \
with reason out_of_scope. For requests that are NOT financial-analysis questions \
answerable from the filings — writing tasks, opinions, investment/buy-sell advice, \
forecasts or predictions, or real-time market data like stock prices — call `abstain` \
with reason off_topic. Do not call the data tools for these.

7. Do NOT guess which fiscal year is the most recent. When the user gives no year (or \
says "latest"/"most recent"/"year over year"), FIRST call get_financials WITHOUT a \
fiscal_year — it returns the latest available year — then use that year (and the year \
before it for YoY).

Notes: map a plain year to the fiscal year (e.g. "2024" -> FY2024); fiscal years differ \
across companies (NVDA's fiscal year ends in January). Be concise and always cite tickers \
and accessions."""


def build_agent(model: str = None, temperature: float = 0.0):
    model = model or os.environ.get("GEN_LLM_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=temperature)
    return create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)


def _extract_trace(messages) -> List[Dict]:
    trace = []
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            trace.append({"tool": tc["name"], "args": tc["args"]})
    return trace


def run_agent(question: str, agent=None, verbose: bool = False,
              recursion_limit: int = DEFAULT_RECURSION_LIMIT) -> Dict:
    """Run the agent on a question. Returns {answer, trace, tools_used}."""
    agent = agent or build_agent()
    try:
        result = agent.invoke({"messages": [("user", question)]},
                              {"recursion_limit": recursion_limit})
        messages = result["messages"]
        answer = messages[-1].content
        trace = _extract_trace(messages)
        # tool outputs (with tool name) — used to verify every number in the answer
        # traces back to get_financials/compute (not read out of search_filings prose).
        tool_outputs = [(getattr(m, "name", "") or "", m.content)
                        for m in messages if isinstance(m, ToolMessage)]
    except GraphRecursionError:
        answer = ("I couldn't resolve this within the step limit — please narrow the "
                  "question.")
        trace, tool_outputs = [], []
    if verbose:
        for step in trace:
            print(f"  🔧 {step['tool']}({step['args']})")
    return {"answer": answer, "trace": trace, "tool_outputs": tool_outputs,
            "tools_used": [t["tool"] for t in trace]}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "How did NVIDIA's revenue change year over year, and what does management attribute it to?"
    print(f"Q: {q}\n")
    out = run_agent(q, verbose=True)
    print("\nA:", out["answer"])
