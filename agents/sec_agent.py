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
import re
from typing import Dict, List

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage, trim_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError

from agents.finance_tools import get_financials as _get_financials, compute as _compute
from agents.filings_retrieval import search_filings as _search_filings
from agents import observability

DEFAULT_RECURSION_LIMIT = 15
MAX_HISTORY_MSGS = 12  # keep the last ~6 turns of prior conversation (Model B: client sends history)
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

7. Do NOT guess or compute which fiscal year is the most recent, and do NOT infer a year \
from your sense of today's date — your internal sense of the current date is likely out of \
date. When the user gives no year, or uses ANY relative-time term ("latest", "most recent", \
"last year", "this year", "recently", "currently", "year over year"), treat it as the most \
recent fiscal year IN THE DATA: call get_financials WITHOUT a fiscal_year (it returns the \
latest available year), then use that year (and the year before it for YoY). For a comparison \
across companies with no year given, get each company's own latest fiscal year separately \
(fiscal calendars differ), and state which fiscal year you used for each.

7b. MULTI-TURN follow-ups: if an earlier turn in THIS conversation already fixed a fiscal year \
and the user's new message just continues the same line of questions without a new year \
(e.g. "and Microsoft?", "what about its net income?"), reuse that SAME fiscal year for \
consistency — do NOT switch to the latest. (An explicit year, or "latest"/"most recent", still \
overrides.) E.g. after "Apple's revenue in fiscal 2024", "and Microsoft?" means Microsoft FY2024.

Notes: map a plain year to the fiscal year (e.g. "2024" -> FY2024); fiscal years differ \
across companies (NVDA's fiscal year ends in January). Be concise and always cite tickers \
and accessions."""

# Appended only when a user has uploaded documents (build_agent(user_id=...)).
USER_DOCS_PROMPT = """

ADDITIONAL TOOL — search_my_documents: the user has uploaded their OWN private documents (e.g. an \
internal financial statement or memo for a company that is NOT one of the 7 public companies above). \
This OVERRIDES Rule 6: a company not in the public list is NOT automatically out_of_scope — it may \
be covered by the user's uploaded documents. So before abstaining out_of_scope, FIRST call \
search_my_documents; if it returns relevant passages, answer from them and cite the [filename · page]. \
Numbers found in the user's documents are acceptable to report (cite the filename · page); only abstain \
(not_in_filings) if search_my_documents finds nothing relevant. Still abstain off_topic for non-analysis \
requests (advice, forecasts, writing)."""


def _user_docs_tool(user_id: str):
    """Bind search_my_documents to a user_id via a closure — the user_id comes from the request
    context, NOT the LLM, so a user can only ever search their own uploaded documents."""
    from langchain_core.tools import tool as _tool
    from agents.user_docs_retrieval import search_my_documents as _smd

    @_tool
    def search_my_documents(query: str) -> str:
        """Search the USER'S OWN uploaded private documents (e.g. an internal financial
        statement or memo they provided) for relevant passages, each tagged [filename · page]
        to cite. Use this when the question is about a file the user uploaded, not a public
        SEC filing."""
        return _smd(query, user_id=user_id)
    return search_my_documents


def build_agent(model: str = None, temperature: float = 0.0, user_id: str = None):
    model = model or os.environ.get("GEN_LLM_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=temperature)
    # only add the private-docs tool when a user is in scope, so eval/public demo are unchanged
    tools = TOOLS + [_user_docs_tool(user_id)] if user_id else TOOLS
    prompt = SYSTEM_PROMPT + USER_DOCS_PROMPT if user_id else SYSTEM_PROMPT
    return create_react_agent(llm, tools, prompt=prompt)


def _extract_trace(messages) -> List[Dict]:
    trace = []
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            trace.append({"tool": tc["name"], "args": tc["args"]})
    return trace


def _build_messages(question: str, history) -> List:
    """Model B: the client (browser / open-webui) sends the prior conversation, the agent is
    stateless. Convert the history to LangChain messages, trim to the last few turns to bound
    token cost/latency (trim_messages, count-based), then append the new question."""
    prior = []
    for m in history or []:
        role, content = (m.get("role"), m.get("content")) if isinstance(m, dict) else (None, None)
        if not content:
            continue
        if role == "user":
            prior.append(HumanMessage(content))
        elif role == "assistant":
            prior.append(AIMessage(content))
        # ignore system/tool roles coming from the client
    prior = trim_messages(prior, token_counter=len, max_tokens=MAX_HISTORY_MSGS,
                          strategy="last", include_system=False, allow_partial=False)
    return prior + [HumanMessage(question)]


def run_agent(question: str, agent=None, history=None, verbose: bool = False,
              recursion_limit: int = DEFAULT_RECURSION_LIMIT) -> Dict:
    """Run the agent on a question. `history` (optional) is prior turns [{role, content}, ...]
    for multi-turn (Model B). Returns {answer, trace, tools_used, tool_outputs}."""
    agent = agent or build_agent()
    usage, tool_outputs = None, []
    # the Langfuse span (if enabled) wraps the invoke, so its duration is the real latency
    with observability.trace_agent(question) as record:
        try:
            result = agent.invoke({"messages": _build_messages(question, history)},
                                  {"recursion_limit": recursion_limit})
            messages = result["messages"]
            answer = messages[-1].content
            trace = _extract_trace(messages)
            # tool outputs (with tool name) — used to verify every number in the answer
            # traces back to get_financials/compute (not read out of search_filings prose).
            tool_outputs = [(getattr(m, "name", "") or "", m.content)
                            for m in messages if isinstance(m, ToolMessage)]
            usage = observability.sum_usage(messages)
        except GraphRecursionError:
            answer = ("I couldn't resolve this within the step limit — please narrow the "
                      "question.")
            trace = []
        tools_used = [t["tool"] for t in trace]
        # audit trail: which filings were cited + whether/why it abstained
        accns = sorted({a for _, c in tool_outputs
                        for a in re.findall(r"\d{10}-\d{2}-\d{6}", c)})
        # retrieved passages stored so the offline scorer (eval/score_traces.py) can
        # grade faithfulness on production traces (the eval <-> observability loop).
        # Keep the full retrieved block: an earlier 1500-char cap dropped most passages, which
        # made faithfulness/domain_judge see "unsupported" claims and emit false BADs (verified:
        # -0.38 mean faithfulness, 9/12 good->bad flips). 8000 holds a full k=5 search result.
        contexts = [c[:8000] for name, c in tool_outputs if name == "search_filings"][:6]
        audit = {"accessions_cited": accns, "abstained": "abstain" in tools_used,
                 "abstain_reason": next((t["args"].get("reason")
                                         for t in trace if t["tool"] == "abstain"), None),
                 "contexts": contexts}
        record(answer=answer, tools_used=tools_used, usage=usage, audit=audit)
    if verbose:
        for step in trace:
            print(f"  🔧 {step['tool']}({step['args']})")
    return {"answer": answer, "trace": trace, "tool_outputs": tool_outputs,
            "tools_used": tools_used}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "How did NVIDIA's revenue change year over year, and what does management attribute it to?"
    print(f"Q: {q}\n")
    out = run_agent(q, verbose=True)
    print("\nA:", out["answer"])
