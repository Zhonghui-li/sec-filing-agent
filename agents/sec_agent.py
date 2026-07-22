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

from agents.finance_tools import (get_financials as _get_financials, compute as _compute,
                                  get_ratio as _get_ratio, get_growth as _get_growth,
                                  compute_formula as _compute_formula)
from agents.filings_retrieval import search_filings as _search_filings
from agents.guardrail import guardrail
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


TOOLS = [tool(_get_financials), tool(_compute), tool(_get_ratio), tool(_get_growth),
         tool(_compute_formula), tool(_search_filings), tool(abstain)]

SYSTEM_PROMPT = f"""You are a financial-analysis assistant that answers questions about \
public companies' SEC 10-K filings. For NUMBERS (exact figures, ratios, year-over-year growth) \
you cover ANY U.S. public company — the tools fetch its XBRL data live by ticker. For QUALITATIVE \
filing text (risk factors, strategy, management's discussion) you have full-text search only for \
these companies: {_COMPANY_LIST}. Map any company name to its ticker before calling tools \
(e.g. "JPMorgan" -> JPM, "Alphabet" -> GOOGL).

Use the tools; never rely on memory for facts or figures:
- get_financials: any exact financial number (revenue, net income, assets, EPS, ...). For a \
QUARTERLY figure (e.g. "Q2 revenue"), pass quarter=1/2/3 with the fiscal_year (Q4 isn't filed on \
its own — use the full year). Omit quarter for the annual (10-K) figure.
- compute: ad-hoc arithmetic — differences, a one-off ratio between two figures you already have.
- get_growth: year-over-year (YoY) change of a metric. Prefer this for ANY "year over year" / \
"how did X change" question — it fetches the year AND its immediately preceding year itself, so \
the comparison is always between consecutive years (do NOT fetch two years and call compute, \
which lets a wrong baseline slip in).
- get_ratio: a STANDARD financial ratio (gross/operating/net margin, cogs_pct, roa, roe, \
current_ratio, quick_ratio, payout_ratio, debt_to_equity, and activity ratios dpo, dso, dio, \
asset_turnover, fixed_asset_turnover, capex_pct_revenue, interest_coverage). ALWAYS prefer this \
over assembling a ratio from parts yourself — multi-step ratios (esp. days-outstanding like DPO) \
are error-prone by hand; the formulas and conventions (ROA uses AVERAGE assets, "debt" means \
long-term debt not total liabilities, days ratios use 365 × average balance) are fixed in the tool.
- search_filings: qualitative content — risk factors, strategy, management's discussion, and \
recent CORPORATE EVENTS from 8-K / quarterly 10-Q filings (a debt/notes issuance, a buyback \
authorization, a dividend action, a material agreement, an executive change).
- abstain: call this (instead of answering) whenever you cannot answer from the data.

HARD RULES (this is finance — wrong or unsupported numbers are unacceptable):
1. EVERY financial number must come from get_financials. NEVER state, estimate, or \
recall a figure yourself, and NEVER read a number out of search_filings text.
2. EVERY calculation must go through compute. Do NOT do arithmetic in your head.
3. For a year-over-year change of one metric, use get_growth (never pick the two years \
yourself). For other comparisons (across companies, or two specific figures), call \
get_financials for each, THEN compute.
4. CITE your sources: include the filing accession that the tools return for every \
factual claim (number or quote).
5. If you cannot answer from the data, call the `abstain` tool with the matching reason \
(not_reported / year_unavailable / not_in_filings), THEN briefly explain. But you MUST \
verify BEFORE abstaining: for not_reported or year_unavailable, FIRST call get_financials \
(the data may include years/metrics you don't expect — e.g. a fiscal year ending early next \
year may already be filed); for not_in_filings, FIRST call search_filings. Only abstain if \
the tool confirms it. Never fabricate a number, ratio, rate, or fact, and NEVER substitute \
a different metric as a "proxy" for the one asked (e.g. do NOT report revenue when asked \
for gross profit; "debt" means interest-bearing debt (long_term_debt), NOT total liabilities — \
do NOT use total liabilities to answer a question about debt) — abstain instead.
6. NUMBERS cover ANY U.S. public company. For a financial figure / ratio / growth about a \
company NOT in the list above, still call get_financials / get_ratio / get_growth with its \
ticker (the tool fetches its XBRL live) — do NOT abstain out_of_scope. If the company is DELISTED \
or RENAMED (its ticker no longer trades, e.g. Activision, or Square which became Block), pass its \
full COMPANY NAME instead of a ticker — the tool resolves the name (including former names) to the \
right filer. Only abstain out_of_scope if the tool returns no data (an unknown company or a \
private company). QUALITATIVE filing text (search_filings) ALSO covers ANY U.S. public company — \
for a risk / strategy / MD&A question, OR a question about a recent CORPORATE EVENT (notes \
issued, buyback authorized, dividend changed, agreement signed, executive appointed), still CALL \
search_filings with its ticker (it fetches and indexes that company's 10-K, recent quarterly 10-Q, \
and recent 8-K narrative live on first use). The corpus INCLUDES recent filings, so do NOT assume \
an event is "too recent" or "not in my data" and abstain — SEARCH FIRST; only if search_filings \
then returns no relevant passages should you say it isn't available. NEVER abstain on a narrative \
or corporate-event question WITHOUT calling search_filings first, and never fabricate narrative. For requests that are NOT financial-analysis questions \
answerable from filings — writing tasks, opinions, investment/buy-sell advice, forecasts or \
predictions, or real-time market data like stock prices — call `abstain` with reason off_topic. \
Do not call the data tools for these.

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

8. DERIVED / MULTI-STEP metrics: use get_ratio if it lists the metric. If it does NOT — e.g. cash \
conversion cycle, EBITDA / EBITDA margin, or a formula spelled out in the question — use \
compute_formula: write the WHOLE formula ONCE with our metric names as variables plus the helpers \
avg() / delta() / prev() (it fetches every figure from XBRL and evaluates the whole expression \
deterministically). For a MULTI-YEAR span (an N-year CAGR or an N-year average) the helpers take a \
years-back argument — prev(metric, n), avg(metric, n) — so use compute_formula for these too (a \
2-year CAGR to FY2022 is "(revenue / prev(revenue, 2)) ** (1/2) - 1" with fiscal_year=2022); \
get_growth is adjacent-year only. Do NOT hand-assemble a metric by fetching parts and chaining \
several compute calls — that mis-orders operands (e.g. 365 / ratio instead of 365 x, or sum \
instead of average) and produces wrong numbers. Abstain (not_reported) ONLY if \
get_ratio/compute_formula reports a required figure isn't available. A wrong number is worse than \
an honest abstention.
9. SANITY-CHECK every number before reporting it: if it is implausible for its kind (a \
days-outstanding ratio over a few hundred days, a turnover above ~20x, a margin above 100%), you \
have made an error — do NOT report that number; abstain instead.

Notes: map a plain year to the fiscal year (e.g. "2024" -> FY2024); fiscal years differ \
across companies (NVDA's fiscal year ends in January). Be concise and always cite tickers \
and accessions."""

# Appended only when a user has uploaded documents (build_agent(user_id=...)).
USER_DOCS_PROMPT = """

ADDITIONAL TOOLS for the user's OWN uploaded private documents (e.g. an internal financial \
statement or memo for a company that is NOT one of the 7 public companies above):
- get_my_financials: EXACT numbers from the document's tables (the analogue of get_financials).
- get_my_ratio: a standard RATIO from the document, computed in code (the analogue of get_ratio).
- get_my_growth: year-over-year change of a metric from the document (the analogue of get_growth).
- search_my_documents: qualitative passages from the document.

Route the uploaded-document tools EXACTLY as you route the public ones: a RATIO (gross margin, \
ROE, current ratio, ...) -> get_my_ratio; a year-over-year change -> get_my_growth; a single \
figure -> get_my_financials; narrative -> search_my_documents. NEVER fetch pieces with \
get_my_financials and divide/compare them yourself when get_my_ratio / get_my_growth applies.

This OVERRIDES Rule 6. A company not in the public list of 7 is NOT automatically out_of_scope — \
the user may have uploaded documents about it. When a question names a company or subject that is \
NOT one of the 7 public tickers and the user has uploaded documents, treat it as being about those \
documents and call the get_my_* tools FIRST. You MUST NEVER abstain out_of_scope for a company \
without FIRST calling the relevant get_my_* tool to check the user's own documents. If the name is \
ambiguous across several uploaded files, ASK the user which file they mean rather than guessing or \
abstaining. Only if the tools return nothing relevant may you abstain (then use not_in_filings).
For ANY number from an uploaded document — including a figure for a specific business segment, \
division, or line item (e.g. "how much did the railroad earn", "insurance underwriting", "operating \
earnings") — you MUST use get_my_financials, NOT search_my_documents. These are table figures: \
get_my_financials reads them from the table with cell-level provenance. Rule 1 still holds: NEVER \
read a figure out of search_my_documents prose, and route arithmetic through compute. Use \
search_my_documents ONLY for non-numeric narrative (commentary, risks, descriptions); cite \
[filename · page]. Still abstain off_topic for non-analysis requests (advice, forecasts, writing)."""


def _user_docs_tools(user_id: str, scope_doc: str = None):
    """Bind the private-document tools to a user_id via closures — the user_id comes from the
    request context, NOT the LLM, so a user only ever touches their own uploaded documents.
    `scope_doc` (A): an explicit document the user selected in the UI; when set it HARD-filters
    every lookup to that file (overrides any LLM hint), the safe default for multi-document use.
    The tools also accept an optional `document` arg (B) the LLM may pass when the question names
    a file/company, to disambiguate when the user hasn't picked one."""
    from langchain_core.tools import tool as _tool
    from agents.user_docs_retrieval import (search_my_documents as _smd,
                                            get_my_financials as _gmf,
                                            get_my_ratio as _gmr,
                                            get_my_growth as _gmg)

    def _filter(document):
        return scope_doc or document   # explicit UI selection (A) wins over the LLM hint (B)

    @_tool
    def search_my_documents(query: str, document: str = None) -> str:
        """Search the USER'S OWN uploaded private documents for QUALITATIVE passages, each tagged
        [filename · page] to cite. Use for narrative/risk/commentary in an uploaded file — NOT for
        exact numbers (use get_my_financials). If the question is about a SPECIFIC uploaded file,
        pass its name (or the company) as `document` to search only that file."""
        return _smd(query, user_id=user_id, doc_filter=_filter(document))

    @_tool
    def get_my_financials(metric: str, period: str = None, document: str = None) -> str:
        """Exact financial numbers from the TABLES in the user's uploaded documents (the
        private-data analogue of get_financials). `metric` matches a table row label (e.g.
        "net income"); optional `period` matches a column (e.g. "FY2025"). Use this for ANY number
        from an uploaded document — never read a figure out of search_my_documents prose. If the
        question is about a SPECIFIC uploaded file, pass its name (or the company) as `document`
        so the figure comes from the right file (important when several files are uploaded)."""
        return _gmf(metric, user_id=user_id, period=period, doc_filter=_filter(document))

    @_tool
    def get_my_ratio(ratio: str, document: str = None, period: str = None) -> str:
        """A standard financial RATIO from the user's uploaded documents (the private-data
        analogue of get_ratio; same fixed formulas, computed in code). Supports gross_margin,
        operating_margin, net_margin, cogs_pct, roa, roe, current_ratio, quick_ratio,
        payout_ratio, debt_to_equity. Use this for ANY ratio about an uploaded file — do NOT
        fetch the pieces with get_my_financials and divide yourself. Pass the file name (or
        company) as `document` when several files are uploaded."""
        return _gmr(ratio, user_id=user_id, period=period, doc_filter=_filter(document))

    @_tool
    def get_my_growth(metric: str, document: str = None, period: str = None) -> str:
        """Year-over-year (YoY) change of a metric from the user's uploaded documents (the
        private-data analogue of get_growth). The tool fetches the year and the immediately
        preceding year itself, so the compared years are always consecutive. Use this for ANY
        'year over year' / 'how did X change' question about an uploaded file — never pick the
        two years yourself. Pass the file name as `document` when several files are uploaded."""
        return _gmg(metric, user_id=user_id, period=period, doc_filter=_filter(document))

    return [search_my_documents, get_my_financials, get_my_ratio, get_my_growth]


def _uploaded_doc_names(user_id: str):
    """Filenames the user has uploaded — injected into the prompt so the agent KNOWS the docs
    exist (and won't dismiss their subject as out_of_scope)."""
    try:
        import psycopg
        with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
            cur.execute("select distinct filename from user_chunks where user_id=%s", (user_id,))
            return [r[0] for r in cur.fetchall()]
    except Exception:
        return []


def build_agent(model: str = None, temperature: float = 0.0, user_id: str = None,
                scope_doc: str = None):
    model = model or os.environ.get("GEN_LLM_MODEL", "o4-mini")
    # o-series reasoning models (o1/o3/o4-...) reject a non-default temperature and instead take a
    # reasoning_effort knob; only the chat models (gpt-4o, ...) get a temperature.
    if re.match(r"^o\d", model):
        llm = ChatOpenAI(model=model, reasoning_effort=os.environ.get("REASONING_EFFORT", "medium"))
    else:
        llm = ChatOpenAI(model=model, temperature=temperature)
    # only add the private-docs tools when a user is in scope, so eval/public demo are unchanged
    if user_id:
        tools = TOOLS + _user_docs_tools(user_id, scope_doc=scope_doc)
        names = _uploaded_doc_names(user_id)
        have = (" The user has uploaded these documents: " + ", ".join(names) +
                ". Treat their subjects as IN scope and answerable from those documents."
                if names else "")
        if scope_doc:
            have += (f" The user has SELECTED the document '{scope_doc}' and is asking about it. "
                     f"Treat EVERY financial question as being about this document, even when no "
                     f"company is named: use get_my_ratio (ratios), get_my_growth (year-over-year), "
                     f"get_my_financials (single figures), or search_my_documents (narrative) on "
                     f"THIS document. Do NOT use the public-company tools, and NEVER ask which "
                     f"company they mean — this document is the subject.")
        elif len(names) > 1:
            have += (" Several documents are uploaded: when a question is about a specific one, "
                     "pass its name as the `document` argument so figures come from the right file, "
                     "and always make clear which file each figure is from.")
        prompt = SYSTEM_PROMPT + USER_DOCS_PROMPT + have
    else:
        tools, prompt = TOOLS, SYSTEM_PROMPT
    return create_react_agent(llm, tools, prompt=prompt)


def _extract_trace(messages, max_chars=600) -> List[Dict]:
    # pair each tool call with the output it produced (by tool_call_id) so the audit trail can show
    # what a tool RETURNED — the cited figure, the abstain, the [CHECK] convention note — not just
    # how it was called. Long retrieval prose is trimmed (max_chars) for the UI; pass max_chars=None
    # for the full output, which the guardrail needs to verify a $ figure traces to retrieved prose.
    outputs = {getattr(m, "tool_call_id", None): m.content
               for m in messages if isinstance(m, ToolMessage)}
    trace = []
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            out = outputs.get(tc.get("id"))
            out = str(out)[:max_chars] if out is not None else None
            trace.append({"tool": tc["name"], "args": tc["args"], "output": out})
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
            trace = _extract_trace(messages)                        # UI-trimmed for the audit trail
            full_trace = _extract_trace(messages, max_chars=None)   # untrimmed for the guardrail
            # tool outputs (with tool name) — used to verify every number in the answer
            # traces back to get_financials/compute (not read out of search_filings prose).
            tool_outputs = [(getattr(m, "name", "") or "", m.content)
                            for m in messages if isinstance(m, ToolMessage)]
            usage = observability.sum_usage(messages)
        except GraphRecursionError:
            answer = ("I couldn't resolve this within the step limit — please narrow the "
                      "question.")
            trace = full_trace = []
        tools_used = [t["tool"] for t in trace]
        # full_trace (untrimmed): the guardrail must see the whole retrieved passage to confirm a $
        # figure traces to it — the 600-char UI trim would starve the check and false-abstain.
        answer = guardrail(answer, tools_used, full_trace)   # hard backstop against bad numbers
        # audit trail: which filings were cited + whether/why it abstained
        accns = sorted({a for _, c in tool_outputs
                        for a in re.findall(r"\d{10}-\d{2}-\d{6}", c)})
        # what the offline scorer (eval/score_traces.py) grounds faithfulness/domain_judge
        # against (the eval <-> observability loop). Include BOTH the numeric tool outputs
        # (get_financials/get_growth/get_ratio/compute) AND the qualitative passages
        # (search_filings): without the numeric outputs the judge can't see where a figure came
        # from and wrongly flags a real, tool-sourced number as "fabricated" (a verified false BAD).
        # Keep the full retrieved block: an earlier 1500-char cap dropped most passages, which
        # made faithfulness/domain_judge see "unsupported" claims and emit false BADs (verified:
        # -0.38 mean faithfulness, 9/12 good->bad flips). 8000 holds a full k=5 search result.
        _NUMERIC_TOOLS = {"get_financials", "get_growth", "get_ratio", "compute"}
        contexts = ([c for name, c in tool_outputs if name in _NUMERIC_TOOLS] +
                    [c[:8000] for name, c in tool_outputs if name == "search_filings"][:6])
        audit = {"accessions_cited": accns, "abstained": "abstain" in tools_used,
                 "abstain_reason": next((t["args"].get("reason")
                                         for t in trace if t["tool"] == "abstain"), None),
                 "contexts": contexts}
        trace_id = record(answer=answer, tools_used=tools_used, usage=usage, audit=audit)
    if verbose:
        for step in trace:
            print(f"  🔧 {step['tool']}({step['args']})")
    return {"answer": answer, "trace": trace, "tool_outputs": tool_outputs,
            "tools_used": tools_used, "trace_id": trace_id}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "How did NVIDIA's revenue change year over year, and what does management attribute it to?"
    print(f"Q: {q}\n")
    out = run_agent(q, verbose=True)
    print("\nA:", out["answer"])
