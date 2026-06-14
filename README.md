# SEC Filing Agent — a finance-grade tool-calling agent over SEC filings

[![Eval](https://github.com/Zhonghui-li/sec-filing-agent/actions/workflows/eval.yml/badge.svg)](https://github.com/Zhonghui-li/sec-filing-agent/actions/workflows/eval.yml)

A LangGraph ReAct agent that answers questions about public companies' SEC 10-K filings —
financials, risk factors, MD&A — built to the **finance bar**, where a wrong or
unsupported number is unacceptable:

- **Every number comes from structured XBRL data (a tool), never from the LLM** doing
  arithmetic or reading a figure out of prose — so financial figures are exact and auditable.
- **Every claim is cited** to the source filing, with a **clickable EDGAR link** to the 10-K.
- The agent **abstains** (a structured, machine-readable signal) when it can't answer —
  metric not reported, company out of scope, off-topic — instead of fabricating.
- All of it is **gated by a 10-metric eval suite in CI**.

> Reuses the retrieval + eval engineering from [Slug Advisor](https://github.com/Zhonghui-li/Agentic-RAG)
> (hybrid retrieval, CrossEncoder reranker, eval-in-CI), re-pointed from course advising to
> financial-document analysis and raised to the finance bar.

## Example

```
Q: How did NVIDIA's revenue change year over year, and what does management attribute it to?

A: NVIDIA's revenue grew +65.5% year over year, from $130.5B (FY2025) to $215.9B (FY2026),
   driven by data-center / accelerated-computing demand for AI.
   🔧 get_financials(NVDA, revenue) · get_financials(NVDA, revenue, 2025) · compute(yoy) · search_filings(NVDA)
   [source: 10-K accession 0001045810-26-000021 → https://www.sec.gov/Archives/edgar/data/1045810/...]

Q: What is JPMorgan's gross profit?
A: 🔧 get_financials(JPM, gross_profit) → not reported · abstain(not_reported)
   JPMorgan doesn't report gross profit (banks have no cost-of-goods line). I can't provide it.
```

The agent **decides which tools to call**: exact figures via `get_financials` (XBRL), math
via `compute`, qualitative context via `search_filings`, and `abstain` when it shouldn't answer.

## How it works

**Design principle — text vs numbers.** Unstructured narrative (risk factors, MD&A) goes
through **retrieval** (`search_filings`, pgvector + reranker); exact figures go through
**tools** that read **XBRL** (`get_financials`) and a deterministic `compute` (YoY, ratios).
The LLM never does arithmetic on financials, and never reads a number out of filing prose.

**Tools:** `get_financials` · `compute` · `search_filings` · `abstain(reason)` — orchestrated
by `create_react_agent` (LangGraph), with a system prompt enforcing the finance-bar rules.

## Evaluation — the part most agent demos skip

A **self-grounded** test set (answers derived from the real XBRL + filings, so they're correct
by construction), organized as a **capability × robustness matrix** (lookup / compute /
qualitative / combined × happy / edge / adversarial), scored by a **complementary suite** —
each metric catches a failure the others miss:

| metric | type | catches |
|---|---|---|
| **numerical** | deterministic | wrong number (2.5% tolerance, FinanceBench-style) |
| **citation** | deterministic | wrong/missing source filing |
| **grounded** | deterministic | a number not traceable to a tool (invented / read from prose) |
| **tool** | deterministic | wrong tool trajectory (e.g. mental math instead of `compute`) |
| **abstain** | deterministic | answering when it should refuse, or vice-versa |
| **reason** | deterministic | wrong abstain category (out_of_scope / not_reported / …) |
| **facts** | deterministic | qualitative answer missing the key points |
| **forbid** | deterministic | prompt-injection / planted false claim |
| **faithfulness** | LLM-judged (Ragas) | qualitative narrative not grounded in the cited text |
| **answer_relevancy** | LLM-judged (Ragas) | off-topic / evasive answers |

**Eval-in-CI (two layers):**
- **L1** — deterministic unit tests for the tools + scorer logic (no LLM, no DB, no secrets);
  runs on **every push/PR**, seconds, free.
- **L2** — the full agent + Ragas eval, gated against a committed baseline (per-metric
  tolerance, exits non-zero on regression); runs **on demand** (so the shared key isn't run
  unattended in a public repo).

Current baseline (24-case set): deterministic metrics **100%**, faithfulness **0.84**,
answer_relevancy **0.86**.

> Built eval-driven: the failures the eval surfaced drove the agent — e.g. it once *guessed*
> fiscal years, *substituted revenue as a "proxy" for gross profit*, and (after a structured
> `abstain` tool was added) *over-refused* questions it could answer. Each was caught by the
> eval and fixed; several "failures" were scorer bugs, not agent bugs — distinguishing the two
> is the core skill.

## Data (public SEC EDGAR — no PII / compliance scope)
- **Numbers**: SEC `companyfacts` XBRL API → exact annual financials (`data/financials.json`).
- **Text**: 10-K Business / Risk Factors / MD&A via `edgartools` → chunked into **pgvector**
  with provenance.
- Companies: AAPL, MSFT, NVDA, AMZN, JPM, TSLA, KO (latest 2 fiscal years).

## Run
```bash
pip install -r requirements.txt

# build the data layer (once)
python scripts/fetch_financials.py                          # XBRL exact numbers
python scripts/fetch_filings.py                             # 10-K sections
DATABASE_URL=... OPENAI_API_KEY=... python scripts/build_filings_store.py   # -> pgvector

# ask the agent
DATABASE_URL=... OPENAI_API_KEY=... python -m agents.sec_agent "How did NVIDIA's revenue change YoY?"

# eval
python -m pytest tests/                                     # L1 (deterministic, no keys)
DATABASE_URL=... OPENAI_API_KEY=... python -m eval.score --quality   # L2 (full agent + Ragas)
```

## Known data-quality notes (real-world filing messiness)
- JPM (a bank) reports no gross profit / operating income / R&D, and its MD&A isn't under the
  standard section — surfaced as "not reported" (feeds the abstain behavior); numbers stay exact.
- A company can migrate the same metric across XBRL tags between years (e.g. NVDA revenue →
  `Revenues`); the extractor merges across candidate tags to avoid gaps.
