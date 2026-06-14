# SEC Filing Agent — a finance-grade tool-calling agent over SEC filings

A tool-calling agent that answers questions about public companies' SEC 10-K filings —
financials, risk factors, MD&A — built to the **finance bar**:

- **Every number comes from structured XBRL data (a tool), never from the LLM** reading
  a figure out of retrieved text — so financial figures are exact and auditable.
- **Every claim is cited** to the source filing (ticker · fiscal year · section · accession).
- The agent **abstains** when something isn't in the filings (e.g. a metric a company
  doesn't report) instead of fabricating.
- Quality is gated by an eval suite measuring **numerical accuracy** and **citation
  accuracy** (the things that matter in finance), aligned with the **FinanceBench** benchmark.

> Reuses the retrieval + eval engineering from [Slug Advisor](https://github.com/Zhonghui-li/Agentic-RAG)
> (hybrid BM25 + pgvector retrieval, CrossEncoder reranker, eval-in-CI, Langfuse), re-pointed
> from course advising to financial-document analysis and raised to the finance bar.

## Design principle — text vs numbers
Unstructured narrative (risk factors, MD&A) goes through **retrieval** (`search_filings`);
exact financial figures go through **tools** that read **XBRL** (`get_financials`) and a
deterministic `compute` (YoY, ratios). The LLM never does arithmetic on financials.

## Data (public SEC EDGAR — no PII / compliance scope)
- **Numbers**: SEC `companyfacts` XBRL API → exact annual financials (`data/financials.json`).
- **Text**: 10-K Business / Risk Factors / MD&A sections via `edgartools` → chunked into
  **pgvector** with provenance for citations.
- Companies (P0): AAPL, MSFT, NVDA, AMZN, JPM, TSLA, KO (latest 2 fiscal years).

## Status
- **P0 data layer — done & verified.** 1,094 exact financial rows (spot-checked vs official
  10-Ks; caught & fixed a real XBRL tag-migration bug on NVDA revenue). 2,857 filing-text
  chunks in pgvector; retrieval returns relevant, cited passages.
- **P1 agent — done & verified.** LangGraph ReAct agent over the 3 tools (`get_financials`,
  `compute`, `search_filings`) with a finance-bar system prompt. Verified end-to-end:
  exact numbers from tools, YoY/ratios via `compute`, source citations, honest abstention
  on unreported metrics (e.g. a bank's gross profit) and out-of-scope companies.
- Next: P2 guardrails (stricter citation / numbers-from-tools enforcement / abstain) →
  P3 eval-in-CI (+ FinanceBench) → P4 QoQ/YoY comparison → P5 demo + audit-trail observability.

## Run (P0)
```bash
pip install -r requirements.txt
python scripts/fetch_financials.py        # XBRL exact numbers -> data/financials.json
python scripts/fetch_filings.py           # 10-K sections     -> data/filings.jsonl
DATABASE_URL=postgresql://... OPENAI_API_KEY=... \
    python scripts/build_filings_store.py # chunk + embed -> pgvector
```

## Known data-quality notes (real-world filing messiness)
- JPM (a bank) has no gross profit / operating income / R&D tags, and its MD&A isn't under
  the standard section — handled as "not reported" (feeds the abstain behavior), numbers
  still exact from XBRL.
- A company can migrate the same metric across XBRL tags between years (e.g. NVDA revenue
  → `Revenues`); the extractor merges across candidate tags to avoid gaps.
