# SEC Filing Agent
*A finance-grade tool-calling agent over SEC filings — and your own documents.*

[![Eval](https://github.com/Zhonghui-li/sec-filing-agent/actions/workflows/eval.yml/badge.svg)](https://github.com/Zhonghui-li/sec-filing-agent/actions/workflows/eval.yml)

Answers financial questions about **any U.S. public company** from its SEC 10-K filings — built to
the **finance bar**, where a wrong or unsupported number is unacceptable.

> **Validated on FinanceBench** (an external benchmark we didn't write): **addressable coverage
> 44% → 87% across five data-driven iterations, hallucination rate ≈ 0.**
> → [full brief](eval/financebench/REPORT.md)

## The finance bar

| Rule | How |
|---|---|
| **Numbers only from tools** | every figure from structured XBRL, never LLM arithmetic — exact & auditable |
| **Every claim cited** | a clickable EDGAR link to the source 10-K |
| **Abstains, never fabricates** | a structured signal when a metric isn't reported / is out of scope |
| **Same bar for your docs** | uploaded statements → structured table extraction, cell-level citations |

## Coverage at a glance

- **Any U.S. public company** — figures fetched live from SEC XBRL; delisted/renamed firms resolve
  by **name** (Activision, Square→Block), not a dead ticker.
- **23 statement line-items + 18 ratios**, plus any spelled-out formula via a Program-of-Thought evaluator.
- **As-reported basis** — returns the figure as originally filed and *flags* later restatements
  (never mixes bases across years).
- **Export to CSV** — pull metrics × years into Excel, straight from the tools (no LLM → exact).

## Example

```
Q: How did NVIDIA's revenue change year over year, and what does management attribute it to?
A: +65.5% YoY, from $130.5B (FY2025) to $215.9B (FY2026), driven by data-center AI demand.
   🔧 get_growth(revenue, NVDA) · search_filings(NVDA)   [source: 10-K 0001045810-26-000021 → sec.gov/…]

Q: What is Activision's FY2019 fixed-asset turnover?
A: 24.26x (revenue / average net PP&E).   🔧 get_ratio(fixed_asset_turnover, "Activision Blizzard", 2019)
   (delisted — resolved by name, not the retired ATVI ticker)

Q: What is JPMorgan's gross profit?
A: 🔧 get_financials(JPM, gross_profit) → not reported · abstain(not_reported)
   JPMorgan doesn't report gross profit (banks have no cost-of-goods line).
```

## How it works — text vs numbers

Unstructured narrative (risk factors, MD&A) goes through **retrieval**; exact figures go through
**deterministic tools** over XBRL. The LLM never does arithmetic on financials and never reads a
number out of prose.

| Tool | What it does |
|---|---|
| `get_financials` | an exact figure from XBRL — any public company, fetched live |
| `get_ratio` | 18 standard ratios from fixed formulas (the right base metric baked in) |
| `get_growth` | YoY change, always consecutive years — no multi-year span mislabeled as YoY |
| `compute_formula` | a spelled-out formula, **Program-of-Thought**: the model writes it once, code fetches every figure and evaluates — the model never transcribes a number |
| `search_filings` | qualitative context (pgvector + CrossEncoder reranker) |
| `abstain(reason)` | a structured, machine-readable refusal |

*Ratios and growth are computed, not assembled.* LLMs reliably pick the **wrong base metric** —
total liabilities for "debt", period-end vs average assets for ROA (both surfaced by FinanceBench).
Baking the conventions into fixed formulas removes that whole class of error.

## Evaluation — the part most agent demos skip

**1 · Internal test set (CI gate).** 69 self-grounded cases (answers derived from the XBRL, correct
by construction) over 7 curated companies, scored by a **complementary suite** — `numerical`,
`grounded` (a number traceable to a tool), `citation`, `tool` (trajectory), `abstain`, `context_recall`,
`forbid` (injection). The **9 deterministic** metrics gate CI; 3 LLM-judge metrics monitor only
(they're systematically biased against the honest hedging a regulated domain *should* use).

**2 · Calibrated domain judge.** A rubric judge that scores honest hedging / appropriate abstention
as *good*, calibrated against a **balanced human-labeled set** (good/bad, including *subtle*
hallucinations) — **Cohen's κ = 0.76** ("substantial agreement"), with **zero false-positives on
hedged/abstaining answers**. Fixes the generic metric's bias (JPMorgan capital answer: generic
relevancy `0.0` vs domain judge `1.0`). Runs on live traces too.

**3 · External benchmark — [FinanceBench](https://github.com/patronus-ai/financebench) (Patronus AI).**
150 questions / 32 companies we did *not* write. **Addressable coverage 44% → 87% across five
iterations, hallucination ≈ 0.** Reproducible harness with a tracked runs log.
→ [`eval/financebench/`](eval/financebench/) · [REPORT.md](eval/financebench/REPORT.md)

> Built eval-driven: the eval surfaced the failures that drove the agent — it once *guessed* fiscal
> years, *substituted revenue as a proxy for gross profit*, *over-refused* answerable questions, and
> mislabeled a 3-year span as YoY. Several "failures" were scorer bugs, not agent bugs — telling the
> two apart is the core skill.

## Bring your own data (private documents)

Upload a statement (an internal report, a non-public company) and the agent answers from it — the
**same finance bar, applied to data with no XBRL**:

- **Structured extraction** ([Docling](https://github.com/docling-project/docling)) → figures read
  from **table cells**, never prose; `get_my_financials` is the private-data analogue of XBRL.
- **Cell-level citations** — `filename · page · row/col`, deep-linked to the source PDF page.
- **Per-user isolation** (optional Google sign-in) + **multi-document scoping** (metadata filtering,
  so a number never gets attributed to the wrong file).
- **Async ingestion** in an isolated venv (Docling's heavy deps never touch the agent). See
  [`ingest/README.md`](ingest/README.md).

## Serving

FastAPI service + a static chat UI, with in-memory rate/quota controls for a cheap single-instance demo.

- **`/ask`** — the answer + clickable EDGAR citations + a collapsible **audit trail** of tool calls.
- **`/export`** — deterministic CSV of metrics × years for any company (no LLM → exact).
- **`/upload`** + **`/file/{doc_id}`** — upload, see what Docling extracted, then ask; citations deep-link to the page.
- **`/v1/chat/completions`** — an **OpenAI-compatible** endpoint, so the agent drops into open-webui / LibreChat.
- Default model: **o4-mini** (reasoning) — deliberate tool-call planning reduces routing/naming slips.

```bash
DATABASE_URL=… OPENAI_API_KEY=… uvicorn service.app:app --port 8100   # http://localhost:8100
```
Deployed on Cloud Run with `max-instances=1`, so the in-memory daily quota is an exact global cap.

## Run

```bash
pip install -r requirements.txt

python -m pytest tests/                                                    # deterministic tests (no keys)
GEN_LLM_MODEL=o4-mini OPENAI_API_KEY=… python -m eval.financebench.run_open   # external benchmark
DATABASE_URL=… OPENAI_API_KEY=… python -m agents.sec_agent "How did NVIDIA's revenue change YoY?"
```

## Data & real-world messiness

- **Numbers**: SEC `companyfacts` XBRL — live for any company; 7 companies cached in
  `data/financials.json` as the deterministic eval baseline.
- **Text**: 10-K Business / Risk / MD&A via `edgartools` → **pgvector** (indexed for the curated set).
- **Tag drift** — a company migrates the same line to a different XBRL tag across years; the extractor
  merges candidate tags (a footnote component can never substitute for the total line).
- **As-reported vs restated** — returns the figure as originally filed (matching the source & the
  benchmark) and flags a later restatement; one consistent basis, never mixed across years.
- **Banks** — JPM reports no gross profit / COGS → surfaced as "not reported" (feeds abstain); numbers stay exact.

> Reuses retrieval + eval engineering from [Slug Advisor](https://github.com/Zhonghui-li/Agentic-RAG)
> (hybrid retrieval, reranker, eval-in-CI), re-pointed from course advising to financial-document
> analysis and raised to the finance bar.
