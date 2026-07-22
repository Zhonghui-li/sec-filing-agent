# FinanceBench — external-benchmark eval harness

An **independent** benchmark (we did not write it) that measures the agent end-to-end on real
questions about ~32 public companies. Used to validate the agent on the **dynamic path** (any
public company, fetched live) and to avoid "grading our own homework" — a complement to the
internal `eval/testset.jsonl` gate, which covers only the 7 curated companies.

- **Source**: FinanceBench open set, Patronus AI. 150 questions (50 metrics-generated,
  50 domain-relevant, 50 novel-generated). Downloaded once and cached to `_open_cache.jsonl`
  (gitignored; re-downloaded if missing).

## Run it

```bash
VENV=/path/to/venv/bin/python              # the shared agentic_rag venv
GEN_LLM_MODEL=o4-mini REASONING_EFFORT=low \
  DATABASE_URL=... OPENAI_API_KEY=... \
  $VENV -m eval.financebench.run_open [--limit N]
```

- `OPENAI_API_KEY` — required (the agent's LLM).
- `DATABASE_URL` — **optional**. Only `search_filings` (narrative/text-extraction questions) needs
  it. Without it, `search_filings` degrades gracefully to "not indexed" and those questions abstain;
  **the numeric coverage headline is unaffected** (numeric questions never touch the DB).
- `--limit N` runs the first N questions (smoke test).
- On o4-mini the full 150 run takes ~40–60 min (reasoning model); ~$1–2.

## What it reports (the numbers that matter)

Numeric questions are scored three ways (never just "did it answer") to keep the finance bar:

- **correct** — the answer matches the gold number (2.5% tolerance).
- **abstain** — declined. Split into **FIXABLE** (we should be able to answer → a design gap) and
  **LEGIT** (needs segment/quarterly data XBRL can't give → correctly declined).
- **hallucinated** — asserted a *wrong* number instead of abstaining. This is the finance-bar
  failure we most want at zero.

**ADDRESSABLE coverage = correct / answerable** (excludes legit-impossible questions, but still
counts fixable-abstains against us so it can't be gamed by abstaining). This is the primary metric.

## Tracked history

Every run appends a record to **`runs_log.jsonl`** (date, model, config, coverage, hallucinations),
so the improvement over time is a reproducible artifact, not scattered numbers. The five-round
progression (44% → 62% → 65% → 73% → 87%) lives there. See `REPORT.md` for the written brief.

## Coverage / limits (honest)

- FinanceBench exercises the **dynamic path** end-to-end (32 companies) — this is where the agent's
  recent work shows up. The internal `testset.jsonl` gate is deterministic but only 7 curated
  companies; the domain judge's κ=0.76 was calibrated on those 7 (generalization is an assumption).
- Narrative/text-extraction questions need `DATABASE_URL`; without it they abstain (not a regression).
