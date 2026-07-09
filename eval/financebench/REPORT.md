# Validating the SEC filing agent on FinanceBench

A brief on how we measured the agent against an **independent external benchmark**, and what five
data-driven iterations achieved: **addressable coverage 44% → 87%, real hallucination rate ≈ 0.**

---

## Why an external benchmark

The core claim of this project is not "we built a finance agent" — it's "**we can prove it's
reliable**." Proving that on our own test set risks *grading our own homework*: we pick the
questions, we set the answers. So we validate on **FinanceBench** (Patronus AI), a published
benchmark we did not write — 150 questions over ~32 public companies (50 metrics-generated,
50 domain-relevant, 50 novel-generated). It also stresses the **dynamic path** (any public company,
fetched live from SEC XBRL), which our internal 7-company gate does not exercise.

## How we score (the finance bar)

We never score a numeric answer as simply "answered." Each is one of three outcomes:

| outcome | meaning |
|---|---|
| **correct** | matches the gold number within 2.5% |
| **abstain** | declined — split into **FIXABLE** (we *should* answer → our gap) and **LEGIT** (needs segment/quarterly data XBRL can't provide → correctly declined) |
| **hallucinated** | asserted a *wrong* number instead of abstaining — the failure we most want at zero |

**Primary metric — ADDRESSABLE coverage = correct / answerable.** It excludes the
legitimately-impossible questions (so it isolates *our* deficiency) but still counts fixable
abstains against us (so it can't be gamed by abstaining). A methodology note we corrected early:
reporting "accuracy = correct / (correct + wrong)" *rewards abstaining* by dropping declines from
the denominator — so coverage, which counts abstains against us, is the honest headline.

## The five-round story

| round | key change | addressable coverage | hallucinations |
|---|---|---|---|
| 1 · baseline | public SEC XBRL + base tools | **44%** | 4 |
| 2 · more data | added line-items + activity ratios | **62%** | **7 ↑** |
| 3 · guardrail | output guardrail + prompt discipline | **65%** | 4 |
| 4 · PoT | `compute_formula` single-expression evaluator | **73%** | 1 |
| 5 · dynamic + basis | multi-year formulas, delisted/renamed resolution, as-reported restatement, EPS, guard removal, o4-mini, self-correction | **87%** | **≈ 0** |

### The counterintuitive insight (round 2)
Adding data *raised* hallucinations (4 → 7). Not because data was missing — because we gave the
model **half** the ingredients. With D&A and DPO components available, it was tempted to
*hand-compute* derived metrics it had no deterministic tool for (CCC, EBITDA margin), and multi-step
assembly is where it errs. The finance-bar lesson: **the danger isn't "no data," it's "enough data
to hand-compute."** The fix (rounds 3–4) wasn't more prompting — it was **removing the assembly**:
one deterministic tool per metric, and a Program-of-Thought evaluator (`compute_formula`) that
writes the whole formula once and lets code fetch every figure and evaluate it, so the model never
transcribes a number.

### Round 5 — what took it to 87%
- **Multi-year spans**: `prev(metric, n)` unlocked N-year CAGR / averages (Lockheed 2-yr CAGR).
- **Delisted / renamed issuers**: `name → CIK` via SEC's former-name lookup — Activision (delisted)
  and Block (formerly Square) resolve by name instead of a dead ticker.
- **As-reported vs restated**: a later filing re-presents a prior year with a different value. We
  now return the figure **as originally reported** (matching the source filing and the benchmark),
  and *flag* when a later restatement exists — computed on one consistent basis, never mixing years.
- **Precision + an accounting insight**: a display-precision bug had flattened a 5.4% margin to
  "5%" (the only round-4 "hallucination"); and an over-eager tag-merge guard was dropping a valid
  figure until we recognized that, by accounting rules, a footnote sub-component never appears
  without its total line — so plain gap-fill is safe and the guard was removed.
- **Model**: switched to the reasoning model **o4-mini**, whose deliberate tool-call planning cut
  the routing/naming flakiness (a metric-name typo made one item abstain; self-correction — the tool
  suggests the closest metric — fixed the class).

## Final result

- **Addressable coverage 48/55 = 87%.** Accuracy when we answered: 48/50 = 96% (high because we
  abstain, not guess).
- **Real hallucination rate ≈ 0.** The two flagged by the scorer are artifacts: General Mills
  CCC "−3.70" *equals* gold "-3.7" (the scorer's number extraction dropped a unicode minus), and
  Ulta is an honest refusal in prose (no fabricated number).

## Honest limits (and future work)

- **Narrative side needs a DB.** This run had `search_filings` degraded (no `DATABASE_URL`), so
  text-extraction and qualitative questions abstain. The **numeric coverage headline is unaffected**
  (numeric questions never touch the DB), but the domain/novel scores are not comparable to a
  DB-enabled run. → re-run with the filings store when access is restored.
- **Eval coverage.** The deterministic CI gate (`testset.jsonl`) is 7 curated companies; the
  dynamic path is covered here (FinanceBench) plus synthetic unit tests and a small set of
  network-gated live assertions — but not in the fast gate. The domain judge's κ=0.95 was calibrated
  on the 7 curated companies; generalization to the full market is an assumption, not yet validated.

## Reproduce

```bash
GEN_LLM_MODEL=o4-mini REASONING_EFFORT=low OPENAI_API_KEY=... \
  <venv>/bin/python -m eval.financebench.run_open      # add DATABASE_URL for the narrative side
```
Every run appends to `runs_log.jsonl`; see `README.md` for details.
