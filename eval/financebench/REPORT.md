# Validating the SEC filing agent on FinanceBench

A brief on how we measured the agent against an **independent external benchmark**, and what five
data-driven iterations achieved: **addressable coverage 44% → 87%, real hallucination rate ≈ 0**
(the numeric side). A **narrative scorecard** (78% answered-correct where the evidence is in the text
we index) and an eval-gated retrieval study — whose honest conclusion was to *not* ship a
+22pp-recall technique that hurt end-to-end answers — follow below.

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

## Narrative side (the qualitative questions)

The 87% above is the **numeric** headline; the questions that need *qualitative* filing text were
unmeasured in that run (the narrative store was offline). Measured now — the real production agent
(all tools, **year-aware retrieval**, default config) on the **61 FinanceBench 10-K narrative
questions** (non-numeric gold, evidence drawn from 10-Ks), scored on the same finance bar:

| | correct | wrong | abstain |
|---|---|---|---|
| **overall (61)** | **29 (48%)** | 15 (25%) | 17 (28%) |
| evidence in narrative prose (23) | 14 | 4 | 5 |
| evidence in a statement/table (38) | 15 | 11 | 12 |

Reading it honestly:

- **The agent answers 72% and abstains 28%** — far less than a narrative-only pipeline (a plain
  retrieve-then-answer baseline over these questions abstained ~64%), because the agent also routes
  to the deterministic numeric tools and iterates.
- **On questions whose evidence is genuinely in the narrative we index, answered-correctness is 78%
  (14/18)** — the clean "narrative capability" number.
- **Its weakness is coverage, not retrieval tuning.** 38/61 questions keep their evidence in a
  financial statement or segment table the narrative path doesn't index; of the 17 abstains only **5
  are fixable** (evidence was in narrative, agent still declined) and **12 are legit** (the data isn't
  in narrative at all — the report-level-data gap, future work).
- **Different failure mode from numeric.** The 25% "wrong" are qualitative judgments (e.g. "is X
  capital-intensive?") — a mix of genuine error and debatable-conclusion disagreement with the LLM
  judge, **not fabricated numbers**. The ≈0-hallucination result is about *numbers* and still holds.

### What moved the narrative number — and what didn't (an eval-gated retrieval study)

A rebuilt, trustworthy retrieval eval-gate (year-controlled, LLM-judged gold) showed dense recall@10
was only ~45% — near the ~55–60% oracle ceiling reported for this task, i.e. a genuinely hard, open
problem. Against that gate we A/B'd the usual levers:

| lever | effect on the gate | shipped? |
|---|---|---|
| cross-encoder reranker | recall +2 (churny wash) | no |
| contextual (metadata) chunks | recall +1 | no |
| **Multi-HyDE** | **recall@10 48% → 70% (+22pp)** | **no** |
| **year-aware ingestion** (retrieve the *asked* fiscal year, not just the latest) | **answer-correct +21pp, 0 regressions** | **yes** |

Multi-HyDE won decisively on *recall* — but a clean end-to-end re-test with the real guardrailed agent
showed it **lowered** answer quality (−3, by surfacing plausible-but-off passages the agent then
abstained or erred on), so it ships **default-off** (`FILINGS_HYDE=1` to enable). The only change that
cleanly improved answers was a **correctness fix, not a retrieval technique**: narrative retrieval had
only ever indexed the *latest* 10-K, so a "FY2022" question read the wrong year's filing; scoping
retrieval to the asked year lifted answered-correctness +21pp with zero regressions. **The lesson:
recall is a misleading proxy once a capable agent is in the loop — the win was fixing a bug, not
adding a technique.**

## Honest limits (and future work)

- **Narrative scorecard caveats.** The 61-case narrative run above is DB-enabled and year-controlled,
  but it is smaller than the numeric set and graded by an LLM judge on *qualitative* conclusions, where
  the correct/wrong line is fuzzier than a 2.5% numeric match (the 25% "wrong" includes debatable
  judgments, not fabricated numbers). Coverage classification (prose vs statement/table) is itself an
  LLM label. Treat 78% answered-correct-on-prose as the robust signal and the overall 48% as
  directional.
- **Lazy narrative cache growth (now bounded).** `filing_chunks` ingests on demand; with year-aware
  retrieval (multiple years per company) it grew to fill the 512 MB store during this study and had to
  be truncated. It is now bounded by **size + freshness**: a `last_accessed` touch on retrieval drives
  a per-filing **LRU** eviction (drop the least-recently-used *accessions* until under
  `FILING_CHUNKS_MAX`, default 20 000 chunks ≈ 320 MB), and an `ingested_at` **TTL** prunes entries
  older than `FILING_TTL_DAYS` (default 30) so a re-query re-fetches the newest filing.
- **Eval coverage.** The deterministic CI gate (`testset.jsonl`) is 7 curated companies; the
  dynamic path is covered here (FinanceBench) plus synthetic unit tests and a small set of
  network-gated live assertions — but not in the fast gate. The domain judge's κ=0.76 was calibrated
  on the 7 curated companies; generalization to the full market is an assumption, not yet validated.

## Reproduce

```bash
GEN_LLM_MODEL=o4-mini REASONING_EFFORT=low OPENAI_API_KEY=... \
  <venv>/bin/python -m eval.financebench.run_open      # add DATABASE_URL for the narrative side
```
Every run appends to `runs_log.jsonl`; see `README.md` for details.
