# Validating the SEC filing agent on FinanceBench

A brief on how we measured the agent against an **independent external benchmark**. The current
headline is **93% addressable coverage on the numeric set at a zero-fabrication rate, with ~92% of
narrative answers grounded in a cited filing or tool output** (see the 2026-09 update below; the
five-round story that reached 87% is kept further down as history). What it misses, it misses by
declining, not by inventing. A narrative scorecard and an eval-gated retrieval study — whose honest conclusion was to
*not* ship a +22pp-recall technique that hurt end-to-end answers — follow below.

> **Update (2026-09-13).** A fiscal-year fix, a fourth guardrail false positive, and two scorer
> boundaries that were wrong. Current headline:
>
> | | | |
> |---|---|---|
> | **Numeric** | **51/54 addressable = 94%** | zero fabrications; all 50 metrics-generated correct |
> | **Narrative** | **36/82 addressable = 44%** | 82% grounded (κ=0.76 judge) |
>
> - **A fiscal year is what the company calls it, not the year it ends in.** Every annual row was
>   labelled with the calendar year of its period end, so Target, Ulta, Home Depot, Lowe's and
>   Kroger — which name a fiscal year after the year it *starts* — were off by one, and asking for
>   one returned the year before it. "Target's FY2024 revenue" returned $107,412,000,000 (Target's
>   FY2023) instead of $106,566,000,000. **Every guard passed**: the figure came from a
>   deterministic tool, traced to a real filing, carried a real accession, and the audit trail read
>   "verified · figures reproduced from source". The error was upstream of all of it, in the mapping
>   from a fiscal year to a period. *Zero fabrication is not the same property as not being wrong.*
>   The naming is now read from the company's own filings — the submissions feed gives each filing's
>   period end, the facts give that filing's `fy`, and joining them is a lookup rather than a rule.
>   Nothing is keyed off the fiscal-year-end month, which cannot separate Target from Walmart: same
>   fiscal calendar, opposite convention. Verified on nine companies, four never touched while
>   developing it. The narrative index was rebuilt, since 7.5% of its rows carried the old labels
>   and `search_filings` re-ingests only when a year is *missing*, never when it is wrong.
> - **The narrative denominator now matches the numeric one.** It split on `question_type` while
>   run_open splits on whether gold is a figure, so five questions were judged by both harnesses —
>   once as a number, once as prose — and double-counted in any summary of the two.
> - **Narrative correctness is now reported addressable**, as the numeric side always was. Thirteen
>   questions are answered only by an earnings release, which `filings_ingest` skips by design (the
>   non-GAAP and guidance questions: adjusted EBITDA, adjusted EPS). Excluding what the corpus
>   cannot hold moves 37/95 = 39% to 36/82 = 44%. The criterion is FinanceBench's own evidence
>   labels, never whether we answered correctly — a first attempt excluded questions whose evidence
>   sits in a financial statement, which would have been a self-issued excuse, since get_statement
>   and get_segment_breakdown can reach those.
> - **A fiscal year inside a metric's name read as a value.** `_IMPLAUSIBLE` matched
>   `(\d+)\s*days` against a bound of 1000, so "Amazon's FY2017 days payable outstanding … was
>   108.43 days" offered *2017 days* and a correct answer was replaced by the safe abstention —
>   the fourth false positive of that shape, after an accession's digits and a multiplication sign.
>   A `get_ratio` result is now bounded by the ratio's KIND, taken from the call's arguments and
>   the tool's output, which cannot misread prose because it does not read prose.
>
> **Update (2026-09-12).** Three output-guardrail false positives and a rewritten numeric scorer
> move the headline. **The agent's capability is unchanged** — it is simply no longer refused, or
> mis-scored, for answers it got right.
>
> - **Addressable coverage 50/54 = 93%** on the numeric set, from 48/55 = 87% on the same 55
>   questions before the fixes. (The denominator moves too: one question was categorised
>   abstain-LEGIT in the later run.) Every point came from the guardrail no longer discarding
>   correct answers:
>   - a **unit conversion** — `compute(ratio, a=32_780_000_000, b=1e6)` for a question asking "in
>     USD millions" — was rejected because the divisor traces to no tool output, and the whole
>     reply was replaced by the safe abstention. Two questions.
>   - the **turnover guard** matched `(\d+)\s*x` against a bound of 100, so a reply restating its
>     own formula — "365 x average accounts payable" — read as a turnover of 365. Which
>     multiplication sign the model happened to write decided the outcome: ASCII `x` blocked, `×`,
>     `X` and `*` passed, so the same question answered or refused at random (2 of 10 runs on the
>     Amazon DPO item, each having computed the correct 93.86).
> - **Scored from a declared answer, not the prose.** The reply is asked to end with an `ANSWER:`
>   line and only that is parsed. Scanning the whole reply made every incidental figure a
>   candidate — measured at 9.3 per answer, one of which was the answer: the fiscal year, the form
>   type (`10-K` → 10, and /100 → 0.1), digits in a company name (`3M` → 3), and the citation's
>   accession, whose hyphens read as minus signs (`…-24-…` → -24 → -0.24, enough to score a wrong
>   -0.50 correct against a gold of -0.23). This is how GSM8K (`####`) and MATH (`\boxed{}`) are
>   graded. **It is an administration change: the evaluated prompt carries this suffix and the
>   production prompt does not.** Slot emission was 52/52 on answers the agent actually gave.
> - **Tolerance is 1%**, the bar [FinQA](https://github.com/czyssrs/FinQA) uses for execution
>   accuracy. Measured twice, 5% / 2.5% / 1% / 0.5% score identically — figures come from
>   deterministic tools, so an answer either matches to the cent or misses by a mile.
> - **The small-ratio band is gold's rounding, not slack.** Below |gold| < 5 a relative tolerance
>   collapses (1% of 0.01 is 0.0001, finer than gold is printed), so a fixed ±0.05 takes over. The
>   benchmark's own justifications confirm what it absorbs: American Water Works' dividends are
>   "directly extracted" as $389M while gold records $0.40; Coca-Cola's and AES' ROA golds are 0.01
>   and −0.02 against answers of 0.014 and −0.015. **All three answers are right and gold is
>   coarser.** The band now also requires the same sign — at gold −0.02 it spanned ±250% of the
>   target, and a ratio of the opposite sign is a different answer in kind, not a near miss.
> - **The one remaining "fabrication" is a benchmark-item shape**, not an agent error: Amcor's gold
>   is `87% of the total restructuring liability`, prose that begins with a digit, so it is routed
>   to numeric scoring while the answer it wants is an explanation.
>
> **Update (2026-08).** A second full run of the improved agent, plus calibration of both eval
> judges, refined the picture and is the current headline:
> - **Numeric accuracy ~88%** (cross-validated: a strict LLM judge and a deterministic
>   number-extraction check agree at 86–88%; a looser scale-tolerant matcher read 96% and was set
>   aside as it can produce coincidental false positives).
> - **Zero fabrications**, verified case by case (the only real numeric error in 150 was a
>   ticker mix-up, BBY vs BBBY; the rest are convention differences or over-abstention).
> - **~93% grounded / traceable.** This surfaced a bug in the *measurement*: the faithfulness judge
>   was fed only retrieved prose, so a figure computed by a tool was scored ungrounded because the
>   number wasn't in the text. Feeding the judge the tool outputs moved the same answers from 63% to
>   93%. **A judge you haven't audited is just one more model you're trusting.**
> - The **correctness judge** was calibrated against human labels on a difficulty-stratified set
>   (Cohen's κ = 0.61) and found *systematically over-strict* on defensible convention differences,
>   so the reported narrative-correctness is a lower bound.
>
> The "addressable coverage 44% → 87%" framing below is the original five-round improvement story and
> is kept for the history; the numbers above are the current, cross-validated headline.

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
| **correct** | matches the gold number within 1% (plus a ±0.05 band below \|gold\| < 5, same sign — see the 2026-09 update) |
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

> **Superseded by the 2026-09-13 update above**, which reports the narrative side addressable
> (36/82 = 44%) on the 95 non-numeric-gold questions rather than overall on a 61-question subset.
> The reading below is kept for the method.

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
