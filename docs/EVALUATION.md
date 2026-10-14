# How the harness works

What each layer checks, what it cannot see, and how to run it. The README says what is measured;
this says how, and what the measurements are worth.

The organising idea: **a layer that cannot say what it misses will be trusted too far.** Every
section below names its own blind spot, because each one was found by being surprised.

---

## The six layers

| # | Layer | Runs | Needs | Cost |
|---|---|---|---|---|
| 1 | Unit tests — 226 tests, 17 files | every push, and locally in ~3s | nothing | free |
| 2 | Image contracts — build, smoke, the suite re-run inside the image, `/health` | every push | Docker (CI has it) | free |
| 3 | Regression gate — 144 cases, 17 gated metrics | before a release | OpenAI key, database | ~$0.40 |
| 4 | Red team — 50 questions, 17 trap categories | before a release | OpenAI key | ~$0.30 |
| 5 | FinanceBench — 150 questions / 32 companies | before a release | OpenAI key, database | ~$5 |
| 6 | Production — miss queue, traces, a manual check | continuously | — | — |

### 1 · Unit tests

Tool logic, scorer arithmetic, guardrail rules, the regression gate's own statistics, the routing
table that decides which knob each model accepts, and the two consistency checks that exist
because this project got burned without them: `test_requirements_agree.py` and
`test_abstain_labels.py`.

**Blind to:** anything that needs a model to answer. It cannot tell you whether the agent is right,
only whether the machinery around it is.

CI installs `pytest` and `pandas` and nothing else, deliberately — so a test that quietly depends
on a project dependency fails there rather than in production. The cost is that tests importing
langchain skip in CI; they run locally and in layer 2.

### 2 · Image contracts

Layers 1 and 3–5 run against `requirements.txt`. The container installs `service/requirements.txt`.
Nothing compared them until this layer existed, and on 2026-10-11 that cost two outages in one
afternoon.

It does four things: builds the image the way Cloud Build does; runs `service/smoke.py` **inside**
it; re-runs the whole test suite there, against the versions that actually ship; and starts the
container to see it serve `/health`.

`service/smoke.py` asserts contracts, never versions — versions belong in the requirements files,
where layer 1 compares them. Its current checks:

- the heavy imports the request path makes, **including the lazy ones** (`pyarrow`, `pandas`,
  `edgar`), because those are reached only by a narrative question and far too late to discover
  they are incompatible
- `AIMessage.text` **is** a string, rather than something that becomes one when called
- `ChatOpenAI` honours `use_responses_api`

**Blind to:** whether the answers are any good. And `/health` is weaker than it looks — see below.

### 3 · Regression gate

`python -m eval.score` runs 144 labelled cases and scores 17 metrics plus 4 monitor-only ones. It
compares against `eval/baseline_run.jsonl`, a recorded per-case run, and exits non-zero on a
regression.

**Blind to:** anything the 144 cases do not contain. This is not a small caveat — see
*[What a slice cannot see](#what-a-slice-cannot-see)*.

### 4 · Red team

50 questions, 46 traps across 17 categories and 4 answerable controls. The categories are the
shapes that *tempt* fabrication: nonexistent metrics and years, wrong company, false premises,
anchoring, hand-computable derived figures, over-precision, prompt injection, quarter-vs-year
confusion, currency and unit traps, non-GAAP, restatements.

Scored three ways — fabrication, *safe over-abstention*, pass — so declining a trap counts as a
pass rather than a miss. It drove the fabrication rate from 11% to 0.

**Blind to:** traps nobody thought to write. The suite is a list of known temptations, not a proof.

### 5 · FinanceBench

An external benchmark (Patronus AI) we did not write: 150 questions over 32 companies, split into
a numeric set scored deterministically and a narrative set scored by judges.

Numeric answers are read from a declared `ANSWER:` line rather than by scanning prose for numbers,
at FinQA's 1% tolerance. *Addressable* coverage excludes questions whose evidence lives only in an
earnings release, which `filings_ingest` skips by design — taken from the benchmark's own evidence
labels, never from whether we got it right, which is what keeps it from being a self-issued excuse.

**Blind to:** it is one benchmark, on one domain, and its narrative half is judged by an LLM whose
own limits are described below.

### 6 · Production

Every metric a user asked for that the tools could not return is appended to
`data/cache/metric_misses.jsonl` **and** printed to stdout — on Cloud Run the container filesystem
dies with the instance, so Cloud Logging is the only durable record. Langfuse traces each run when
keys are present.

**Blind to:** nothing here is automatic. It is a queue someone has to read.

---

## What `/health` cannot catch

On 2026-10-11 the deployed image resolved `pyarrow` 26 against a pinned `numpy` 1.26.4. The two
install together and import fine until something touches them. The app started. `/health` answered.
Numeric questions kept working, because they read a baked JSON. Only narrative questions died — the
ones that reach `edgartools` — with `ImportError: pyarrow requires NumPy 2.0 or newer`.

A liveness probe cannot see a lazy import. That is why layer 2 imports the modules rather than
pinging the server, and why `/health` is listed last among its four checks rather than first.

The same deploy had already failed once that afternoon for a different reason: the image was still
building `langchain-core` 0.3.x three days after the repo moved to 1.x, because
`service/requirements.txt` was never updated alongside `requirements.txt`. `.text` is a property in
1.x and a method in 0.3.x, so every answer became
`TypeError: expected string or bytes-like object, got 'method'`.

**Reproducing the container's dependency resolution locally**, which is the check that was missing:

```bash
# WRONG — resolves against what is already installed and reports "already satisfied"
.venv/bin/pip install --dry-run -r service/requirements.txt

# RIGHT — resolves from scratch, the way a clean image does
.venv/bin/pip install --dry-run --ignore-installed -r service/requirements.txt
```

The first form was run, read as evidence, and proved nothing. The second reproduces the image's
choice exactly.

---

## How the gate reaches a verdict

A percentage cannot compare two metrics whose denominators differ by a factor of fifteen — ten
points is ten cases on `tool` and less than one on `forbid` — and it reads three breaks alongside
three fixes as three breaks.

So the gate is **paired**. For each metric it takes the cases where the two runs disagree and runs
an exact one-sided McNemar test at α = 0.05, with a floor: fewer than **5 disagreements** returns
UNDECIDABLE rather than a verdict. Five all going the wrong way has p = 0.031, just inside α, which
is what sets the floor — to catch a 20% regression you need 5 ≤ 0.2n, so n ≥ 25.

Three verdicts, not two: **OK**, **REGRESSED**, and **UNDECIDABLE**. The third is the point. A
metric that moved by two cases has not told you anything, and a gate that pretends otherwise
converts noise into decisions.

### Which metrics decide

Seven of the 17 are **label-dependent** — `tool`, `step_recall`, `tool_precision`, `arg_correct`,
`order_ok`, `dependency_ok`, `call_budget` — because the golden paths were labelled against one
model's behaviour. A new model taking a legal but unlabelled route scores as wrong. They trigger a
manual look at the trace; they do not decide a model selection. The verdict for that look is also
fixed in advance:

```
"legal but unlabelled"  a recognised tool · a defensible order · the answer is right
"actually wrong"        a necessary step skipped · a forbidden tool · the answer is wrong
```

---

## What a slice cannot see

Choosing the generation model in October 2026 took three rounds, and the first two were each
measuring something other than the model.

**Round 1** ran a 54-case development slice over four candidates. `gpt-5.4-mini` led: five
disagreements across three cases, every one in its favour, none against.

**Round 2** ran the full 144 over the two front-runners. The direction reversed — ten regressions
against five improvements, with `abstain` alone at four regressions and no improvements.

**Three of the four cases that reversed it are not in the 54-case slice at all.**

A development slice is not a smaller version of the suite. It is a *subset with a composition*, and
the question types it omits are the weaknesses it can never report. It is the right tool for
choosing cheaply; it is not evidence about anything it does not contain.

### And round 2 was measuring our own bug

Reading the four `abstain` regressions found one mechanism rather than four:

```
get_segment_breakdown     gpt-5-mini 5 calls    gpt-5.4-mini 2 calls
search_filings            46                    53

every other structured tool, unchanged:
get_financials 76 / 77 · get_ratio 15 / 15 · get_growth 9 / 9
compute 7 / 7 · get_statement 1 / 1
```

A difference one tool wide is the shape of a description problem, not a capability gap. The tool's
description offered three example questions, all about *amounts* — "revenue by segment", "which
segment is largest", "how is revenue split geographically" — while the failing questions asked
which segments *exist*. The tool answers that. It had never said so.

Two models generalised past the gap. One read it literally, and was right to. **Depending on a
model to guess past our documentation is a dependency on that model**, and it broke on the first
day the model changed.

After the fix the call counts matched and `abstain` went from 4 regressions / 0 improvements to
1 / 1. Round 3 then ran both candidates again and found them tied, no metric near the floor.

### Changing a rule mid-selection

Rounds 1 and 2 ran under *"among eligible candidates, pick the cheapest"*. Before round 3 that was
changed to quality and latency first, cost as a tiebreak under a 3× ceiling.

Amending a rule after seeing data is only legitimate under three conditions, and all three are
recorded in `eval/_migration_17_round3/DECISION_RULE.md`:

1. **declared before the next measurement**, and binding whatever it produces
2. **the ceiling fixed before knowing who it excludes** — 3× was chosen while it was still unknown
   that the winner would be 2.09× and the model it ruled out would be 8.4×
3. **the tool-description fix justified independently** — the test applied was "would this be fixed
   if the incumbent had already been chosen and no re-run were planned?", and the answer was yes

The sin is not changing a rule. It is changing it quietly, until a preferred candidate wins.

---

## The judges

| Judge | Env var | Default | Judges |
|---|---|---|---|
| domain / grounding | `DOMAIN_JUDGE_MODEL` | `gpt-4o` | `facts`, `facts_grounded`, and FinanceBench groundedness |
| correctness-vs-gold | `CORRECTNESS_JUDGE_MODEL` | `gpt-4o` | FinanceBench narrative correctness |
| red team | `REDTEAM_JUDGE_MODEL` | `gpt-4o-mini` | fabrication labels |

They are deliberately **not** the model being judged, and deliberately **not** changed when the
agent's model changes — two variables moving at once make any difference uninterpretable.

### Why not an off-the-shelf judge

Ragas was replaced, not merely dropped. Its generic assumptions misfire in a regulated domain:
`answer_relevancy`'s noncommittal classifier penalises exactly the honest "this remains uncertain,
see the filing" hedging a compliant answer *should* use. A fully-grounded JPMorgan answer scored
relevancy 0.0.

The replacement follows G-Eval / RAGalyst: own the judge. The rubric encodes the domain's norms
explicitly — hedging is good, appropriate abstention is good, every claim must sit in the retrieved
context — and the judge reasons step by step before a binary verdict. It is shown the **tool
outputs** as well as the retrieved prose, so a figure that came from a tool counts as grounded even
when no sentence contains it. That one change moved grounded from 63% to 93%; a later bug where the
judge saw its evidence truncated to 600 characters was the same mistake at smaller scale.

### Why the judge model has not been upgraded

An external labelled set says it should be, and that is exactly why this is worth writing down.

```
the original 33-item calibration set is SATURATED
    five candidates, three tied at κ = 1.000 — it cannot tell models apart

PHANTOM (NeurIPS 2025, human-labelled 10-K / DEF14A) replaced it
    200 items to choose on, a disjoint 200 sealed

on the sealed slice:  gpt-5-mini κ = 0.820   vs   gpt-4o κ = 0.660
                      21 items better, 5 worse, p = 0.0012
```

Switched to `gpt-5-mini`, the 95 narrative answers went from **96% grounded to 54%**. The failures
share a shape: the numbers are right and the **evaluative conclusion** is what gets failed —
"liquidity is healthy", "margin is improving", "inventory is not a material concern". Forty-eight
of our 95 narrative questions ask for exactly that kind of judgement, and PHANTOM contains none of
it: it builds a hallucination by perturbing a fact, which cannot produce *every fact correct,
conclusion overreaches*. It contains no abstentions either, which is what the
zero-false-positives-on-hedged-answers claim rests on.

**A benchmark proving A beats B on its own task does not mean A is better on yours. The question is
whether the way it constructs failure is the way you actually fail.**

So `gpt-4o` stays until a labelled set exists containing evaluative conclusions and abstentions.
`DOMAIN_JUDGE_MODEL` switches it for anyone who wants the stronger factual detector and accepts
that cost. The spent PHANTOM rows are listed in `eval/labeling/phantom_slices_spent.json` — reusing
them turns held-out data back into tuning data, which is how the previous held-out set was lost.

### On the κ numbers

Three figures exist and they are not interchangeable:

| κ | What it is |
|---|---|
| 1.000 | three candidates on the original 33-item set — the number that proves the set is saturated |
| 0.76 | `gpt-4o` on the balanced calibration set, against 0.947 on the same set left imbalanced (39 good / 13 bad inflates it) |
| 0.660 | `gpt-4o` on PHANTOM's sealed slice — the better-designed measurement, and the lower number |

The README quotes none of them, on purpose. 0.76 is a real measurement on a set the repo has since
judged inadequate, and quoting it while 0.660 exists is the same move as quoting a benchmark's
best historical run.

---

## Metrics that currently cannot be computed

`ragas` was removed on 2026-10-08: it imports `langchain_community.chat_models.vertexai`, which
`langchain-community` removed, and holding that back pins `langchain-core` to 0.3.x, which nothing
else can live with.

Three monitor metrics went with it — `faithfulness`, `answer_relevancy`, `context_precision` — and
`eval/score.py --quality` now raises `ModuleNotFoundError` by design.

`eval/baseline.json` carried `faithfulness: 0.936` and `answer_relevancy: 0.862` forward from a
run before the removal, merged into every later baseline by a rule meant to stop a
deterministic-only run from dropping the monitor metrics. Nothing could reproduce them, so they
were deleted on 2026-10-14: a number no measurement can produce is a claim rather than a result,
and carrying it forward is how it survives long enough to be quoted. Of the three, only
`context_precision` was judged worth reimplementing.

---

## Running each layer

```bash
# 1 · unit tests (no keys, ~3s)
env -u DATABASE_URL .venv/bin/python -m pytest tests/ -q

# 2 · what the image resolves, without Docker
.venv/bin/pip install --dry-run --ignore-installed -r service/requirements.txt
#     with Docker, the whole layer:
docker build -f service/Dockerfile -t sec-filing-agent:ci .
docker run --rm sec-filing-agent:ci python -m service.smoke

# 3 · regression gate (writes nothing; --update-baseline to adopt a run as the new reference)
.venv/bin/python -m eval.score --run-dir eval/runs/$(date +%Y%m%d-%H%M%S)
#     a subset, reusing a recorded slice's exact ids:
IDS=$(python -c "import json;print(','.join(json.loads(l)['id'] for l in \
      open('eval/_migration_lc1x/o4-mini/run1.jsonl')))")
.venv/bin/python -m eval.score --only "$IDS" --run-dir eval/runs/slice
#     measure the noise floor instead of guessing it:
.venv/bin/python -m eval.score --repeat 3 --run-dir eval/runs/variance

# 4 · red team
.venv/bin/python -m eval.redteam.run_redteam

# 5 · FinanceBench — numeric, then narrative
.venv/bin/python -m eval.financebench.run_open
.venv/bin/python -m eval.financebench.judge_narrative

# judge calibration itself
.venv/bin/python -m eval.judge
```

Every run writes a `config.json` beside its rows recording the model, reasoning effort, API path,
index knobs and commit — because a directory of per-case results that cannot say what produced it
will eventually be read as something it is not. That happened on 2026-10-12: the narrative capture
was keyed on question text alone, so hours after the production model changed a rerun printed
"cached" for all 95 questions and judged answers two generations old. Captures now carry
`produced_by` and reuse only what matches.
