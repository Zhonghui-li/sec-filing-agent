# Phase 1.7 · reselection on /v1/responses — decision rule

Written 2026-10-12, BEFORE any arm was run. Per note 37 §二.

## The question

o4-mini is shut down 2026-10-23. Which model carries production, measured on the architecture we
actually run now — /v1/responses — rather than the one Phase 1 measured?

Phase 1 (2026-10-08) picked gpt-5-mini on chat/completions. That result is not void, but it is
stale in two ways this round exists to repair: reasoning was dropped between turns on that path,
so every candidate was measured in a per-turn-amnesia state, and two candidates were excluded for
reasons that no longer hold.

```
fixed      commit f031ff8 · RESPONSES_API default (/v1/responses) · REASONING_EFFORT=low
           same 54 dev-slice ids · same index · judges unchanged (gpt-4o / gpt-4o-mini)
           1 run per model
variable   GEN_LLM_MODEL, and nothing else
```

## Candidates, and why each is here

| model | why |
|---|---|
| o4-mini | the incumbent, re-run at THIS commit as the control. Not a candidate — it is the line everything else is measured against, and it is the thing being replaced. |
| gpt-5-mini | Phase 1's pick. Its numbers were taken with reasoning dropped between turns, so they describe a model we would not be deploying. |
| gpt-5.4-mini | eliminated in Phase 1 for behaviour (clarify 2/7, grounded 17/19) that was measured with reasoning DISABLED by the API path. That elimination reason is void. |
| gpt-6-astra | could not call tools at all on the old path. It can now, on both. |

## What decides, and what does not

Inherited from §七 unchanged, because the reason still holds: the golden paths were labelled
against o4-mini's behaviour, so a new model taking a legal-but-unlabelled route scores as wrong.

```
DECIDES (label-independent)
  numerical · citation · grounded · facts · facts_grounded
  abstain · reason · context_recall · clarify · forbid
  plus cost, latency, and the format profile

DOES NOT DECIDE (label-dependent) — triggers a MANUAL look at the trace, nothing more
  tool · step_recall · tool_precision · arg_correct · order_ok · dependency_ok · call_budget
```

Manual-review verdict, also fixed in advance:

```
"legal but unlabelled"  a recognised tool · a defensible order · the answer is right
"actually wrong"        a necessary step skipped · a forbidden tool · the answer is wrong
```

## The rule

1. A candidate is **eligible** only if it is no worse than o4-mini on every deciding metric,
   allowing a shortfall of **2 cases or fewer** — §七 measured the noise floor of this slice at
   2–3 cases with one run each, so a 1–2 case gap is not a difference.
2. Among eligible candidates, pick the **cheapest**.
3. If a candidate fails on exactly one deciding metric by 3 or more, it is **not** eliminated
   silently: look at those cases and record what the failures actually are before deciding.
4. If no candidate is eligible, gpt-5-mini is the fallback by default, because Phase 1 already
   validated it as safe on the old path and 10-23 is a hard date.

## Stated before the data, so it cannot be rationalised afterwards

- **Phase 1's conclusion is superseded by this round, not defended by it.** If gpt-5.4-mini now
  wins, that is the correct outcome, not an embarrassment — the earlier elimination was made
  under a condition (reasoning off) that was nobody's intent.
- **gpt-6-astra's price is unknown to us.** It gets recorded like any other number. A candidate
  is not disqualified for being expensive at this step; cost decides only between eligible ones.
- **Cost will be higher than Phase 1's figures across the board**, because /v1/responses preserves
  reasoning across turns (note 37 §九 measured +30% reasoning on o4-mini). Comparisons are against
  o4-mini *in this round*, never against Phase 1's table.

## A limit of this design, stated up front

One run per model on 54 cases. This separates models from each other no better than the noise
floor allows, which is why rule 1 carries a 2-case tolerance rather than demanding strict
superiority. A winner here is a candidate for Phase 4, not a verified improvement; Phase 4
re-measures it on all 144.
