# Phase 1.6 · /v1/responses — decision rule

Written 2026-10-11, BEFORE either arm was run. Per note 37 §二.

## The question

Does changing ONLY the API path change the agent's behaviour?

```
fixed      GEN_LLM_MODEL=o4-mini · REASONING_EFFORT=low · same index · same commit (bc24fa7)
           same 54-case set · 1 run per arm · judges unchanged (gpt-4o / gpt-4o-mini)
variable   RESPONSES_API=0  ->  chat/      (control)
           RESPONSES_API=1  ->  responses/ (experiment)
```

## The rule

| outcome | decision |
|---|---|
| 0 metrics REGRESSED | adopt the migration, proceed to Phase 1.7 (reselection over the widened candidate set) |
| any metric REGRESSED | STOP. Investigate before 1.7. Do not adopt. |
| discordant pairs < 5 (`gate.MIN_DISCORDANT`) | UNDECIDABLE — counts as neither pass nor regression |

## Measured, NOT gated

- **reasoning tokens / cost.** Note 37 §八 predicts a rise, because /v1/responses preserves
  reasoning across turns while chat/completions drops it. A rise is the expected consequence of
  the thing being migrated to, not a reason to reject it. Record it; decide separately.
- **latency.** Same reason. Arms run sequentially so the numbers stay comparable.

## A limit of this design, stated up front

One run per arm measures the architecture change AND run-to-run noise together. So a REGRESSED
verdict here is a reason to repeat the arm, not yet a finding — asserting before verifying is the
mistake recorded in note 35 and repeated on 2026-10-08. A clean result is informative; a dirty one
needs a second run before it is called anything.
