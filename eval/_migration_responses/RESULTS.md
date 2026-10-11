# Phase 1.6 · /v1/responses — result

Run 2026-10-11, commit bc24fa7, o4-mini / effort low, 144 cases per arm, identical case ids.
Rule was fixed before either arm ran: see DECISION_RULE.md.

## Verdict: 0 REGRESSED -> adopt, proceed to Phase 1.7

```
OK            11 metrics
UNDECIDABLE    6 metrics   (abstain 4 · grounded 3 · context_recall 2 · tool_precision 2 ·
                            step_recall 1 · tool 1 discordant pairs; threshold is 5)
REGRESSED      0
```

No metric moved enough to clear the gate in either direction. The six UNDECIDABLE are two-sided
noise (`abstain` broke 2 / fixed 2; `grounded` broke 1 / fixed 2), not a one-way drift.

## Measured, not gated

|                  | chat | responses | delta |
|---|---|---|---|
| output tokens    | 40,960 | 53,513 | **+31%** |
| reasoning tokens | 15,808 | 20,608 | **+30%** |
| cases spending any reasoning | 83 | 100 | +17 |
| input tokens     | 1,990,970 | 1,885,765 | -5% |
| cached input     | 1,849,984 | 1,710,848 | -8% |
| tool calls       | 208 | 199 | -4% |
| latency median   | 3.58s | 3.91s | +9% |
| salvaged         | 1 | 1 | 0 |
| guardrail fired  | 2 | 0 | -2 |

The +30% reasoning is the direction note 37 §八 predicted, and for the stated reason:
/v1/responses preserves reasoning across turns while chat/completions drops it. It shows up as
more cases spending reasoning at all (83 -> 100), not as a few cases spending much more. Budget
for it before Phase 1.7, where the candidate models are more expensive than o4-mini.

## What this run does NOT establish

- **One run per arm.** Architecture and run-to-run noise are not separated. A clean result is
  informative; it is not a measurement of the noise floor. `--repeat` exists for that.
- Judges were unchanged (gpt-4o / gpt-4o-mini on chat/completions). Whether THEY should migrate
  is Phase 5, deliberately untouched.
- The guardrail 2 -> 0 and the 17 extra reasoning cases are single-run observations. Nothing here
  licenses a claim that /v1/responses makes the agent better.

## Incident during the run

The control arm died at case 63 (entering Q13) on a bare `openai.PermissionDeniedError: 403`
with an EMPTY message body. It was transient, not the 2026-10-23 shutdown arriving early: a
chat/completions call with o4-mini succeeded minutes later. `score.py`'s per-case resume (it skips
ids already in run<N>.jsonl) finished the remaining 81 cases without re-spending on the first 63.
Both arms' id sets are identical, verified.
