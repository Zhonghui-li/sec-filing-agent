# Phase 1.7 — result: gpt-5.4-mini

Three rounds, 2026-10-12. Rules for rounds 1–2 in `eval/_migration_17/DECISION_RULE.md`, for
round 3 in this directory's `DECISION_RULE.md`, each written before the data it judges.

## The rounds, and why there were three

```
round 1   54-case dev slice, 4 models      gpt-5.4-mini: 3 disagreements, all in its favour
round 2   144 cases, 2 front-runners       gpt-5.4-mini: 10 regressions, 5 improvements — REVERSED
round 3   144 cases, after fixing OUR bug  gpt-5.4-mini: 8 regressions, 6 improvements, diffuse
```

**Round 1's result was an artifact of the slice.** Three of the four cases where round 2 found
gpt-5.4-mini weaker — Q11, Q24, Q29 — are not in the 54-case dev slice at all. The slice can
select a model; what it contains decides what it can see.

**Round 2's result was an artifact of our own tool description.** The four abstain failures
shared one mechanism: gpt-5.4-mini called `get_segment_breakdown` twice where gpt-5-mini called
it five times, searched the filing text instead, and abstained. Every other structured tool was
used identically — get_financials 76 vs 77, get_ratio 15 vs 15, get_growth 9 vs 9, compute 7 vs
7, get_statement 1 vs 1. One tool wide, which is the shape of a description problem rather than a
model deficiency. The description's three examples were all about AMOUNTS while the failing
questions asked which segments EXIST, something the tool answers and never claimed to.

After the fix (b2b0104), gpt-5.4-mini calls it 5 times — the same as gpt-5-mini — and abstain goes
from 4 regressions / 0 improvements to 1 / 1.

## Verdict under the round-3 rule

| rule | outcome |
|---|---|
| 1. eligible — no deciding metric REGRESSED vs the repo baseline | both pass |
| 2a. head-to-head, if any metric clears the 5-pair floor | none does → **tied on quality** |
| 2b. median latency | **gpt-5.4-mini 3.29s** vs 4.37s |
| 2c. median answer length | **gpt-5.4-mini 163** vs 264 characters |
| 2d. cost | $0.400 vs $0.191 = 2.09x, inside the 3x ceiling fixed in advance |

**gpt-5.4-mini.**

## Stated against our own conclusion

- The raw count still leans the other way: 8 regressions against 6 improvements across the
  deciding metrics. The rule reads that as tied because no single metric approaches the
  discordance floor, and because summing across metrics is not a valid test — one case can break
  several, and the metrics are not independent. Both numbers are here so the verdict can be
  argued with.
- The remaining differences are **diffuse**: 8 regressions over 6 distinct cases, 6 improvements
  over 5, with no shared mechanism. That is what distinguishes them from round 2, where four
  failures traced to one cause. A diffuse difference below every threshold is what noise looks
  like; it is not proof that none of it is real.
- **One run per model.** `--repeat` would measure the noise floor directly and has not been run.
- Phase 1's selection of gpt-5-mini was not wrong on its own terms. It was made on an
  architecture we no longer run, against a tool description that has since been corrected.

## What production actually changed to, and from

Rounds 1 to 3 compare *candidates to each other*, and the cost table below is gpt-5-mini against
gpt-5.4-mini. That is not the change production made. Production ran **o4-mini**, which is being
retired; a reader checking this file finds a 2.09x cost increase and has no way to see what shipping
actually did. So, measured on the round-1 dev slice at one commit with the model as the only
variable:

| 54 cases, same commit | o4-mini (replaced) | gpt-5.4-mini (shipped) | |
|---|---|---|---|
| cost | $0.356 | $0.165 | **54% cheaper** |
| per question | $0.0066 | $0.0031 | |
| p50 latency | 4.66s | 3.53s | **24% faster** |
| median answer length | 154 chars | 136 chars | |
| prompt-cache hit | 89% | 87% | |

Two things this does not say.

**gpt-5.4-mini was not the cheapest candidate.** gpt-5-mini was, by a further 2.09x, and it was
eligible. The rule in this directory's DECISION_RULE.md ranks quality and latency ahead of cost
under a 3x ceiling, and that rule was fixed before the winner was known. Shipping the more
expensive of two tied candidates is the decision that rule produces, deliberately.

**The o4-mini comparison is on 54 cases, not 144.** The full set was run for the two candidates,
not for the incumbent, because o4-mini is being retired and re-measuring it at 144 would have cost
more than the comparison was worth. Anything quoting these figures should say which set they are
from.

## What choosing gpt-5.4-mini over gpt-5-mini costs

Against the other CANDIDATE, not against what production ran. The comparison with the replaced
model is above.

```
per question          $0.0013 -> $0.0028
at DAILY_QUOTA=150    $0.20/day -> $0.42/day
a full month at quota ~$6 -> ~$12.6
```

Bought: 1.1s off the median answer, and answers 38% shorter — which on this project is a quality
property and not only a cost one, since prose beyond what the citations cover is surface for
claims nothing supports.

## The part worth keeping

The defect that decided round 2 was ours, not the model's. o4-mini and gpt-5-mini both generalised
past an incomplete tool description; gpt-5.4-mini read it literally and was penalised for being
right. Depending on a model's willingness to guess past our documentation is an implicit
dependency on that model, and it surfaced the first time the model changed — which is the same
failure, in a different costume, as the deploy environment that had quietly diverged from the test
environment two days earlier.
