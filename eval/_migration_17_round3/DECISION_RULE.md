# Phase 1.7 round 3 — amended rule, and why it is being amended

Written 2026-10-12, BEFORE the tool description is changed and before either model is re-run.

## What round 2 established

On all 144 cases, gpt-5.4-mini lost 4 abstain cases to gpt-5-mini and gained none. Reading those
cases showed one mechanism, not four: it calls `get_segment_breakdown` on 2 questions where
gpt-5-mini calls it on 5, searches the filing text instead, and then abstains. Every other
structured tool is used identically — get_financials 76 vs 77, get_ratio 15 vs 15, get_growth
9 vs 9, compute 7 vs 7, get_statement 1 vs 1. The divergence is one tool wide.

The tool's description is the reason. Its three example questions are all about AMOUNTS —
"revenue by segment/region", "which segment is largest", "how is revenue split geographically" —
while Q24 and Q29 ask which segments EXIST. The tool answers that; the description never says so.
o4-mini and gpt-5-mini generalised past the gap. gpt-5.4-mini read it literally.

## Change 1 — the description, which would be fixed regardless of model choice

The test applied before deciding this was legitimate: would this change be made if gpt-5-mini had
already been selected and no re-run were planned? Yes. Depending on a model to generalise past an
incomplete tool description is an implicit dependency on that model's habits, and it broke the
first time the model changed — which is the lesson of the whole 2026-10-11/12 sequence.

It changes routing for EVERY model, so round 2's comparison and the Phase 4 baseline both have to
be rebuilt on top of it. That is a cost of fixing it, not a reason not to.

## Change 2 — the selection rule

Round 1 and 2 ran under "among eligible candidates, pick the cheapest". That is being changed to
quality and latency first, with cost as a tiebreak and a ceiling. The reason is a product one,
stated plainly: this service is a live demo whose visitors wait for each answer, so response time
and answer concision are part of what it is being judged on, and the price gap between the
candidates is small in absolute terms ($0.19 vs $0.42 per 144 questions).

Changing a rule after seeing data is only legitimate if the change is declared before the next
measurement and binds whatever it produces. This file is that declaration.

```
1. ELIGIBLE      no deciding metric REGRESSED against the repo baseline, per eval/gate.py
2. RANK          a) head-to-head on deciding metrics, if any metric clears the 5-pair
                    discordance floor; if none does, the two count as TIED on quality
                 b) median latency
                 c) median answer length (shorter wins — more prose is more surface for
                    statements no citation covers)
                 d) cost
3. COST CEILING  a candidate costing more than 3x the cheapest eligible candidate is out
                 regardless of how it ranks. Set now, at 3x, before knowing who wins:
                 gpt-5.4-mini is 2.2x and passes, gpt-6-astra was 8.4x and would not.
```

Deciding metrics are unchanged from round 1: numerical, citation, grounded, facts,
facts_grounded, abstain, reason, context_recall, clarify, forbid. The seven label-dependent
metrics still only trigger a manual look, for the reason §七 gave: the golden paths were labelled
against o4-mini's behaviour.

## Bound in advance

- **The outcome is accepted either way.** If gpt-5.4-mini still loses abstain cases after the
  description is fixed, it loses, and the description fix is kept anyway because it was justified
  on its own.
- **If the fix changes gpt-5-mini's behaviour too**, that is a result, not a confound — both
  models are measured on the same prompt, which is the only thing fairness requires here.
- **No further prompt edits before this measurement.** Editing again after seeing the re-run
  would be tuning the harness until a preferred candidate wins, which is worse than never having
  fixed the description.
