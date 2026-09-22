"""Trajectory scoring: did the answer come through a correct, verifiable, non-wasteful tool path?

The existing metrics ask whether the answer is right. These ask how it was reached. A run that
lands on the right number through a redundant or out-of-order path is still a production failure
(the "lucky pass" problem), and nothing in the suite could see it before.

Every metric returns True / False / None. **None means not applicable or not decidable, and is
excluded from the denominator** — deliberately, because a metric that guesses when it can't tell
becomes the next "score dropped but nothing broke" (that happened four times on 2026-09-21 alone).

Contract and schema: Obsidian note 33.
"""
from typing import Any, Dict, List, Optional

# Metrics defined here. Listed so eval/score.py can report them without hardcoding names twice.
# `step_recall` is not a rename of score.py's `tool`. That one asks whether the right KINDS of tool
# were used and is a set comparison, so it can't see that a path wanted two searches and got one.
# This asks whether every declared STEP happened. They coincide on single-step paths and diverge
# exactly where a path calls one tool more than once.
TRAJECTORY_METRICS = ["step_recall", "tool_precision", "arg_correct", "order_ok",
                      "dependency_ok", "call_budget"]
# Reported, never gated. Efficiency has no threshold worth setting until there is a real
# distribution to set it from — the same reason cost stays out of the CI gate.
EFFICIENCY_METRICS = ["parallel_rate"]

# Trajectory metrics are report-only for now, for the same reason. A run's path carries randomness
# the answer metrics don't — tool choice, query wording, how many searches it takes to convince
# itself, which accepted path it lands on — and the 10pp tolerance was calibrated on answer metrics
# over 84 cases. Inheriting it here would be a guess. Gate them once repeated runs of one unchanged
# build show what each metric's natural spread actually is; they may well need different tolerances
# from each other.
REPORT_ONLY = set(TRAJECTORY_METRICS) | set(EFFICIENCY_METRICS)


def _canon(v: Any) -> Any:
    """Resolve a metric alias to the name the tool actually resolves it to.

    The tools accept several spellings of a metric ("research and development expense",
    "r&d", "rd_expense") and normalise them internally. Comparing the raw string would mark a
    correct call wrong for choosing a different accepted synonym — which is what happened to L27
    on the first run of this scorer, where the annotation was wrong and the agent was right.
    """
    if not isinstance(v, str):
        return v
    try:                                    # reuse the tools' own normaliser, never a second copy
        from agents.finance_tools import _canon as _tool_canon
        return _tool_canon(v)
    except Exception:
        return v.strip().lower()


def _arg_matches(expected: Any, actual: Any) -> bool:
    """A list of expected values means any of them is acceptable (e.g. several abstain reasons)."""
    if isinstance(expected, list):
        return any(_arg_matches(e, actual) for e in expected)
    if isinstance(expected, str) and isinstance(actual, str):
        return _canon(expected) == _canon(actual)
    try:
        return int(expected) == int(actual)
    except (TypeError, ValueError):
        return expected == actual


def _match_steps(steps: List[Dict], calls: List[Dict]) -> Dict[str, Optional[int]]:
    """Map each declared sid to the index of the call that satisfies it, or None if unmatched.

    Greedy, left to right, first unused call of the right tool whose declared args all match.
    Falls back to the first unused call of that tool so a wrong-argument call is reported as a
    wrong argument rather than vanishing as a missing tool.
    """
    taken, out = set(), {}
    for st in steps:
        hit = None
        for i, c in enumerate(calls):
            if i in taken or c.get("tool") != st["tool"]:
                continue
            if all(_arg_matches(v, (c.get("args") or {}).get(k))
                   for k, v in (st.get("args") or {}).items()):
                hit = i
                break
        if hit is None:                       # right tool, wrong args — still this step's call
            hit = next((i for i, c in enumerate(calls)
                        if i not in taken and c.get("tool") == st["tool"]), None)
        if hit is not None:
            taken.add(hit)
        out[st["sid"]] = hit
    return out


def _score_path(path: Dict, calls: List[Dict], forbidden: List[str], slack: int) -> Dict:
    """Score one declared path.

    The rule every metric below follows: **a step that never happened is reported once, by
    step_recall, and does not cascade.** Anything downstream evaluates only the steps that actually
    ran and matched. Without that rule a single missing search came back as four failures —
    step_recall, arg_correct, order_ok and dependency_ok all red — which reads as four problems and
    is one. Constraints are therefore scored individually and aggregated over the evaluable ones,
    rather than the whole metric collapsing because one endpoint is absent.
    """
    steps = path["steps"]
    where = _match_steps(steps, calls)
    matched = {i for i in where.values() if i is not None}
    res: Dict[str, Optional[Any]] = {}

    # The only metric that speaks to a missing step.
    res["step_recall"] = all(i is not None for i in where.values())

    extra = [c for i, c in enumerate(calls) if i not in matched]
    res["tool_precision"] = not extra and not any(c.get("tool") in (forbidden or []) for c in calls)

    # arg_correct: over declared arg keys, plus any `args_differ_from` constraint. A step with
    # neither contributes nothing, so a tool whose only argument is free text can't fail on wording.
    #
    # `args_differ_from` is how a step says its arguments must NOT repeat an earlier step's. A
    # second search that rephrases the query is verifying a negative — a filing may say "digital
    # assets" where the question said "crypto mining", and one miss is weak evidence of absence.
    # A second search with the SAME query adds no information. Both are two calls, so counting
    # calls can't separate them; whether the second one asked anything new is what does.
    checks = []
    for st in steps:
        idx = where[st["sid"]]
        if idx is None:
            continue                                   # missing: step_recall said so already
        if st.get("args"):
            checks.append(all(_arg_matches(v, (calls[idx].get("args") or {}).get(k))
                              for k, v in st["args"].items()))
        for prior in st.get("args_differ_from", []):
            j = where.get(prior)
            if j is None:
                continue                               # the step to differ from never ran
            a = {k: str(v).strip().lower() for k, v in (calls[idx].get("args") or {}).items()}
            b = {k: str(v).strip().lower() for k, v in (calls[j].get("args") or {}).items()}
            checks.append(a != b)
    res["arg_correct"] = all(checks) if checks else None

    # order_ok: declared order, minus pairs the case marked interchangeable, minus pairs where
    # either endpoint is absent — those are unevaluable, not violated.
    free = {frozenset(g) for g in path.get("unordered", [])}
    verdicts = [where[a["sid"]] < where[b["sid"]]
                for x, a in enumerate(steps) for b in steps[x + 1:]
                if not any({a["sid"], b["sid"]} <= g for g in free)
                and where[a["sid"]] is not None and where[b["sid"]] is not None]
    res["order_ok"] = all(verdicts) if verdicts else None

    # dependency_ok: three-valued, per constraint. A dependent step must run after what it depends
    # on and should visibly carry something that step produced. When its arguments are free text
    # with no literal overlap we can't tell, so we say so rather than guess — a metric that guesses
    # when it can't see becomes the next score that drops while nothing has broken.
    dep: List[Optional[bool]] = []
    for st in steps:
        for d in st.get("depends_on", []):
            i, j = where.get(d), where.get(st["sid"])
            if i is None or j is None:
                continue                               # unevaluable, not a violation
            if j < i:
                dep.append(False)
                continue
            src = str(calls[i].get("output") or "")
            dst = " ".join(str(v) for v in (calls[j].get("args") or {}).values())
            tokens = [t for t in dst.replace(",", " ").split() if len(t) > 3]
            dep.append(True if any(t.lower() in src.lower() for t in tokens) else None)
    decided = [v for v in dep if v is not None]
    res["dependency_ok"] = None if not decided else all(decided)
    res["_dep_decided"], res["_dep_total"] = len(decided), len(dep)

    # call_budget is relative to THIS path. A question answerable two ways has a different
    # reasonable call count on each, so the budget follows whichever path the run actually took.
    res["call_budget"] = len(calls) <= len(steps) + slack

    # parallel_rate: of the groups this case says COULD go out together, how many actually did.
    # Not parallel-calls-over-total-calls — most calls have a real dependency and were never
    # eligible, so that ratio would punish correct sequencing. Serial execution of an eligible
    # group is slower, not wrong, which is why this is reported and never gated.
    groups = path.get("parallelizable", [])
    if not groups or all(c.get("turn") is None for c in calls):
        # No eligible group, or a trace recorded before `turn` existed. Either way there is nothing
        # to measure — and saying 0.0 would report "never parallelised" for a run we simply cannot
        # see, which is the failure mode this module exists to avoid.
        res["parallel_rate"] = None
    else:
        together = sum(
            1 for g in groups
            if all(where.get(sid) is not None for sid in g)
            and len({calls[where[sid]].get("turn") for sid in g}) == 1)
        res["parallel_rate"] = together / len(groups)
    return res


def score_trajectory(case: Dict, trace: List[Dict]) -> Dict[str, Optional[Any]]:
    """Score a run against the case's declared paths, or all-None when the case declares none.

    Cases without a `trajectory` block — every existing case today — are unaffected: each metric
    comes back None and stays out of the denominator. That is also the right answer for an
    ordinary refusal, where calling no tool is correct behaviour and there is no path to compare.

    Also returns `matched_path_id`: which accepted path the run took. A question answerable two
    ways will be answered both ways across runs, and both are correct — but a build that shifts
    from mostly-A to mostly-B has changed strategy, and aggregating this field over runs shows
    that with no scoring and no extra schema.
    """
    spec = case.get("trajectory")
    if not spec or not spec.get("paths"):
        return {m: None for m in TRAJECTORY_METRICS + EFFICIENCY_METRICS + ["matched_path_id"]}

    calls = [{"tool": t.get("tool"), "args": t.get("args") or {}, "output": t.get("output"),
              "turn": t.get("turn")}          # carried through — parallel_rate reads nothing else
             for t in (trace or [])]
    forbidden, slack = spec.get("forbidden_tools", []), spec.get("call_slack", 1)

    # Deterministic choice among accepted paths: most steps matched, then fewest extra calls, then
    # declaration order. Paths are never blended — a half-of-A-half-of-B run scores against
    # whichever single path fits best, and if that hybrid is legitimate it gets declared as its own.
    def rank(p):
        r = _score_path(p, calls, forbidden, slack)
        hit = sum(1 for i in _match_steps(p["steps"], calls).values() if i is not None)
        return (-hit, len(calls) - hit), p.get("id"), r

    best = sorted((rank(p) for p in spec["paths"]), key=lambda x: x[0])[0]
    out = {m: best[2].get(m) for m in TRAJECTORY_METRICS + EFFICIENCY_METRICS}
    out["matched_path_id"] = best[1]
    out["_dep_decided"] = best[2].get("_dep_decided", 0)
    out["_dep_total"] = best[2].get("_dep_total", 0)
    return out
