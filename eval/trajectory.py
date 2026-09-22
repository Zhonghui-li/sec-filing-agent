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
TRAJECTORY_METRICS = ["tool_precision", "arg_correct", "order_ok", "dependency_ok", "call_budget"]


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
    steps = path["steps"]
    where = _match_steps(steps, calls)
    by_sid = {st["sid"]: st for st in steps}
    matched = {i for i in where.values() if i is not None}

    res: Dict[str, Optional[bool]] = {}
    res["tool_recall"] = all(i is not None for i in where.values())   # internal, for path choice

    extra = [c for i, c in enumerate(calls) if i not in matched]
    res["tool_precision"] = not extra and not any(c.get("tool") in (forbidden or []) for c in calls)

    # arg_correct: only over declared arg keys. A step with no declared args contributes nothing,
    # so a tool whose only argument is free text (a search query) can't fail on wording.
    checked = [(st, where[st["sid"]]) for st in steps if st.get("args")]
    if not checked:
        res["arg_correct"] = None
    else:
        res["arg_correct"] = all(
            idx is not None and all(_arg_matches(v, (calls[idx].get("args") or {}).get(k))
                                    for k, v in st["args"].items())
            for st, idx in checked)

    # order_ok: declared order, minus pairs the case marked interchangeable.
    free = {frozenset(g) for g in path.get("unordered", [])}
    pairs = [(a["sid"], b["sid"]) for x, a in enumerate(steps) for b in steps[x + 1:]
             if not any({a["sid"], b["sid"]} <= g for g in free)]
    seq = [(where[a], where[b]) for a, b in pairs]
    res["order_ok"] = None if not seq else all(
        i is not None and j is not None and i < j for i, j in seq)

    # dependency_ok: three-valued. A dependent step must come after what it depends on, and should
    # visibly carry something that step produced. When its arguments are free text with no literal
    # overlap we cannot tell, so we say so instead of guessing — see note 33.
    verdicts = []
    for st in steps:
        for dep in st.get("depends_on", []):
            i, j = where.get(dep), where.get(st["sid"])
            if i is None or j is None or j < i:
                verdicts.append(False)
                continue
            src = str(calls[i].get("output") or "")
            dst = " ".join(str(v) for v in (calls[j].get("args") or {}).values())
            if not dst.strip():
                verdicts.append(None)         # nothing to look for the dependency in
                continue
            tokens = [t for t in dst.replace(",", " ").split() if len(t) > 3]
            verdicts.append(True if any(t.lower() in src.lower() for t in tokens) else None)
    real = [v for v in verdicts if v is not None]
    res["dependency_ok"] = None if not verdicts else (False if False in verdicts
                                                      else (all(real) if real else None))

    res["call_budget"] = len(calls) <= len(steps) + slack
    return res


def score_trajectory(case: Dict, trace: List[Dict]) -> Dict[str, Optional[bool]]:
    """Score a run against the case's declared paths, or all-None when the case declares none.

    Cases without a `trajectory` block — every existing case today — are unaffected: each metric
    comes back None and stays out of the denominator. That is also the right answer for an
    ordinary refusal, where calling no tool is the correct behaviour and has no path to compare.
    """
    spec = case.get("trajectory")
    if not spec or not spec.get("paths"):
        return {m: None for m in TRAJECTORY_METRICS}

    calls = [{"tool": t.get("tool"), "args": t.get("args") or {}, "output": t.get("output")}
             for t in (trace or [])]
    forbidden, slack = spec.get("forbidden_tools", []), spec.get("call_slack", 1)

    # Deterministic choice among accepted paths: most steps matched, then fewest extra calls, then
    # declaration order. Paths are never blended — a half-of-A-half-of-B run scores against
    # whichever single path fits best, and if that hybrid is legitimate it gets declared as its own.
    def rank(p):
        r = _score_path(p, calls, forbidden, slack)
        hit = sum(1 for i in _match_steps(p["steps"], calls).values() if i is not None)
        return (-hit, len(calls) - hit), r

    scored = sorted((rank(p) for p in spec["paths"]), key=lambda x: x[0])
    best = scored[0][1]
    return {m: best.get(m) for m in TRAJECTORY_METRICS}
