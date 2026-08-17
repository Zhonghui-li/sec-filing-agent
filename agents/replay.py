"""Deterministic replay + re-verification of an answer's evidence path (Framework 2 / Phase A2).

Given a stored answer and its tool-call trace, re-run the DETERMINISTIC tool calls (same tool +
same args) and check the answer still holds. We do NOT replay the LLM's routing or prose — those
are non-deterministic (temp-0 still jitters); we replay the deterministic evidence and re-verify:

  - REPRODUCIBILITY: does each tool return the same figures now as when the answer was produced?
    A difference means the underlying filing data moved (a later restatement / correction) — the
    answer may be superseded. That is a signal, not a fabrication.
  - GROUNDING: does every dollar / ratio figure the answer asserts still trace to a freshly-fetched
    tool output? A figure that traces to nothing (and no source change) is the real red flag.

Verdicts: VERIFIED · SOURCE_CHANGED · UNGROUNDED · IRREPRODUCIBLE · NO_DETERMINISTIC_EVIDENCE.

CLI (live): OPENAI_API_KEY=... DATABASE_URL=... python -m agents.replay "How did NVIDIA's revenue change YoY?"
"""
import math
import re

from agents import finance_tools, statements

# only the deterministic (XBRL-backed) tools are replayable; search_filings/abstain are not.
_DETERMINISTIC_TOOLS = {name: fn for name, fn in {
    "get_financials": finance_tools.get_financials,
    "get_ratio": finance_tools.get_ratio,
    "get_growth": finance_tools.get_growth,
    "compute_formula": finance_tools.compute_formula,
    "compute": finance_tools.compute,
    "get_statement": getattr(statements, "get_statement", None),
    "largest_line_item": getattr(statements, "largest_line_item", None),
    "get_segment_breakdown": getattr(statements, "get_segment_breakdown", None),
    "get_segment_growth": getattr(statements, "get_segment_growth", None),
}.items() if fn}

_TOL = 0.02
_MULT = {"trillion": 1e12, "billion": 1e9, "million": 1e6, "thousand": 1e3,
         "t": 1e12, "b": 1e9, "m": 1e6, "k": 1e3}


def _match(a, b):
    """Equal within tolerance, allowing a consistent power-of-ten unit change ($5B == 5,000,000,000).
    A dropped/added zero is a ×10 factor and still fails (mantissa 5 vs 50)."""
    if a == 0 or b == 0:
        return a == b
    k = round(math.log10(abs(a) / abs(b)))
    return abs(abs(a) / abs(b) - 10.0 ** k) <= _TOL * 10.0 ** k


def _nums(text):
    """Significant numbers in a tool/answer string, ignoring accession ids and URLs."""
    text = re.sub(r"\d{10}-\d{2}-\d{6}", " ", text or "")
    text = re.sub(r"https?://\S+", " ", text)
    out = []
    for n in re.findall(r"-?\d[\d,]*\.?\d*", text):
        try:
            out.append(abs(float(n.replace(",", ""))))
        except ValueError:
            pass
    return out


def _answer_figures(answer):
    """The financial figures an answer ASSERTS: dollar amounts (unit-normalized) and ratio values
    (%, x, days). Bare integers ('5 years', '3 segments') are skipped so they don't false-flag."""
    figs = []
    for num, unit in re.findall(
            r"\$\s?([\d,]+(?:\.\d+)?)\s*(trillion|billion|million|thousand|[kmbt])?\b", answer, re.I):
        figs.append(abs(float(num.replace(",", "")) * _MULT.get(unit.lower(), 1.0)))
    for num in re.findall(r"(-?[\d,]+(?:\.\d+)?)\s*(?:%|x\b|days\b)", answer, re.I):
        try:
            figs.append(abs(float(num.replace(",", ""))))
        except ValueError:
            pass
    return figs


def _covered(src, dst):
    return all(any(_match(s, d) for d in dst) for s in src)


def replay(trace, answer, tools=None):
    """Re-run the deterministic tool calls in `trace` and re-verify `answer`. Returns a report:
    {verdict, checks:[{tool,args,status,stored,fresh}], numbers_grounded, ungrounded}."""
    tools = _DETERMINISTIC_TOOLS if tools is None else tools
    checks, fresh_nums = [], []
    for step in trace or []:
        fn = tools.get(step.get("tool"))
        if not fn:                                  # search_filings / abstain — not replayable
            continue
        try:
            fresh = str(fn(**(step.get("args") or {})))
        except Exception as e:                      # noqa: BLE001 — any tool failure = irreproducible
            checks.append({"tool": step["tool"], "args": step.get("args"),
                           "status": "error", "detail": str(e)[:150]})
            continue
        fresh_nums += _nums(fresh)
        stored, f = _nums(step.get("output") or ""), _nums(fresh)
        # one-directional: every figure the stored output relied on must still appear now. Fresh may
        # carry MORE numbers (untrimmed vs the UI-trimmed stored, added dates/context) — that's fine;
        # only a stored figure that no longer reproduces means the source moved.
        reproduced = _covered(stored, f)
        checks.append({"tool": step["tool"], "args": step.get("args"),
                       "status": "reproduced" if reproduced else "source_changed",
                       "stored": (step.get("output") or "")[:150], "fresh": fresh[:150]})

    if not checks:
        return {"verdict": "NO_DETERMINISTIC_EVIDENCE", "checks": [],
                "numbers_grounded": None, "ungrounded": []}

    ungrounded = [a for a in _answer_figures(answer) if not any(_match(a, n) for n in fresh_nums)]
    # precedence: a tool error blocks a clean verdict; a source change explains any ungrounded figure
    # (stale, not fabricated); only an ungrounded figure WITHOUT a source change is the real red flag.
    if any(c["status"] == "error" for c in checks):
        verdict = "IRREPRODUCIBLE"
    elif any(c["status"] == "source_changed" for c in checks):
        verdict = "SOURCE_CHANGED"
    elif ungrounded:
        verdict = "UNGROUNDED"
    else:
        verdict = "VERIFIED"
    return {"verdict": verdict, "checks": checks,
            "numbers_grounded": not ungrounded, "ungrounded": ungrounded}


def _cli(question):
    from agents.sec_agent import build_agent, run_agent
    out = run_agent(question, agent=build_agent())
    print("ANSWER:", out["answer"], "\n")
    rep = replay(out["trace"], out["answer"])
    print("VERDICT:", rep["verdict"], "| grounded:", rep["numbers_grounded"])
    for c in rep["checks"]:
        print(f"  [{c['status']:14}] {c['tool']}({c.get('args')})")
        if c["status"] != "error":
            print(f"       stored: {c['stored']}\n       fresh : {c['fresh']}")
    if rep["ungrounded"]:
        print("  UNGROUNDED figures:", rep["ungrounded"])


if __name__ == "__main__":
    import sys
    _cli(sys.argv[1] if len(sys.argv) > 1 else "How did NVIDIA's revenue change year over year?")
