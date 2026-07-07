"""Deterministic output guardrail — the "hard" layer of the finance bar (a rule the model can't
talk its way past). Pure string/regex logic, no LLM or heavy deps, so it runs in the L1 tests.

It only flags PHYSICALLY IMPOSSIBLE magnitudes, so a genuine outlier — which comes from a
deterministic tool (get_ratio on real XBRL), not hand arithmetic — is never rejected. It catches
the mis-composed multi-step numbers seen on FinanceBench (DPO 1419 days, cash-conversion-cycle
4760 days) and dollar figures asserted with no data tool called at all (fabricated from memory).
"""
import re
from typing import List

_IMPLAUSIBLE = [
    (re.compile(r"(-?\d[\d,]*\.?\d*)\s*days\b", re.I), 1000),   # days-outstanding ratios are bounded
    (re.compile(r"(-?\d[\d,]*\.?\d*)\s*x\b"), 100),             # turnover ratios are bounded
]
_DATA_TOOLS = {"get_financials", "get_ratio", "get_growth", "compute_formula",
               "get_my_financials", "get_my_ratio", "get_my_growth"}

_SAFE = ("I can't give a reliable figure for this — it needs a computation I don't have a "
         "deterministic tool for, and I won't report a hand-derived number that may be wrong. "
         "Ask for the underlying figures (I can give those exactly), or rephrase.")


def guardrail(answer: str, tools_used: List[str]) -> str:
    """Return the answer, or a safe abstention if it reports an untrustworthy number:
    (a) a physically-impossible magnitude (the model mis-composed a formula by hand), or
    (b) a dollar figure asserted with NO data tool called at all (fabricated from memory).
    Percentages are never thresholded (growth can legitimately exceed 100%)."""
    if "abstain" in tools_used:
        return answer
    for rx, limit in _IMPLAUSIBLE:
        for m in rx.finditer(answer):
            try:
                if abs(float(m.group(1).replace(",", ""))) > limit:
                    return _SAFE
            except ValueError:
                continue
    if not (set(tools_used) & _DATA_TOOLS) and re.search(r"\$\s?\d", answer):
        return _SAFE
    return answer
