"""Deterministic output guardrail — the "hard" layer of the finance bar (a rule the model can't
talk its way past). Pure string/regex logic, no LLM or heavy deps, so it runs in the L1 tests.

It only flags PHYSICALLY IMPOSSIBLE magnitudes, so a genuine outlier — which comes from a
deterministic tool (get_ratio on real XBRL), not hand arithmetic — is never rejected. It catches
the mis-composed multi-step numbers seen on FinanceBench (DPO 1419 days, cash-conversion-cycle
4760 days) and dollar figures asserted with no data tool called at all (fabricated from memory).
"""
import math
import re
from typing import Dict, List

_IMPLAUSIBLE = [
    (re.compile(r"(-?\d[\d,]*\.?\d*)\s*days\b", re.I), 1000),   # days-outstanding ratios are bounded
    (re.compile(r"(-?\d[\d,]*\.?\d*)\s*x\b"), 100),             # turnover ratios are bounded
]
_DATA_TOOLS = {"get_financials", "get_ratio", "get_growth", "compute_formula",
               "get_statement", "largest_line_item",
               "get_my_financials", "get_my_ratio", "get_my_growth"}

_SAFE = ("I can't give a reliable figure for this — it needs a computation I don't have a "
         "deterministic tool for, and I won't report a hand-derived number that may be wrong. "
         "Ask for the underlying figures (I can give those exactly), or rephrase.")

# A $ amount, with an optional scale word (so "$550 million" == "$550,000,000"). The \b keeps a
# stray trailing letter ("$5 the") from being read as a unit.
_MONEY_RX = re.compile(
    r"\$\s?([\d,]+(?:\.\d+)?)\s*(trillion|billion|million|thousand|tn|bn|mn|[kmbt])?\b", re.I)
_MULT = {"trillion": 1e12, "tn": 1e12, "t": 1e12, "billion": 1e9, "bn": 1e9, "b": 1e9,
         "million": 1e6, "mn": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}
# A cited figure must land within 2% of a source figure. Unit rendering is exact (0%) and light
# rounding stays inside; a dropped/added zero is a ~90% miss, so it's caught.
_MONEY_TOL = 0.02


def _parse_money(text) -> List[float]:
    """Every $ amount in `text`, unit-normalized to a plain number ($1.5 billion -> 1.5e9)."""
    out = []
    for num, unit in _MONEY_RX.findall(text or ""):
        try:
            out.append(float(num.replace(",", "")) * _MULT.get(unit.lower(), 1.0))
        except ValueError:
            pass
    return out


def _fetched_figures(trace) -> List[float]:
    """Every number a fetch tool actually produced — the only legitimate operands for a later
    `compute` call (compute takes raw numbers the model TYPES, so its inputs need vouching for)."""
    figs = []
    for t in trace or []:
        if t.get("tool") in _DATA_TOOLS and t.get("output"):
            for tok in re.findall(r"-?\d[\d,]*\.?\d*", t["output"]):
                try:
                    figs.append(abs(float(tok.replace(",", ""))))
                except ValueError:
                    pass
    return figs


def _scale_exp(x: float, figs: List[float]):
    """The power-of-ten k with |x| == fig * 10**k for some fetched fig (prefer the smallest |k|);
    None if x traces to no fetched figure. Lets a consistent unit change (dollars vs millions) pass
    while still exposing a dropped/added zero — see _hand_typed_operand."""
    x = abs(x)
    if x == 0:
        return 0 if any(f == 0 for f in figs) else None
    best = None
    for f in figs:
        if f == 0:
            continue
        k = round(math.log10(x / f))
        if abs(x / f - 10.0 ** k) <= 1e-6 * 10.0 ** k and (best is None or abs(k) < abs(best)):
            best = k
    return best


def _hand_typed_operand(trace) -> bool:
    """True if a `compute` call has an operand that doesn't trace to a fetched figure, or the two
    operands are rescaled INCONSISTENTLY. compute launders a mistyped operand into a fresh result
    that looks tool-produced (grounding can't catch it), so we verify provenance here. A dropped
    zero is exactly a ×10 factor — indistinguishable from a unit change by one operand alone — so
    we require both operands to share the same power-of-ten scale (both raw, or both millions)."""
    figs = _fetched_figures(trace)
    for t in trace or []:
        if t.get("tool") != "compute":
            continue
        exps = []
        for key in ("a", "b"):
            v = (t.get("args") or {}).get(key)
            try:
                v = float(v)
            except (ValueError, TypeError):
                continue
            k = _scale_exp(v, figs)
            if k is None:
                return True                # operand traces to no fetched figure -> typed by hand
            exps.append(k)
        if len(set(exps)) > 1:             # a and b scaled differently -> a dropped/added zero
            return True
    return False


def guardrail(answer: str, tools_used: List[str], trace: List[Dict] = None) -> str:
    """Return the answer, or a safe abstention if it reports an untrustworthy number:
    (a) a physically-impossible magnitude (the model mis-composed a formula by hand),
    (b) a dollar figure asserted with NO data tool called at all (fabricated from memory), or
    (c) a `compute` whose operands don't trace to a fetched figure (a hand-typed number laundered
        into a fresh result). Percentages are never thresholded (growth can exceed 100%)."""
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
        # No numeric tool ran, but a $ amount may still be legitimately quoted from filing prose —
        # an 8-K debt/buyback figure, say, that XBRL doesn't carry. Allow it only if it traces (unit-
        # normalized, within tolerance) to a figure in a retrieved passage; otherwise it's memory.
        prose = [v for t in (trace or [])
                 if t.get("tool") == "search_filings" and t.get("output")
                 for v in _parse_money(t["output"])]
        for a in _parse_money(answer):
            if not any(abs(a - s) <= _MONEY_TOL * max(a, s, 1.0) for s in prose):
                return _SAFE
    if _hand_typed_operand(trace):
        return _SAFE
    return answer
