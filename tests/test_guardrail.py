"""L1 deterministic tests for the output guardrail — no LLM, no deps. Pins that it blocks the
hand-computed impossible numbers seen on FinanceBench (DPO 1419 days, CCC 4760 days) and fabricated
figures, while never rejecting a legitimate answer (incl. a real outlier that came from a tool).
"""
from agents.guardrail import guardrail, _SAFE


def _blocked(ans, tools):
    return guardrail(ans, tools) == _SAFE


# --- should block ---
def test_block_impossible_dpo():
    assert _blocked("Amazon's DPO for FY2017 is approximately 1419.68 days.",
                    ["get_financials", "compute"])


def test_block_impossible_ccc():
    assert _blocked("The cash conversion cycle is approximately 4760.96 days.",
                    ["get_financials", "compute"])


def test_block_dollar_figure_with_no_data_tool():
    assert _blocked("Revenue was $391 billion.", [])


# --- should NOT block ---
def test_pass_normal_dpo():
    assert not _blocked("AAPL DPO for FY2024 = 76.92 days.", ["get_ratio"])


def test_pass_real_outlier_from_tool():
    # a high but possible days value that came from the deterministic tool — must be trusted
    assert not _blocked("Distressed co DPO = 420.00 days.", ["get_ratio"])


def test_pass_growth_over_100pct():
    # percentages are never thresholded (growth can exceed 100%)
    assert not _blocked("NVDA revenue grew 700.5% year over year.", ["get_growth"])


def test_pass_turnover():
    assert not _blocked("Asset turnover = 1.09x.", ["get_ratio"])


def test_pass_dollar_with_data_tool():
    assert not _blocked("Apple revenue FY2024 was $391,035,000,000.", ["get_financials"])


def test_pass_abstention_untouched():
    ans = "I can't answer; that company isn't covered."
    assert guardrail(ans, ["abstain"]) == ans


# --- (c) compute operand provenance: a compute operand must trace to a fetched figure ---
def _gf(out):
    return {"tool": "get_financials", "args": {}, "output": out}


def _comp(a, b, op="ratio"):
    return {"tool": "compute", "args": {"op": op, "a": a, "b": b}, "output": ""}


_REV = _gf("AAPL revenue for FY2024: $391,035,000,000.")
_PPE = _gf("AAPL ppe_net for FY2024: $45,680,000,000.")


def test_pass_compute_operands_traceable():
    trace = [_REV, _PPE, _comp(391035000000, 45680000000)]
    assert guardrail("Fixed-asset turnover 8.56x.", ["get_financials", "compute"], trace) != _SAFE


def test_block_compute_operand_not_fetched():
    # a hand-typed operand that appears in no tool output -> laundered into a fresh ratio
    trace = [_REV, _PPE, _comp(500000000000, 45680000000)]
    assert guardrail("Turnover 10.9x.", ["get_financials", "compute"], trace) == _SAFE


def test_block_compute_dropped_zero():
    # a dropped a zero (39,103,500,000 vs 391,035,000,000) -> scaled inconsistently with b
    trace = [_REV, _PPE, _comp(39103500000, 45680000000)]
    assert guardrail("Turnover 0.86x.", ["get_financials", "compute"], trace) == _SAFE


def test_pass_compute_consistent_units():
    # both operands in millions (units cancel in the ratio) -> legitimate, not a dropped zero
    trace = [_REV, _PPE, _comp(391035, 45680)]
    assert guardrail("Turnover 8.56x.", ["get_financials", "compute"], trace) != _SAFE


def test_pass_compute_cross_company():
    # compute's real niche: two DIFFERENT tickers, both figures fetched
    trace = [_gf("AAPL revenue FY2024: $391,035,000,000."),
             _gf("MSFT revenue FY2024: $245,122,000,000."),
             _comp(391035000000, 245122000000, op="diff")]
    assert guardrail("Apple's revenue is $145,913,000,000 higher.",
                     ["get_financials", "compute"], trace) != _SAFE
