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
