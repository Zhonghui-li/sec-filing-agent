"""L1 deterministic tests for the scorer's number logic — no LLM, no DB, no secrets."""
from eval.score import extract_numbers, near, expected


def test_extract_billion():
    nums = extract_numbers("revenue was $391.035 billion")
    assert any(abs(v - 391_035_000_000) < 1 and not p for v, p in nums)


def test_extract_percent():
    nums = extract_numbers("a year-over-year increase of 65.5%")
    assert (65.5, True) in [(round(v, 1), p) for v, p in nums]


def test_extract_plain_dollars():
    nums = extract_numbers("$1,234,567")
    assert any(abs(v - 1_234_567) < 1 for v, _ in nums)


def test_near_within_tolerance():
    assert near([391_035_000_000], 391_000_000_000)        # 0.009% off -> within 2.5%
    assert not near([100], 200)
    assert not near([], 100)


def test_expected_lookup():
    val, is_pct, accns = expected(
        {"number": {"ticker": "AAPL", "metric": "revenue", "fiscal_year": 2024}})
    assert abs(val - 391_035_000_000) < 1 and not is_pct and accns


def test_expected_yoy_is_percent():
    val, is_pct, _ = expected(
        {"number": {"op": "yoy", "ticker": "NVDA", "metric": "revenue",
                    "year_a": 2026, "year_b": 2025}})
    assert is_pct and val > 0   # NVDA FY2026 grew vs FY2025
