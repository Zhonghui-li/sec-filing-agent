"""The regression gate: McNemar over the cases two runs disagree on.

Replaces a percentage threshold that could not mean the same thing on two metrics — ten points is
ten cases on `tool` (106 scored) and less than one on `forbid` (7) — and that read three breaks
alongside three fixes as three breaks.
"""
from eval.gate import MIN_DISCORDANT, binom_tail, compare, mcnemar


def _pairs(broke, fixed, unchanged=20):
    return [(True, False)] * broke + [(False, True)] * fixed + [(True, True)] * unchanged


def test_five_breaks_and_no_fixes_is_the_smallest_detectable_regression():
    """P(5 of 5 disagreements going the wrong way) = 0.031, just inside alpha. This is what sets
    how many cases a metric needs: to catch a 20% regression you need 5 <= 0.2n, so n >= 25."""
    assert binom_tail(5, 5) == 0.03125
    assert mcnemar(_pairs(5, 0))["verdict"] == "REGRESSED"
    assert mcnemar(_pairs(4, 0))["verdict"] == "UNDECIDABLE"      # 4 disagreements is below the floor


def test_breaks_offset_by_fixes_are_not_a_regression():
    """Three cases breaking while three others get fixed is the build moving, not degrading — the
    failure the old count-the-failures threshold would have reported."""
    r = mcnemar(_pairs(3, 3))
    assert r["verdict"] == "OK" and r["regressed"] == 3 and r["fixed"] == 3


def test_a_stable_build_is_ok_not_undecidable():
    """Nothing broke, so the answer to "did this get worse" is no. Reporting UNDECIDABLE would
    show yellow forever on a metric that never moves."""
    assert mcnemar(_pairs(0, 0))["verdict"] == "OK"
    assert mcnemar(_pairs(0, 4))["verdict"] == "OK"               # only improvements


def test_too_few_disagreements_says_so_instead_of_passing():
    """`forbid` is scored over 7 cases, so it can rarely produce five disagreements and no test can
    speak for it. A green light there would be manufactured — the honest output is that the sample
    cannot answer."""
    r = mcnemar(_pairs(3, 0))
    assert r["verdict"] == "UNDECIDABLE" and r["discordant"] == 3 < MIN_DISCORDANT


def test_only_cases_both_runs_judged_are_paired():
    """None means the metric did not apply to that case, and a case in one run but not the other
    was added or deleted rather than changed. Neither is evidence about behaviour."""
    base = {"a": {"m": True}, "b": {"m": True}, "c": {"m": None}, "gone": {"m": True}}
    cur = {"a": {"m": False}, "b": {"m": True}, "c": {"m": False}, "new": {"m": False}}
    r = compare(base, cur)["m"]
    assert r["scored"] == 2                    # a and b; c is None in the baseline, gone/new unpaired
    assert r["regressed"] == 1 and r["fixed"] == 0
