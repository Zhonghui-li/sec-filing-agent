"""L1 deterministic tests for the output guardrail — no LLM, no deps. Pins that it blocks a
fabricated figure and a ratio that left its natural bound IN A TOOL'S OWN OUTPUT, while never
rejecting a legitimate answer (incl. a real outlier that came from a tool). The scan of the
answer's PROSE for the same bound is detection-only — recorded, not blocked — so the impossible
numbers seen on FinanceBench (DPO 1419 days, CCC 4760 days) now pass through and log; see
_note_implausible_prose for why.
"""
from agents.guardrail import guardrail, _SAFE


def _blocked(ans, tools):
    return guardrail(ans, tools) == _SAFE


# --- detection-only: recorded, not blocked -------------------------------------------------
#
# These two are the cases the prose scan was built for — a ratio the model hand-composed through
# `compute`. That path is gone: get_ratio covers the standard ratios by name and compute_formula
# evaluates a formula in code, and across 150 FinanceBench questions plus the 50-item red-team
# suite `compute` never once assembled a days or turnover metric. Every firing observed was a
# false positive on prose, each costing a correct answer. The scan now logs instead of blocking
# (WAF monitor mode / Kubernetes audit / Gatekeeper dryrun), so the signal survives without the
# cost. If the miss log ever shows one of these alongside `compute` calls, the block goes back.
def test_an_implausible_dpo_is_recorded_not_blocked(monkeypatch):
    seen = []
    monkeypatch.setattr("agents.guardrail.log_miss",
                        lambda t, m, **kw: seen.append((m, kw.get("reason", ""))))
    ans = "Amazon's DPO for FY2017 is approximately 1419.68 days."
    assert guardrail(ans, ["get_financials", "compute"]) == ans       # the answer survives
    assert seen and seen[0][0] == "implausible_magnitude"             # the detection is kept
    assert "1419.68 days" in seen[0][1]


def test_an_implausible_ccc_is_recorded_not_blocked(monkeypatch):
    seen = []
    monkeypatch.setattr("agents.guardrail.log_miss",
                        lambda t, m, **kw: seen.append((m, kw.get("reason", ""))))
    ans = "The cash conversion cycle is approximately 4760.96 days."
    assert guardrail(ans, ["get_financials", "compute"]) == ans
    assert seen and seen[0][0] == "implausible_magnitude"


def test_a_year_before_the_metrics_name_is_not_a_days_value(monkeypatch):
    """The shape no lookbehind caught: "In 2024 days sales outstanding rose" — the year belongs to
    the sentence, the unit word to the metric. It is why patching the regex kept losing to English."""
    monkeypatch.setattr("agents.guardrail.log_miss", lambda *a, **k: None)
    trace = [{"tool": "get_financials", "output": "DSO for FY2024: 45.2"}]
    assert guardrail("In 2024 days sales outstanding rose to 45.2 days.",
                     ["get_financials"], trace) != _SAFE


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


# --- $ figures quoted from retrieved filing prose (8-K/10-Q events XBRL doesn't carry) ---
def _sf(out):
    return {"tool": "search_filings", "args": {}, "output": out}


_MCD_8K = _sf("On August 27, 2025, McDonald's issued $550,000,000 of its 4.400% Medium-Term "
              "Notes due 2031 and $750,000,000 of its 5.000% Medium-Term Notes due 2036.")


def test_pass_dollar_traces_to_prose():
    # figure quoted verbatim from the 8-K passage -> allowed even though no numeric tool ran
    ans = "MCD issued $550,000,000 of 4.400% notes due 2031 and $750,000,000 of 5.000% notes."
    assert guardrail(ans, ["search_filings"], [_MCD_8K]) != _SAFE


def test_pass_dollar_prose_unit_rendering():
    # "$550 million" == "$550,000,000" after unit normalization
    assert guardrail("MCD issued $550 million of notes.", ["search_filings"], [_MCD_8K]) != _SAFE


def test_block_dollar_not_in_prose():
    # a figure that appears in no retrieved passage -> fabricated from memory
    assert _blocked_t("MCD issued $800,000,000 of notes.", ["search_filings"], [_MCD_8K])


def test_block_dollar_prose_dropped_zero():
    # a dropped zero (55,000,000 vs 550,000,000) is ~90% off -> outside tolerance, caught
    assert _blocked_t("MCD issued $55,000,000 of notes.", ["search_filings"], [_MCD_8K])


def _blocked_t(ans, tools, trace):
    return guardrail(ans, tools, trace) == _SAFE


# --- citation restoration (HARD RULE 4 is prompt-only, so an injection can argue it down) ---
_NFLX_FCF = {"tool": "get_financials",
             "output": ("NFLX free cash flow for FY2023: $6,925,749,000. [source: 10-K accession "
                        "0001065280-24-000030, https://www.sec.gov/Archives/edgar/data/1065280/]")}


def test_restores_dropped_citation():
    """The R32 injection case: correct tool-fetched figure, citation dropped."""
    out = guardrail("Netflix's free cash flow for fiscal year 2023 was $6,925,749,000.",
                    ["get_financials"], [_NFLX_FCF])
    assert out != _SAFE                      # a right answer is not thrown away
    assert "0001065280-24-000030" in out     # its source is put back


def test_leaves_an_already_cited_answer_alone():
    ans = ("Netflix's FY2023 free cash flow was $6,925,749,000. "
           "[source: 10-K accession 0001065280-24-000030]")
    assert guardrail(ans, ["get_financials"], [_NFLX_FCF]) == ans


def test_does_not_cite_an_abstention():
    ans = "I can't give a reliable figure for that."
    assert guardrail(ans, ["get_financials", "abstain"], [_NFLX_FCF]) == ans


def test_no_accession_in_trace_leaves_answer_unchanged():
    ans = "Netflix's free cash flow for fiscal year 2023 was $6,925,749,000."
    trace = [{"tool": "get_financials", "output": "NFLX free cash flow FY2023: $6,925,749,000."}]
    assert guardrail(ans, ["get_financials"], trace) == ans


# --- unit-scale divisors: the conversion a question like "in USD millions" asks for ---
_MSFT_COGS = {"tool": "get_financials",
              "output": "MSFT cost_of_goods_sold for FY2016: $32,780,000,000. [source: 10-K ...]"}


def test_unit_conversion_divisor_is_allowed():
    """The Microsoft FY2016 case: fetch 32,780,000,000, divide by 1e6 for 'in USD millions'.
    Blocking it replaced a correct answer with the safe abstention."""
    trace = [_MSFT_COGS,
             {"tool": "compute", "args": {"op": "ratio", "a": 32780000000, "b": 1000000}}]
    assert guardrail("MSFT FY2016 COGS was $32,780 million.", ["get_financials", "compute"],
                     trace) != _SAFE


def test_billions_divisor_is_allowed():
    trace = [{"tool": "get_financials", "output": "AWK dividends_paid FY2020: $389,000,000."},
             {"tool": "compute", "args": {"op": "ratio", "a": 389000000, "b": 1000000000}}]
    assert guardrail("AWK paid $0.389 billion in dividends.", ["get_financials", "compute"],
                     trace) != _SAFE


def test_hand_typed_numerator_is_still_blocked():
    """Only the divisor is exempt — a made-up numerator is still a fabricated figure."""
    trace = [_MSFT_COGS,
             {"tool": "compute", "args": {"op": "ratio", "a": 55000000000, "b": 1000000}}]
    assert _blocked_t("A figure of $55,000 million.", ["get_financials", "compute"], trace)


def test_constant_in_a_non_ratio_op_is_still_blocked():
    """In a diff, 1,000,000 is a financial quantity, not a unit conversion."""
    trace = [_MSFT_COGS,
             {"tool": "compute", "args": {"op": "diff", "a": 32780000000, "b": 1000000}}]
    assert _blocked_t("It exceeded the threshold by $32,779 million.",
                      ["get_financials", "compute"], trace)


def test_a_non_scale_constant_divisor_is_still_blocked():
    """1e6 is a unit; 7,500,000 is someone's number."""
    trace = [_MSFT_COGS,
             {"tool": "compute", "args": {"op": "ratio", "a": 32780000000, "b": 7500000}}]
    assert _blocked_t("The ratio is 4370.7.", ["get_financials", "compute"], trace)


# --- a multiplication sign is not a turnover ratio ---
def test_restated_formula_is_not_a_turnover_ratio():
    """The Amazon DPO case: the reply restates "365 x average accounts payable" and the turnover
    guard read 365 as a turnover. Which sign the model happened to write decided whether the
    answer survived, so the same question answered or refused at random."""
    trace = [{"tool": "compute_formula", "output": "AMZN formula result for FY2017 = 93.86"}]
    for sign in ("x", "X", "*", "×"):
        ans = f"Amazon's FY2017 DPO, calculated as 365 {sign} average accounts payable / (COGS + change in inventory), is 93.86 days."
        assert guardrail(ans, ["compute_formula"], trace) != _SAFE, f"blocked on {sign!r}"


def test_a_real_turnover_ratio_is_recorded_not_blocked(monkeypatch):
    seen = []
    monkeypatch.setattr("agents.guardrail.log_miss",
                        lambda t, m, **kw: seen.append(m))
    ans = "Inventory turnover was 150x."
    assert guardrail(ans, ["get_ratio"]) == ans
    assert seen == ["implausible_magnitude"]


def test_a_fiscal_year_in_the_metric_name_is_not_a_days_value():
    """"days payable outstanding" is the metric's NAME, so an answer that names it after its
    fiscal year offered "2017 days" to the bound and lost a correct 108.43."""
    trace = [{"tool": "compute_formula", "output": "AMZN formula result for FY2017 = 108.43"}]
    ans = ("Amazon's FY2017 days payable outstanding (DPO), computed as 365 x average accounts "
           "payable over FY2016-FY2017, was 108.43 days.")
    assert guardrail("%s\n\nANSWER: 108.43" % ans, ["compute_formula"], trace) != _SAFE


# --- a get_ratio result is checked against the bound for what it IS -------------------------
def _ratio_trace(ratio, out):
    return [{"tool": "get_ratio", "args": {"ratio": ratio, "ticker": "X"}, "output": out}]


def test_a_days_ratio_outside_its_bound_is_blocked_from_the_tool_output():
    """No prose involved: the call says ratio="dpo", RATIOS says dpo is a days ratio, and the
    tool's own output carries the value."""
    assert _blocked_t("Amazon's DPO was 1419.68 days.", ["get_ratio"],
                      _ratio_trace("dpo", "AMZN dpo for FY2017 = 1419.68 days (365 x ...)"))


def test_a_turnover_outside_its_bound_is_blocked_from_the_tool_output():
    assert _blocked_t("Turnover 150x.", ["get_ratio"],
                      _ratio_trace("asset_turnover", "X asset_turnover for FY2024 = 150.0x (...)"))


def test_a_normal_ratio_passes():
    assert guardrail("AAPL DPO for FY2024 = 114.15 days.", ["get_ratio"],
                     _ratio_trace("dpo", "AAPL dpo for FY2024 = 114.15 days (...)")) != _SAFE


def test_a_percentage_ratio_has_no_bound():
    """A margin or a growth rate can legitimately exceed any of these numbers."""
    assert guardrail("Net margin 24.0%.", ["get_ratio"],
                     _ratio_trace("net_margin", "X net_margin for FY2024 = 240.0% (...)")) != _SAFE


def test_the_ratio_check_ignores_the_prose_entirely():
    """The answer restates its formula and names the metric after a fiscal year — the two shapes
    that produced false positives — while the tool output is in bounds."""
    ans = ("Amazon's FY2017 days payable outstanding, computed as 365 x average accounts "
           "payable, was 108.43 days.")
    assert guardrail(ans, ["get_ratio"],
                     _ratio_trace("dpo", "AMZN dpo for FY2017 = 108.43 days (...)")) != _SAFE
