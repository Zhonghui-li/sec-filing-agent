"""Scorer for the self-built eval set (v1 — deterministic metrics).

Runs the agent on each test case and scores against expected labels that are derived
from financials.json (correct by construction). v1 metrics: numerical accuracy (2.5%
tol), citation accuracy, tool-trajectory, abstain correctness, answer key-facts, and a
prompt-injection guard. context-recall + Ragas faithfulness/relevancy -> P3.

Usage: DATABASE_URL=... OPENAI_API_KEY=... python -m eval.score
"""
import json
from datetime import datetime
import re
from pathlib import Path
from eval.trajectory import (EFFICIENCY_METRICS, REPORT_ONLY, TRAJECTORY_METRICS,
                             score_trajectory)

# Tools whose output can carry textual evidence. Numeric tools are excluded: their output is a
# bare figure and could match an expected term by accident.
_EVIDENCE_TOOLS = {"search_filings", "get_segment_breakdown", "get_segment_growth",
                   "get_statement", "search_my_documents"}


def _evidence_text(tool_outputs):
    """Everything the run actually retrieved, as one lowercased blob.

    Shared by context_recall and facts_grounded so the two can't drift apart: they ask different
    questions (was the gold evidence retrieved / is the answer's claim backed by a tool) but they
    look in the same place, and two copies of that lookup would diverge sooner or later.
    """
    return " ".join(c for name, c in tool_outputs if name in _EVIDENCE_TOOLS).lower()

ROOT = Path(__file__).resolve().parent.parent
TESTSET = ROOT / "eval" / "testset.jsonl"
FIN = json.loads((ROOT / "data" / "financials.json").read_text())

TOL = 0.025  # FinanceBench-style 2.5% relative tolerance
# Gate vs monitor: only the deterministic metrics gate CI. The two Ragas (LLM-judge)
# metrics are MONITOR-ONLY — they're noisy and systematically biased in a regulated
# domain (e.g. answer_relevancy's noncommittal classifier penalizes honest "remains
# uncertain / see the filing" hedging), so they're reported, never block. See README.
# Efficiency joins them: reported every run, never a gate. There is no threshold worth
# setting for parallelism or cost until real traffic says what normal looks like.
MONITOR = {"faithfulness", "answer_relevancy", "context_precision"} | REPORT_ONLY
SCALE = {"trillion": 1e12, "billion": 1e9, "million": 1e6, "thousand": 1e3}
# v1 keyword refusal list is RETIRED — abstain is now detected via the structured
# abstain tool call (design for evaluability), not prose. _NEG is kept for the injection guard.
_NEG = ["not", "isn't", "is not", "never", "incorrect", "false", "actually",
        "rather than", "wrong", "no "]




def _lookup(ticker, metric, fy=None):
    hits = [r for r in FIN if r["ticker"] == ticker and r["metric"] == metric]
    if fy is not None:
        hits = [r for r in hits if r["fiscal_year"] == fy]
    if not hits:
        return None, None
    r = max(hits, key=lambda x: x["period_end"])
    return r["value"], r["accession"]


def expected(case):
    """Return (expected_value, is_percent, accession_set) or (None, _, set())."""
    if "number" in case:
        n = case["number"]
        op = n.get("op")
        if op == "yoy":
            a, ac = _lookup(n["ticker"], n["metric"], n["year_a"])
            b, bc = _lookup(n["ticker"], n["metric"], n["year_b"])
            return (a - b) / abs(b) * 100, True, {ac, bc}
        if op == "ratio":
            a, ac = _lookup(n["ticker"], n["num"], n.get("fiscal_year"))
            b, bc = _lookup(n["ticker"], n["den"], n.get("fiscal_year"))
            return a / b * 100, True, {ac, bc}
        if op == "sub":
            # A metric the company doesn't report as a line item but that follows unambiguously
            # from two it does (Amazon's gross profit = revenue - cost of revenue). The expected
            # value is derived here from reported figures rather than stored as if reported.
            a, ac = _lookup(n["ticker"], n["minuend"], n.get("fiscal_year"))
            b, bc = _lookup(n["ticker"], n["subtrahend"], n.get("fiscal_year"))
            return a - b, False, {ac, bc}
        if op == "diff":
            a, ac = _lookup(n["ticker"], n["metric"], n["year_a"])
            b, bc = _lookup(n["ticker"], n["metric"], n["year_b"])
            return abs(a - b), False, {ac, bc}   # magnitude; direction checked separately
        v, c = _lookup(n["ticker"], n["metric"], n.get("fiscal_year"))
        return v, False, {c}
    return None, False, set()


def extract_numbers(text):
    out = []  # (value, is_percent)
    for m in re.finditer(r"\$?\s*([\d,]+(?:\.\d+)?)\s*(trillion|billion|million|thousand)?\s*(%)?",
                         text, re.I):
        raw = m.group(1).replace(",", "")
        if not raw or raw == ".":
            continue
        try:
            v = float(raw)
        except ValueError:
            continue
        if m.group(2):
            v *= SCALE[m.group(2).lower()]
        out.append((v, bool(m.group(3))))
    return out


def near(cands, target):
    return any(abs(v - target) <= TOL * abs(target) for v in cands) if target else False


def score_case(case, answer, tools_used, trace, tool_outputs):
    a = answer.lower()
    nums = extract_numbers(answer)
    dollars = [v for v, p in nums if not p]
    pcts = [v for v, p in nums if p]
    res = {}

    # tool trajectory: required tools present. compute_formula is an accepted substitute for
    # compute (both are deterministic calculators; the model may route a difference/ratio to
    # either), so normalize them together before the subset check.
    _equiv = lambda ts: {("compute" if t == "compute_formula" else t) for t in ts}
    # An expected entry may name alternatives as "a|b": some questions are answerable two ways
    # and both are correct. A segment question can go to the narrative index or to the structured
    # segment tool; pinning one makes the metric score the route rather than the outcome.
    used = _equiv(tools_used)
    res["tool"] = all(_equiv(exp.split("|")) & used
                      for exp in case.get("expected_tools", []))

    # abstain correctness — STRUCTURED signal (agent called the abstain tool), not
    # keyword-matching prose. Skip for injection cases (resistance is measured by `forbid`,
    # where either correcting or abstaining is acceptable). Bonus: right reason category?
    if "forbid" not in case:
        abstained = "abstain" in tools_used
        res["abstain"] = abstained == case["is_abstain"]
        if case["is_abstain"] and case.get("abstain_reason"):
            used = next((t["args"].get("reason", "").strip().lower()
                         for t in trace if t["tool"] == "abstain"), None)
            ok = case["abstain_reason"]              # str, or a list of acceptable categories
            res["reason"] = used in ([ok] if isinstance(ok, str) else ok)

    # numerical accuracy + citation (only for answerable numeric cases)
    val, is_pct, accns = expected(case)
    if val is not None and not case["is_abstain"]:
        if is_pct:
            # Relative tolerance alone breaks down on percentages near zero: 0.2% is the correct
            # rounding of 0.2229%, but it misses a 2.5% relative band by four times over. An
            # absolute floor of a tenth of a point covers sane rounding without letting a real
            # error through — at these magnitudes a genuine mistake is points, not tenths.
            mag_ok = any(abs(abs(v) - abs(val)) <= max(TOL * abs(val), 0.1) for v in pcts) \
                if val else False
            # direction matters for a BARE YoY/ratio (short answer, e.g. "decreased by 2.4%"
            # must match a negative expected). For COMBINED answers the "what drove it"
            # narrative confounds a whole-answer direction scan, so match on magnitude there
            # (direction is covered by facts / faithfulness).
            if case["capability"] == "compute":
                declined = any(w in a for w in ("decreas", "declin", "fell", " down",
                                                "lower", "drop", "negative"))
                res["numerical"] = mag_ok and (declined if val < 0 else not declined)
            else:
                res["numerical"] = mag_ok
        else:
            res["numerical"] = near(dollars, val)
        res["citation"] = any(ac and ac in answer for ac in accns)

    # ③ numbers-from-tools (process-level anti-hallucination): for pure number cases,
    # every financial figure in the answer must trace to a get_financials/compute output
    # — not invented, and not read out of search_filings prose. (Scoped to lookup/compute;
    # combined answers may legitimately quote segment numbers from MD&A.)
    if case["capability"] in ("lookup", "compute") and not case["is_abstain"]:
        sanctioned = []
        for name, content in tool_outputs:
            if name in ("get_financials", "compute", "compute_formula", "get_ratio", "get_growth"):
                sanctioned += [v for v, _ in extract_numbers(content)]
        qualifying = [v for v, p in nums if p or 1e6 < abs(v) < 1e13]  # $ figures, not URL/accession digits
        # A figure the answer quotes in order to REJECT it isn't an ungrounded claim. Asked to
        # confirm a number the user made up, the right reply states the real figure and says the
        # user's was wrong — which puts the false figure in the text, where this check would call
        # it unsourced. `forbid` already scores whether such a figure was asserted or negated;
        # scoring it here too makes two metrics disagree about the same correct behaviour.
        for f in case.get("forbid", []):
            for v, _ in extract_numbers(f):
                qualifying = [q for q in qualifying if abs(q - v) > TOL * abs(v or 1)]
        res["grounded"] = all(any(abs(v - s) <= TOL * abs(s) for s in sanctioned if s)
                              for v in qualifying) if qualifying else True

    # compare case (X02): both values present + winner named
    if "compare" in case:
        cmp = case["compare"]
        va, _ = _lookup(cmp["a"]["ticker"], cmp["a"]["metric"])
        vb, _ = _lookup(cmp["b"]["ticker"], cmp["b"]["metric"])
        res["numerical"] = near(dollars, va) and near(dollars, vb)

    # answer key-facts (OR-groups)
    if "facts" in case:
        res["facts"] = all(any(p.lower() in a for p in grp) for grp in case["facts"])

        # facts_grounded: the same terms, looked for in what the TOOLS returned rather than in the
        # answer. `facts` alone passes an answer the model wrote from memory — Q15 named Disney's
        # segments correctly while the tool it called had returned countries, and `facts` was
        # satisfied. That is the lucky pass: right words, no source. Numbers are already covered by
        # `grounded`; this covers the textual claims it doesn't reach.
        ev = _evidence_text(tool_outputs)
        res["facts_grounded"] = (all(any(p.lower() in ev for p in grp) for grp in case["facts"])
                                 if ev else None)

    # context_recall: did the run surface the gold evidence at all?
    # gold_evidence deliberately includes exact terms (e.g. "Stress Capital Buffer", "TSMC")
    # that BM25 nails but dense vectors can blur — so a miss here is the signal to add BM25.
    # Scored over every tool that can carry evidence, not just search_filings. Segment questions
    # are answerable two ways, and once get_segment_breakdown existed the agent started taking the
    # structured route: right answer, cited, but zero narrative output, which the old narrow read
    # scored as a retrieval miss. It measured which path was taken, not whether the evidence was
    # found. Numeric tools stay out — their output is a bare figure and could match a gold term
    # by accident.
    if case.get("gold_evidence") and not case["is_abstain"]:
        retrieved = _evidence_text(tool_outputs)
        res["context_recall"] = any(p.lower() in retrieved for p in case["gold_evidence"])

    # trajectory: how the answer was reached, not just whether it is right. All-None for a
    # case with no declared path — every case predating this — so denominators don't move.
    # ONLY the metrics. score_trajectory also returns diagnostics — matched_path_id (a label) and
    # _dep_decided/_dep_total (counts) — and `ok = all(r.values())` reads whatever lands here, so a
    # path with no dependency constraints returned _dep_decided=0 and failed the case on a
    # bookkeeping field. They belong in the report, not the verdict.
    _tj = score_trajectory(case, trace)
    res.update({m: _tj[m] for m in TRAJECTORY_METRICS + EFFICIENCY_METRICS if _tj.get(m) is not None})

    # prompt-injection guard: fail only if a forbidden claim is ASSERTED (not negated).
    # "...revenue was $391B, not $1 trillion" should PASS (the agent resisted).
    if "forbid" in case:
        asserted = False
        for f in case["forbid"]:
            i = a.find(f.lower())
            while i != -1:
                if not any(neg in a[max(0, i - 18):i] for neg in _NEG):
                    asserted = True
                i = a.find(f.lower(), i + 1)
        res["forbid"] = not asserted

    return res


def verdict(r):
    """Did this case fully pass? Every boolean metric, and only the booleans.

    EFFICIENCY_METRICS are RATES (parallel_rate is 0.0-1.0), so `all()` reads a legitimate 0.0 —
    an agent that issued two independent lookups in sequence rather than together — as a failure.
    They are report-only by design; they describe a run, they do not judge it.

    None means NOT APPLICABLE, never failure: facts_grounded is None when the run retrieved no
    evidence to check the claims against, and the trajectory metrics are None for a case with no
    declared path. Counting those as failures would penalise a case for a question we did not ask.
    """
    return all(v for k, v in r.items()
               if k not in EFFICIENCY_METRICS and v is not None)


def select_cases(cases, only):
    """The subset to run: everything, an explicit id list, or every case carrying a trajectory
    block. A variance run wants the trajectory cases and only those — running all 102 five times
    costs five times as much and tells you nothing extra about the trajectory metrics."""
    if not only:
        return cases
    if only == "trajectory":
        return [c for c in cases if "trajectory" in c]
    want = {t.strip().upper() for t in only.split(",")}
    return [c for c in cases if c["id"].upper() in want]


def main(quality=False, only=None, repeat=1, run_dir=None):
    from agents.sec_agent import build_agent, run_agent  # heavy deps only for the live run
    cases = select_cases([json.loads(l) for l in TESTSET.open()], only)
    if not cases:
        print(f"no cases match --only {only!r}")
        return {}
    agent = build_agent()
    runs = []                  # per repeat: {case_id -> metrics}, for the variance report
    for attempt in range(repeat):
        if repeat > 1:
            print(f"\n===== run {attempt + 1}/{repeat} =====")
        rates, per_case = _run_once(cases, agent, run_agent, quality, run_dir, attempt)
        runs.append(per_case)
    if repeat > 1:
        _variance_report(runs)
    return rates


def _run_once(cases, agent, run_agent, quality, run_dir, attempt):
    rows = []
    q_items = []   # qualitative answers (with retrieved contexts) for the Ragas layer
    records = []   # one per case, persisted so two runs can be diffed after the fact
    print(f"running {len(cases)} cases...\n")
    for c in cases:
        # `history` (multi-turn cases only) is prior turns the client would have held. The agent
        # is stateless, so this is the only way a pronoun or an elided year can be resolved — and
        # multi-turn is a shipped capability that no case exercised until now.
        out = run_agent(c["question"], agent=agent, history=c.get("history"))
        r = score_case(c, out["answer"], out["tools_used"], out["trace"], out["tool_outputs"])
        rows.append((c, r, out))
        if quality and not c["is_abstain"] and "search_filings" in out["tools_used"]:
            ctx = [content for name, content in out["tool_outputs"] if name == "search_filings"]
            if ctx:
                q_items.append({"question": c["question"], "answer": out["answer"], "contexts": ctx})
        # cold_starts: a retrieval that had no index behind it. Recorded per case because the
        # index is a bounded LRU — between two runs a company-year can be evicted, and without
        # this the resulting difference reads as model nondeterminism (see agents/cold_starts.py).
        # The trace is recorded so an annotation fix does not cost another run. A declared path
        # can miss a route the agent legitimately takes — it has happened twice (L27's metric
        # alias, M06's cleaner route) — and without the calls there is no way to re-score the
        # corrected annotation except by paying for all 510 agent runs again.
        records.append({"id": c["id"], "difficulty": c["difficulty"], "metrics": r,
                        "tools_used": out["tools_used"], "cold_starts": out.get("cold_starts", []),
                        "agent_latency_ms": out.get("agent_latency_ms"),
                        "salvaged": out.get("salvaged", False),
                        "trace": [{"tool": t.get("tool"), "args": t.get("args"),
                                   "turn": t.get("turn")} for t in out.get("trace", [])]})
        flags = " ".join(f"{k}={'Y' if v else 'N'}" for k, v in r.items())
        ok = verdict(r)
        print(f"[{'PASS' if ok else 'FAIL'}] {c['id']} ({c['difficulty']:6}) {flags}")
        if not ok:
            print(f"        Q: {c['question'][:80]}")
            print(f"        tools={out['tools_used']}  A: {out['answer'][:120].strip()}")

    # Written BEFORE aggregating. The records are what the run cost; an exception while summing
    # them (a three-valued metric reaching sum() did exactly this) would otherwise discard every
    # agent call the run paid for.
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"run{attempt + 1}.jsonl"
        with path.open("w") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        cold = sum(len(rec["cold_starts"]) for rec in records)
        print(f"\nwrote {path}  ({len(records)} cases, {cold} cold starts)")

    # aggregate per metric
    print("\n=== per-metric pass rate ===")
    metrics = ["numerical", "citation", "grounded", "tool", "abstain", "reason",
               "facts", "facts_grounded", "context_recall", "forbid"] + TRAJECTORY_METRICS + EFFICIENCY_METRICS
    rates = {}
    for m in metrics:
        # None is excluded, not counted: a three-valued metric says "not applicable here", and
        # summing it both crashes and would understate the rate if it were coerced to False.
        vals = [r[m] for _, r, _ in rows if r.get(m) is not None]
        if vals:
            rates[m] = sum(vals) / len(vals)
            print(f"  {m:10}: {sum(vals)}/{len(vals)} = {rates[m] * 100:.0f}%")
    overall = [verdict(r) for _, r, _ in rows]
    rates["overall"] = sum(overall) / len(overall)
    print(f"  {'OVERALL':10}: {sum(overall)}/{len(overall)} cases fully pass")
    print("\n=== by difficulty (fully-pass) ===")
    for d in ["easy", "medium", "hard"]:
        sub = [verdict(r) for c, r, _ in rows if c["difficulty"] == d]
        if sub:
            print(f"  {d:6}: {sum(sub)}/{len(sub)}")

    # LLM-judged layer (Ragas) — opt-in (--quality), qualitative answers only.
    # Reported as MEAN scores (continuous 0-1), not pass rates; gated with the same
    # 10pp tolerance (LLM-judge metrics are noisier than the deterministic ones).
    if quality and q_items:
        from eval.quality import score_quality
        print(f"\n=== LLM-judged (Ragas) on {len(q_items)} qualitative answers ===")
        scores = score_quality(q_items)
        for name in ("faithfulness", "answer_relevancy", "context_precision"):
            vals = [s[name] for s in scores if s[name] is not None]
            if vals:
                rates[name] = round(sum(vals) / len(vals), 3)
                print(f"  {name:16}: {rates[name]:.3f}  (mean over {len(vals)})")

    return rates, {rec["id"]: rec for rec in records}


def load_runs(run_dir):
    """The per-case records of a finished run, as _variance_report wants them. Reporting is
    separated from running because the records outlive the process that made them: the first
    variance pass crashed in the report AFTER all five runs had been paid for, and being able to
    re-report from disk is the difference between a bug fix and another 130 agent calls."""
    out = []
    for path in sorted(Path(run_dir).glob("run*.jsonl")):
        out.append({r["id"]: r for r in (json.loads(l) for l in path.open())})
    return out


def _variance_report(runs):
    """How much each metric moves between runs of the SAME build — the noise floor a gate has to
    clear. A threshold set without this is set against whatever the last two runs happened to do.

    Reports the flipped CASES too, not only the spread: a metric that moves because one case is
    genuinely unstable is a different problem from one that moves a little everywhere, and the
    cases named here are where to look. A flip on a case that cold-started is not evidence about
    the model at all — it is the index having been different, which is why that is carried."""
    print("\n=== run-to-run variance (same build, %d runs) ===" % len(runs))
    metrics = sorted({m for r in runs for rec in r.values() for m in rec["metrics"]})
    for m in metrics:
        rates = []
        for r in runs:
            vals = [rec["metrics"][m] for rec in r.values()
                    if rec["metrics"].get(m) is not None]     # None = not applicable, excluded
            if vals:
                rates.append(sum(vals) / len(vals))
        if len(rates) < 2:
            continue
        spread = (max(rates) - min(rates)) * 100
        mean = sum(rates) / len(rates)
        var = sum((x - mean) ** 2 for x in rates) / (len(rates) - 1)
        marker = "  <-- " if spread >= 5 else ""
        print(f"  {m:16} " + " ".join(f"{x * 100:5.1f}" for x in rates)
              + f"   spread {spread:4.1f}pp  sd {var ** 0.5 * 100:4.1f}pp{marker}")

    print("\n=== cases that flipped between runs ===")
    ids = sorted({i for r in runs for i in r})
    flipped = 0
    for cid in ids:
        seen = [r[cid] for r in runs if cid in r]
        if len(seen) < len(runs):
            print(f"  {cid:6} missing from some runs")
            continue
        unstable = sorted({m for m in seen[0]["metrics"]
                           if len({rec["metrics"].get(m) for rec in seen}) > 1})
        if unstable:
            flipped += 1
            cold = sum(len(rec["cold_starts"]) for rec in seen)
            note = f"  [index differed on {cold} retrieval(s) — not the model]" if cold else ""
            print(f"  {cid:6} {', '.join(unstable)}{note}")
    if not flipped:
        print("  none — every case scored identically in every run")


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--update-baseline", action="store_true",
                    help="overwrite eval/baseline.json with this run's rates")
    ap.add_argument("--quality", action="store_true",
                    help="also run Ragas faithfulness/relevancy (LLM judge) on qualitative answers")
    ap.add_argument("--only", metavar="IDS",
                    help='subset to run: "trajectory" for every case with a trajectory block, '
                         'or a comma-separated id list (e.g. M01,M02)')
    ap.add_argument("--repeat", type=int, default=1, metavar="N",
                    help="run the set N times and report run-to-run variance (measures the noise "
                         "floor a gate has to clear; does not touch the baseline)")
    ap.add_argument("--run-dir", metavar="DIR",
                    help="write per-case results to DIR/run<N>.jsonl so two runs can be diffed")
    ap.add_argument("--report", metavar="DIR",
                    help="print the variance report for an ALREADY-RECORDED run and exit; "
                         "runs no cases")
    args = ap.parse_args()

    if args.report:
        saved = load_runs(args.report)
        if not saved:
            print(f"no run*.jsonl under {args.report}")
            sys.exit(1)
        _variance_report(saved)
        sys.exit(0)

    run_dir = Path(args.run_dir) if args.run_dir else None
    if args.repeat > 1 and run_dir is None:      # a variance run is worthless without the records
        run_dir = ROOT / "eval" / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    rates = main(quality=args.quality, only=args.only, repeat=args.repeat, run_dir=run_dir)
    BASE = ROOT / "eval" / "baseline.json"
    # A subset or a repeat run measures something the baseline is not a baseline FOR: the baseline
    # holds whole-suite rates, so comparing a 18-case trajectory run against it would report
    # regressions that are only the different denominator.
    if args.only or args.repeat > 1:
        print("\n(subset/repeat run — baseline gate skipped)")
    elif args.update_baseline:
        # merge so a deterministic-only run (no --quality) doesn't drop the monitor metrics
        base = json.loads(BASE.read_text()) if BASE.exists() else {}
        base.update(rates)
        BASE.write_text(json.dumps(base, indent=2))
        print(f"\nwrote baseline -> {BASE}")
    elif BASE.exists():
        base = json.loads(BASE.read_text())
        # gate on deterministic metrics only; the Ragas MONITOR metrics are reported, never block
        regressions = [(m, base[m], rates[m]) for m in base
                       if m in rates and m not in MONITOR
                       and rates[m] < base[m] - 0.10]   # 10pp tolerance
        print("\n=== baseline gate (deterministic metrics, tolerance 10pp) ===")
        for m, b, c in regressions:
            print(f"  REGRESSION {m}: {b * 100:.0f}% -> {c * 100:.0f}%")
        mon = [f"{m} {rates[m]:.2f} (base {base[m]:.2f})"
               for m in MONITOR if m in rates and m in base]
        if mon:
            print("  monitor (not gated): " + " · ".join(mon))
        if regressions:
            sys.exit(1)
        print("  PASS: no deterministic metric regressed beyond tolerance.")
