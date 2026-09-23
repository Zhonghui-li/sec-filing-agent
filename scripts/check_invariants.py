#!/usr/bin/env python3
"""Check the extracted rows against properties that must hold whatever the filings say.

An eval set can only ask about the answers someone thought to write down. These are the other
half: rules the data itself has to obey, checkable on every covered company for the cost of the
SEC fetch and no LLM call at all. Both fiscal-year bugs fixed in this file were found here rather
than by a failing eval case — the eval set had no Walmart-2013 question, and would not have had
one, because nobody writes a question about a year they don't know went missing.

    python scripts/check_invariants.py                  # the eval-set companies
    python scripts/check_invariants.py WMT KR ULTA      # just these

Exit status is 1 when anything is flagged, so it can gate a build.
"""
import collections
import sys
from pathlib import Path

# allow `import agents...` when run as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.companyfacts import METRICS, company_rows

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "WMT", "TGT", "COST",
           "HD", "LOW", "KR", "ULTA", "NKE", "SBUX", "MCD", "AMD", "INTC", "CRM",
           "ORCL", "ADBE", "PEP", "KO", "XOM"]


def check(ticker):
    """[(rule, detail)] — empty when the company's rows are self-consistent."""
    # company_rows, not extract_rows: the scan has to see what the TOOLS see, including the
    # predecessor fallback for a ticker that has moved to a successor registrant. Checking the
    # extraction in isolation reported XOM as empty long after the tools could answer for it.
    try:
        rows = company_rows(ticker)
    except Exception as e:
        return [("fetch_failed", f"{type(e).__name__}: {e}")]
    if not rows:
        return [("no_rows", "covered company extracts nothing — a retired ticker or a tag change")]

    out = []
    # The company's own year end, from the duration rows (a full-year duration can only be the
    # fiscal year; an instant is exactly what we're trying to validate, so it can't be the ruler).
    ends = [r["period_end"] for r in rows if METRICS[r["metric"]][1] == "duration"]
    near = set()
    if ends:
        month = collections.Counter(e[5:7] for e in ends).most_common(1)[0][0]
        near = {f"{(int(month) + d - 1) % 12 + 1:02d}" for d in (-1, 0, 1)}

    by_fy = collections.defaultdict(set)
    for r in rows:
        # 1. An annual figure sits at the company's own year end. A mid-year balance sheet
        #    reported on a 10-K is indistinguishable from a year-end one by form or fp alone.
        if near and r["period_end"][5:7] not in near:
            out.append(("mid_year_end", f'{r["metric"]} FY{r["fiscal_year"]} {r["period_end"]}'))
        # 2. A fiscal-year label is within a year of the period it names.
        if r["fiscal_year"] - int(r["period_end"][:4]) not in (-1, 0):
            out.append(("label_far_from_end",
                        f'{r["metric"]} FY{r["fiscal_year"]} {r["period_end"]}'))
        by_fy[(r["metric"], r["fiscal_year"])].add(r["period_end"])

    # 3. One fiscal year, one period end. Two means a label was applied to a year it doesn't
    #    name — and somewhere else a real year is then missing.
    for (metric, fy), es in sorted(by_fy.items()):
        if len(es) > 1:
            out.append(("fy_multiple_ends", f"{metric} FY{fy} {sorted(es)}"))

    # 4. No gap in the years. Revenue is reported by every company every year, so a jump is a
    #    label collision upstream, not a company that didn't file.
    fys = sorted(fy for (metric, fy) in by_fy if metric == "revenue")
    for a, b in zip(fys, fys[1:]):
        if b - a != 1:
            out.append(("fy_gap", f"revenue {a} -> {b}"))
    return out


def main():
    tickers = sys.argv[1:] or TICKERS
    totals = collections.Counter()
    for ticker in tickers:
        for rule, detail in check(ticker):
            totals[rule] += 1
            print(f"{ticker:6} {rule:20} {detail}")
    if not totals:
        print(f"{len(tickers)} companies, no violations")
        return 0
    print("\n" + "\n".join(f"{rule:20} {n}" for rule, n in totals.most_common()))
    return 1


if __name__ == "__main__":
    sys.exit(main())
