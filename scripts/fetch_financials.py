"""Build data/financials.json — the offline snapshot of exact annual financials for the curated
companies, from SEC EDGAR XBRL (companyfacts API).

This is the ground-truth numeric layer: every figure the agent reports must come from here (a
tool), never from the LLM reading a number out of retrieved text. The extraction logic (candidate
tag mapping, duration/instant handling, restatement dedup) lives in agents/companyfacts.py and is
shared with the on-demand path that fetches ANY company live — so a dynamically fetched company
matches this curated set exactly.

Usage:  python scripts/fetch_financials.py   ->   data/financials.json
"""
import json
import sys
import time
from pathlib import Path

# allow `import agents...` when run as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.companyfacts import METRICS, ticker_to_cik_map, fetch_facts, extract_rows

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "JPM", "TSLA", "KO"]
OUT = Path(__file__).resolve().parent.parent / "data" / "financials.json"


def main():
    cik_map = ticker_to_cik_map()
    rows = []
    for tk in TICKERS:
        cik = cik_map[tk]
        gaap = fetch_facts(cik)
        tk_rows = extract_rows(gaap, tk, cik)
        rows += tk_rows
        found = {r["metric"] for r in tk_rows}
        missing = [m for m in METRICS if m not in found]
        print(f"{tk} (CIK {cik}): {len(found)}/{len(METRICS)} metrics"
              + (f"  MISSING: {missing}" if missing else ""))
        time.sleep(0.2)   # be polite to SEC

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {len(rows)} (ticker,metric,year) rows -> {OUT}")


if __name__ == "__main__":
    main()
