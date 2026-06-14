"""Fetch 10-K narrative sections (Business / Risk Factors / MD&A) for the RAG corpus.

Text goes through retrieval (search_filings); exact numbers come from the financials
table (fetch_financials.py), NEVER from the LLM reading numbers out of this text.
We pull the latest 2 fiscal years per company so YoY comparison has both periods.

Usage:  python scripts/fetch_filings.py   ->   data/filings.jsonl
"""
import json
from pathlib import Path

from edgar import set_identity, Company

set_identity("Zhonghui Li lizhonghui923@gmail.com")  # SEC requires a contact

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "JPM", "TSLA", "KO"]
N_YEARS = 2
SECTIONS = {           # canonical name -> TenK attribute
    "business": "business",
    "risk_factors": "risk_factors",
    "mda": "management_discussion",
}
OUT = Path(__file__).resolve().parent.parent / "data" / "filings.jsonl"


def main():
    records = []
    for tk in TICKERS:
        # dedupe to one filing per FISCAL YEAR (not "latest N filings") — a company can
        # have an amendment or a duplicate 10-K for the same year that would otherwise
        # crowd out an earlier year (this is what dropped TSLA's prior year). Exclude
        # 10-K/A amendments; newest accession wins per year; take the latest N years.
        by_year = {}
        for f in Company(tk).get_filings(form="10-K"):
            if f.form != "10-K":
                continue
            yr = str(getattr(f, "period_of_report", "") or "")[:4]
            if yr.isdigit() and (yr not in by_year or f.accession_no > by_year[yr].accession_no):
                by_year[yr] = f
        filings = [by_year[y] for y in sorted(by_year, reverse=True)[:N_YEARS]]
        for f in filings:
            period = str(getattr(f, "period_of_report", "") or "")
            fy = int(period[:4]) if period[:4].isdigit() else int(str(f.filing_date)[:4]) - 1
            tenk = f.obj()
            kept = []
            for name, attr in SECTIONS.items():
                text = getattr(tenk, attr, None)
                if not text or len(str(text)) < 200:
                    continue
                records.append({
                    "ticker": tk, "fiscal_year": fy, "section": name,
                    "form": f.form, "filing_date": str(f.filing_date),
                    "accession": f.accession_no, "text": str(text),
                })
                kept.append(f"{name}({len(str(text))//1000}k)")
            print(f"{tk} FY{fy} [{f.accession_no}]: {', '.join(kept)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(records)} section records -> {OUT}")


if __name__ == "__main__":
    main()
