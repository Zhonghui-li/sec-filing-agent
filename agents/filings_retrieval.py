"""search_filings: RAG over 10-K narrative sections (pgvector).

Returns passages each tagged with its source (ticker · fiscal year · section · accession) so the
agent can CITE them. Dense pgvector (cosine) retrieval, with an optional ticker filter that keeps
a company's question on its own filings. Companies not yet indexed are fetched & cached live
(agents.filings_ingest), so narrative retrieval is "any company", not a fixed corpus.

Dense-only, data-driven: a gold_evidence eval on lazily-ingested filings showed the CrossEncoder
reranker (ms-marco, trained for web QA) was DEMOTING the right passages on dry 10-K narrative —
dense ranked "Latin America" / "water" / "new stores" at positions 0-1-4, the reranker pushed them
to 4-7-9 (out of the top-k). So the reranker was dropped here (kept where it helps, e.g. course
codes in Slug Advisor). Same restraint as not adding an RL router — remove complexity the eval
shows hurts, not just avoid adding it.
"""
import os

import psycopg
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from agents.finance_tools import edgar_url, cik_for

_EMB_MODEL = os.environ.get("EMB_MODEL", "text-embedding-3-small")
_emb = None


def _embed(q: str):
    global _emb
    if _emb is None:
        _emb = OpenAIEmbeddings(model=_EMB_MODEL)
    return "[" + ",".join(f"{x:.8f}" for x in _emb.embed_query(q)) + "]"


def _dense(cur, qv, ticker, k):
    tflt, tp = ("where ticker=%s", (ticker,)) if ticker else ("", ())
    cur.execute("select ticker,fiscal_year,section,accession,chunk_text "
                f"from filing_chunks {tflt} order by embedding<=>%s::vector limit %s", tp + (qv, k))
    return cur.fetchall()


def search_filings(query: str, ticker: str = None, k: int = 5) -> str:
    """Search companies' 10-K narrative sections (business, risk factors, MD&A) for
    QUALITATIVE information — risks, strategy, management's discussion/explanations.
    Pass `ticker` (e.g. "AAPL") to restrict to one company. Returns passages, each
    tagged [ticker · FY · section · accession] so you can cite the source. This does
    NOT return exact financial figures — use get_financials for any number."""
    if not os.environ.get("DATABASE_URL"):        # no filings index available -> abstain, don't crash
        return (f"No indexed filing narrative available for '{query}'"
                + (f" ({ticker})." if ticker else ".") + " (narrative search is not configured.)")
    qv = _embed(query)
    tk = ticker.strip().upper() if ticker else None
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        rows = _dense(cur, qv, tk, k)

    if not rows and tk:                     # company not indexed yet -> fetch & cache it live,
        from agents.filings_ingest import ingest_ticker   # so narrative is "any company" too
        if ingest_ticker(tk):
            with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
                rows = _dense(cur, qv, tk, k)

    if not rows:
        return f"No filing passages found for '{query}'" + (f" ({ticker})." if ticker else ".")

    out = []
    for r in rows:
        ticker_, fy, section, accession, text = r
        cik = cik_for(ticker_)
        url = f" · {edgar_url(cik, accession)}" if cik else ""
        out.append(f"[{ticker_} · FY{fy} · {section} · {accession}{url}]\n{text}")
    return "\n\n".join(out)


if __name__ == "__main__":
    print(search_filings("supply chain and single-source component risk", ticker="AAPL", k=2))
    print("\n---\n")
    print(search_filings("regulatory capital and stress testing requirements", ticker="JPM", k=2))
