"""search_filings: RAG over 10-K narrative sections (pgvector).

Returns passages each tagged with its source (ticker · fiscal year · section ·
accession) so the agent can CITE them. Dense pgvector (cosine) search + a CrossEncoder
reranker (reused from the Slug Advisor stack). An optional ticker filter keeps a
company's question on its own filings.

BM25/hybrid is intentionally omitted for now — filing queries are semantic, not
exact-code lookups, and the ticker filter matters more here. Add hybrid only if eval
shows exact-term recall is lacking (same data-driven restraint as not adding an RL router).
"""
import os

import psycopg
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from agents.finance_tools import edgar_url, cik_for

_EMB_MODEL = os.environ.get("EMB_MODEL", "text-embedding-3-small")
_RERANK = os.environ.get("RERANK", "1") == "1"
_emb = None
_reranker = None


def _embed(q: str):
    global _emb
    if _emb is None:
        _emb = OpenAIEmbeddings(model=_EMB_MODEL)
    return "[" + ",".join(f"{x:.8f}" for x in _emb.embed_query(q)) + "]"


def _get_reranker():
    global _reranker
    if _reranker is None and _RERANK:
        try:
            from agents.reranker import create_cross_encoder_reranker
            _reranker = create_cross_encoder_reranker()
        except Exception as e:
            print(f"[search_filings] reranker disabled ({type(e).__name__}: {e})")
            _reranker = False
    return _reranker or None


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
    kk = max(k * 3, k)
    if ticker:
        sql = ("select ticker,fiscal_year,section,accession,chunk_text,"
               "embedding<=>%s::vector as d from filing_chunks where ticker=%s "
               "order by d limit %s")
        params = (qv, ticker.strip().upper(), kk)
    else:
        sql = ("select ticker,fiscal_year,section,accession,chunk_text,"
               "embedding<=>%s::vector as d from filing_chunks order by d limit %s")
        params = (qv, kk)

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    if not rows and ticker:                 # company not indexed yet -> fetch & cache it live,
        from agents.filings_ingest import ingest_ticker   # so narrative is "any company" too
        if ingest_ticker(ticker):
            with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

    docs = [Document(page_content=r[4],
                     metadata={"ticker": r[0], "fiscal_year": r[1],
                               "section": r[2], "accession": r[3]})
            for r in rows]
    rr = _get_reranker()
    docs = rr(query, docs)[:k] if (rr and docs) else docs[:k]

    if not docs:
        return f"No filing passages found for '{query}'" + (f" ({ticker})." if ticker else ".")

    out = []
    for d in docs:
        m = d.metadata
        cik = cik_for(m["ticker"])
        url = f" · {edgar_url(cik, m['accession'])}" if cik else ""
        out.append(f"[{m['ticker']} · FY{m['fiscal_year']} · {m['section']} · "
                   f"{m['accession']}{url}]\n{d.page_content}")
    return "\n\n".join(out)


if __name__ == "__main__":
    print(search_filings("supply chain and single-source component risk", ticker="AAPL", k=2))
    print("\n---\n")
    print(search_filings("regulatory capital and stress testing requirements", ticker="JPM", k=2))
