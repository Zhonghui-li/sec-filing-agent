"""search_filings: RAG over 10-K narrative sections (pgvector).

Returns passages each tagged with its source (ticker · fiscal year · section · accession) so the
agent can CITE them. Dense pgvector (cosine) retrieval, with an optional ticker filter that keeps
a company's question on its own filings. Companies not yet indexed are fetched & cached live
(agents.filings_ingest), so narrative retrieval is "any company", not a fixed corpus.

Dense-only reranking was dropped (data-driven): a gold_evidence eval showed the ms-marco CrossEncoder
DEMOTED the right passages on dry 10-K narrative. But a rebuilt, trustworthy eval-gate (FinanceBench
10-K narrative, year-controlled, LLM-judged gold) later showed dense recall@10 is only ~45% — the
answer-bearing MD&A passage is often IN the corpus but ranked far down, because the question asks
about an OUTCOME (operating margin) while the evidence describes CAUSES (SG&A, restructuring). Of the
levers A/B'd against that gate, a reranker (+2) and contextual chunks (+1) were washes, but
**Multi-HyDE won decisively on RECALL: recall@10|covered 48%->70% (+22pp, gained 11 / regressed 1)** —
matching the literature's finding that HyDE is the one net-positive cheap lever here (arXiv 2602.17981).
BUT a year-controlled end-to-end (question->answer) eval showed that +22pp recall did NOT translate to
answer quality: answered-correct moved only 28%->31% (within noise) while WRONG rose 6->9, including
abstain->wrong flips — a finance-bar regression (fabricating over abstaining). Recall is necessary, not
sufficient; the bottleneck moved to synthesis + the ~30% coverage-miss (evidence in statements/tables).
So HyDE is wired but DEFAULT OFF (FILINGS_HYDE=1 to enable) — not shipped on a recall win that didn't
clear the end-to-end bar. (Caveat: that harness used a plain answerer without our guardrails, which may
understate the benefit / overstate the wrong-answer risk vs the real agent.)
"""
import os

import psycopg
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from agents.finance_tools import edgar_url, cik_for

_EMB_MODEL = os.environ.get("EMB_MODEL", "text-embedding-3-small")
_HYDE = os.environ.get("FILINGS_HYDE", "0") == "1"      # Multi-HyDE: +22pp recall but no end-to-end gain -> default OFF
_HYDE_MODEL = os.environ.get("HYDE_MODEL", "gpt-4o-mini")
_emb = None
_oai = None


def _emb_model():
    global _emb
    if _emb is None:
        _emb = OpenAIEmbeddings(model=_EMB_MODEL)
    return _emb


def _fmt(v):
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def _embed(q: str):
    return _fmt(_emb_model().embed_query(q))


def _hyde_embed(query: str):
    """Multi-HyDE: generate 3 hypothetical 10-K-style answer passages (in the filing's cause-vocabulary),
    mean-pool their embeddings with the query's, and retrieve with that vector. Bridges the outcome-word
    query vs cause-word evidence gap that sinks dense recall on MD&A. Any failure falls back to the plain
    query embedding, so retrieval never breaks."""
    import numpy as np
    e = _emb_model()
    qv = e.embed_query(query)
    try:
        global _oai
        if _oai is None:
            from openai import OpenAI
            _oai = OpenAI()
        p = ("Write a brief passage (2-3 sentences) that could appear in a US public company's SEC 10-K "
             "and would DIRECTLY answer the question below. Use the concrete financial line-items, drivers, "
             "and terminology a filing actually uses (e.g. cost of sales, SG&A, restructuring charges, "
             "segment results). Just write the passage; do not hedge or name a specific company.\n\n"
             "Question: " + query)
        r = _oai.chat.completions.create(model=_HYDE_MODEL, temperature=0.8, n=3, max_tokens=160,
                                         messages=[{"role": "user", "content": p}])
        hyps = [c.message.content for c in r.choices if c.message.content]
        if hyps:
            return _fmt(np.mean([qv] + e.embed_documents(hyps), axis=0))
    except Exception:
        pass
    return _fmt(qv)


def _dense(cur, qv, ticker, k, fiscal_year=None):
    where, tp = [], []
    if ticker:
        where.append("ticker=%s"); tp.append(ticker)
    if fiscal_year is not None:
        where.append("fiscal_year=%s"); tp.append(int(fiscal_year))
    wc = ("where " + " and ".join(where)) if where else ""
    cur.execute("select ticker,fiscal_year,section,accession,chunk_text "
                f"from filing_chunks {wc} order by embedding<=>%s::vector limit %s", tuple(tp) + (qv, k))
    return cur.fetchall()


def search_filings(query: str, ticker: str = None, k: int = 5, fiscal_year: int = None) -> str:
    """Search companies' 10-K narrative sections (business, risk factors, MD&A) for
    QUALITATIVE information — risks, strategy, management's discussion/explanations.
    Pass `ticker` (e.g. "AAPL") to restrict to one company. For a question about a SPECIFIC
    fiscal year (e.g. "what drove FY2022 margin change"), pass `fiscal_year` so it retrieves
    that year's 10-K, not just the latest. Returns passages, each tagged
    [ticker · FY · section · accession] so you can cite the source. This does NOT return exact
    financial figures — use get_financials for any number."""
    if not os.environ.get("DATABASE_URL"):        # no filings index available -> abstain, don't crash
        return (f"No indexed filing narrative available for '{query}'"
                + (f" ({ticker})." if ticker else ".") + " (narrative search is not configured.)")
    qv = _hyde_embed(query) if _HYDE else _embed(query)
    tk = ticker.strip().upper() if ticker else None
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        rows = _dense(cur, qv, tk, k, fiscal_year)

    if not rows and tk:                     # company/year not indexed yet -> fetch & cache it live,
        from agents.filings_ingest import ingest_ticker   # so narrative is "any company / any year"
        if ingest_ticker(tk, fiscal_year=fiscal_year):
            with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
                rows = _dense(cur, qv, tk, k, fiscal_year)

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
