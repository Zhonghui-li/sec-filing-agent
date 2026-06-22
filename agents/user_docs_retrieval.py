"""search_my_documents: RAG over a USER's uploaded private documents (pgvector `user_chunks`).

The private-data counterpart of search_filings. Same dense + reranker stack, but reads the
`user_chunks` table and filters by user_id, so each user only ever retrieves their own data.
Docling did the parsing offline (ingest/ingest.py, isolated venv); this side has no docling
dependency — the two are decoupled through pgvector, mirroring the public path.

Step-1 (minimal): cites filename (+ page when available). Step-2 upgrade: ground numbers in
extracted table cells with cell-level citations (the finance bar on private data).
"""
import os

import psycopg
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

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
            print(f"[search_my_documents] reranker disabled ({type(e).__name__}: {e})")
            _reranker = False
    return _reranker or None


def search_my_documents(query: str, user_id: str, k: int = 5) -> str:
    """Search the USER'S OWN uploaded documents (private files they provided, e.g. an internal
    financial statement or memo) for relevant passages. Returns passages tagged
    [filename · page] so you can cite the source. Only this user's documents are searched.
    Use this when the question is about a file the user uploaded, not a public SEC filing."""
    qv = _embed(query)
    kk = max(k * 3, k)
    sql = ("select filename,page,chunk_text,embedding<=>%s::vector as d "
           "from user_chunks where user_id=%s order by d limit %s")
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute(sql, (qv, user_id, kk))
        rows = cur.fetchall()

    docs = [Document(page_content=r[2], metadata={"filename": r[0], "page": r[1]}) for r in rows]
    rr = _get_reranker()
    docs = rr(query, docs)[:k] if (rr and docs) else docs[:k]

    if not docs:
        return ("No passages found in your uploaded documents for "
                f"'{query}'. (You may not have uploaded a relevant document yet.)")

    out = []
    for d in docs:
        m = d.metadata
        tag = m["filename"] + (f" · p.{m['page']}" if m.get("page") is not None else "")
        out.append(f"[{tag}]\n{d.page_content}")
    return "\n\n".join(out)


def get_my_financials(metric: str, user_id: str, period: str = None) -> str:
    """Exact numbers from the TABLES in the user's uploaded documents (the private-data analogue
    of get_financials/XBRL). Matches the metric against table row labels; optional `period`
    (e.g. "FY2025") matches the column. Returns exact value + cell provenance, never prose.
    Returns the available metrics if no match, so the caller can retry."""
    like = f"%{metric.strip().lower()}%"
    sql = ("select filename,page,metric,period,value,raw,cell from user_facts "
           "where user_id=%s and lower(metric) like %s")
    params = [user_id, like]
    if period:
        sql += " and lower(period) like %s"
        params.append(f"%{period.strip().lower()}%")
    sql += " order by metric, period"
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        if not rows:
            cur.execute("select distinct metric from user_facts where user_id=%s order by metric",
                        (user_id,))
            avail = [r[0] for r in cur.fetchall()]
            if not avail:
                return ("No table figures found in your uploaded documents "
                        f"(nothing matching '{metric}').")
            return (f"No figure matching '{metric}'"
                    f"{' for ' + period if period else ''} in your documents. "
                    f"Available metrics: {', '.join(avail)}.")
    out = []
    for fn, page, m, per, val, raw, cell in rows:
        loc = f"{fn}" + (f" · p.{page}" if page is not None else "") + f" · {cell}"
        out.append(f"{m} ({per}): {raw} (exact value {val:g}). [source: {loc}]")
    return "\n".join(out)
