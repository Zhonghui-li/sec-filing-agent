"""search_my_documents: RAG over a USER's uploaded private documents (pgvector `user_chunks`).

The private-data counterpart of search_filings. Same dense + reranker stack, but reads the
`user_chunks` table and filters by user_id, so each user only ever retrieves their own data.
Docling did the parsing offline (ingest/ingest.py, isolated venv); this side has no docling
dependency — the two are decoupled through pgvector, mirroring the public path.

Step-1 (minimal): cites filename (+ page when available). Step-2 upgrade: ground numbers in
extracted table cells with cell-level citations (the finance bar on private data).
"""
import os
import re

import psycopg
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from agents.finance_tools import RATIOS, _RATIO_ALIASES, _resolve_operand

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


def _doc_clause(doc_filter):
    """Build a SQL fragment + params to scope a query to one document. doc_filter is a doc_id
    (exact) or a filename substring; None = all of the user's documents (metadata filtering —
    the fix for the 'indistinguishable multi-documents problem')."""
    if not doc_filter:
        return "", []
    return " and (doc_id = %s or lower(filename) like %s)", [doc_filter, f"%{doc_filter.lower()}%"]


def search_my_documents(query: str, user_id: str, k: int = 5, doc_filter: str = None) -> str:
    """Search the USER'S OWN uploaded documents (private files they provided, e.g. an internal
    financial statement or memo) for relevant passages. Returns passages tagged
    [filename · page] so you can cite the source. Only this user's documents are searched.
    Use this when the question is about a file the user uploaded, not a public SEC filing."""
    qv = _embed(query)
    kk = max(k * 3, k)
    clause, dparams = _doc_clause(doc_filter)
    sql = ("select filename,page,chunk_text,embedding<=>%s::vector as d "
           "from user_chunks where user_id=%s" + clause + " order by d limit %s")
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute(sql, [qv, user_id, *dparams, kk])
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


def get_my_financials(metric: str, user_id: str, period: str = None, doc_filter: str = None) -> str:
    """Exact numbers from the TABLES in the user's uploaded documents (the private-data analogue
    of get_financials/XBRL). Matches the metric against table row labels; optional `period`
    (e.g. "FY2025") matches the column. Returns exact value + cell provenance, never prose.
    Returns the available metrics if no match, so the caller can retry."""
    like = f"%{metric.strip().lower()}%"
    clause, dparams = _doc_clause(doc_filter)
    sql = ("select filename,page,metric,period,value,raw,cell from user_facts "
           "where user_id=%s and lower(metric) like %s" + clause)
    params = [user_id, like, *dparams]
    if period:
        sql += " and lower(period) like %s"
        params.append(f"%{period.strip().lower()}%")
    sql += " order by filename, metric, period"
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


# --- Ratios & growth on uploaded documents ---------------------------------------------------
# The private-data analogues of get_ratio / get_growth. They reuse the SAME fixed formulas
# (finance_tools.RATIOS) and operand resolver; only the value getter changes — it reads the
# uploaded table cells (user_facts) instead of the public XBRL rows. So a ratio is still
# computed deterministically in code (the finance bar), with cell-level citations.

def _year_of(period):
    """First 4-digit year in a period label (e.g. 'FY2025' -> 2025), or None."""
    m = re.search(r"(?:19|20)\d{2}", str(period or ""))
    return int(m.group()) if m else None


def _cite(src):
    return (f"{src['filename']}" + (f" · p.{src['page']}" if src.get("page") is not None else "")
            + f" · {src['cell']}")


def _my_value(metric, user_id, doc_filter=None, year=None):
    """Private analogue of finance_tools._value: one base metric's numeric value from the user's
    uploaded tables (user_facts), matched as a row-label substring. Prefers an exact label match,
    then the shortest label (closest to the base metric, so 'revenue' isn't shadowed by 'cost of
    revenue'), then the latest fiscal year. Returns (float value, src dict) or (None, None)."""
    target = re.sub(r"\s+", " ", str(metric).replace("_", " ").strip().lower())
    clause, dparams = _doc_clause(doc_filter)
    sql = ("select value, filename, period, cell, page, lower(metric) from user_facts "
           "where user_id=%s and lower(metric) like %s" + clause)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute(sql, [user_id, f"%{target}%", *dparams])
        rows = cur.fetchall()
    cand = [{"value": float(v), "filename": fn, "period": per, "cell": c, "page": pg,
             "label": re.sub(r"\s+", " ", lbl.strip()), "fiscal_year": _year_of(per)}
            for (v, fn, per, c, pg, lbl) in rows]
    if year is not None:
        cand = [r for r in cand if r["fiscal_year"] == int(year)]
    if not cand:
        return None, None
    exact = [r for r in cand if r["label"] == target]
    pool = exact or cand
    r = min(pool, key=lambda x: (len(x["label"]), -(x["fiscal_year"] or -1)))
    return r["value"], r


def get_my_ratio(ratio: str, user_id: str, period: str = None, doc_filter: str = None) -> str:
    """Compute a standard financial RATIO from the user's UPLOADED documents deterministically —
    the private-data analogue of get_ratio, using the SAME fixed formulas. Supports the same set
    (gross_margin, operating_margin, net_margin, cogs_pct, roa, roe, current_ratio, quick_ratio,
    payout_ratio, debt_to_equity). It fetches the formula's base figures from the document's table
    cells and divides in code, returning the value, the formula, and cell-level citations. Use
    this for ANY ratio about an uploaded file instead of fetching pieces and dividing yourself."""
    name = _RATIO_ALIASES.get(ratio.strip().lower(), ratio.strip().lower().replace(" ", "_"))
    if name not in RATIOS:
        return f"Unknown ratio '{ratio}'. Supported: {', '.join(sorted(RATIOS))}."
    (num_spec, _op, den_spec), kind, definition = RATIOS[name]
    getval = lambda m, y: _my_value(m, user_id, doc_filter, y)
    year = _year_of(period)
    if year is None:   # pin both operands to the numerator's latest available fiscal year
        base = num_spec.replace("avg:", "").split("-")[0]
        _, nsrc = getval(base, None)
        year = nsrc["fiscal_year"] if nsrc else None
    num, src = _resolve_operand(num_spec, year, getval)
    den, _ = _resolve_operand(den_spec, year, getval)
    if num is None or den is None:
        missing = num_spec if num is None else den_spec
        return (f"Cannot compute {name} from your uploaded documents: a required figure "
                f"('{missing}') isn't in the tables. Abstain rather than guess.")
    if den == 0:
        return f"Cannot compute {name}: denominator is 0."
    raw = num / den
    out = f"{raw * 100:.1f}%" if kind == "pct" else f"{raw:.2f}"
    fy = f" for FY{src['fiscal_year']}" if src and src.get("fiscal_year") else ""
    return f"{name}{fy} = {out} ({definition}). [source: {_cite(src)}]"


def get_my_growth(metric: str, user_id: str, period: str = None, doc_filter: str = None) -> str:
    """Year-over-year change of a metric from the user's UPLOADED documents, deterministic — the
    private-data analogue of get_growth. Fetches the given/latest fiscal year AND the immediately
    preceding year from the document's tables, so the two years compared are ALWAYS consecutive;
    abstains if the prior year isn't in the document rather than using a non-adjacent one."""
    year = _year_of(period)
    cur, rc = _my_value(metric, user_id, doc_filter, year)
    if cur is None:
        return get_my_financials(metric, user_id, period=period, doc_filter=doc_filter)
    cy = rc["fiscal_year"]
    if cy is None:
        return f"Cannot compute YoY for '{metric}': the document's period label has no year. Abstain."
    prev, rp = _my_value(metric, user_id, doc_filter, cy - 1)
    if prev is None:
        return (f"Cannot compute YoY for '{metric}' FY{cy}: the prior year (FY{cy - 1}) isn't in "
                f"your uploaded documents, so there's no adjacent year to compare. Abstain.")
    if prev == 0:
        return f"Cannot compute YoY for '{metric}': prior-year value is 0."
    pct = (cur - prev) / abs(prev) * 100
    return (f"{metric} grew {pct:+.1f}% year over year, from {prev:g} (FY{cy - 1}) to {cur:g} "
            f"(FY{cy}). [sources: {_cite(rp)}; {_cite(rc)}]")
