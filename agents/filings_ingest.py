"""Lazy, on-demand ingestion of a company's 10-K narrative into the pgvector RAG corpus.

Mirrors the numeric side (XBRL fetched live for ANY company): when search_filings hits a
company whose filings aren't indexed yet, we fetch its latest 10-K sections, chunk, embed, and
cache them in filing_chunks — so narrative retrieval is "any company", not a fixed 7-company
corpus. Reuses the fetch (fetch_filings) and chunk/embed (build_filings_store) logic; the first
query for a new company pays the ingest, subsequent queries hit the cache.
"""
import os

import psycopg
from edgar import set_identity, Company
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

set_identity("Zhonghui Li lizhonghui923@gmail.com")  # SEC requires a contact

_EMB_MODEL = os.environ.get("EMB_MODEL", "text-embedding-3-small")
_SECTIONS = {"business": "business", "risk_factors": "risk_factors",
             "mda": "management_discussion"}

_ENSURE = """
create extension if not exists vector;
create table if not exists filing_chunks (
    id serial primary key,
    ticker text, fiscal_year int, section text, accession text,
    chunk_text text, embedding vector(1536)
);
"""


def _vec(v):
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def _fetch_sections(ticker, years):
    """Latest `years` fiscal years of 10-K narrative sections (one 10-K per FY, amendments
    excluded, newest accession wins). Returns [(fy, section, accession, text), ...]."""
    by_year = {}
    for f in Company(ticker).get_filings(form="10-K"):
        if f.form != "10-K":
            continue
        yr = str(getattr(f, "period_of_report", "") or "")[:4]
        if yr.isdigit() and (yr not in by_year or f.accession_no > by_year[yr].accession_no):
            by_year[yr] = f
    out = []
    for y in sorted(by_year, reverse=True)[:years]:
        f = by_year[y]
        period = str(getattr(f, "period_of_report", "") or "")
        fy = int(period[:4]) if period[:4].isdigit() else int(str(f.filing_date)[:4]) - 1
        tenk = f.obj()
        for name, attr in _SECTIONS.items():
            text = getattr(tenk, attr, None)
            if text and len(str(text)) >= 200:
                out.append((fy, name, f.accession_no, str(text)))
    return out


def ingest_ticker(ticker, years=1):
    """Fetch, chunk, embed, and cache a company's latest 10-K narrative in filing_chunks.
    Idempotent (skips if the ticker is already indexed; a per-ticker advisory lock serializes
    concurrent first-queries). Returns the number of chunks inserted (0 if already present,
    nothing found, or on failure — the caller falls back to abstain, never fabricates)."""
    tk = ticker.strip().upper()
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        return 0
    try:
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(_ENSURE)
            cur.execute("select 1 from filing_chunks where ticker=%s limit 1", (tk,))
            if cur.fetchone():
                return 0                       # already indexed -> cache hit, nothing to do
        sections = _fetch_sections(tk, years)
        if not sections:
            return 0
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
        chunks = [(fy, sec, accn, piece)
                  for (fy, sec, accn, text) in sections
                  for piece in splitter.split_text(text)]
        vectors = OpenAIEmbeddings(model=_EMB_MODEL).embed_documents([c[3] for c in chunks])
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("select pg_advisory_xact_lock(hashtext(%s))", (tk,))  # serialize per ticker
            cur.execute("select 1 from filing_chunks where ticker=%s limit 1", (tk,))
            if cur.fetchone():
                return 0                       # another request ingested it while we embedded
            for (fy, sec, accn, text), v in zip(chunks, vectors):
                cur.execute(
                    "insert into filing_chunks (ticker,fiscal_year,section,accession,"
                    "chunk_text,embedding) values (%s,%s,%s,%s,%s,%s::vector)",
                    (tk, fy, sec, accn, text, _vec(v)))
            conn.commit()
        return len(chunks)
    except Exception as e:
        print(f"[filings_ingest] {tk} ingestion failed: {type(e).__name__}: {e}")
        return 0


if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "PEP"
    print(f"ingesting {t} ... inserted {ingest_ticker(t)} chunks")
