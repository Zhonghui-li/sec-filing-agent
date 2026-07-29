"""Lazy, on-demand ingestion of a company's filing narrative into the pgvector RAG corpus.

Mirrors the numeric side (XBRL fetched live for ANY company): when search_filings hits a
company whose filings aren't indexed yet, we fetch its latest 10-K sections, recent 10-Q MD&A,
and recent (bounded) 8-K events, chunk, embed, and cache them in filing_chunks — so narrative
retrieval is "any company", not a fixed 7-company corpus. The 8-K path is what lets the agent
answer event questions (a debt issuance, a buyback authorization) whose figures XBRL doesn't
carry. The first query for a new company pays the ingest, subsequent queries hit the cache.
"""
import os

import psycopg
from edgar import set_identity, Company
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agents.companyfacts import cik_for

set_identity("Zhonghui Li lizhonghui923@gmail.com")  # SEC requires a contact

_EMB_MODEL = os.environ.get("EMB_MODEL", "text-embedding-3-small")
# Bounded LRU cache: this table is a lazy, on-demand cache that otherwise grows without limit (each
# (company, year) narrative adds ~100-700 vector rows and is never evicted), which filled the 512 MB
# store once year-aware retrieval put multiple years per company in it. Cap the row count and drop the
# least-recently-used companies when over it. ~20k rows ≈ 320 MB, well under the free-tier limit.
_MAX_CHUNKS = int(os.environ.get("FILING_CHUNKS_MAX", "20000"))
# Freshness TTL: an entry older than this is pruned so a re-query re-fetches the newest filing (a new
# 10-Q/10-K may have been filed). Year-pinned filings are immutable, so pruning just harmlessly
# re-fetches the same content on next use.
_TTL_DAYS = int(os.environ.get("FILING_TTL_DAYS", "30"))
_SECTIONS = {"business": "business", "risk_factors": "risk_factors",
             "mda": "management_discussion"}

_ENSURE = """
create extension if not exists vector;
create table if not exists filing_chunks (
    id serial primary key,
    ticker text, fiscal_year int, section text, accession text,
    chunk_text text, embedding vector(1536)
);
alter table filing_chunks add column if not exists last_accessed timestamptz not null default now();
alter table filing_chunks add column if not exists ingested_at timestamptz not null default now();
alter table filing_chunks add column if not exists pinned boolean not null default false;
"""


def _prune_stale(cur, ttl_days):
    """Freshness: drop perishable latest-snapshot entries older than the TTL, so the next query
    re-fetches the newest filing. Year-pinned filings are immutable → exempt (kept until LRU-evicted)."""
    cur.execute("delete from filing_chunks where not pinned and ingested_at < now() - make_interval(days => %s)",
                (ttl_days,))


def _evict_lru(cur, cap, protect=None):
    """If the cache exceeds `cap` chunks, drop least-recently-used FILINGS (one accession at a time,
    oldest access first) until back under cap. Per-accession eviction keeps every cached filing whole
    and retains a company's other filings; an evicted filing re-ingests on its next query. `protect`
    (the just-ingested ticker) is spared, so a single company larger than the cap is kept, not deleted
    right after ingesting it."""
    cur.execute("select count(*) from filing_chunks")
    n = cur.fetchone()[0]
    while n > cap:
        cur.execute("select accession, count(*) from filing_chunks where ticker <> %s group by accession "
                    "order by max(last_accessed) asc, count(*) desc limit 1", (protect or "",))
        row = cur.fetchone()
        if not row:
            break                          # only the protected ticker remains; keep it even if over cap
        accn, vc = row
        cur.execute("delete from filing_chunks where accession=%s", (accn,))
        n -= vc


def _vec(v):
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def _company(ticker):
    """Resolve a ticker OR company name (incl. former names of delisted/renamed issuers) to an
    edgartools Company, via the same CIK resolution the numeric side uses. Returns None on an
    unknown/ambiguous name, so the caller skips ingestion (agent abstains) instead of edgartools
    raising CompanyNotFoundError on a raw name the model passed."""
    cik = cik_for(ticker)
    return Company(int(cik)) if cik else None


def _fetch_sections(ticker, years=1, fiscal_year=None):
    """10-K narrative sections (one 10-K per FY, amendments excluded, newest accession wins).
    With `fiscal_year`, the 10-K FOR that fiscal year (year-aware — so a historical narrative question
    retrieves the right year's filing, not just the latest; ±1 tolerates non-December fiscal-year-ends,
    and rows are tagged with the REQUESTED year so retrieval filters to it exactly). Otherwise the
    latest `years` fiscal years. Returns [(fy, section, accession, text), ...]."""
    by_year = {}
    co = _company(ticker)
    if co is None:
        return []
    for f in co.get_filings(form="10-K"):
        if f.form != "10-K":
            continue
        yr = str(getattr(f, "period_of_report", "") or "")[:4]
        if yr.isdigit() and (yr not in by_year or f.accession_no > by_year[yr].accession_no):
            by_year[yr] = f
    if fiscal_year is not None:                       # year-aware: the 10-K for this fiscal year
        fy = int(fiscal_year)
        pick = next((by_year[y] for y in (str(fy), str(fy + 1), str(fy - 1)) if y in by_year), None)
        chosen = [(fy, pick)] if pick else []         # tag with REQUESTED year -> exact retrieval filter
    else:
        chosen = [(None, by_year[y]) for y in sorted(by_year, reverse=True)[:years]]
    out = []
    for tag_fy, f in chosen:
        period = str(getattr(f, "period_of_report", "") or "")
        fy = tag_fy if tag_fy is not None else (int(period[:4]) if period[:4].isdigit()
                                                else int(str(f.filing_date)[:4]) - 1)
        tenk = f.obj()
        for name, attr in _SECTIONS.items():
            text = getattr(tenk, attr, None)
            if text and len(str(text)) >= 200:
                out.append((fy, name, f.accession_no, str(text)))
    return out


def _tenq_mda(tenq):
    """MD&A text (Item 2) from a TenQ. The item key varies by filer/edgartools version ('Part I,
    Item 2' vs 'Item 2') and TenQ.items is unreliable, so we try candidate keys and confirm the text
    is actually MD&A (rather than silently returning the wrong item)."""
    for k in ("Part I, Item 2", "Item 2"):
        try:
            t = tenq[k]
        except Exception:
            continue
        if t and len(str(t)) >= 200 and "discussion and analysis" in str(t)[:3000].lower():
            return str(t)
    return None


def _fetch_10q_mda(ticker, quarters, target_year=None):
    """Latest `quarters` 10-Q MD&A sections (the quarterly narrative). Returns
    [(fy, section, accession, text)].

    With `target_year`, fetch that fiscal year's quarters instead of the most recent — the year-aware
    path (mirrors 10-K/8-K), so a historical-quarter question can reach that year's 10-Q narrative."""
    out = []
    co = _company(ticker)
    if co is None:
        return out
    for f in co.get_filings(form="10-Q"):
        if f.form != "10-Q":
            continue
        period = str(getattr(f, "period_of_report", "") or "")
        fy = int(period[:4]) if period[:4].isdigit() else int(str(f.filing_date)[:4])
        if target_year is not None:
            if fy > target_year:           # newer than target (newest-first) -> keep scanning
                continue
            if fy < target_year:           # passed the target year -> done
                break
        try:
            mda = _tenq_mda(f.obj())
        except Exception:
            continue
        if mda:
            out.append((fy, "mda_10q", f.accession_no, mda))
        if len(out) >= quarters:
            break
    return out


def _fetch_8k(ticker, months=18, cap=12, target_year=None):
    """Recent 8-K events, bounded to ~`months` back and `cap` filings. Emits ONE row per item (the
    concise event description), not per filing: an 8-K bundles a high-signal event item (e.g. Item
    8.01 "issued $550M of notes") with boilerplate (Item 9.01 exhibit lists) that, chunked together,
    dilutes the event's embedding and sinks its retrieval rank — per-item keeps each event's signal
    clean. Skips the large earnings exhibits that `.text()` would pull in. Filings come newest-first,
    so we stop once past the cutoff. Returns [(fy, section, accession, text), ...].

    With `target_year`, fetch that CALENDAR year's 8-Ks instead of the recent-`months` window — this is
    the year-aware path (mirrors 10-K), so a historical-event question ("the 8-K dated July 2022") can
    be answered rather than falling outside the recency window."""
    from datetime import date, timedelta
    cutoff = date.today() - timedelta(days=months * 30)
    out, n_filings = [], 0
    co = _company(ticker)
    if co is None:
        return out
    for f in co.get_filings(form="8-K"):
        if f.form != "8-K":
            continue
        try:
            fd = date.fromisoformat(str(f.filing_date)[:10])
        except Exception:
            continue
        if target_year is not None:
            if fd.year > target_year:      # newer than the target year -> keep scanning (newest-first)
                continue
            if fd.year < target_year:      # passed the target year -> done
                break
        elif fd < cutoff:
            break
        try:
            ek = f.obj()
            items = [str(ek[it]) for it in ek.items]
        except Exception:
            continue
        for text in items:
            if text and len(text) >= 100:
                out.append((fd.year, "8k", f.accession_no, text))
        n_filings += 1
        if n_filings >= cap:
            break
    return out


def ingest_ticker(ticker, years=1, fiscal_year=None):
    """Fetch, chunk, embed, and cache a company's filing narrative in filing_chunks. With
    `fiscal_year`, ingest just that fiscal year's 10-K (year-aware, keyed on (ticker, fiscal_year));
    otherwise the latest 10-K + recent 10-Q/8-K. Idempotent (skips if already indexed; an advisory
    lock serializes concurrent first-queries). Returns the number of chunks inserted (0 if already
    present, nothing found, or on failure — the caller falls back to abstain, never fabricates)."""
    tk = ticker.strip().upper()
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        return 0
    fy_req = int(fiscal_year) if fiscal_year is not None else None
    pinned = fy_req is not None                         # a year-pinned filing vs the perishable latest snapshot

    def _present(cur):
        # Idempotency key: a year-aware ingest is keyed on (ticker, fiscal_year); a no-year ingest on the
        # LATEST SNAPSHOT ONLY (`not pinned`) — otherwise a cached year-pinned entry would falsely satisfy
        # the no-year check and the latest 10-K/10-Q/8-K would never be fetched for that company.
        if fy_req is not None:
            cur.execute("select 1 from filing_chunks where ticker=%s and fiscal_year=%s limit 1", (tk, fy_req))
        else:
            cur.execute("select 1 from filing_chunks where ticker=%s and not pinned limit 1", (tk,))
        return cur.fetchone()

    try:
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(_ENSURE)
            if _present(cur):
                return 0                       # already indexed -> cache hit, nothing to do
        if fy_req is not None:
            sections = _fetch_sections(tk, fiscal_year=fy_req)   # that year's 10-K
            sections += _fetch_8k(tk, target_year=fy_req, cap=30)   # + that year's 8-K events (historical)
            sections += _fetch_10q_mda(tk, quarters=4, target_year=fy_req)  # + that year's 10-Q MD&A
        else:
            sections = _fetch_sections(tk, years)
            sections += _fetch_10q_mda(tk, quarters=4)   # last 4 quarters of 10-Q MD&A
            sections += _fetch_8k(tk)                     # recent bounded 8-K events
        if not sections:
            return 0
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
        chunks = []
        for (fy, sec, accn, text) in sections:
            # An 8-K is a short, self-contained event (a debt issuance, a buyback authorization);
            # keep each whole so the figure and its context stay in ONE retrievable chunk, instead
            # of being split across 1200-char windows that dilute and scatter the event.
            pieces = [text] if sec == "8k" else splitter.split_text(text)
            chunks += [(fy, sec, accn, p) for p in pieces]
        vectors = OpenAIEmbeddings(model=_EMB_MODEL).embed_documents([c[3] for c in chunks])
        lock_key = f"{tk}:{fy_req}" if fy_req is not None else tk
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("select pg_advisory_xact_lock(hashtext(%s))", (lock_key,))  # serialize per key
            if _present(cur):
                return 0                       # another request ingested it while we embedded
            for (fy, sec, accn, text), v in zip(chunks, vectors):
                cur.execute(
                    "insert into filing_chunks (ticker,fiscal_year,section,accession,"
                    "chunk_text,embedding,pinned) values (%s,%s,%s,%s,%s,%s::vector,%s)",
                    (tk, fy, sec, accn, text, _vec(v), pinned))
            conn.commit()
            _prune_stale(cur, _TTL_DAYS)               # freshness: drop entries past the TTL
            _evict_lru(cur, _MAX_CHUNKS, protect=tk)   # size: drop LRU filings if over cap
            conn.commit()
        return len(chunks)
    except Exception as e:
        print(f"[filings_ingest] {tk} ingestion failed: {type(e).__name__}: {e}")
        return 0


if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "PEP"
    print(f"ingesting {t} ... inserted {ingest_ticker(t)} chunks")
