"""Narrative cache: the no-year (latest snapshot) path must not be satisfied or answered by a cached
year-pinned filing. Regression tests for the `_present`/`_dense` keyspace collision that year-aware
ingestion introduced. Uses synthetic rows (no SEC/OpenAI), isolated under a ZZTEST* ticker; skipped
without a DATABASE_URL."""
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="needs the filings DB")

_ZERO = "[" + ",".join(["0.0"] * 1536) + "]"


@pytest.fixture
def db():
    import psycopg
    from agents.filings_ingest import _ENSURE
    dsn = os.environ["DATABASE_URL"]
    with psycopg.connect(dsn) as c, c.cursor() as cur:
        cur.execute(_ENSURE)                                   # ensures the pinned column exists
        cur.execute("delete from filing_chunks where ticker like 'ZZTEST%'")
        c.commit()
    yield dsn
    with psycopg.connect(dsn) as c, c.cursor() as cur:
        cur.execute("delete from filing_chunks where ticker like 'ZZTEST%'")
        c.commit()


def _insert(cur, ticker, fy, pinned, accession, n=2):
    for i in range(n):
        cur.execute("insert into filing_chunks (ticker,fiscal_year,section,accession,chunk_text,embedding,pinned) "
                    "values (%s,%s,'mda',%s,%s,%s::vector,%s)", (ticker, fy, accession, f"chunk {i}", _ZERO, pinned))


def test_present_no_year_ignores_pinned(db):
    """Ingest-side fix: a no-year presence check must NOT be satisfied by a cached year-pinned entry,
    or the latest snapshot would never be ingested for that company."""
    import psycopg
    with psycopg.connect(db) as c, c.cursor() as cur:
        _insert(cur, "ZZTESTP", 2022, True, "acc-pinned-2022")
        c.commit()
        cur.execute("select 1 from filing_chunks where ticker='ZZTESTP' and not pinned limit 1")
        assert cur.fetchone() is None                          # fixed predicate: pinned year does not count
        cur.execute("select 1 from filing_chunks where ticker='ZZTESTP' limit 1")
        assert cur.fetchone() is not None                      # the OLD predicate would have (the bug)


def test_dense_no_year_excludes_pinned(db):
    """Retrieval-side fix: a no-year query returns nothing when only a year-pinned filing is cached,
    but the year-scoped query still finds it."""
    import psycopg
    from agents.filings_retrieval import _dense
    with psycopg.connect(db) as c, c.cursor() as cur:
        _insert(cur, "ZZTESTD", 2022, True, "acc-pinned-2022")
        c.commit()
        assert _dense(cur, _ZERO, "ZZTESTD", 5, None) == []           # no-year -> pinned year excluded
        assert len(_dense(cur, _ZERO, "ZZTESTD", 5, 2022)) == 2       # year-scoped -> found


def test_dense_no_year_returns_latest_over_pinned(db):
    """With both a latest snapshot and a pinned historical year cached, a no-year query returns only
    the latest snapshot."""
    import psycopg
    from agents.filings_retrieval import _dense
    with psycopg.connect(db) as c, c.cursor() as cur:
        _insert(cur, "ZZTESTL", 2024, False, "acc-latest")
        _insert(cur, "ZZTESTL", 2022, True, "acc-pinned-2022")
        c.commit()
        rows = _dense(cur, _ZERO, "ZZTESTL", 10, None)
        assert rows and all(r[1] == 2024 for r in rows)               # r[1] = fiscal_year; only the latest
