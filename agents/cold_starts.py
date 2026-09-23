"""Which retrievals in a turn had no index behind them.

A search_filings call that misses the index ingests the filing inline and queries again, so a cold
start returns the same passages a cache hit would and nothing downstream can tell them apart. That
matters for measurement: filing_chunks is a bounded LRU, so between two runs of the same eval a
company-year can be evicted, and the case that had to cold-start is the one whose retrieval,
latency and possibly answer differed for a reason unrelated to the code.

A ContextVar, not a module-level list: service/app.py declares `ask` as a sync def, so FastAPI runs
run_agent in a THREADPOOL and concurrent requests share the module. A shared list would file one
request's cold start in another request's audit trail.

Its own module, rather than living next to the retrieval that records it, so the L1 test lane can
import it: agents/filings_retrieval.py imports psycopg at module level, which CI's deterministic
job does not install, and a collection-time ImportError there fails the whole suite rather than
skipping one file. Nothing here needs a database.
"""
import contextvars

_cold_starts = contextvars.ContextVar("filings_cold_starts")


def reset_cold_starts():
    """Begin collecting cold starts for this turn (run_agent calls this per request)."""
    _cold_starts.set([])


def take_cold_starts():
    """The cold starts since reset_cold_starts(), clearing them. [] when nobody is collecting."""
    try:
        out = _cold_starts.get()
    except LookupError:
        return []
    _cold_starts.set([])
    return out


def note_cold_start(ticker, fiscal_year, chunks):
    """Record one. Called even when `chunks` is 0: a cold start that found nothing says the
    retrieval had no index behind it at all, which is the more informative case."""
    try:
        _cold_starts.get().append({"ticker": ticker, "fiscal_year": fiscal_year, "chunks": chunks})
    except LookupError:
        pass                                  # nobody is collecting — a direct tool call, or a test
