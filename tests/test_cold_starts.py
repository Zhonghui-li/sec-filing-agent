"""A retrieval that misses the index ingests inline and queries again, returning the same passages
a cache hit would — so the two are indistinguishable downstream. filing_chunks is a bounded LRU, so
between two runs of the same eval a company-year can be evicted, and the case that had to cold-start
is the one whose retrieval differed for a reason unrelated to the code. These pin the record that
makes that visible; the DB is never touched.
"""
import threading

from agents.cold_starts import note_cold_start, reset_cold_starts, take_cold_starts


def test_nothing_is_recorded_until_someone_is_collecting():
    """A direct tool call, or a test, runs with no turn around it. That must not raise, and must not
    leak a record into whatever collects next."""
    assert take_cold_starts() == []
    note_cold_start("AAPL", 2024, 0)          # no reset first
    assert take_cold_starts() == []


def test_take_reports_once_then_clears():
    reset_cold_starts()
    note_cold_start("XOM", 2019, 240)
    assert take_cold_starts() == [{"ticker": "XOM", "fiscal_year": 2019, "chunks": 240}]
    assert take_cold_starts() == []            # a second turn starts empty, not with turn one's


def test_a_failed_cold_start_is_recorded_too():
    """chunks=0 is the MORE informative case — the retrieval had no index behind it at all, rather
    than a stale one — so it must not be filtered out as 'nothing happened'."""
    reset_cold_starts()
    note_cold_start("ZZZZ", None, 0)
    assert take_cold_starts() == [{"ticker": "ZZZZ", "fiscal_year": None, "chunks": 0}]


def test_concurrent_turns_do_not_see_each_others_cold_starts():
    """service/app.py declares `ask` as a sync def, so FastAPI runs run_agent in a threadpool and
    concurrent requests share the module. A plain module-level list would file one request's cold
    start in another request's audit trail — which is why this is a ContextVar."""
    seen, barrier = {}, threading.Barrier(2)

    def turn(name, ticker):
        reset_cold_starts()
        barrier.wait()                         # both turns are open at once
        note_cold_start(ticker, 2024, 100)
        barrier.wait()                         # both have recorded before either collects
        seen[name] = take_cold_starts()

    threads = [threading.Thread(target=turn, args=("a", "AAPL")),
               threading.Thread(target=turn, args=("b", "MSFT"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [r["ticker"] for r in seen["a"]] == ["AAPL"]
    assert [r["ticker"] for r in seen["b"]] == ["MSFT"]
