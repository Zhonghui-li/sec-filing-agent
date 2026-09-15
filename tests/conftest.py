"""Keep the test suite out of the real metric miss queue.

`data/cache/metric_misses.jsonl` is production signal: it drives which metrics get added next, and
an `implausible_magnitude` record carrying `compute` is the trigger to re-enable the prose magnitude
block (agents/guardrail.py:_note_implausible_prose). A test run appending to it corrupts both — the
first version of these guardrail tests wrote three such records with fixture numbers, which
scripts/check_misses.py then reported as a live reversal trigger. Redirect the log for every test.
"""
import pytest

import agents.companyfacts as cf


@pytest.fixture(autouse=True)
def miss_log(tmp_path, monkeypatch):
    """Path the miss log is redirected to; read it to assert a miss was recorded."""
    path = tmp_path / "metric_misses.jsonl"
    monkeypatch.setattr(cf, "_MISS_LOG", path)
    return path
