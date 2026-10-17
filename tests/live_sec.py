"""Skip, rather than fail, when SEC itself is the thing that is unavailable.

Four tests here resolve tickers against SEC's live endpoints, and that is deliberate: a mocked CIK
lookup cannot catch the ticker file changing shape, or a retired ticker quietly being reassigned,
which is exactly what they exist to catch.

The cost is that SEC answers 429 to a shared CI egress address that has already run ten times that
day. When it does, a red L1 reports SEC's rate limiter rather than anything about this repository,
and a build that goes red for reasons the committer cannot act on is a build people learn to
ignore.

`SourceUnavailable` is the type this codebase already uses to keep "could not fetch it" apart from
"it is not reported" -- the same separation belongs here. A transport failure skips; anything else,
including an assertion about what SEC returned, still fails.
"""
import functools
import urllib.error

import pytest

from agents.companyfacts import SourceUnavailable

_TRANSPORT = (SourceUnavailable, urllib.error.HTTPError, urllib.error.URLError, TimeoutError)


def live_sec(fn):
    """Mark a test as depending on SEC being reachable."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except _TRANSPORT as e:
            pytest.skip(f"SEC unreachable ({type(e).__name__}: {str(e)[:90]}) — "
                        f"not a failure of this repository")
    return wrapper
