"""Abuse / cost controls for the public demo (OpenAI + GCP are shared lab accounts,
so the app itself caps how many LLM calls the public can trigger).

Three layers, all tunable via env vars:
  - MAX_INPUT_CHARS    : reject over-long questions (token-inflation abuse)
  - RATE_LIMIT_PER_MIN : per-IP requests/minute
  - DAILY_QUOTA        : hard ceiling on requests/day -> bounds daily $ cost

Counters are in-memory (per instance). Deploy with --max-instances=1 so the in-memory
quota is an EXACT global cap (and it's cheaper).
"""
import os
import time
import threading
from collections import defaultdict, deque
from datetime import date

MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "500"))
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "6"))
DAILY_QUOTA = int(os.environ.get("DAILY_QUOTA", "150"))

_lock = threading.Lock()
_hits = defaultdict(deque)
_day = {"date": date.today(), "count": 0}


def check_rate_limit(ip: str) -> bool:
    now = time.time()
    with _lock:
        dq = _hits[ip]
        while dq and now - dq[0] > 60:
            dq.popleft()
        if len(dq) >= RATE_LIMIT_PER_MIN:
            return False
        dq.append(now)
        return True


def check_daily_quota() -> bool:
    with _lock:
        today = date.today()
        if _day["date"] != today:
            _day["date"], _day["count"] = today, 0
        if _day["count"] >= DAILY_QUOTA:
            return False
        _day["count"] += 1
        return True
