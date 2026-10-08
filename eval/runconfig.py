"""What produced a set of eval results.

A run directory full of per-case rows does not say which model wrote them, at what reasoning
effort, against which index, or from which commit. That was fine while there was one model and
one person; it stopped being fine the week o4-mini was scheduled for shutdown and every number in
the repo became "produced by something that no longer exists". The 2026-10-07 freeze needed a
CONFIG.json and it was written by hand, which is the failure mode this module removes: a snapshot
nobody has to remember to take.

Secrets are deliberately absent. DATABASE_URL is recorded as present/absent, never its value.
"""
import os
import subprocess
from datetime import datetime, timezone

_MODEL_VARS = ("GEN_LLM_MODEL", "REASONING_EFFORT", "DOMAIN_JUDGE_MODEL",
               "CORRECTNESS_JUDGE_MODEL", "RAGAS_JUDGE_MODEL", "HYDE_MODEL", "EMB_MODEL")
_KNOB_VARS = ("FILINGS_HYDE", "FILING_CHUNKS_MAX", "FILING_TTL_DAYS", "RERANK")

# The defaults the code falls back to, so a snapshot taken with nothing exported still records
# what actually ran. Keep in step with the modules that read these.
_DEFAULTS = {"GEN_LLM_MODEL": "o4-mini", "REASONING_EFFORT": "low",
             "DOMAIN_JUDGE_MODEL": "gpt-4o", "CORRECTNESS_JUDGE_MODEL": "gpt-4o",
             "RAGAS_JUDGE_MODEL": "gpt-4o-mini", "HYDE_MODEL": "gpt-4o-mini",
             "EMB_MODEL": "text-embedding-3-small", "FILINGS_HYDE": "0",
             "FILING_CHUNKS_MAX": "20000", "FILING_TTL_DAYS": "30"}


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except Exception:
        return ""


def snapshot():
    """The configuration behind a run, as a plain dict. No secrets."""
    return {
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": {k: os.environ.get(k, _DEFAULTS.get(k)) for k in _MODEL_VARS},
        "knobs": {k: os.environ.get(k, _DEFAULTS.get(k)) for k in _KNOB_VARS},
        "git": {"branch": _git("branch", "--show-current"),
                "head": _git("rev-parse", "HEAD"),
                "subject": _git("log", "-1", "--format=%s"),
                # A dirty tree means the commit does not describe what ran.
                "dirty": bool(_git("status", "--porcelain"))},
        "database_url_set": bool(os.environ.get("DATABASE_URL")),
    }
