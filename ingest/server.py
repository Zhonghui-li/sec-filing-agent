"""HTTP wrapper that turns the docling parse into its own Cloud Run service.

Why it is a separate service rather than part of the agent image. docling drags in torch, opencv,
rapidocr, scipy and transformers -- about 1.2 GB -- and the agent's image is already 0.64 GB, of
which a cold start spends 15 to 60 seconds just getting to the first line of Python. Baking the
parser in would make that worse for every visitor, including the overwhelming majority who never
upload anything. Here the weight sits behind an endpoint that scales to zero, and its cold start
is paid only by an upload, which is asynchronous and whose UI already says it can take minutes.

It also keeps the isolation the repo already decided on: ingest.py runs in a process that never
shares an interpreter with the agent, which is the same reason it used to live in ingest/.venv.
ingest.py itself is unchanged and still works from the command line.

This service writes the final status itself. The caller cannot be relied on to: /upload returns
immediately and the agent instance that started the parse may be recycled long before the parse
ends, which would strand the row in 'processing' forever.
"""
import json
import os
import subprocess
import sys
import tempfile

import psycopg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPT = os.path.join(_HERE, "ingest.py")
_TIMEOUT = int(os.environ.get("PARSE_TIMEOUT", "1500"))

app = FastAPI(title="SEC Filing Agent — document parser")


class ParseReq(BaseModel):
    user_id: str
    doc_id: str
    filename: str
    locator: str          # gs://bucket/key, or a path when running without object storage


def _set_status(user_id, doc_id, status, error=None, parse_quality=None):
    try:
        with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
            cur.execute("update user_documents set status=%s, error=%s, "
                        "parse_quality=coalesce(%s, parse_quality) "
                        "where user_id=%s and doc_id=%s",
                        (status, error, parse_quality, user_id, doc_id))
            conn.commit()
    except Exception as e:                      # a status write must never mask the parse result
        print(f"[parser] status write failed: {type(e).__name__}: {e}", flush=True)


def _fetch(locator, into_dir):
    """Materialise the upload locally for docling, which reads a path."""
    if not locator.startswith("gs://"):
        return locator
    from google.cloud import storage
    bucket_name, _, key = locator[len("gs://"):].partition("/")
    local = os.path.join(into_dir, os.path.basename(key))
    storage.Client().bucket(bucket_name).blob(key).download_to_filename(local)
    return local


@app.post("/parse")
def parse(req: ParseReq):
    """Parse one uploaded document, then record the outcome. Synchronous on purpose: the caller
    already runs this in a background thread, and a second queue here would add a failure mode
    without adding a capability."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            path = _fetch(req.locator, tmp)
        except Exception as e:
            _set_status(req.user_id, req.doc_id, "failed", error=f"could not read upload: {e}")
            raise HTTPException(400, f"could not read upload: {type(e).__name__}")
        try:
            proc = subprocess.run(
                [sys.executable, _SCRIPT, "--user", req.user_id, "--doc-id", req.doc_id,
                 "--file", path, "--name", req.filename],   # no --max-pages: parse the whole doc
                cwd=os.path.dirname(_HERE), capture_output=True, text=True, timeout=_TIMEOUT)
        except subprocess.TimeoutExpired:
            _set_status(req.user_id, req.doc_id, "failed",
                        error=f"parsing took longer than {_TIMEOUT}s")
            raise HTTPException(504, "parse timed out")

    if proc.returncode != 0:
        err = (proc.stderr or "")[-400:] or "ingestion failed"
        print(f"[parser] {req.doc_id} failed: {err}", flush=True)
        _set_status(req.user_id, req.doc_id, "failed", error=err)
        return {"status": "failed", "error": err}

    pq = next((ln[len("PARSE_QUALITY="):] for ln in proc.stdout.splitlines()
               if ln.startswith("PARSE_QUALITY=")), None)
    _set_status(req.user_id, req.doc_id, "ready", parse_quality=pq)
    return {"status": "ready", "parse_quality": json.loads(pq) if pq else None}


@app.get("/health")
def health():
    return {"ok": True}
