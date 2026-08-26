"""FastAPI service for the SEC Filing Agent — a clickable live demo that exposes the
agent over HTTP and surfaces its tool-call trace + cited filings.

Run locally (from repo root, with OPENAI_API_KEY + DATABASE_URL set):
    uvicorn service.app:app --reload --port 8100
Then open http://localhost:8100
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import psycopg
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.sec_agent import build_agent, run_agent
from service.limits import MAX_INPUT_CHARS, check_rate_limit, check_daily_quota

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_STATIC = os.path.join(os.path.dirname(__file__), "static")
_INGEST_PY = os.path.join(_ROOT, "ingest", ".venv", "bin", "python")
_INGEST_SCRIPT = os.path.join(_ROOT, "ingest", "ingest.py")
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB demo cap
# docling's precise parse is ~2s/page on CPU; cap pages so a sync upload stays responsive.
# Production would parse the whole document asynchronously instead (roadmap §16).
_MAX_PAGES = int(os.environ.get("UPLOAD_MAX_PAGES", "15"))
# original uploads are kept so the citation can link back to the source page (auditability).
# local demo: a folder on disk; production would use object storage (GCS/S3) + encryption.
_UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(_ROOT, "ingest", "uploads"))
# Optional Google sign-in: set GOOGLE_CLIENT_ID to require login (the user_id then becomes the
# verified email, so data is isolated per account and persists across sessions). Unset = the
# demo stays open with an anonymous per-browser session_id (current behavior, fully testable).
_GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
_agent = None
_user_agents = {}   # user_id -> agent (built lazily once a user has uploaded docs)


def _verify_google(token):
    """Verify a Google ID token (JWT) and return the user's email, or None if invalid."""
    if not token:
        return None
    try:
        from google.oauth2 import id_token
        from google.auth.transport import requests as g_requests
        info = id_token.verify_oauth2_token(token, g_requests.Request(), _GOOGLE_CLIENT_ID)
        return info.get("email") if info.get("email_verified") else None
    except Exception:
        return None


def _user_id(token, session_id):
    """The effective per-user id. With Google sign-in on: the verified email (raises 401 if the
    token is missing/invalid). Off: the anonymous browser session_id."""
    if _GOOGLE_CLIENT_ID:
        email = _verify_google(token)
        if not email:
            raise HTTPException(401, "Sign-in required.")
        return email
    return session_id


_ACC = re.compile(r"\d{10}-\d{2}-\d{6}")
_EDGAR_URL = re.compile(r"https://www\.sec\.gov/Archives/edgar/\S+?-index\.htm")


def _sources(answer: str, tool_outputs):
    """Build clickable sources deterministically: map each accession the answer cites to the
    EDGAR URL the tools returned. Independent of how the model formatted its inline citation."""
    url_by_acc = {}
    for _, content in tool_outputs:
        for url in _EDGAR_URL.findall(content or ""):
            m = _ACC.search(url)
            if m:
                url_by_acc[m.group(0)] = url
    out, seen = [], set()
    for acc in _ACC.findall(answer or ""):
        if acc in url_by_acc and acc not in seen:
            seen.add(acc)
            out.append({"accession": acc, "url": url_by_acc[acc]})
    return out


_PRIV_SRC = re.compile(r"\[source:\s*([^\]·]+?)(?:\s*·\s*p\.(\d+))?"
                       r"(?:\s*·\s*(row[^\]]+?))?\]")


def _private_sources(tool_outputs, session_id=None):
    """Deterministic cell-level provenance for uploaded-doc answers: parse the
    [source: filename · p.N · row '...' / col '...'] tags the user-doc tools emit, so the cell
    citation is always shown regardless of how the LLM phrased its answer (mirrors _sources).
    Attaches doc_id (looked up by filename) so the citation can deep-link to the source page."""
    out, seen = [], set()
    docid_by_name = {}
    if session_id:
        with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
            cur.execute("select distinct filename, doc_id from user_chunks where user_id=%s",
                        (session_id,))
            docid_by_name = {fn: did for fn, did in cur.fetchall()}
    for name, content in tool_outputs:
        if name not in ("get_my_financials", "search_my_documents"):
            continue
        for fn, page, cell in _PRIV_SRC.findall(content or ""):
            fn = fn.strip()
            key = (fn, page, cell.strip())
            if key in seen:
                continue
            seen.add(key)
            out.append({"filename": fn, "page": int(page) if page else None,
                        "cell": cell.strip() or None, "doc_id": docid_by_name.get(fn)})
    return out


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    _ensure_doc_schema()    # the async-ingestion status table
    _agent = build_agent()  # loads the reranker once, on startup
    yield


app = FastAPI(title="SEC Filing Agent", lifespan=lifespan)


class Query(BaseModel):
    question: str
    history: Optional[List[dict]] = None   # prior turns [{role, content}] for multi-turn (Model B)
    session_id: Optional[str] = None       # anonymous browser session (when sign-in is off)
    token: Optional[str] = None            # Google ID token (when sign-in is on)
    doc: Optional[str] = None              # selected document to scope to (A); None = all docs


def _agent_for(user_id, scope_doc=None):
    """Use the per-user agent (with the user's private-doc tools) if they've uploaded anything;
    otherwise the shared public-only agent. `scope_doc` (A) restricts lookups to one selected
    document. Agents are cached by (user_id, scope_doc); lazily rebuilt if not cached but the
    user has docs in the DB — so a returning user (new login / restart) can still query their
    persisted documents."""
    if not user_id:
        return _agent
    key = (user_id, scope_doc)
    if key in _user_agents:
        return _user_agents[key]
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute("select 1 from user_chunks where user_id=%s limit 1", (user_id,))
        has_docs = cur.fetchone() is not None
    if has_docs:
        _user_agents[key] = build_agent(user_id=user_id, scope_doc=scope_doc)
        return _user_agents[key]
    return _agent


@app.get("/auth/config")
def auth_config():
    """Tells the frontend whether to show a Google sign-in screen, and the client id to use."""
    return {"auth_required": bool(_GOOGLE_CLIENT_ID), "client_id": _GOOGLE_CLIENT_ID}


class Ident(BaseModel):
    session_id: Optional[str] = None
    token: Optional[str] = None


@app.post("/mydocs")
def mydocs(ident: Ident):
    """The current user's previously uploaded documents (persisted, keyed by account) — so a
    returning user sees their docs are still there after signing back in."""
    user_id = _user_id(ident.token, ident.session_id)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute("select distinct filename from user_chunks where user_id=%s order by filename",
                    (user_id,))
        return {"documents": [r[0] for r in cur.fetchall()]}


class DocRef(BaseModel):
    session_id: Optional[str] = None
    token: Optional[str] = None
    filename: str


@app.post("/mydocs/delete")
def delete_doc(ref: DocRef):
    """Remove one of the user's uploaded documents (DB rows + stored original file), then drop
    their cached agents so the next request rebuilds without it. Deletes by filename, the same
    key the doc list and selector use (one filename = one document, since re-upload replaces)."""
    user_id = _user_id(ref.token, ref.session_id)
    _remove_prior_uploads(user_id, ref.filename)   # same teardown re-upload uses (chunks+facts+file)
    for k in [k for k in _user_agents if k[0] == user_id]:
        del _user_agents[k]
    return {"ok": True}


# Response cache: an identical public single-turn question returns the already-built answer instead of
# re-running the agent (the slow, costly LLM step). Keyed on the normalized question; a short TTL bounds
# staleness so a newly-filed or restated figure isn't served from an old entry. Cached only when the
# request is safely shareable: no conversation history (multi-turn answers depend on it) and no private
# uploaded-doc scope (those answers are user-specific). In-memory suffices at single-instance scale; a
# distributed store (Redis) would only be needed once serving fans out across instances.
_RESP_CACHE = {}                                              # normalized question -> (expires_at, response)
_RESP_CACHE_TTL = int(os.environ.get("RESPONSE_CACHE_TTL", "3600"))
_RESP_CACHE_MAX = 1000


def _cache_key(q: "Query", user_id):
    if q.history or user_id or q.doc:                         # not safely shareable -> don't cache
        return None
    return " ".join(q.question.lower().split())               # normalize whitespace/case


@app.post("/ask")
def ask(q: Query, request: Request):
    if len(q.question) > MAX_INPUT_CHARS:
        raise HTTPException(400, f"Question too long (max {MAX_INPUT_CHARS} characters).")
    if not check_rate_limit(_client_ip(request)):
        raise HTTPException(429, "Too many requests — please wait a minute and try again.")
    user_id = _user_id(q.token, q.session_id)
    key = _cache_key(q, user_id)
    if key is not None:
        hit = _RESP_CACHE.get(key)
        if hit and hit[0] > time.time():
            return {**hit[1], "cached": True}                 # served without an LLM call
    if not check_daily_quota():                               # only real (uncached) answers cost budget
        raise HTTPException(429, "This demo has reached its daily limit. Please try again tomorrow.")
    # sync def -> FastAPI runs it in a threadpool, so the blocking LangGraph/LLM calls
    # don't block the event loop
    try:
        out = run_agent(q.question, agent=_agent_for(user_id, scope_doc=q.doc), history=q.history)
    except Exception as e:
        # public testing: an LLM/timeout error should be a friendly bubble, not a 500
        print(f"[/ask] agent error: {type(e).__name__}: {e}")
        return {"answer": "Sorry — I hit an error answering that. Please try again in a moment.",
                "trace": [], "tools_used": [], "sources": [], "doc_sources": [], "trace_id": None}
    resp = {"answer": out["answer"], "trace": out["trace"], "tools_used": out["tools_used"],
            "sources": _sources(out["answer"], out["tool_outputs"]),
            "doc_sources": _private_sources(out["tool_outputs"], user_id),
            "trace_id": out.get("trace_id")}
    if key is not None:
        if len(_RESP_CACHE) >= _RESP_CACHE_MAX:
            _RESP_CACHE.pop(next(iter(_RESP_CACHE)))          # simple bound: evict the oldest entry
        _RESP_CACHE[key] = (time.time() + _RESP_CACHE_TTL, resp)
    return resp


class VerifyReq(BaseModel):
    trace: List[dict] = []
    answer: str = ""


@app.post("/verify")
def verify(v: VerifyReq, request: Request):
    """Re-run the deterministic tool calls in a stored answer's trace and re-verify the answer
    (reproducibility + grounding). On-demand (a button), so it doesn't cost anything per question."""
    if not check_rate_limit(_client_ip(request)):
        raise HTTPException(429, "Too many requests — please wait a minute and try again.")
    from agents.replay import replay
    try:
        return replay(v.trace, v.answer)
    except Exception as e:
        print(f"[/verify] error: {type(e).__name__}: {e}")
        return {"verdict": "ERROR", "checks": [], "numbers_grounded": None, "ungrounded": []}


@app.get("/export")
def export_table(request: Request, ticker: str, metrics: str, start: int, end: int):
    """Deterministic CSV of metrics/ratios × fiscal years, straight from the XBRL tools (no LLM in
    the loop, so the export is exact). Public data -> no auth; light rate limit."""
    if not check_rate_limit(_client_ip(request)):
        raise HTTPException(429, "Too many requests — please wait a minute and try again.")
    ticker = (ticker or "").strip()
    mlist = [m for m in (metrics or "").split(",") if m.strip()]
    if not ticker or not mlist:
        raise HTTPException(400, "Provide a ticker (or company name) and at least one metric.")
    if not (1990 <= start <= end <= 2100) or end - start > 30:
        raise HTTPException(400, "Provide a sensible fiscal-year range (start ≤ end, ≤ 30 years).")
    from agents.finance_tools import financial_table_csv, _rows_for
    if not _rows_for(ticker):
        raise HTTPException(404, f"No company found for '{ticker}'. If delisted/renamed, use its full name.")
    csv = financial_table_csv(ticker, mlist, list(range(start, end + 1)))
    fname = f"{ticker.upper().replace(' ', '_')}_{start}-{end}.csv"
    return Response(csv, media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


class Feedback(BaseModel):
    session_id: Optional[str] = None
    token: Optional[str] = None
    trace_id: str
    value: int                      # 1 = thumbs up, 0 = thumbs down
    comment: Optional[str] = None
    question: Optional[str] = None   # carried so the score row is self-explanatory in the UI
    answer: Optional[str] = None


@app.post("/feedback")
def feedback(fb: Feedback):
    """Capture a user's thumbs up/down on an answer as a Langfuse score (name=user_feedback),
    so community testing yields LABELED eval data, not just unlabeled traces. The question/answer
    are attached as the score's metadata so a reviewer can read what was rated without opening the
    trace (and to find the thumbs-down cases worth investigating)."""
    _user_id(fb.token, fb.session_id)   # gate behind sign-in (raises 401 if required & absent)
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return {"ok": False, "reason": "observability disabled"}
    try:
        from langfuse import get_client
        lf = get_client()
        meta = {k: v[:500] for k, v in (("question", fb.question), ("answer", fb.answer)) if v}
        lf.create_score(name="user_feedback", value=float(1 if fb.value >= 1 else 0),
                        trace_id=fb.trace_id, data_type="NUMERIC",
                        comment=(fb.comment or None), metadata=(meta or None))
        lf.flush()
    except Exception as e:
        print(f"[/feedback] {type(e).__name__}: {e}")
        return {"ok": False}
    return {"ok": True}


def _remove_prior_uploads(session_id, filename):
    """Delete this user's previous upload(s) of the same filename: DB rows (chunks + facts) and
    the stored original file(s), so re-uploading replaces rather than duplicates."""
    if not filename:
        return
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute("select distinct doc_id from user_chunks where user_id=%s and filename=%s",
                    (session_id, filename))
        old_ids = [r[0] for r in cur.fetchall()]
        for did in old_ids:
            cur.execute("delete from user_chunks where user_id=%s and doc_id=%s", (session_id, did))
            cur.execute("delete from user_facts where user_id=%s and doc_id=%s", (session_id, did))
            cur.execute("delete from user_documents where user_id=%s and doc_id=%s", (session_id, did))
        conn.commit()
    user_dir = os.path.join(_UPLOAD_DIR, session_id)
    for did in old_ids:
        for fn in (os.listdir(user_dir) if os.path.isdir(user_dir) else []):
            if fn.startswith(did + "."):
                try:
                    os.unlink(os.path.join(user_dir, fn))
                except OSError:
                    pass


def _find_duplicate(user_id, sha):
    """Return (doc_id, filename) of an already-uploaded file with byte-identical content, or None.
    Compares the SHA-256 of the new upload against this user's stored original files — so an exact
    re-upload is detected and reused instead of re-parsed (parsing is slow + costs embeddings)."""
    user_dir = os.path.join(_UPLOAD_DIR, user_id)
    if not os.path.isdir(user_dir):
        return None
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute("select distinct doc_id, filename from user_chunks where user_id=%s", (user_id,))
        names = {d: f for d, f in cur.fetchall()}
    for fn in os.listdir(user_dir):
        path = os.path.join(user_dir, fn)
        if not os.path.isfile(path):
            continue
        did = os.path.splitext(fn)[0]
        if did not in names:  # only reuse a fully-parsed doc; skip files still processing / failed / orphaned
            continue
        try:
            with open(path, "rb") as fh:
                if hashlib.sha256(fh.read()).hexdigest() == sha:
                    return did, names[did]
        except OSError:
            continue
    return None


def _doc_payload(user_id, doc_id, filename):
    """Read back an ingested document (parsed-text preview + extracted table facts) in the shape
    the upload UI expects. Shared by a fresh ingest and a dedup hit."""
    PREVIEW_CHARS = 3000
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute("select chunk_text from user_chunks where user_id=%s and doc_id=%s order by id",
                    (user_id, doc_id))
        full = "\n".join(r[0] for r in cur.fetchall())
        cur.execute("select max(page) from user_chunks where user_id=%s and doc_id=%s",
                    (user_id, doc_id))
        n_pages = cur.fetchone()[0]
        cur.execute("select metric,period,raw,cell,page from user_facts where user_id=%s "
                    "and doc_id=%s order by id", (user_id, doc_id))
        facts = [{"metric": m, "period": p, "value": v, "cell": c, "page": pg}
                 for (m, p, v, c, pg) in cur.fetchall()]
    return {"filename": filename, "doc_id": doc_id,
            "parsed_markdown": full[:PREVIEW_CHARS], "truncated": len(full) > PREVIEW_CHARS,
            "n_chars": len(full), "n_pages": n_pages, "facts": facts, "n_facts": len(facts),
            "page_cap": _MAX_PAGES}


def _ensure_doc_schema():
    """One status row per uploaded document, so ingestion can run asynchronously and the UI can
    poll. (Production-grade would back this with a durable queue + a separate worker so jobs
    survive instance restarts; here a thread + this table is enough for a single-instance demo.)"""
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute("""create table if not exists user_documents (
                user_id text not null, doc_id text not null, filename text,
                status text not null default 'processing',  -- processing | ready | failed
                error text, n_pages int, parse_quality text,
                created_at timestamptz default now(),
                primary key (user_id, doc_id))""")
        conn.commit()


def _doc_set_status(user_id, doc_id, status, error=None, parse_quality=None):
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute("update user_documents set status=%s, error=%s, "
                    "parse_quality=coalesce(%s, parse_quality) where user_id=%s and doc_id=%s",
                    (status, error, parse_quality, user_id, doc_id))
        conn.commit()


def _doc_get(user_id, doc_id):
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute("select filename, status, error, parse_quality from user_documents "
                    "where user_id=%s and doc_id=%s", (user_id, doc_id))
        return cur.fetchone()


def _ingest_async(user_id, doc_id, filename, file_path):
    """Background worker: parse the FULL document (no 15-page cap) and mark it ready/failed.
    Runs in a thread so /upload returns immediately — the slow docling parse never blocks the
    request, which is what lets us drop the page cap and accept real (100+ page) filings."""
    try:
        proc = subprocess.run(
            [_INGEST_PY, _INGEST_SCRIPT, "--user", user_id, "--doc-id", doc_id,
             "--file", file_path, "--name", filename],   # no --max-pages -> parse the whole doc
            cwd=_ROOT, env={**os.environ}, capture_output=True, text=True, timeout=1200)
        if proc.returncode != 0:
            _doc_set_status(user_id, doc_id, "failed", error=(proc.stderr[-400:] or "ingestion failed"))
            return
        pq = next((ln[len("PARSE_QUALITY="):] for ln in proc.stdout.splitlines()
                   if ln.startswith("PARSE_QUALITY=")), None)
        _doc_set_status(user_id, doc_id, "ready", parse_quality=pq)
        # doc list changed -> drop cached agents so they rebuild with the new document
        for k in [k for k in _user_agents if k[0] == user_id]:
            del _user_agents[k]
    except Exception as e:
        _doc_set_status(user_id, doc_id, "failed", error=f"{type(e).__name__}: {e}")


@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...),
                 session_id: str = Form(None), token: str = Form(None)):
    """Bring-your-own-data: ingest a user's document into their private space, then return a
    preview of what docling parsed (markdown + extracted table facts) so the parse is visible.
    Ingestion runs in the isolated docling venv via subprocess — the service never imports it."""
    if not check_rate_limit(_client_ip(request)):
        raise HTTPException(429, "Too many requests — please wait a minute and try again.")
    session_id = _user_id(token, session_id)   # verified email if sign-in is on
    data = await file.read()
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(400, "File too large (max 10 MB for this demo).")
    # content dedup: an exact re-upload (byte-identical) reuses the existing parse instead of
    # re-running docling. A same-NAME but changed file has a different hash, so it falls through
    # to the replace path below and re-parses (treated as an updated version).
    dup = _find_duplicate(session_id, hashlib.sha256(data).hexdigest())
    if dup:
        return {**_doc_payload(session_id, dup[0], dup[1]),
                "already_uploaded": True, "parse_quality": None, "status": "ready"}
    # re-uploading the same filename replaces the prior copy (DB rows + original file on disk),
    # so a user never ends up with stale duplicates of the same document
    _remove_prior_uploads(session_id, file.filename)
    doc_id = uuid.uuid4().hex[:12]
    suffix = os.path.splitext(file.filename or "upload.pdf")[1] or ".pdf"
    # save the original now (so /file works + the worker reads it), record a 'processing' row,
    # kick off the docling parse in a thread, and return immediately — the UI polls /docstatus.
    # No page cap on this path: parsing the full document is exactly what async buys us.
    user_dir = os.path.join(_UPLOAD_DIR, session_id)
    os.makedirs(user_dir, exist_ok=True)
    file_path = os.path.join(user_dir, doc_id + suffix)
    with open(file_path, "wb") as f:
        f.write(data)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute("insert into user_documents (user_id, doc_id, filename, status) "
                    "values (%s, %s, %s, 'processing')", (session_id, doc_id, file.filename))
        conn.commit()
    threading.Thread(target=_ingest_async,
                     args=(session_id, doc_id, file.filename or "upload.pdf", file_path),
                     daemon=True).start()
    return {"doc_id": doc_id, "filename": file.filename, "status": "processing"}


class DocStatusReq(BaseModel):
    session_id: Optional[str] = None
    token: Optional[str] = None
    doc_id: str


@app.post("/docstatus")
def docstatus(req: DocStatusReq):
    """Poll an async upload: returns 'processing'/'failed', or the full parse preview once 'ready'
    (facts + markdown), so the UI renders the result when the background parse finishes."""
    user_id = _user_id(req.token, req.session_id)
    row = _doc_get(user_id, req.doc_id)
    if not row:
        return {"status": "unknown"}
    filename, status, error, pq = row
    if status == "ready":
        return {**_doc_payload(user_id, req.doc_id, filename), "status": "ready",
                "parse_quality": (json.loads(pq) if pq else None)}
    return {"doc_id": req.doc_id, "filename": filename, "status": status, "error": error}


@app.get("/file/{doc_id}")
def get_file(doc_id: str, session_id: str):
    """Serve a user's original uploaded file so a citation can deep-link to its source page
    (#page=N). Access is checked: the doc must belong to this session (per-user isolation)."""
    if not re.fullmatch(r"[a-f0-9]{12}", doc_id):
        raise HTTPException(400, "bad doc id")
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute("select 1 from user_chunks where user_id=%s and doc_id=%s limit 1",
                    (session_id, doc_id))
        if not cur.fetchone():
            raise HTTPException(404, "not found")   # not this user's doc (or doesn't exist)
    user_dir = os.path.join(_UPLOAD_DIR, session_id)
    for fn in os.listdir(user_dir) if os.path.isdir(user_dir) else []:
        if fn.startswith(doc_id + "."):
            return FileResponse(os.path.join(user_dir, fn), media_type="application/pdf")
    raise HTTPException(404, "file not found")


def _answer_with_sources(out) -> str:
    """For OpenAI-compatible clients (open-webui) that only get the message text: append the
    cited filings as markdown links so they stay clickable (the rich tool-trace stays demo-only)."""
    ans = out["answer"]
    srcs = _sources(out["answer"], out["tool_outputs"])
    if srcs:
        ans += "\n\n**Sources:** " + " · ".join(f"[{s['accession']}]({s['url']})" for s in srcs)
    return ans


# --- OpenAI-compatible endpoints: lets the same agent plug into open-webui / LibreChat ---
# (Model B: the shell sends the full conversation; the agent is stateless and consumes it.)
@app.get("/v1/models")
def list_models():
    return {"object": "list",
            "data": [{"id": "sec-filing-agent", "object": "model", "created": 0, "owned_by": "local"}]}


@app.post("/v1/chat/completions")
def chat_completions(body: dict, request: Request):
    if not check_rate_limit(_client_ip(request)):
        raise HTTPException(429, "Too many requests — please wait a minute and try again.")
    if not check_daily_quota():
        raise HTTPException(429, "This demo has reached its daily limit. Please try again tomorrow.")
    messages = body.get("messages") or []
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        raise HTTPException(400, "no user message in 'messages'.")
    question = user_msgs[-1].get("content", "")          # latest user turn = the question
    history = [m for m in messages if m is not user_msgs[-1]]  # everything else = prior history
    out = run_agent(question, agent=_agent, history=history)
    content = _answer_with_sources(out)
    model = body.get("model", "sec-filing-agent")

    if body.get("stream"):
        return StreamingResponse(_sse_chunks(content, model), media_type="text/event-stream")
    return {"id": "chatcmpl-" + uuid.uuid4().hex[:24], "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}]}


def _sse_chunks(content: str, model: str):
    """Minimal OpenAI streaming: role chunk, one content chunk, stop, [DONE]."""
    cid, ts = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())

    def frame(delta, finish=None):
        return "data: " + json.dumps({
            "id": cid, "object": "chat.completion.chunk", "created": ts, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n"
    yield frame({"role": "assistant"})
    yield frame({"content": content})
    yield frame({}, finish="stop")
    yield "data: [DONE]\n\n"


@app.get("/health")
def health():
    return {"status": "ok", "agent_loaded": _agent is not None}


@app.get("/")
def index():
    return FileResponse(os.path.join(_STATIC, "index.html"))
