"""FastAPI service for the SEC Filing Agent — a clickable live demo that exposes the
agent over HTTP and surfaces its tool-call trace + cited filings.

Run locally (from repo root, with OPENAI_API_KEY + DATABASE_URL set):
    uvicorn service.app:app --reload --port 8100
Then open http://localhost:8100
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import psycopg
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
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
_agent = None
_user_agents = {}   # session_id -> agent (built lazily once a user has uploaded docs)


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
    _agent = build_agent()  # loads the reranker once, on startup
    yield


app = FastAPI(title="SEC Filing Agent", lifespan=lifespan)


class Query(BaseModel):
    question: str
    history: Optional[List[dict]] = None   # prior turns [{role, content}] for multi-turn (Model B)
    session_id: Optional[str] = None       # browser session -> isolates this user's uploaded docs


def _agent_for(session_id):
    """Use the per-session agent (with the user's private-doc tools) if they've uploaded
    anything; otherwise the shared public-only agent."""
    return _user_agents.get(session_id, _agent) if session_id else _agent


@app.post("/ask")
def ask(q: Query, request: Request):
    if len(q.question) > MAX_INPUT_CHARS:
        raise HTTPException(400, f"Question too long (max {MAX_INPUT_CHARS} characters).")
    if not check_rate_limit(_client_ip(request)):
        raise HTTPException(429, "Too many requests — please wait a minute and try again.")
    if not check_daily_quota():
        raise HTTPException(429, "This demo has reached its daily limit. Please try again tomorrow.")
    # sync def -> FastAPI runs it in a threadpool, so the blocking LangGraph/LLM calls
    # don't block the event loop
    out = run_agent(q.question, agent=_agent_for(q.session_id), history=q.history)
    return {"answer": out["answer"], "trace": out["trace"], "tools_used": out["tools_used"],
            "sources": _sources(out["answer"], out["tool_outputs"]),
            "doc_sources": _private_sources(out["tool_outputs"], q.session_id)}


@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...), session_id: str = Form(...)):
    """Bring-your-own-data: ingest a user's document into their private space, then return a
    preview of what docling parsed (markdown + extracted table facts) so the parse is visible.
    Ingestion runs in the isolated docling venv via subprocess — the service never imports it."""
    if not check_rate_limit(_client_ip(request)):
        raise HTTPException(429, "Too many requests — please wait a minute and try again.")
    data = await file.read()
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(400, "File too large (max 10 MB for this demo).")
    doc_id = uuid.uuid4().hex[:12]
    suffix = os.path.splitext(file.filename or "upload.pdf")[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        proc = subprocess.run(
            [_INGEST_PY, _INGEST_SCRIPT, "--user", session_id, "--doc-id", doc_id,
             "--file", tmp_path, "--name", file.filename or "upload.pdf",
             "--max-pages", str(_MAX_PAGES)],
            cwd=_ROOT, env={**os.environ}, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            os.unlink(tmp_path)
            raise HTTPException(500, f"Ingestion failed: {proc.stderr[-400:]}")
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    # layer-1 parse-quality signal printed by the ingest job (machine-readable line)
    parse_quality = None
    for line in proc.stdout.splitlines():
        if line.startswith("PARSE_QUALITY="):
            try:
                parse_quality = json.loads(line[len("PARSE_QUALITY="):])
            except Exception:
                pass

    # keep the original file so its citations can link back to the source page (method 乙)
    user_dir = os.path.join(_UPLOAD_DIR, session_id)
    os.makedirs(user_dir, exist_ok=True)
    os.replace(tmp_path, os.path.join(user_dir, doc_id + suffix))

    # rebuild this session's agent so it knows about the new document
    _user_agents[session_id] = build_agent(user_id=session_id)

    # read back what was parsed, for a visible preview
    PREVIEW_CHARS = 3000
    facts = []
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        # full parsed text spans multiple chunks for a long doc; stitch + report totals
        cur.execute("select chunk_text from user_chunks where user_id=%s and doc_id=%s order by id",
                    (session_id, doc_id))
        full = "\n".join(r[0] for r in cur.fetchall())
        cur.execute("select max(page) from user_chunks where user_id=%s and doc_id=%s",
                    (session_id, doc_id))
        n_pages = cur.fetchone()[0]
        cur.execute("select metric,period,raw,cell,page from user_facts where user_id=%s "
                    "and doc_id=%s order by id", (session_id, doc_id))
        facts = [{"metric": m, "period": p, "value": v, "cell": c, "page": pg}
                 for (m, p, v, c, pg) in cur.fetchall()]
    preview = full[:PREVIEW_CHARS]
    return {"filename": file.filename, "doc_id": doc_id,
            "parsed_markdown": preview, "truncated": len(full) > PREVIEW_CHARS,
            "n_chars": len(full), "n_pages": n_pages, "facts": facts, "n_facts": len(facts),
            "page_cap": _MAX_PAGES, "parse_quality": parse_quality}


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
