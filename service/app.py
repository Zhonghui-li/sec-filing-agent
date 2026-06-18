"""FastAPI service for the SEC Filing Agent — a clickable live demo that exposes the
agent over HTTP and surfaces its tool-call trace + cited filings.

Run locally (from repo root, with OPENAI_API_KEY + DATABASE_URL set):
    uvicorn service.app:app --reload --port 8100
Then open http://localhost:8100
"""
import json
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.sec_agent import build_agent, run_agent
from service.limits import MAX_INPUT_CHARS, check_rate_limit, check_daily_quota

_STATIC = os.path.join(os.path.dirname(__file__), "static")
_agent = None


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
    out = run_agent(q.question, agent=_agent, history=q.history)
    return {"answer": out["answer"], "trace": out["trace"], "tools_used": out["tools_used"],
            "sources": _sources(out["answer"], out["tool_outputs"])}


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
