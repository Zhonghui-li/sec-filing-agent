"""FastAPI service for the SEC Filing Agent — a clickable live demo that exposes the
agent over HTTP and surfaces its tool-call trace + cited filings.

Run locally (from repo root, with OPENAI_API_KEY + DATABASE_URL set):
    uvicorn service.app:app --reload --port 8100
Then open http://localhost:8100
"""
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.sec_agent import build_agent, run_agent
from service.limits import MAX_INPUT_CHARS, check_rate_limit, check_daily_quota

_STATIC = os.path.join(os.path.dirname(__file__), "static")
_agent = None


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
    out = run_agent(q.question, agent=_agent)
    return {"answer": out["answer"], "trace": out["trace"], "tools_used": out["tools_used"]}


@app.get("/health")
def health():
    return {"status": "ok", "agent_loaded": _agent is not None}


@app.get("/")
def index():
    return FileResponse(os.path.join(_STATIC, "index.html"))
