# Private-document ingestion (bring-your-own-data)

The second retrieval path: a user uploads their own documents (an internal financial
statement, a memo for a non-public company, ...) and the agent answers from them — alongside
the public SEC path, not replacing it.

## Why the parser is kept apart

`docling` pulls heavyweight, conflicting deps (transformers 5.x / numpy 2 / torch ≥2.4) that break
the main agent's reranker, so the agent **never imports it**. The two sides are decoupled through
the `user_chunks` table in pgvector — exactly like the public path (`build_filings_store.py` →
`filing_chunks`).

That separation takes a different shape in each environment, and `INGEST_URL` chooses:

```
unset (a dev machine)
  upload ──▶ ingest/.venv ──▶ pgvector user_chunks ◀── agent (search_my_documents)
             subprocess, parse → chunk → embed            reads, filters by user_id

set (deployed)
  upload ──▶ object storage ──▶ parser service (server.py) ──▶ pgvector user_chunks
                                private, concurrency 1,
                                writes its own status row
```

The deployed half exists because a venv is not something an image has: `ingest/.venv` is gitignored
and the agent image never copied `ingest/` at all, so every upload to Cloud Run failed on a missing
interpreter — unnoticed for months, because a dev machine and the deployed service share one
database, and documents parsed locally appear in the hosted demo and answer questions there.

`ingest.py` is the same program in both. `server.py` only wraps it: fetch the upload, run it, record
what happened.

## Setup (once, for local use)

```bash
python3.10 -m venv ingest/.venv
ingest/.venv/bin/pip install -r ingest/requirements.txt
```

The deployed parser needs no venv — the image IS the environment. Build and deploy it with
`bash ingest/deploy.sh` (from the repo root), then point the agent at the URL it prints.

## Ingest a document

```bash
DATABASE_URL=... OPENAI_API_KEY=... \
  ingest/.venv/bin/python ingest/ingest.py --user <user_id> --file path/to/report.pdf
```

docling parses PDF/DOCX/XLSX (tables preserved as markdown), chunks + embeds, and inserts into
`user_chunks` tagged with `user_id` (data is isolated per user). Re-ingesting the same
`--doc-id` replaces it.

The agent picks up the documents automatically: `build_agent(user_id=...)` binds a
`search_my_documents` tool scoped to that user.

## Status / next step

Minimal version (path-B step 1): numbers are read from the retrieved text. **Step 2** (the
finance bar on private data): ground numbers in docling's extracted **table cells** with
cell-level citations, mirroring how the public path grounds numbers in XBRL.

This is run manually here (offline). To serve it to multiple users, wrap the same logic in a
small `/ingest` HTTP service in its own container (see roadmap note §13: C1 → C2).
