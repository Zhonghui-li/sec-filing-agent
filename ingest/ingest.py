"""Ingest a user-uploaded document into pgvector (the private-data path, Model B step 1).

Runs in an ISOLATED venv (ingest/.venv) because docling pulls heavyweight, conflicting deps
(transformers 5.x / numpy 2 / torch >=2.4) that would break the main agent's reranker. The
agent never imports docling — the two sides are decoupled through the `user_chunks` table,
exactly like the public path (build_filings_store.py -> filing_chunks).

Pipeline: docling parses any format (PDF/DOCX/XLSX) -> markdown (tables preserved) ->
chunk -> embed -> insert into user_chunks tagged with user_id + filename + page.

Usage (from repo root):
    DATABASE_URL=... OPENAI_API_KEY=... \
      ingest/.venv/bin/python ingest/ingest.py --user demo --file path/to/report.pdf
"""
import argparse
import os
import uuid

import psycopg
from docling.document_converter import DocumentConverter
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMB_MODEL = os.environ.get("EMB_MODEL", "text-embedding-3-small")

DDL = """
create extension if not exists vector;
create table if not exists user_chunks (
    id serial primary key,
    user_id text not null,
    doc_id text not null,
    filename text,
    page int,
    chunk_text text,
    embedding vector(1536)
);
create index if not exists user_chunks_user_idx on user_chunks (user_id);
"""


def _vec(v):
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def parse_document(path: str):
    """docling -> list of (page, text). Minimal version: one markdown blob per document
    (page tracking is coarse; per-page/table-cell grounding is the path-B step-2 upgrade)."""
    conv = DocumentConverter()
    doc = conv.convert(path).document
    md = doc.export_to_markdown()
    return [(None, md)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="owner user_id (data is isolated per user)")
    ap.add_argument("--file", required=True, help="path to the document to ingest")
    ap.add_argument("--doc-id", default=None, help="optional stable id (default: random)")
    args = ap.parse_args()

    doc_id = args.doc_id or uuid.uuid4().hex[:12]
    filename = os.path.basename(args.file)
    print(f"parsing {filename} (docling)...")
    pages = parse_document(args.file)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = []  # (page, text)
    for page, text in pages:
        for piece in splitter.split_text(text):
            chunks.append((page, piece))
    print(f"{len(chunks)} chunks; embedding ({EMB_MODEL})...")

    emb = OpenAIEmbeddings(model=EMB_MODEL)
    vectors = emb.embed_documents([c[1] for c in chunks])

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute(DDL)
        # re-ingesting the same doc_id replaces it (idempotent uploads)
        cur.execute("delete from user_chunks where user_id=%s and doc_id=%s", (args.user, doc_id))
        for (page, text), v in zip(chunks, vectors):
            cur.execute(
                "insert into user_chunks (user_id,doc_id,filename,page,chunk_text,embedding) "
                "values (%s,%s,%s,%s,%s,%s::vector)",
                (args.user, doc_id, filename, page, text, _vec(v)))
        conn.commit()
        cur.execute("select count(*) from user_chunks where user_id=%s", (args.user,))
        print(f"ingested doc_id={doc_id} for user={args.user}; "
              f"{cur.fetchone()[0]} total chunks for this user.")


if __name__ == "__main__":
    main()
