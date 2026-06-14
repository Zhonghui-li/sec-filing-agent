"""Chunk the 10-K sections and embed them into pgvector (the RAG corpus).

Each chunk keeps ticker / fiscal_year / section / accession so retrieval results
carry their provenance — the agent cites the exact filing it grounded an answer in.

Usage:  DATABASE_URL=... OPENAI_API_KEY=... python scripts/build_filings_store.py
"""
import os
import json
from pathlib import Path

import psycopg
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

SRC = Path(__file__).resolve().parent.parent / "data" / "filings.jsonl"
EMB_MODEL = os.environ.get("EMB_MODEL", "text-embedding-3-small")

DDL = """
create extension if not exists vector;
drop table if exists filing_chunks;
create table filing_chunks (
    id serial primary key,
    ticker text, fiscal_year int, section text, accession text,
    chunk_text text, embedding vector(1536)
);
"""


def _vec(v):
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def main():
    records = [json.loads(l) for l in SRC.open()]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)

    chunks = []   # (ticker, fy, section, accession, text)
    for r in records:
        for piece in splitter.split_text(r["text"]):
            chunks.append((r["ticker"], r["fiscal_year"], r["section"],
                           r["accession"], piece))
    print(f"{len(records)} sections -> {len(chunks)} chunks; embedding ({EMB_MODEL})...")

    emb = OpenAIEmbeddings(model=EMB_MODEL)
    vectors = []
    B = 256
    for i in range(0, len(chunks), B):
        vectors.extend(emb.embed_documents([c[4] for c in chunks[i:i + B]]))
        print(f"  embedded {min(i + B, len(chunks))}/{len(chunks)}")

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute(DDL)
        for (tk, fy, sec, accn, text), v in zip(chunks, vectors):
            cur.execute(
                "insert into filing_chunks (ticker,fiscal_year,section,accession,"
                "chunk_text,embedding) values (%s,%s,%s,%s,%s,%s::vector)",
                (tk, fy, sec, accn, text, _vec(v)))
        cur.execute("create index on filing_chunks using hnsw (embedding vector_cosine_ops);")
        conn.commit()
        cur.execute("select count(*) from filing_chunks;")
        print(f"Inserted {cur.fetchone()[0]} filing chunks into pgvector.")


if __name__ == "__main__":
    main()
