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
import re
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

-- structured numbers extracted from the document's TABLES (the private-data analogue of XBRL):
-- the agent reads exact figures from here, never from prose. cell = "row x col" provenance.
create table if not exists user_facts (
    id serial primary key,
    user_id text not null,
    doc_id text not null,
    filename text,
    page int,
    metric text,        -- row label, e.g. "Net Income"
    period text,        -- column label, e.g. "FY2025"
    value numeric,      -- parsed exact value
    raw text,           -- original cell string, e.g. "4,980"
    cell text           -- provenance, e.g. "row 'Net Income' / col 'FY2025'"
);
create index if not exists user_facts_user_idx on user_facts (user_id);
"""


_NUM = re.compile(r"^-?\(?\$?\s*[\d,]+(?:\.\d+)?\)?$")


def _parse_value(s: str):
    """Parse a table cell to a number, or None if it isn't one. Handles $, commas, and
    accounting negatives (parentheses). Conservative: only cells that are clearly numeric."""
    s = (s or "").strip()
    if not s or not _NUM.match(s):
        return None
    neg = s.startswith("(") and s.endswith(")")
    cleaned = s.replace("$", "").replace(",", "").replace("(", "").replace(")", "").strip()
    try:
        v = float(cleaned)
    except ValueError:
        return None
    return -v if neg else v


def extract_table_facts(doc):
    """docling tables -> [(page, metric, period, value, raw, cell)]. First column is the metric
    (row label); each remaining numeric column is a period. Values stay exact; non-numeric cells
    are skipped. This is the structured source the agent grounds numbers in (mirrors XBRL)."""
    facts = []
    for t in doc.tables:
        df = t.export_to_dataframe(doc=doc)
        if df.shape[0] == 0 or df.shape[1] < 2:
            continue
        page = None
        prov = getattr(t, "prov", None)
        if prov and hasattr(prov[0], "page_no"):
            page = prov[0].page_no
        metric_col = df.columns[0]
        period_cols = list(df.columns[1:])
        for _, row in df.iterrows():
            metric = str(row[metric_col]).strip()
            if not metric:
                continue
            for pcol in period_cols:
                raw = str(row[pcol]).strip()
                v = _parse_value(raw)
                if v is None:
                    continue
                facts.append((page, metric, str(pcol).strip(), v, raw,
                              f"row '{metric}' / col '{pcol}'"))
    return facts


def _vec(v):
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def parse_document(path: str, max_pages: int = None):
    """docling -> (markdown_pages, table_facts). markdown for narrative retrieval; table_facts
    for exact-number grounding. `max_pages` caps how many pages are parsed — docling's precise
    parsing is slow (~2s/page on CPU), so a long doc would block a sync upload; the demo caps it
    while production runs the same parse asynchronously (see roadmap §16). Parses the file once."""
    # page_range=(1, N) parses only the first N pages (max_num_pages instead REJECTS longer docs)
    kwargs = {"page_range": (1, max_pages)} if max_pages else {}
    doc = DocumentConverter().convert(path, **kwargs).document
    md = doc.export_to_markdown()
    return [(None, md)], extract_table_facts(doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="owner user_id (data is isolated per user)")
    ap.add_argument("--file", required=True, help="path to the document to ingest")
    ap.add_argument("--doc-id", default=None, help="optional stable id (default: random)")
    ap.add_argument("--name", default=None, help="original filename to record (default: basename)")
    ap.add_argument("--max-pages", type=int, default=None,
                    help="cap pages parsed (demo: keep sync upload fast; prod parses all async)")
    args = ap.parse_args()

    doc_id = args.doc_id or uuid.uuid4().hex[:12]
    filename = args.name or os.path.basename(args.file)
    print(f"parsing {filename} (docling{', max '+str(args.max_pages)+' pages' if args.max_pages else ''})...")
    pages, facts = parse_document(args.file, max_pages=args.max_pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = []  # (page, text)
    for page, text in pages:
        for piece in splitter.split_text(text):
            chunks.append((page, piece))
    print(f"{len(chunks)} narrative chunks, {len(facts)} table facts; embedding ({EMB_MODEL})...")

    emb = OpenAIEmbeddings(model=EMB_MODEL)
    vectors = emb.embed_documents([c[1] for c in chunks])

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute(DDL)
        # re-ingesting the same doc_id replaces it (idempotent uploads)
        cur.execute("delete from user_chunks where user_id=%s and doc_id=%s", (args.user, doc_id))
        cur.execute("delete from user_facts where user_id=%s and doc_id=%s", (args.user, doc_id))
        for (page, text), v in zip(chunks, vectors):
            cur.execute(
                "insert into user_chunks (user_id,doc_id,filename,page,chunk_text,embedding) "
                "values (%s,%s,%s,%s,%s,%s::vector)",
                (args.user, doc_id, filename, page, text, _vec(v)))
        for (page, metric, period, value, raw, cell) in facts:
            cur.execute(
                "insert into user_facts (user_id,doc_id,filename,page,metric,period,value,raw,cell) "
                "values (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (args.user, doc_id, filename, page, metric, period, value, raw, cell))
        conn.commit()
        cur.execute("select count(*) from user_chunks where user_id=%s", (args.user,))
        nchunks = cur.fetchone()[0]
        cur.execute("select count(*) from user_facts where user_id=%s", (args.user,))
        nfacts = cur.fetchone()[0]
        print(f"ingested doc_id={doc_id} for user={args.user}; "
              f"{nchunks} chunks, {nfacts} table facts for this user.")


if __name__ == "__main__":
    main()
