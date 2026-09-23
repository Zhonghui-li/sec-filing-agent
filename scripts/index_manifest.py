#!/usr/bin/env python3
"""Record what the filings index contains, so its state can be asserted instead of assumed.

`filing_chunks` is filled lazily: a question about a company-year that isn't indexed triggers
`ingest_ticker`, which fetches, chunks, embeds and inserts it. Nothing else writes to it. So its
contents are a by-product of which questions happened to be asked, and there is no record anywhere
of what it is supposed to hold — the equivalent of a venv with no requirements.txt. It works until
something has to be rebuilt, and then the coverage is gone along with the knowledge of what the
coverage was.

That matters most for evaluation. A retrieval-dependent case behaves differently depending on
whether its company-year happens to be indexed, so the same commit can score differently on two
days and there is no way to tell a code change from a cache state. The LRU cap makes this live
rather than theoretical: at FILING_CHUNKS_MAX the next new company evicts an existing filing.

Two distinctions the output keeps separate, because only one of them is a problem:

  PINNED entries are year-aware ingests — a specific fiscal year's 10-K, immutable, reachable only
  when somebody asks about that year. They disappear only by LRU eviction, and they are the
  coverage that is expensive to get back.

  UNPINNED entries are the perishable latest-year snapshot. They are SUPPOSED to churn: a TTL
  prunes them so the next query re-fetches the newest filing, and a new 10-K moves the year. Drift
  here is the system working.

    python scripts/index_manifest.py dump     # write data/index_manifest.json (commit it)
    python scripts/index_manifest.py check    # compare the live index against it

`check` exits 1 when pinned coverage has been lost, so it can gate an evaluation run. Losing
unpinned coverage is reported and does not fail: it self-heals on the next query. This does not
restore anything — restoring is a separate problem, and a harder one, because ingest_ticker skips
whatever is already present and its no-year path only fetches the latest year.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg

OUT = Path(__file__).resolve().parent.parent / "data" / "index_manifest.json"


def read_index(dsn):
    """[{ticker, fiscal_year, pinned, chunks, accessions}] — one entry per indexed company-year."""
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("""select ticker, fiscal_year, pinned, count(*), count(distinct accession)
                       from filing_chunks group by 1, 2, 3 order by 1, 2, 3""")
        return [{"ticker": t, "fiscal_year": fy, "pinned": p, "chunks": n, "accessions": a}
                for t, fy, p, n, a in cur.fetchall()]


def key(e):
    return (e["ticker"], e["fiscal_year"], e["pinned"])


def dump(dsn):
    entries = read_index(dsn)
    OUT.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "filing_chunks_max": int(os.environ.get("FILING_CHUNKS_MAX", "20000")),
        "total_chunks": sum(e["chunks"] for e in entries),
        "entries": entries,
    }, indent=2) + "\n")
    pinned = sum(1 for e in entries if e["pinned"])
    print(f"{OUT.relative_to(OUT.parent.parent)}: {len(entries)} company-years "
          f"({pinned} pinned), {sum(e['chunks'] for e in entries)} chunks")
    return 0


def check(dsn):
    if not OUT.exists():
        print(f"no manifest at {OUT} — run `dump` first")
        return 1
    saved = json.loads(OUT.read_text())
    want = {key(e): e for e in saved["entries"]}
    have = {key(e): e for e in read_index(dsn)}

    lost = sorted(k for k in want if k not in have)
    new = sorted(k for k in have if k not in want)
    lost_pinned = [k for k in lost if k[2]]

    total = sum(e["chunks"] for e in have.values())
    cap = saved["filing_chunks_max"]
    print(f"manifest {saved['generated_at']} · {len(want)} company-years -> live {len(have)}")
    print(f"chunks {total}/{cap} ({total / cap * 100:.1f}% of the LRU cap)")

    for k in lost:
        print(f"  LOST    {k[0]:6} FY{k[1]} {'pinned' if k[2] else 'latest'} "
              f"({want[k]['chunks']} chunks)")
    for k in new:
        print(f"  NEW     {k[0]:6} FY{k[1]} {'pinned' if k[2] else 'latest'} "
              f"({have[k]['chunks']} chunks) — re-dump to record it")
    if not lost and not new:
        print("  in sync")

    if lost_pinned:
        print(f"\nevicted pinned coverage: {len(lost_pinned)} — historical company-years an "
              f"evaluation may depend on, back only when somebody asks for that year again.")
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["dump", "check"])
    args = ap.parse_args()
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL is not set")
        return 1
    return dump(dsn) if args.mode == "dump" else check(dsn)


if __name__ == "__main__":
    sys.exit(main())
