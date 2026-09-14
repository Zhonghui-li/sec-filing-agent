#!/usr/bin/env python3
"""Read the metric miss queue and answer the two questions it exists to answer.

1. WHICH METRICS TO ADD NEXT — the queue's original job. Real traffic asking for something the
   tools can't return is a better expansion signal than guessing at the METRICS table.

2. IS THE PROSE MAGNITUDE GUARD NEEDED AGAIN — the reversal trigger. That guard used to replace
   any answer holding an out-of-bound ratio; it was downgraded to detection-only because the
   failure it caught (the model hand-composing a ratio through `compute`) stopped happening once
   get_ratio and compute_formula covered those metrics, while it kept destroying correct answers
   over a multiplication sign or a fiscal year. An `implausible_magnitude` miss whose tools include
   `compute` means that failure is back — and this time with a real case to write the rule against.
   See agents/guardrail.py:_note_implausible_prose.

    python scripts/check_misses.py                      # the local queue
    gcloud logging read 'textPayload:METRIC_MISS' --project sec-filing-agent --limit 500 \
        --format='value(textPayload)' | python scripts/check_misses.py -   # production
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent.parent / "data" / "cache" / "metric_misses.jsonl"


def _records(lines):
    for line in lines:
        line = line.strip()
        if "METRIC_MISS " in line:            # a Cloud Logging payload carries the prefix
            line = line.split("METRIC_MISS ", 1)[1]
        if line.startswith("{"):
            try:
                yield json.loads(line)
            except ValueError:
                continue


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", default=str(DEFAULT), help="jsonl file, or - for stdin")
    args = ap.parse_args()

    src = sys.stdin if args.path == "-" else open(args.path)
    with src as f:
        rows = list(_records(f))
    if not rows:
        print(f"no misses in {args.path}")
        return 0

    print(f"{len(rows)} misses in {args.path}\n")
    print("top requested-but-missing metrics:")
    for (metric, reason), n in Counter((r["metric"], r.get("reason", "")) for r in rows
                                       if r["metric"] != "implausible_magnitude").most_common(10):
        print(f"  {n:>4}  {metric:<28} {reason}")

    impl = [r for r in rows if r["metric"] == "implausible_magnitude"]
    hand = [r for r in impl if "'compute'" in r.get("reason", "")]
    print(f"\nimplausible_magnitude: {len(impl)} recorded, {len(hand)} with `compute` in the trace")
    for r in hand[-10:]:
        print(f"  {r['ts']}  {r['reason']}")
    if hand:
        print("\n  ^ REVERSAL TRIGGER: the model is hand-composing ratios again. Re-enable the"
              "\n    block in agents/guardrail.py, written against these cases.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
