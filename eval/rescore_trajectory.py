#!/usr/bin/env python3
"""Re-score the trajectory metrics of a recorded run against the CURRENT annotations.

A declared path can miss a route the agent legitimately takes: L27's annotation named a metric
alias the tools normalise away, and M06 was annotated as deriving gross profit from components
when Tesla reports it directly and the agent found the cleaner route. Both times the annotation
was wrong and the agent was right. Finding that out is most of what a first run is for — but
fixing the annotation afterwards would mean paying for every agent call again, because the scores
alone cannot be recomputed.

So eval/score.py records each case's tool calls, and this replays them. Fix an annotation, re-run
this, and the corrected numbers come back for free.

    python -m eval.rescore_trajectory eval/runs/20260924-1200        # every run in the directory

Prints the per-run trajectory rates and, for each case, whether the correction changed it.
"""
import argparse
import json
import sys
from pathlib import Path

from eval.trajectory import EFFICIENCY_METRICS, TRAJECTORY_METRICS, score_trajectory

ROOT = Path(__file__).resolve().parent.parent
TESTSET = ROOT / "eval" / "testset.jsonl"
METRICS = TRAJECTORY_METRICS + EFFICIENCY_METRICS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="a directory written by score.py --run-dir")
    args = ap.parse_args()

    cases = {json.loads(l)["id"]: json.loads(l) for l in TESTSET.open()}
    files = sorted(Path(args.run_dir).glob("run*.jsonl"))
    if not files:
        print(f"no run*.jsonl under {args.run_dir}")
        return 1

    changed = {}
    for path in files:
        recs = [json.loads(l) for l in path.open()]
        if "trace" not in (recs[0] if recs else {}):
            print(f"{path.name}: recorded before traces were kept — cannot re-score")
            continue
        rates = {}
        for rec in recs:
            case = cases.get(rec["id"])
            if case is None:                    # a case deleted or renamed since the run
                continue
            now = score_trajectory(case, rec["trace"])
            for m in METRICS:
                if now[m] is not None:
                    rates.setdefault(m, []).append(now[m])
                was = rec["metrics"].get(m)
                if was != now[m] and not (was is None and now[m] is None):
                    changed.setdefault(rec["id"], set()).add(m)
        print(f"\n{path.name}")
        for m in METRICS:
            if rates.get(m):
                v = rates[m]
                print(f"  {m:16} {sum(v)}/{len(v)} = {sum(v) / len(v) * 100:5.1f}%")

    print("\n=== cases whose trajectory score changed under the current annotations ===")
    if changed:
        for cid, ms in sorted(changed.items()):
            print(f"  {cid:6} {', '.join(sorted(ms))}")
    else:
        print("  none — the annotations in use are the ones the run was scored with")
    return 0


if __name__ == "__main__":
    sys.exit(main())
