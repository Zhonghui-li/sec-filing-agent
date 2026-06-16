"""Offline scorer: the P2-eval <-> P5-observability loop.

Fetches recent Langfuse traces (real production runs), scores each with the eval
suite's reference-free Ragas metrics (faithfulness + answer relevancy), and pushes
the scores back to Langfuse. Run periodically to monitor production QUALITY over
time — without adding judge-call latency/cost to user requests.

Reference-free on purpose: production questions have no gold labels, so we use the
metrics that need none (is the answer grounded in the filing text the tools returned?
is it on-topic?). The deterministic graders (numerical / citation / context_recall /
forbid / ...) stay in the offline eval set (eval/score.py), which does have gold.

Each agent run stores its retrieved filing chunks in the trace metadata
(`contexts`, set in agents/sec_agent.py); this script grounds the Ragas judge against
exactly what the agent saw. Abstain answers carry no contexts and are skipped.

Usage (with OPENAI_API_KEY + EMB_MODEL + LANGFUSE_* set):
    python -m eval.score_traces
"""
import argparse

from langfuse import get_client
from eval.quality import score_quality
from eval.judge import domain_judge

# reference-free monitor metrics pushed to each trace; adding one here backfills it onto
# already-scored traces on the next run (the guard skips a trace only if ALL are present).
RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision")
# domain_judge: our calibrated G-Eval judge (good=1.0/bad=0.0). Pushed alongside the Ragas
# metrics so a trace shows both — e.g. an honestly-hedged answer scores answer_relevancy≈0
# (the generic metric's bias) but domain_judge=1.0 (correct). It's the path off answer_relevancy.
PUSH_METRICS = frozenset(RAGAS_METRICS) | {"domain_judge"}


def main(limit=50):
    lf = get_client()
    summaries = lf.api.trace.list(limit=limit).data

    items, targets = [], []   # targets: (trace_id, metrics_already_present)
    for s in summaries:
        t = lf.api.trace.get(s.id)  # full detail (observations + scores)
        already = {sc.name for sc in (getattr(t, "scores", None) or [])}
        if PUSH_METRICS <= already:
            continue  # idempotent: skip only if ALL metrics present (else backfill the missing)
        contexts = None
        for o in (t.observations or []):
            md = getattr(o, "metadata", None) or {}
            if md.get("contexts"):
                contexts = md["contexts"]
                break
        if not contexts or not t.output:
            continue  # nothing to ground against (e.g. abstain / lookup-only answers)
        items.append({"question": t.input, "answer": t.output,
                      "contexts": contexts if isinstance(contexts, list) else [contexts]})
        targets.append((s.id, already))

    if not items:
        print("no new scorable traces.")
        return 0

    print(f"scoring {len(items)} traces ({', '.join(sorted(PUSH_METRICS))})...")
    scores = score_quality(items)
    verdicts = domain_judge(items)

    pushed = 0
    for (tid, already), sc, vd in zip(targets, scores, verdicts):
        for name in RAGAS_METRICS:
            if sc.get(name) is not None and name not in already:  # push only the missing ones
                lf.create_score(name=name, value=float(sc[name]),
                                trace_id=tid, data_type="NUMERIC")
                pushed += 1
        if "domain_judge" not in already:
            lf.create_score(name="domain_judge",
                            value=1.0 if vd["verdict"] == "good" else 0.0,
                            trace_id=tid, data_type="NUMERIC",
                            comment=(vd.get("reasoning") or "")[:480])
            pushed += 1
    lf.flush()
    print(f"pushed {pushed} scores across {len(items)} traces -> Langfuse (Scores).")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    raise SystemExit(main(ap.parse_args().limit))
