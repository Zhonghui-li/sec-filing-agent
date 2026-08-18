"""Run the agent over FinanceBench, capture traces -> (a) a full ROUTING REFERENCE for the eval and
(b) REJECTION-SAMPLED real training data. Two birds:
  - real (question -> tool call) pairs from what the working agent actually did on cases it handled
    well — augments the synthetic templates with genuine human-written questions;
  - a routing reference covering ALL types (not just numeric): a numeric question is verified
    FUNCTIONALLY (answer matches gold), a narrative question's right route is search_filings.

Rejection-sampling rule (only keep GOOD routing as training/reference):
  - numeric gold + answer matches gold      -> (question, the numeric tool call it made)
  - narrative gold + it called search_filings, didn't abstain -> (question, that search call)

Usage: DATABASE_URL=... OPENAI_API_KEY=... GEN_LLM_MODEL=o4-mini python -m finetune.capture_fb_traces [--limit N]
"""
import argparse
import json
import re
from pathlib import Path

from eval.financebench.run import _nums, _abstained
from finetune.fb_routing_eval import _num_match  # reuse scale-aware match


def _gold_is_numeric(answer):
    """The gold answer is a specific figure (starts with a number / $ / paren-negative / %)."""
    a = str(answer).strip().lstrip("~≈").strip()
    return bool(re.match(r"^[-(]?\$?-?\d", a))

HERE = Path(__file__).resolve().parent
_FB_CACHE = HERE.parent / "eval" / "financebench" / "_open_cache.jsonl"
_DATA_TOOLS = {"get_financials", "get_ratio", "get_growth", "compute_formula", "get_statement",
               "largest_line_item", "get_segment_breakdown", "get_segment_growth"}


def _first_call(trace):
    """The agent's first ROUTING decision: the first data/search/abstain tool call (name + args)."""
    for step in trace or []:
        if step["tool"] in _DATA_TOOLS or step["tool"] in ("search_filings", "abstain"):
            return {"name": step["tool"], "arguments": step.get("args", {})}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    from agents.sec_agent import build_agent, run_agent

    cases = [json.loads(l) for l in _FB_CACHE.read_text().splitlines() if l.strip()]
    if args.limit:
        cases = cases[:args.limit]
    agent = build_agent()

    traces, train, reference = [], [], []
    for i, c in enumerate(cases, 1):
        q, gold, qtype = c["question"], str(c["answer"]), c.get("question_type")
        out = run_agent(q, agent=agent)
        answer, tools = out["answer"], out.get("tools_used", [])
        call = _first_call(out["trace"])
        numeric = _gold_is_numeric(gold)
        abstained = _abstained(out)

        # verdict + rejection-sampling
        if numeric:
            correct = bool(_nums(gold)) and _num_match(answer, _nums(gold)[0])
            verdict = "abstain" if abstained else ("correct" if correct else "wrong")
            # LEAK GUARD: the metrics-generated numeric questions ARE the eval set
            # (finetune.fb_routing_eval scores exactly these). Never rejection-sample them into
            # training, or the fine-tuned model would train on its own test — inflated, invalid.
            # These questions still get a routing reference; they just never become training pairs.
            ref_tool = call["name"] if call else ("abstain" if abstained else None)
        else:  # narrative: PURE narrative -> search_filings; a "judgment on a metric" question
            # (capital-intensive? improving margin?) legitimately routes to a numeric tool + reasoning,
            # which we can't cheaply verify -> mark 'ambiguous' and exclude it from the clean reference.
            routed_search = "search_filings" in tools
            verdict = "abstain" if abstained else ("routed_search" if routed_search else "other")
            if verdict == "routed_search" and call and call["name"] == "search_filings":
                train.append({"question": q, "tool": call})
            ref_tool = ("abstain" if abstained else
                        "search_filings" if routed_search else "ambiguous")

        reference.append({"question": q, "question_type": qtype, "gold": gold,
                          "reference_tool": ref_tool, "verdict": verdict})
        traces.append({"question": q, "question_type": qtype, "gold": gold, "answer": answer[:200],
                       "tools_used": tools, "first_call": call, "verdict": verdict})
        print(f"  [{i:>3}/{len(cases)}] {verdict:13} {qtype:18} {(call or {}).get('name','-'):16} {q[:48]}")

    (HERE / "fb_traces.jsonl").write_text("\n".join(json.dumps(t) for t in traces) + "\n")
    (HERE / "routing_train_real.jsonl").write_text("\n".join(json.dumps(t) for t in train) + "\n")
    (HERE / "fb_reference.jsonl").write_text("\n".join(json.dumps(r) for r in reference) + "\n")

    from collections import Counter
    print(f"\ncaptured {len(traces)} | rejection-sampled train pairs {len(train)}")
    print("verdicts:", dict(Counter(t["verdict"] for t in traces)))
    print(f"wrote fb_traces.jsonl, routing_train_real.jsonl ({len(train)}), fb_reference.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
