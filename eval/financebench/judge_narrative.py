"""Correctness of the agent's NARRATIVE answers on FinanceBench, two complementary judges:

  (X) CORRECTNESS-vs-GOLD — the industry-standard way to score free-form QA: an LLM judge is shown
      the QUESTION, the FinanceBench GOLD answer, and the agent's ANSWER, and decides whether the
      answer AGREES with gold (CORRECT / PARTIAL / INCORRECT). This is exactly how FinanceBench's own
      paper scores open-ended answers. NOT κ-calibrated here (honest gap) — it's an accuracy read.
  (Y) GROUNDEDNESS — reuses eval.judge.domain_judge (κ=0.76 calibrated): is every claim supported by
      the retrieved context / is abstention appropriate. Measures "does not fabricate" (the trust bar),
      independent of whether it matches gold.

Step 1 (capture): re-run the agent over the narrative FinanceBench questions, saving the FULL answer
and retrieved contexts (the cached/prior files truncate both). Step 2 (judge): run both judges.

Usage: OPENAI_API_KEY=... GEN_LLM_MODEL=o4-mini python -m eval.financebench.judge_narrative [--limit N]
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_openai import ChatOpenAI

HERE = Path(__file__).resolve().parent
_CACHE = HERE / "_open_cache.jsonl"
_CAP = HERE / "_narrative_answers.jsonl"      # full answers + contexts we capture
JUDGE_MODEL = os.environ.get("DOMAIN_JUDGE_MODEL", "gpt-4o-mini")

CORRECTNESS_RUBRIC = """You are grading an AI assistant's answer to a question about a company's SEC \
filings, against a reference (GOLD) answer written by financial analysts. Decide whether the \
assistant's ANSWER is factually correct RELATIVE TO THE GOLD answer.

Judge on substance, not wording or extra detail:
- CORRECT: the answer reaches the same conclusion / reports the same key fact as gold. For a \
yes/no or directional question (improving? healthy? capital-intensive?), the DIRECTION/verdict must \
match. For a figure, it must match gold within a small rounding/convention tolerance (~2-3%), OR a \
larger gap is acceptable ONLY if it clearly stems from a stated, defensible convention difference \
(e.g. average vs ending balance) and the conclusion is unchanged.
- PARTIAL: the main conclusion matches but a material number is off beyond convention, OR it answers \
only part of a multi-part question.
- INCORRECT: the conclusion/direction contradicts gold, the key figure is materially wrong, or it \
answers a different thing. An abstention/refusal ("I can't determine…") when gold HAS an answer is \
INCORRECT (it failed to answer).

Respond ONLY with JSON: {"reasoning":"<1-3 sentences>","verdict":"CORRECT"|"PARTIAL"|"INCORRECT"}"""


def _correctness_one(llm, item):
    user = (f"QUESTION:\n{item['question']}\n\nGOLD ANSWER:\n{item['gold']}\n\n"
            f"ASSISTANT ANSWER:\n{item['answer']}")
    txt = llm.invoke([("system", CORRECTNESS_RUBRIC), ("user", user)]).content.strip()
    try:
        d = json.loads(txt)
        v = str(d.get("verdict", "")).upper()
    except Exception:
        d, v = {"reasoning": txt[:200]}, txt.upper()
    verdict = "correct" if "CORRECT" in v and "IN" not in v else ("partial" if "PARTIAL" in v else "incorrect")
    return {"verdict": verdict, "reasoning": d.get("reasoning", "")}


def correctness_judge(items):
    llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0,
                     model_kwargs={"response_format": {"type": "json_object"}})
    with ThreadPoolExecutor(max_workers=4) as ex:
        return list(ex.map(lambda it: _correctness_one(llm, it), items))


def capture(limit=None):
    """Re-run the agent over the NARRATIVE questions, saving full answer + contexts."""
    from agents.sec_agent import build_agent, run_agent
    cases = [json.loads(l) for l in _CACHE.read_text().splitlines() if l.strip()]
    narrative = [c for c in cases if c.get("question_type") != "metrics-generated"]
    if limit:
        narrative = narrative[:limit]
    agent = build_agent()
    rows = []
    for i, c in enumerate(narrative, 1):
        out = run_agent(c["question"], agent=agent)
        ctxs, tool_outs = [], []
        for step in out.get("trace", []):
            o = step.get("output")
            if not o:
                continue
            if step["tool"] == "search_filings":
                ctxs.append(str(o))                       # retrieved prose
            elif step["tool"] != "abstain":
                tool_outs.append(f"[{step['tool']}] {o}")  # numeric/deterministic tool output (grounds figures)
        rows.append({"question": c["question"], "question_type": c.get("question_type"),
                     "gold": str(c["answer"]), "answer": out["answer"],
                     "contexts": ctxs, "tool_outputs": tool_outs})
        print(f"  captured [{i}/{len(narrative)}] {c['question'][:60]}")
    _CAP.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {_CAP.name} ({len(rows)} full narrative answers)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--judge-only", action="store_true", help="skip capture, judge existing _narrative_answers.jsonl")
    args = ap.parse_args()

    if args.judge_only and _CAP.exists():
        rows = [json.loads(l) for l in _CAP.read_text().splitlines() if l.strip()]
    else:
        rows = capture(args.limit)

    from eval.judge import domain_judge
    corr = correctness_judge(rows)
    # grounding: feed BOTH retrieved prose AND numeric tool outputs, so a figure grounded in a tool
    # (get_ratio/get_financials) isn't falsely judged ungrounded just because it's not in the prose.
    grnd = domain_judge([{"question": r["question"], "answer": r["answer"],
                          "contexts": (r["contexts"] or []) + (r.get("tool_outputs") or [])}
                         for r in rows])

    from collections import Counter
    cc = Counter(c["verdict"] for c in corr)
    gg = Counter(g["verdict"] for g in grnd)
    n = len(rows)
    strict = cc["correct"] / n
    lenient = (cc["correct"] + cc["partial"]) / n
    print(f"\n=== NARRATIVE ({n} answers) ===")
    print(f"X · CORRECTNESS-vs-GOLD  correct={cc['correct']} partial={cc['partial']} incorrect={cc['incorrect']}")
    print(f"    strict (correct/n)          = {strict:.0%}")
    print(f"    lenient (correct+partial/n) = {lenient:.0%}")
    print(f"Y · GROUNDEDNESS (κ=0.76)  good={gg['good']} bad={gg['bad']}  ->  {gg['good']/n:.0%} grounded")

    out = [{**r, "correctness": corr[i]["verdict"], "correctness_why": corr[i]["reasoning"],
            "grounded": grnd[i]["verdict"]} for i, r in enumerate(rows)]
    (HERE / "_narrative_judged.jsonl").write_text("\n".join(json.dumps(o) for o in out) + "\n")
    print(f"wrote _narrative_judged.jsonl (per-question verdicts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
