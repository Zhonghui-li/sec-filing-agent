"""Domain-tuned LLM judge (G-Eval style) for qualitative answer quality.

Replaces the off-the-shelf Ragas faithfulness/answer_relevancy judges, whose generic
assumptions misfire in a regulated domain — e.g. answer_relevancy's noncommittal classifier
penalizes the honest "this remains uncertain / see the filing" hedging that a compliant
answer SHOULD use (verified: a fully-grounded JPMorgan answer scored relevancy 0.0).

The fix (per G-Eval / RAGalyst): own the judge instead of borrowing it. The rubric encodes
the domain norms explicitly (hedging = good, appropriate abstention = good, every claim must
sit in the retrieved context); the judge reasons step by step (CoT) before a binary verdict.
Calibrated against HUMAN labels via Cohen's kappa (run this file).

Unlike the production eval-score loop, this judge reads the FULL retrieved context (no [:1500]
truncation), so it doesn't penalize claims that the truncated trace metadata happened to drop.

Usage (OPENAI_API_KEY set):
    python -m eval.judge            # run the calibration set, print kappa + confusion
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_openai import ChatOpenAI

JUDGE_MODEL = os.environ.get("DOMAIN_JUDGE_MODEL", "gpt-4o-mini")
ROOT = Path(__file__).resolve().parent.parent

RUBRIC = """You are a senior financial-compliance reviewer auditing an AI assistant that \
answers questions about public companies' SEC 10-K filings. You are given the QUESTION, the \
RETRIEVED CONTEXT (verbatim filing excerpts the assistant was given), and the assistant's \
ANSWER. Decide whether the answer is GOOD or BAD under these domain rules.

GOOD requires ALL of:
1. GROUNDED: every factual claim — every number, statistic, percentage, date, and named \
entity (company, product, regulator) — is supported by the RETRIEVED CONTEXT. The assistant \
may paraphrase and summarize, but must NOT introduce facts from outside the context (i.e. \
from its own general knowledge).
2. HONEST HEDGING IS GOOD, never a defect: "this remains uncertain", "the proposal is not \
yet finalized", "see the filing for detail" are appropriate and desirable in this domain.
3. APPROPRIATE ABSTENTION IS GOOD: if the topic/figure is not in the filings, or the request \
is out of scope (a company not covered) or off-topic (investment advice, forecasts/predictions, \
creative writing, real-time stock prices), then DECLINING to answer is the correct behavior.
4. ON-TOPIC: it addresses what was asked.

BAD if ANY of:
- A claim (number, stat, fact, commitment, named entity) is NOT supported by the retrieved \
context — fabricated or recalled from the model's own knowledge.
- A claim CONTRADICTS the retrieved context.
- It attributes another company's facts, or invents a specific figure/percentage/market share \
not present in the context.
- It FABRICATES a substantive answer when it should have abstained.
- It is off-topic or evasive on a question it could have answered from the context.

Think step by step:
1. List the answer's main factual claims (facts, numbers, named entities).
2. For each, check whether the RETRIEVED CONTEXT supports it. Be strict about specific \
numbers, percentages, and named entities — these are the common hallucinations.
3. If the answer is an abstention/refusal, judge whether abstaining was appropriate.
4. Collect any claim that is unsupported or contradicted.

Respond ONLY with a JSON object:
{"reasoning": "<2-4 sentences>", "unsupported_claims": ["<claim>", ...], "verdict": "GOOD" or "BAD"}"""


def _judge_one(llm, item):
    ctx = "\n\n---\n\n".join(item.get("contexts") or []) or \
        "(no retrieved context — the assistant abstained or the question was off-topic)"
    user = (f"QUESTION:\n{item['question']}\n\n"
            f"RETRIEVED CONTEXT:\n{ctx}\n\n"
            f"ANSWER:\n{item['answer']}")
    txt = llm.invoke([("system", RUBRIC), ("user", user)]).content.strip()
    try:
        d = json.loads(txt)
        verdict = str(d.get("verdict", "")).upper()
    except Exception:
        d, verdict = {"reasoning": txt[:200]}, txt.upper()
    return {"verdict": "bad" if "BAD" in verdict else "good",
            "reasoning": d.get("reasoning", ""),
            "unsupported": d.get("unsupported_claims", [])}


def domain_judge(items):
    """items: [{question, answer, contexts}]. Returns [{verdict: good|bad, reasoning, unsupported}]."""
    llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0,
                     model_kwargs={"response_format": {"type": "json_object"}})
    with ThreadPoolExecutor(max_workers=4) as ex:
        return list(ex.map(lambda it: _judge_one(llm, it), items))


def _kappa(gold, pred):
    n = len(gold)
    po = sum(g == p for g, p in zip(gold, pred)) / n
    gb, pb = gold.count("bad") / n, pred.count("bad") / n
    gg, pg = 1 - gb, 1 - pb
    pe = gg * pg + gb * pb
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po


def main():
    real = json.loads((ROOT / "eval/labeling/judge_calibration.json").read_text())
    syn = json.loads((ROOT / "eval/labeling/synthetic_bad.json").read_text())
    # gold: real cases use the adjudicated draft_label (39 good + C18 bad); synthetic are bad.
    items = real + syn
    gold = [r["draft_label"] for r in real] + [s["label"] for s in syn]

    print(f"judging {len(items)} items ({gold.count('good')} good / {gold.count('bad')} bad) "
          f"with {JUDGE_MODEL}...")
    out = domain_judge(items)
    pred = [o["verdict"] for o in out]

    kappa, po = _kappa(gold, pred)
    tp = sum(g == "bad" and p == "bad" for g, p in zip(gold, pred))
    tn = sum(g == "good" and p == "good" for g, p in zip(gold, pred))
    fp = sum(g == "good" and p == "bad" for g, p in zip(gold, pred))
    fn = sum(g == "bad" and p == "good" for g, p in zip(gold, pred))
    print(f"\n=== Cohen's kappa = {kappa:.3f}  (accuracy {po:.3f}) ===")
    print(f"  confusion: TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"  bad recall  (caught/total bad)  : {tp}/{tp + fn}")
    print(f"  good pass   (passed/total good) : {tn}/{tn + fp}")

    print("\n=== disagreements (judge vs gold) ===")
    for it, g, o in zip(items, gold, out):
        if o["verdict"] != g:
            tag = "FALSE-POS (judge said bad, gold good)" if g == "good" else \
                  "FALSE-NEG (judge said good, gold bad)"
            print(f"  {it['id']} [{tag}] {it['question'][:50]}")
            print(f"      judge: {o['reasoning'][:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
