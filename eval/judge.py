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

# Still gpt-4o — but the calibration behind that choice no longer holds, and the obvious
# replacement turned out to be wrong for half of what this judge actually does.
#
# The 33-item balanced set this was calibrated on is saturated: five candidates were run over it
# and three tied at kappa=1.000, so it cannot tell models apart, and the held-out set that caught
# overfitting in 2026-07 was never committed. PHANTOM (NeurIPS 2025, 10-K/DEF14A, human-labelled)
# replaced it — 200 balanced items to choose on, a disjoint 200 held back — and on the sealed slice
# gpt-5-mini reached kappa=0.820 against gpt-4o's 0.660, 21 items better and 5 worse, p=0.0012.
#
# That result is real and does NOT transfer. Switched to gpt-5-mini, the 95 narrative answers went
# from 96% grounded to 54%, and the failures share a shape: the numbers are right and the
# EVALUATIVE conclusion is what gets failed — "liquidity is healthy", "margin is improving",
# "inventory is not a material concern". Half our narrative questions ask for exactly that kind of
# judgement (48 of 95; they fail at 56% against 36% for the factual ones), and PHANTOM contains
# none of it: it builds a hallucination by perturbing a fact, which cannot produce "every fact
# correct, conclusion overreaches". It also contains no abstentions, which is what the
# zero-false-positives-on-hedged-answers claim rests on.
#
# This is the same strictness that was deliberately rolled back in 2026-07, when tightening the
# rubric to catch evaluative claims scored kappa=1.0 on the tuning set and 0.40 on held-out, false-
# failing three grounded answers. The rubric was never un-rolled-back — the words are identical.
# A different model simply reads them more strictly, which is to say part of this rubric's leniency
# was always the model's disposition rather than the text.
#
# So: gpt-4o stays until an evaluative-answer labelled set exists. DOMAIN_JUDGE_MODEL switches it
# for anyone who wants the stronger factual detector and accepts that cost. Do not re-decide this
# from the PHANTOM slices already spent — their row numbers are in
# eval/labeling/phantom_slices_spent.json (400 of the 994 in the 10k seed; 594 remain there, plus
# the 8k/497k/def14a seeds). They were nearly lost a second way: they lived only in a scratch
# directory the system reaped, so this warning pointed at nothing until they were regenerated from
# their seeds and committed. Reusing them
# turns held-out back into tuning data, which is how the previous held-out was lost in the first
# place.
JUDGE_MODEL = os.environ.get("DOMAIN_JUDGE_MODEL", "gpt-4o")
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
    # NOTE: this judge is pinned to gpt-4o because kappa=0.76 is calibrated against it, and one
    # call carrying retrieved passages plus a full financial statement runs to ~30k tokens. On an
    # OpenAI Tier-1 account gpt-4o allows 30,000 TPM, so a SINGLE call spends the whole minute and
    # the 95-question narrative set cannot complete. Lowering concurrency, raising retries and
    # pacing the sends were all tried and none of them help — the request itself is the budget.
    # The fix is account tier (Tier 2, reached at $50 cumulative spend, raises gpt-4o to 450k TPM),
    # not anything in this file. Do not "solve" it by switching to gpt-4o-mini: that judge is
    # systematically over-strict (kappa 0.61) and the calibration would no longer hold.
    # gpt-5 and the o-series reject an explicit temperature, so pass it only where it is accepted.
    kw = {"model": JUDGE_MODEL, "model_kwargs": {"response_format": {"type": "json_object"}}}
    if not JUDGE_MODEL.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")):
        kw["temperature"] = 0
    llm = ChatOpenAI(**kw)
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
    # Use the BALANCED set (15 good / 18 bad) — it is the honest calibration number. The alternative
    # real+synthetic set is good-heavy (39/13) and inflates kappa (0.76 balanced vs 0.947 imbalanced,
    # the same easy-set inflation seen on the correctness judge).
    items = json.loads((ROOT / "eval/labeling/judge_calibration_balanced.json").read_text())
    gold = [r["label"] for r in items]

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
