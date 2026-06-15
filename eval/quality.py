"""LLM-judged quality metrics (Ragas) for the qualitative answers — the only place an
LLM JUDGE is used. faithfulness = is the narrative grounded in the retrieved filing
text (no hallucination); answer_relevancy = is it on-topic. Reused from the Slug Advisor
stack, with the concurrency throttle that avoids judge-API TimeoutError -> NaN.

Deterministic metrics (numerical/citation/grounded/...) need no judge; these two are
opt-in via `score.py --quality` because they add LLM judge calls (cost + latency + noise).

TODO (deferred): context_recall — needs per-qualitative-case gold-passage labels; it's
what tells us whether to add BM25 back to retrieval.
"""
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (Faithfulness, ResponseRelevancy,
                           LLMContextPrecisionWithoutReference)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

JUDGE_MODEL = os.environ.get("RAGAS_JUDGE_MODEL", "gpt-4o-mini")


def score_quality(items):
    """items: [{question, answer, contexts: list[str]}]. Returns aligned
    [{faithfulness, answer_relevancy, context_precision}] (values may be None if the judge
    timed out). context_precision is rank-aware (mean precision@k) — rewards the retriever
    for ranking the chunks that support the answer higher; reference-free (judges each
    retrieved chunk's usefulness to the response, so no gold passage labels needed)."""
    judge = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0))
    emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=os.environ.get("EMB_MODEL", "text-embedding-3-small")))
    samples = [SingleTurnSample(user_input=it["question"], response=it["answer"] or "",
                                retrieved_contexts=it["contexts"] or ["(no context)"])
               for it in items]
    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=[Faithfulness(llm=judge), ResponseRelevancy(llm=judge, embeddings=emb),
                 LLMContextPrecisionWithoutReference(llm=judge)],
        show_progress=False,
        run_config=RunConfig(timeout=180, max_workers=4),  # throttle -> avoid NaN timeouts
    )
    df = result.to_pandas()
    fcol = next((c for c in df.columns if "faith" in c.lower()), None)
    rcol = next((c for c in df.columns if "relevan" in c.lower()), None)
    pcol = next((c for c in df.columns if "precision" in c.lower()), None)
    out = []
    for _, r in df.iterrows():
        def num(col):
            try:
                f = float(r.get(col) if col else None)
            except (TypeError, ValueError):
                return None
            return None if f != f else round(f, 3)   # f != f detects NaN
        out.append({"faithfulness": num(fcol), "answer_relevancy": num(rcol),
                    "context_precision": num(pcol)})
    return out
