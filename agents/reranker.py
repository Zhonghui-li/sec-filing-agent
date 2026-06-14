# agents/reranker.py
"""Cross-encoder reranker using sentence-transformers."""

from typing import List, Callable, Optional

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


def create_cross_encoder_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: Optional[int] = None,
) -> Callable[[str, List[Document]], List[Document]]:
    """
    Factory: returns a ``rerank(query, docs) -> docs`` function.

    Args:
        model_name: HuggingFace cross-encoder model id.
        top_n: If set, return only top-n after reranking.
               If None, return all docs in reranked order.

    Returns:
        A callable ``rerank(query: str, docs: List[Document]) -> List[Document]``.
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)
    print(f"[Reranker] Loaded cross-encoder: {model_name}")

    def rerank(query: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return docs

        pairs = [(query, d.page_content) for d in docs]
        scores = model.predict(pairs)

        # Attach scores and sort descending
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        limit = top_n if top_n is not None else len(scored)
        results: List[Document] = []
        for doc, score in scored[:limit]:
            meta = dict(doc.metadata) if doc.metadata else {}
            meta["rerank_score"] = float(score)
            results.append(Document(page_content=doc.page_content, metadata=meta))

        return results

    return rerank
