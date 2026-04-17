from sentence_transformers import CrossEncoder
import torch


class RerankerService:
    """
    Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

    Unlike the bi-encoder (MiniLM) which embeds query and chunk separately,
    a cross-encoder reads (query, chunk) together — giving it full attention
    over both. This lets it understand synonyms, paraphrase, and context
    that bi-encoders miss.

    Pipeline:
        BM25 + Semantic → RRF (candidate pool)  →  Cross-Encoder (final rank)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, chunks: list[str], top_k: int = 3) -> list[str]:
        """
        Score every (query, chunk) pair and return top_k chunks by relevance.
        Returns chunks ordered best-first.
        """
        if not chunks:
            return []

        pairs = [(query, chunk) for chunk in chunks]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in ranked[:top_k]]
