from typing import Dict, Any
from .decision import decide_route
import time


class RetrievalService:
    """
    Lazy-loading retrieval service.
    QA, Summarizer, and Reranker models are loaded on first use
    to avoid loading all ~1.6GB into memory at startup.
    """

    def __init__(self, vector_store, embedding_service):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self._qa = None
        self._summarizer = None
        self._reranker = None

    @property
    def qa(self):
        if self._qa is None:
            from .qa import QAService
            self._qa = QAService()
        return self._qa

    @property
    def summarizer(self):
        if self._summarizer is None:
            from .summarizer import SummarizerService
            self._summarizer = SummarizerService()
        return self._summarizer

    @property
    def reranker(self):
        if self._reranker is None:
            try:
                from .reranker import RerankerService
                self._reranker = RerankerService()
            except Exception as e:
                print(f"Reranker unavailable (low memory?): {e}. Using hybrid search only.")
                self._reranker = None
        return self._reranker

    def retrieve_and_answer(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        start_time = time.time()

        # 1. Decide route first — retrieval strategy differs per route
        route = decide_route(question)

        if route == "summarize":
            generic_queries = {"summarize this", "summarize", "summarise this",
                               "summarise", "summary", "give me a summary", "overview"}
            if question.lower().strip() in generic_queries:
                # Generic: sample evenly across the whole document
                all_chunks = self.vector_store.get_all_chunks()
                step = max(1, len(all_chunks) // 8)
                sources = all_chunks[::step][:8] if all_chunks else []
            else:
                # Specific reasoning question: retrieve targeted chunks via hybrid search
                query_emb = self.embedding_service.get_embedding(question)
                candidates = self.vector_store.hybrid_search(query_emb, question, top_k=top_k * 4)
                sources = [res['chunk'] for res in candidates][:6]
        else:
            # QA: precision retrieval via hybrid search + cross-encoder reranker
            query_emb = self.embedding_service.get_embedding(question)
            candidates = self.vector_store.hybrid_search(query_emb, question, top_k=top_k * 4)
            candidate_chunks = [res['chunk'] for res in candidates]
            reranker = self.reranker
            if reranker is not None:
                sources = reranker.rerank(question, candidate_chunks, top_k=top_k)
            else:
                sources = candidate_chunks[:top_k]

        context = "\n".join(sources)

        # 2. Generate answer
        if route == "qa":
            answer = self.qa.answer(question, context)
        else:
            answer = self.summarizer.summarize(question, context)

        latency = (time.time() - start_time) * 1000

        return {
            "answer": answer,
            "type": route,
            "sources": sources,
            "latency_ms": round(latency, 2),
        }
