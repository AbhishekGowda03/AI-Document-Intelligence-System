import faiss
import numpy as np
import os
import pickle
from rank_bm25 import BM25Okapi

VECTOR_DIR = "vector_store"
INDEX_PATH = os.path.join(VECTOR_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(VECTOR_DIR, "metadata.pkl")


class VectorStore:
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata = []
        self.bm25 = None
        self.load_index()

    def _build_bm25(self):
        if self.metadata:
            tokenized = [doc.lower().split() for doc in self.metadata]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None

    def load_index(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
            try:
                self.index = faiss.read_index(INDEX_PATH)
                with open(METADATA_PATH, "rb") as f:
                    self.metadata = pickle.load(f)
                self._build_bm25()
                return True
            except Exception as e:
                print(f"Error loading index: {e}")
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                self.metadata = []
                self.bm25 = None
                return False
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.metadata = []
            self.bm25 = None
            return False

    def add_embeddings(self, embeddings: np.ndarray, chunks: list[str]):
        if len(embeddings) != len(chunks):
            raise ValueError("Mismatched counts between embeddings and chunks.")
        self.index.add(embeddings)
        self.metadata.extend(chunks)
        self._build_bm25()
        self.save_index()

    def save_index(self):
        if not os.path.exists(VECTOR_DIR):
            os.makedirs(VECTOR_DIR)
        faiss.write_index(self.index, INDEX_PATH)
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(self.metadata, f)

    def clear(self):
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = []
        self.bm25 = None
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
        if os.path.exists(METADATA_PATH):
            os.remove(METADATA_PATH)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """Dense semantic search via FAISS. Returns list of (chunk, rank)."""
        if self.index is None or self.index.ntotal == 0:
            return {}

        if len(query_embedding.shape) == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        fetch_k = min(top_k * 4, self.index.ntotal)
        _, indices = self.index.search(query_embedding, fetch_k)

        ranked = {}
        rank = 1
        seen = set()
        for idx in indices[0]:
            if idx == -1 or idx >= len(self.metadata):
                continue
            chunk = self.metadata[idx]
            if chunk in seen:
                continue
            seen.add(chunk)
            ranked[chunk] = rank
            rank += 1
            if rank > top_k:
                break
        return ranked

    def bm25_search(self, query: str, top_k: int = 5):
        """Sparse BM25 lexical search. Returns dict of (chunk, rank)."""
        if self.bm25 is None or not self.metadata:
            return {}

        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        # Get top_k indices sorted by score descending
        top_indices = np.argsort(scores)[::-1]

        ranked = {}
        rank = 1
        seen = set()
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            chunk = self.metadata[idx]
            if chunk in seen:
                continue
            seen.add(chunk)
            ranked[chunk] = rank
            rank += 1
            if rank > top_k:
                break
        return ranked

    def get_all_chunks(self) -> list[str]:
        """Returns all stored chunks in indexed order."""
        return list(self.metadata)

    def hybrid_search(self, query_embedding: np.ndarray, query: str, top_k: int = 3):
        """
        Reciprocal Rank Fusion (RRF) of dense + BM25 results.
        RRF score = 1/(k+rank_semantic) + 1/(k+rank_bm25)
        Higher score = more relevant.
        """
        K = 60  # standard RRF constant
        fetch = top_k * 3

        semantic_ranks = self.search(query_embedding, top_k=fetch)
        bm25_ranks = self.bm25_search(query, top_k=fetch)

        all_chunks = set(semantic_ranks) | set(bm25_ranks)
        rrf_scores = {}
        for chunk in all_chunks:
            s_rank = semantic_ranks.get(chunk, fetch + 1)
            b_rank = bm25_ranks.get(chunk, fetch + 1)
            rrf_scores[chunk] = 1 / (K + s_rank) + 1 / (K + b_rank)

        sorted_chunks = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        return [{"chunk": c, "rrf_score": rrf_scores[c]} for c in sorted_chunks[:top_k]]
