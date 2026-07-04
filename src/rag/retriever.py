"""Multi-strategy retriever with BM25, hybrid search, and reranking."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from src.config.settings import get_settings
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    """Multi-strategy retriever for code understanding.

    Combines vector search with keyword-based BM25 retrieval
    and optional reranking for high-quality results.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder | None = None,
        enable_bm25: bool = True,
enable_reranker: bool = True,
    ):
        self.vector_store = vector_store
        self.embedder = embedder or Embedder()
        self.enable_bm25 = enable_bm25
        self.enable_reranker = enable_reranker

        # BM25 in-memory index (rebuilt on each retrieval)
        self._bm25_index: dict[str, float] = {}
        self._doc_freqs: Counter[str] = Counter()
        self._total_docs: int = 0
        self._chunk_texts: list[str] = []
        self._chunk_ids: list[str] = []

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        n_rerank: int = 20,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant code chunks using hybrid search.

        Strategy:
        1. Vector search (semantic similarity)
        2. Optional BM25 (keyword matching)
        3. Fusion of results
        4. Optional cross-encoder reranking

        Args:
            query: Natural language or code-related query.
            top_k: Final number of results to return.
            n_rerank: Number of candidates to consider before reranking.
            where: Optional metadata filter.

        Returns:
            List of result dicts with content, metadata, and relevance score.
        """
        vector_results = self.vector_store.search(query, n_results=n_rerank, where=where)

        if self.enable_bm25:
            self._build_bm25_index(vector_results)
            bm25_results = self._bm25_search(query, top_k=n_rerank)
            results = self._fusion(vector_results, bm25_results, top_k=top_k)
        else:
            results = vector_results[:top_k]

        # Add relevance score from distance (lower distance = more relevant)
        for r in results:
            r["relevance_score"] = round(1.0 - r.get("distance", 0.0), 4)

        if self.enable_reranker and len(results) > 1:
            results = self._rerank(query, results, top_k=top_k)

        return results

    def _build_bm25_index(self, vector_results: list[dict[str, Any]]) -> None:
        """Build BM25 index from the retrieved documents."""
        self._chunk_texts = [r["content"] for r in vector_results]
        self._chunk_ids = [r["id"] for r in vector_results]
        self._total_docs = len(vector_results)

        self._doc_freqs = Counter()
        for text in self._chunk_texts:
            tokens = set(self._tokenize(text))
            for token in tokens:
                self._doc_freqs[token] += 1

    def _bm25_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Simple BM25 search for keyword matching.

        Uses the formula: BM25 = IDF * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        where f = term frequency, dl = document length, avgdl = average document length.
        """
        if not self._chunk_texts:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        avgdl = sum(len(t.split()) for t in self._chunk_texts) / max(len(self._chunk_texts), 1)
        k1, b = 1.5, 0.75
        N = max(self._total_docs, 1)

        scores: list[tuple[int, float]] = []
        for idx, text in enumerate(self._chunk_texts):
            tokens = self._tokenize(text)
            dl = len(tokens)
            score = 0.0
            tf_counter = Counter(tokens)

            for token in query_tokens:
                if token not in tf_counter:
                    continue
                f = tf_counter[token]
                df = self._doc_freqs.get(token, 1)
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))

            scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = scores[:top_k]

        results = []
        for idx, score in top_indices:
            if r := next(
                (r for r in self.vector_store.search("", n_results=500)
                 if r["id"] == self._chunk_ids[idx]),
                None,
            ):
                results.append({**r, "distance": 1.0 - score / max(s[1] for s in scores if s[1] > 0 or True)})

        return results

    def _fusion(
        self,
        vector_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion (RRF) for combining result lists."""
        k = 60  # RRF constant

        scores: dict[str, float] = {}
        results_by_id: dict[str, dict[str, Any]] = {}

        for rank, result in enumerate(vector_results):
            rid = result["id"]
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
            results_by_id[rid] = result

        for rank, result in enumerate(bm25_results):
            rid = result["id"]
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
            if rid not in results_by_id:
                results_by_id[rid] = result

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused = []
        for rid, score in ranked[:top_k]:
            result = dict(results_by_id[rid])
            result["fusion_score"] = round(score, 4)
            result["distance"] = 1.0 - score  # Normalize for consistency
            fused.append(result)

        return fused

    def _rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Rerank candidates using a cross-encoder model.

        Falls back to score-based reranking if cross-encoder is unavailable.
        """
        try:
            from sentence_transformers import CrossEncoder
            # BGE reranker - good for both English and Chinese code queries
            model = CrossEncoder("BAAI/bge-reranker-base")
            pairs = [(query, c["content"]) for c in candidates]
            scores = model.predict(pairs)
            scored = list(zip(candidates, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            reranked = []
            for c, s in scored[:top_k]:
                c["rerank_score"] = round(float(s), 4)
                c["distance"] = 1.0 - float(s)
                reranked.append(c)
            logger.debug("Reranked %d candidates -> %d results", len(candidates), len(reranked))
            return reranked
        except ImportError:
            logger.warning("CrossEncoder not available; returning top candidates by fusion score")
            return candidates[:top_k]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple tokenizer for code and natural language."""
        import re
        # Split on non-alphanumeric characters, preserve code-specific tokens
        tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|=>|==|!=|>=|<=|->", text)
        return [t.lower() for t in tokens]

    def __repr__(self) -> str:
        return (
            f"Retriever(bm25={self.enable_bm25}, "
            f"reranker={self.enable_reranker})"
        )
