"""
Hybrid retriever: dense (ChromaDB) + sparse (BM25) with RRF fusion and cross-encoder reranking.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config.settings import DEFAULT_SCORE_THRESHOLD, DEFAULT_TOP_K, RERANKER_MODEL_NAME, RRF_K
from rag.embeddings import EmbeddingManager
from rag.vectorstore import VectorStore


def build_bm25_index(vector_store: VectorStore) -> tuple[BM25Okapi | None, list]:
    """
    Pull every document from *vector_store* and build a BM25 index.

    Building from the ChromaDB store (rather than the original list of chunks)
    guarantees that BM25 document IDs are in exact 1-to-1 correspondence with
    the dense-search IDs, which is a precondition for correct RRF fusion.

    Returns:
        (bm25_index, bm25_store)  where ``bm25_store`` is a list of
        ``{"id": str, "content": str, "metadata": dict}`` dicts.
    """
    _tokenize = lambda t: re.findall(r"\w+", t.lower())
    all_docs = vector_store.collection.get()

    if not all_docs.get("documents"):
        print("⚠️ BM25 index skipped — vector store is empty")
        return None, []

    bm25_index = BM25Okapi([_tokenize(t) for t in all_docs["documents"]])
    bm25_store = [
        {"id": doc_id, "content": text, "metadata": meta}
        for doc_id, text, meta in zip(
            all_docs["ids"], all_docs["documents"], all_docs["metadatas"]
        )
    ]
    print(f"✅ BM25 index built — {len(bm25_store)} documents")
    return bm25_index, bm25_store


def load_reranker(model_name: str = RERANKER_MODEL_NAME) -> CrossEncoder:
    """Load and return a CrossEncoder reranker model."""
    reranker = CrossEncoder(model_name)
    print(f"✅ Reranker loaded: {model_name}")
    return reranker


def rerank_documents(
    query: str,
    docs: list,
    reranker: CrossEncoder,
    top_n: int = DEFAULT_TOP_K,
    min_rerank_score: float = -2.0,
) -> list:
    """
    Rerank *docs* using a CrossEncoder and return the top-*top_n* results.

    Documents scoring below *min_rerank_score* are filtered out (deeply
    negative scores indicate irrelevant matches).

    Args:
        reranker:          CrossEncoder instance.
        min_rerank_score:  Floor; docs below this scalar score are dropped.
    """

    def _get_text(d) -> str:
        if hasattr(d, "page_content"):
            return d.page_content
        if isinstance(d, dict):
            return d.get("content", str(d))
        return str(d)

    pairs = [(query, _get_text(doc)) for doc in docs]
    scores = reranker.predict(pairs)

    reranked: list = []
    for doc, score in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_n]:
        if score < min_rerank_score:
            continue
        if isinstance(doc, dict):
            doc = {**doc, "rerank_score": float(score)}
        reranked.append(doc)

    return reranked


class RAGRetriever:
    """
    Hybrid retriever combining dense (ChromaDB) + BM25 sparse search via
    Reciprocal Rank Fusion, with optional cross-encoder reranking.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
        reranker: CrossEncoder | None = None,
        bm25_index: BM25Okapi | None = None,
        bm25_store: list | None = None,
    ):
        self.vector_store      = vector_store
        self.embedding_manager = embedding_manager
        self.reranker          = reranker
        self.bm25_index        = bm25_index
        self.bm25_store        = bm25_store

    # ── BM25 sparse search ────────────────────────────────────────────────────

    def _bm25_search(self, query: str, top_k: int) -> list:
        tokens = re.findall(r"\w+", query.lower())
        scores = self.bm25_index.get_scores(tokens)
        top_idx = scores.argsort()[::-1][:top_k]
        return [
            {
                **self.bm25_store[i],
                "similarity_score": float(scores[i]),
                "distance": 0.0,
                "rank": rank + 1,
            }
            for rank, i in enumerate(top_idx)
            if scores[i] > 0
        ]

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────────

    def _rrf_merge(self, dense: list, sparse: list, k: int = RRF_K) -> list:
        rrf_scores: Dict[str, float] = {}
        all_docs:   Dict[str, dict]  = {}

        for rank, doc in enumerate(dense):
            rrf_scores[doc["id"]] = rrf_scores.get(doc["id"], 0) + 1 / (k + rank + 1)
            all_docs[doc["id"]] = doc

        for rank, doc in enumerate(sparse):
            rrf_scores[doc["id"]] = rrf_scores.get(doc["id"], 0) + 1 / (k + rank + 1)
            all_docs.setdefault(doc["id"], doc)

        return sorted(all_docs.values(), key=lambda d: rrf_scores[d["id"]], reverse=True)

    # ── Main retrieve ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        metadata_filter: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant documents for *query*.

        1. Dense search against ChromaDB.
        2. BM25 sparse search (when index is available) + RRF fusion.
        3. Cross-encoder reranking (when reranker is available).

        Args:
            top_k:            Number of documents to return.
            score_threshold:  Minimum cosine similarity to include a dense result.
            metadata_filter:  ChromaDB ``where`` filter dict.
        """
        initial_k = top_k * 4 if self.reranker else top_k

        # Dense search
        query_embedding = self.embedding_manager.generate_embeddings([query], is_query=True)[0]
        raw = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=initial_k,
            where=metadata_filter,
        )

        dense_docs: List[Dict[str, Any]] = []
        if raw["documents"] and raw["documents"][0]:
            for i, (doc_id, text, meta, dist) in enumerate(
                zip(raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0])
            ):
                sim = 1 - dist  # cosine distance → similarity
                if sim >= score_threshold:
                    dense_docs.append({
                        "id": doc_id, "content": text, "metadata": meta,
                        "similarity_score": sim, "distance": dist, "rank": i + 1,
                    })
        else:
            print(f"No documents retrieved for query: '{query}'")

        # Hybrid BM25 + RRF
        if self.bm25_index and self.bm25_store:
            bm25_results = self._bm25_search(query, top_k=initial_k)
            if metadata_filter:
                bm25_results = [
                    d for d in bm25_results
                    if all(
                        d["metadata"].get(k) == v
                        for k, v in metadata_filter.items()
                        if not isinstance(v, dict)  # skip $in/$gte operators
                    )
                ]
            retrieved = self._rrf_merge(dense_docs, bm25_results)[:initial_k]
        else:
            retrieved = dense_docs

        # Rerank
        if self.reranker and retrieved:
            retrieved = rerank_documents(query, retrieved, reranker=self.reranker, top_n=top_k)

        return retrieved
