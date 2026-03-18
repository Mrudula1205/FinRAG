"""
ChromaDB vector store: create, populate, and query the persistent collection.
"""

from __future__ import annotations

from typing import Any, Dict, List

import chromadb
import numpy as np
from langchain_core.documents import Document

from config.settings import CHROMA_COLLECTION_NAME, VECTORSTORE_DIR

_BATCH_SIZE = 500


class VectorStore:
    """
    Thin wrapper around a ChromaDB persistent collection.

    Uses cosine similarity (``hnsw:space = cosine``).  Documents are upserted
    in batches of ``_BATCH_SIZE`` to stay within ChromaDB's per-call limits.
    """

    def __init__(
        self,
        collection_name: str = CHROMA_COLLECTION_NAME,
        persist_directory: str | None = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory or str(VECTORSTORE_DIR)
        self.client: chromadb.ClientAPI | None = None
        self.collection: chromadb.Collection | None = None
        self._initialize()

    def _initialize(self) -> None:
        import os
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "GDPR Documents Collection",
                "hnsw:space": "cosine",
            },
        )
        print(
            f"✅ Collection '{self.collection_name}' ready. "
            f"Current doc count: {self.collection.count()}"
        )

    def add_documents(
        self,
        documents: List[Document],
        embeddings: np.ndarray,
    ) -> None:
        """
        Upsert documents + their embeddings into the collection.

        Duplicate IDs found in the same batch are disambiguated by appending
        ``__<index>`` to keep ChromaDB happy.

        Args:
            documents:  LangChain Documents to store.
            embeddings: Numpy array with one row per document.
        """
        ids, metadatas, texts, vecs = [], [], [], []
        seen_ids: set[str] = set()

        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            base_id = doc.metadata.get("id") or f"doc_{i}"
            doc_id  = base_id if base_id not in seen_ids else f"{base_id}__{i}"
            seen_ids.add(doc_id)

            meta = dict(doc.metadata)
            meta.update({"doc_index": i, "content_length": len(doc.page_content)})

            ids.append(doc_id)
            # ChromaDB only accepts str / int / float / bool metadata values
            metadatas.append({k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))})
            texts.append(doc.page_content)
            vecs.append(emb.tolist())

        for start in range(0, len(ids), _BATCH_SIZE):
            sl = slice(start, start + _BATCH_SIZE)
            self.collection.upsert(
                ids=ids[sl],
                metadatas=metadatas[sl],
                documents=texts[sl],
                embeddings=vecs[sl],
            )

        print(f"✅ Upserted {len(ids)} documents. Collection total: {self.collection.count()}")

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        where: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Run a nearest-neighbour query against the collection.

        Returns the raw ChromaDB result dict
        (keys: ``ids``, ``documents``, ``metadatas``, ``distances``).
        """
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)
