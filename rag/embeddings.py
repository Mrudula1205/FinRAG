"""
Embedding management: thin wrapper around HuggingFaceEmbeddings that
preserves the BGE asymmetric encoding contract.
"""

from __future__ import annotations

from typing import List

import numpy as np
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from config.settings import EMBEDDING_MODEL_NAME


class EmbeddingManager:
    """
    Wraps a LangChain HuggingFaceEmbeddings model.

    BGE asymmetric encoding contract:
      - Documents → ``embed_documents()``   (no instruction prefix)
      - Queries   → ``embed_query()``       (model applies its own query prefix)

    HuggingFaceEmbeddings with BAAI/bge-large-en-v1.5 handles the instruction
    prefix for queries automatically via ``embed_query()``, so we delegate
    directly rather than prepending it manually.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = HuggingFaceEmbeddings(model_name=model_name)
        print(f"✅ EmbeddingManager ready — model: {self.model.model_name}")

    def generate_embeddings(
        self,
        texts: List[str],
        is_query: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts:    Strings to embed.
            is_query: If ``True``, uses ``embed_query()`` which applies the BGE
                      instruction prefix required for query embeddings.
                      If ``False``, uses ``embed_documents()`` for passage embeddings.
                      Using the wrong method degrades retrieval quality.

        Returns:
            Float32 numpy array of shape ``(len(texts), embedding_dim)``.
        """
        if is_query:
            vectors = [self.model.embed_query(t) for t in texts]
        else:
            vectors = self.model.embed_documents(texts)
        return np.array(vectors, dtype=np.float32)


def load_embedding_manager(model_name: str = EMBEDDING_MODEL_NAME) -> EmbeddingManager:
    """Convenience factory — creates and returns an EmbeddingManager."""
    return EmbeddingManager(model_name=model_name)
