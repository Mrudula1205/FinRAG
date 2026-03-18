"""
Document chunking: semantic chunking for PDF/HTML docs; GDPR articles pass through as-is.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE, SEMANTIC_THRESHOLD


def build_chunkers(embedding_model) -> tuple[SemanticChunker, RecursiveCharacterTextSplitter]:
    """
    Return a (semantic_chunker, safety_splitter) pair.

    Args:
        embedding_model: A LangChain-compatible embeddings instance used by SemanticChunker.
    """
    semantic_chunker = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=SEMANTIC_THRESHOLD,
    )
    safety_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return semantic_chunker, safety_splitter


def chunk_documents(
    documents: List[Document],
    embedding_model,
) -> List[Document]:
    """
    Chunk a list of documents.

    - ``gdpr_article`` documents are already atomic — returned unchanged.
    - All other documents (PDF, HTML) are semantically chunked and then
      capped at ``CHUNK_SIZE`` characters by a safety splitter.

    Args:
        documents:       List of enriched LangChain Documents.
        embedding_model: Embeddings instance forwarded to SemanticChunker.

    Returns:
        Flat list of chunked Documents.
    """
    semantic_chunker, safety_splitter = build_chunkers(embedding_model)
    all_chunks: List[Document] = []

    for i, doc in enumerate(documents):
        if doc.metadata.get("doc_type") == "gdpr_article":
            all_chunks.append(doc)
        else:
            for sem_chunk in semantic_chunker.split_documents([doc]):
                all_chunks.extend(safety_splitter.split_documents([sem_chunk]))

        if (i + 1) % 50 == 0 or (i + 1) == len(documents):
            print(f"  chunked {i + 1}/{len(documents)} docs → {len(all_chunks)} chunks so far")

    print(f"✅ {len(all_chunks)} total chunks from {len(documents)} documents")
    return all_chunks
