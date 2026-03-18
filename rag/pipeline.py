"""
RAG query functions: format retrieved context and invoke the LLM.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_groq import ChatGroq

from config.settings import DEFAULT_SCORE_THRESHOLD, DEFAULT_TOP_K
from rag.retriever import RAGRetriever

_SYSTEM_PROMPT = (
    "You are a financial analysis assistant specialising in SEC 10-K filings. "
    "Answer the question using ONLY the provided context.\n"
    "If the context does not contain sufficient information to answer, say "
    '"The provided documents do not address this directly."\n'
    "Where possible, refer to the relevant section titles from the filing (for example, "
    '"Item 1A. Risk Factors").'
)


# ── Context formatting ────────────────────────────────────────────────────────

def extract_content_for_rag(retrieved_docs: List[Dict[str, Any]]) -> str:
    """
    Format a list of retrieved documents into a single context string.

    For 10-K filings, prefixes each passage with ``[source - section]`` when
    possible, where *section* comes from markdown headers. Falls back to just
    ``[source]`` when no header metadata is present.
    """
    parts: List[str] = []
    for doc in retrieved_docs:
        meta    = doc["metadata"]
        source  = meta.get("source", "")
        section = (
            meta.get("Header 3")
            or meta.get("Header 2")
            or meta.get("Header 1")
            or ""
        )

        if source and section:
            ref = f"[{source} - {section}]"
        elif source:
            ref = f"[{source}]"
        else:
            ref = ""

        parts.append(f"{ref}\n{doc['content']}" if ref else doc["content"])

    return "\n\n".join(parts)


def _build_prompt(context: str, question: str) -> str:
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


# ── Query functions ───────────────────────────────────────────────────────────

def simple_rag_query(
    query: str,
    retriever: RAGRetriever,
    llm: ChatGroq,
    top_k: int = DEFAULT_TOP_K,
) -> str:
    """
    Retrieve context and return the raw LLM answer string.

    Useful for quick interactive queries where you don't need source metadata.
    """
    docs = retriever.retrieve(query, top_k=top_k)
    if not docs:
        return "No relevant documents found."

    context = extract_content_for_rag(docs)
    return llm.invoke(_build_prompt(context, query)).content


def rag_enhanced_query(
    query: str,
    retriever: RAGRetriever,
    llm: ChatGroq,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_SCORE_THRESHOLD,
    return_context: bool = False,
    metadata_filter: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Retrieve context, generate an answer, and return a structured result dict.

    Returns a dict with keys:
      - ``answer``              : LLM response string
      - ``top_retrieval_score`` : highest cosine similarity among retrieved docs
      - ``sources``             : list of source dicts with article/clause/score/preview
      - ``context``             : (only when *return_context* is True) raw context string
    """
    docs = retriever.retrieve(query, top_k=top_k, score_threshold=min_score, metadata_filter=metadata_filter)

    if not docs:
        result: Dict[str, Any] = {
            "answer": "No relevant documents found.",
            "top_retrieval_score": 0.0,
            "sources": [],
        }
        if return_context:
            result["context"] = ""
        return result

    context = extract_content_for_rag(docs)
    answer  = llm.invoke(_build_prompt(context, query)).content

    sources = [
        {
            "source":  doc["metadata"].get("source", "N/A"),
            "section": (
                doc["metadata"].get("Header 3")
                or doc["metadata"].get("Header 2")
                or doc["metadata"].get("Header 1")
                or "N/A"
            ),
            "score":   round(doc["similarity_score"], 3),
            "preview": doc["content"][:150] + "...",
        }
        for doc in docs
    ]

    result = {
        "answer": answer,
        "top_retrieval_score": round(max(d["similarity_score"] for d in docs), 3),
        "sources": sources,
    }
    if return_context:
        result["context"] = context

    return result
