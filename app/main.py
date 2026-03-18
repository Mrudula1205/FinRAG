"""Entry point for the 10-K RAG application.

Usage (build vector store from scratch):
    python -m app.main --ingest

Usage (ask a single question):
    python -m app.main --query "What are the main risk factors in this 10-K?"

Usage (run evaluation):
    python -m app.main --evaluate
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

from config.settings import (
    DEFAULT_TOP_K,
    JUDGE_LLM_MODEL,
    JUDGE_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    RAG_LLM_MODEL,
)
from evaluation.evaluator import EVAL_DATASET, LLMJudgeEvaluator, print_scorecard, run_evaluation
from rag.embeddings import EmbeddingManager
from rag.ingestion import load_10k_documents
from rag.pipeline import rag_enhanced_query
from rag.retriever import RAGRetriever, build_bm25_index, load_reranker
from rag.vectorstore import VectorStore


# ── Component factory ─────────────────────────────────────────────────────────

def build_pipeline(skip_ingestion: bool = True):
    """Instantiate every component and return a ready-to-use (retriever, llm) pair.

    Args:
        skip_ingestion: When False, (re)parse the 10-K PDF, embed the chunks,
                        and upsert them into ChromaDB before continuing.
                        Set to True (default) when the vector store is already
                        populated and you only want to query.
    """
    # Embeddings & vector store
    embedding_manager = EmbeddingManager()
    vector_store      = VectorStore()

    if not skip_ingestion:
        print("\n📥 Ingestion mode — parsing 10-K PDF …")
        # Load markdown-based chunks from the 10-K filing
        import asyncio

        docs = asyncio.run(load_10k_documents())

        texts      = [d.page_content for d in docs]
        embeddings = embedding_manager.generate_embeddings(texts)
        vector_store.add_documents(docs, embeddings)
        print("✅ Vector store populated with 10-K chunks.\n")

    # BM25 + reranker
    bm25_index, bm25_store = build_bm25_index(vector_store)
    reranker               = load_reranker()

    # Retriever
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        reranker=reranker,
        bm25_index=bm25_index,
        bm25_store=bm25_store,
    )

    # LLM
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=RAG_LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

    return retriever, llm


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="10-K RAG command-line interface")
    parser.add_argument(
        "--ingest", action="store_true",
        help="Re-ingest all source documents and rebuild the vector store."
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Ask a single question about the 10-K filing."
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run the evaluation pipeline on the built-in test dataset."
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"Number of documents to retrieve (default: {DEFAULT_TOP_K})."
    )
    args = parser.parse_args()

    retriever, llm = build_pipeline(skip_ingestion=not args.ingest)

    if args.query:
        print(f"\n🔍 Query: {args.query}\n")
        result = rag_enhanced_query(
            args.query, retriever, llm,
            top_k=args.top_k, return_context=False
        )
        print("Answer:\n")
        print(result["answer"])
        print(f"\nTop retrieval score: {result['top_retrieval_score']}")
        print("\nSources:")
        for s in result["sources"]:
            print(f"  - Article {s['article']}, Clause {s['clause']} | score={s['score']} | {s['preview'][:80]}...")

    if args.evaluate:
        print("\n📊 Running evaluation pipeline …\n")
        groq_api_key = os.getenv("GROQ_API_KEY")
        judge_llm = ChatGroq(
            api_key=groq_api_key,
            model_name=JUDGE_LLM_MODEL,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        judge = LLMJudgeEvaluator(judge_llm=judge_llm)
        df = run_evaluation(EVAL_DATASET, retriever, llm, judge, top_k=args.top_k)
        print_scorecard(df)

    if not args.query and not args.evaluate and not args.ingest:
        parser.print_help()


if __name__ == "__main__":
    main()
