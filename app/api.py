"""FastAPI backend for the 10-K RAG application.

Startup:  loads embeddings, vector store, BM25 index, reranker, and LLM once.
Endpoints:
    POST /api/query        	— ask a question about the 10-K filing
    GET  /api/eval-results 	— return saved evaluation CSV as JSON
    GET  /health           	— liveness check
    GET  /                 	— serves the static frontend
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_groq import ChatGroq
from pydantic import BaseModel

load_dotenv()

from config.settings import (
    EVAL_RESULTS_PATH,
    JUDGE_LLM_MODEL,
    JUDGE_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    RAG_LLM_MODEL,
)
from rag.embeddings import EmbeddingManager
from rag.pipeline import rag_enhanced_query
from rag.retriever import RAGRetriever, build_bm25_index, load_reranker
from rag.vectorstore import VectorStore

# ── Global pipeline state (initialised once at startup) ───────────────────────
_state: Dict[str, Any] = {}

STATIC_DIR = Path(__file__).parent / "static"


# ── Lifespan: initialise heavy components once ────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Loading RAG pipeline components …")

    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()

    if vector_store.collection.count() == 0:
        print(
            "⚠️  Vector store is empty. "
            "Run `python -m app.main --ingest` first to populate it."
        )

    bm25_index, bm25_store = build_bm25_index(vector_store)
    reranker = load_reranker()

    retriever = RAGRetriever(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        reranker=reranker,
        bm25_index=bm25_index,
        bm25_store=bm25_store,
    )

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=RAG_LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

    _state["retriever"] = retriever
    _state["llm"] = llm
    print("✅ Pipeline ready — server is accepting requests.")

    yield  # ── server runs here ──

    _state.clear()
    print("🛑 Shutdown — pipeline unloaded.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="10-K RAG API",
    description="Ask questions about a 10-K filing powered by retrieval-augmented generation.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the React dev server (Vite on http://localhost:5173) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML / CSS / JS)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Pydantic models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    min_score: float = 0.1
    # Optional metadata-based filter (reserved for future use)
    doc_filter: Optional[str] = None


class SourceItem(BaseModel):
    source: str
    section: str
    score: float
    preview: str


class QueryResponse(BaseModel):
    answer: str
    top_retrieval_score: float
    sources: List[SourceItem]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the single-page frontend."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(str(index_path))


@app.get("/health")
async def health():
    """Liveness check — also reports vector store document count."""
    retriever: RAGRetriever = _state.get("retriever")
    doc_count = retriever.vector_store.collection.count() if retriever else 0
    return {"status": "ok", "vector_store_docs": doc_count}


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a question about the 10-K filing.

    Returns the generated answer, the top retrieval similarity score, and a
    list of source passages used to construct the answer.
    """
    retriever: RAGRetriever = _state.get("retriever")
    llm: ChatGroq = _state.get("llm")

    if not retriever or not llm:
        raise HTTPException(status_code=503, detail="Pipeline not initialised.")

    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    metadata_filter = {"doc": request.doc_filter} if request.doc_filter else None

    result = rag_enhanced_query(
        query=request.question,
        retriever=retriever,
        llm=llm,
        top_k=request.top_k,
        min_score=request.min_score,
        metadata_filter=metadata_filter,
    )

    return QueryResponse(
        answer=result["answer"],
        top_retrieval_score=result["top_retrieval_score"],
        sources=[SourceItem(**s) for s in result["sources"]],
    )


@app.get("/api/eval-results")
async def eval_results():
    """
    Return the saved evaluation CSV as a list of JSON objects.
    Returns an empty list if no results file exists yet.
    """
    if not EVAL_RESULTS_PATH.exists():
        return {"results": [], "summary": {}}

    df = pd.read_csv(EVAL_RESULTS_PATH)

    # Replace NaN with None so JSON serialisation works
    df = df.where(pd.notna(df), None)

    score_cols = [
        "relevance_score", "correctness_score",
        "completeness_score", "faithfulness_score", "mean_score",
    ]
    existing_score_cols = [c for c in score_cols if c in df.columns]

    summary = {}
    if existing_score_cols:
        summary = df[existing_score_cols].mean().round(2).to_dict()

    return {
        "results": df.to_dict(orient="records"),
        "summary": summary,
    }
