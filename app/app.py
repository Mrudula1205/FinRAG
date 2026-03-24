"""FastAPI backend for the 10-K RAG application.

Startup:  loads embeddings, vector store, BM25 index, reranker, and LLM once.
Endpoints:
    POST /api/query        	— ask a question about the 10-K filing
    POST /api/upload       	— upload a PDF and trigger ingestion
    GET  /health           	— liveness check
    GET  /                 	— serves the static frontend
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_groq import ChatGroq
from pydantic import BaseModel

load_dotenv()

from config.settings import (
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    RAG_LLM_MODEL,
    RAW_DATA_DIR,
)
from rag.embeddings import EmbeddingManager
from rag.ingestion import load_10k_documents
from rag.pipeline import rag_enhanced_query
from rag.retriever import RAGRetriever, build_bm25_index, load_reranker
from rag.vectorstore import VectorStore

# ── Global pipeline state (initialised once at startup) ───────────────────────
_state: Dict[str, Any] = {}

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


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
    _state["embedding_manager"] = embedding_manager
    _state["vector_store"] = vector_store
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


class UploadResponse(BaseModel):
    message: str
    filename: str


class ResetResponse(BaseModel):
    message: str
    vector_store_docs: int


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


async def _ingest_pdf(path: Path) -> int:
    """Parse a 10-K PDF and upsert its chunks into the shared vector store.

    This reuses the same ingestion pipeline used by the CLI, but operates on
    a specific *path* uploaded by the user instead of a fixed local file.
    """

    docs = await load_10k_documents(pdf_path=path)

    embedding_manager: EmbeddingManager = _state["embedding_manager"]
    vector_store: VectorStore = _state["vector_store"]

    texts = [d.page_content for d in docs]
    embeddings = embedding_manager.generate_embeddings(texts)
    vector_store.add_documents(docs, embeddings)

    # Rebuild BM25 index to include the new documents
    bm25_index, bm25_store = build_bm25_index(vector_store)
    retriever: RAGRetriever = _state["retriever"]
    retriever.bm25_index = bm25_index
    retriever.bm25_store = bm25_store

    return len(docs)


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


@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload a 10-K PDF and trigger background ingestion.

    The file is stored under ``data/raw`` and parsed with Llama Cloud. Its
    chunks are embedded and upserted into the shared ChromaDB collection.
    """

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DATA_DIR / file.filename

    contents = await file.read()
    with dest.open("wb") as f:
        f.write(contents)

    # Ingest asynchronously so the request returns quickly.
    background_tasks.add_task(_ingest_pdf, dest)

    return UploadResponse(
        message="Upload accepted. Ingestion is running in the background.",
        filename=file.filename,
    )


@app.post("/api/reset-index", response_model=ResetResponse)
async def reset_index():
    """Reset the vector index so each session can start fresh in dev mode."""
    retriever: RAGRetriever = _state.get("retriever")
    vector_store: VectorStore = _state.get("vector_store")

    if not retriever or not vector_store:
        raise HTTPException(status_code=503, detail="Pipeline not initialised.")

    vector_store.reset_collection()

    bm25_index, bm25_store = build_bm25_index(vector_store)
    retriever.bm25_index = bm25_index
    retriever.bm25_store = bm25_store

    return ResetResponse(
        message="Vector index cleared. Upload documents to build a fresh session.",
        vector_store_docs=vector_store.collection.count(),
    )
