"""
Central configuration for the GDPR-RAG project.
All path and model constants live here so other modules never hardcode them.
"""

from pathlib import Path

# ── Project root (one level above this file) ───────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data paths ──────────────────────────────────────────────────────────────
DATA_DIR        = PROJECT_ROOT / "data"
RAW_DATA_DIR    = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

# ── Chunking ────────────────────────────────────────────────────────────────
CHUNK_SIZE         = 800
CHUNK_OVERLAP      = 100
SEMANTIC_THRESHOLD = 80          # percentile breakpoint for SemanticChunker

# ── Embedding model ─────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "finrag_documents"

# ── Reranker ────────────────────────────────────────────────────────────────
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# ── LLM (Groq) ───────────────────────────────────────────────────────────────
RAG_LLM_MODEL   = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS  = 2048

# ── Retrieval defaults ───────────────────────────────────────────────────────
DEFAULT_TOP_K         = 3
DEFAULT_SCORE_THRESHOLD = 0.1
RRF_K                 = 60      # Reciprocal Rank Fusion constant

# ── Document source metadata ─────────────────────────────────────────────────
# (Reserved for future use with multiple filings or richer metadata.)
