"""
Central configuration for the GDPR-RAG project.
All path and model constants live here so other modules never hardcode them.
"""

from pathlib import Path

# ── Project root (one level above this file) ───────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data paths ──────────────────────────────────────────────────────────────
DATA_DIR          = PROJECT_ROOT / "data"
RAW_DATA_DIR      = DATA_DIR / "raw"
PROCESSED_DIR     = DATA_DIR / "processed"
VECTORSTORE_DIR   = DATA_DIR / "vectorstore"
EVAL_RESULTS_PATH = DATA_DIR / "eval_results.csv"

GDPR_CLAUSES_JSON = RAW_DATA_DIR / "gdpr_clauses.json"

# ── Chunking ────────────────────────────────────────────────────────────────
CHUNK_SIZE         = 800
CHUNK_OVERLAP      = 100
SEMANTIC_THRESHOLD = 80          # percentile breakpoint for SemanticChunker

# ── Embedding model ─────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "gdpr_documents"

# ── Reranker ────────────────────────────────────────────────────────────────
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# ── LLM (Groq) ───────────────────────────────────────────────────────────────
RAG_LLM_MODEL    = "llama-3.3-70b-versatile"
JUDGE_LLM_MODEL  = "openai/gpt-oss-120b"
LLM_TEMPERATURE  = 0.1
JUDGE_TEMPERATURE = 0.0
LLM_MAX_TOKENS   = 2048

# ── Retrieval defaults ───────────────────────────────────────────────────────
DEFAULT_TOP_K         = 3
DEFAULT_SCORE_THRESHOLD = 0.1
RRF_K                 = 60      # Reciprocal Rank Fusion constant

# ── Document source metadata ─────────────────────────────────────────────────
PDF_SOURCE_MAP = {
    "google_privacy_policy_en.pdf": {
        "doc_type": "company_policy", "source": "Google",
        "jurisdiction": "EU", "version": "N/A", "effective_date": "N/A"
    },
    "LinkedIn Privacy Policy.pdf": {
        "doc_type": "company_policy", "source": "LinkedIn",
        "jurisdiction": "EU", "version": "N/A", "effective_date": "N/A"
    },
    "Microsoft Privacy Statement \u2013 Microsoft privacy.pdf": {
        "doc_type": "company_policy", "source": "Microsoft",
        "jurisdiction": "EU", "version": "N/A", "effective_date": "N/A"
    },
    "Regulation_EU_2016.pdf": {
        "doc_type": "gdpr_regulation", "source": "EU Official Journal",
        "jurisdiction": "EU", "version": "2016/679", "effective_date": "2018-05-25"
    },
}

HTML_METADATA_BASE = {
    "doc_type": "gdpr_regulation",
    "source": "EU Official Journal",
    "jurisdiction": "EU",
    "version": "2016/679",
    "effective_date": "2018-05-25",
}
