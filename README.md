# FinRAG

Production-style retrieval-augmented generation system for SEC 10-K analysis.

FinRAG ingests annual filings, parses them into structured markdown, builds a
hybrid retrieval index, and serves grounded answers through a FastAPI backend
and React frontend. The project is designed to answer filing-specific
questions with traceable source passages rather than free-form financial
commentary.

## What This System Does

- Ingests 10-K PDF filings with LlamaCloud parsing.
- Normalizes parsed markdown into section-aware chunks with metadata.
- Stores embeddings in a persistent Chroma collection.
- Combines dense retrieval, BM25 sparse retrieval, RRF fusion, and cross-encoder reranking.
- Serves answers with retrieved source passages through an API and web UI.

## Architecture

```text
PDF upload / local filing
        |
        v
LlamaCloud parse -> page markdown -> normalized sections -> rechunked chunks
        |
        v
Embeddings (BAAI/bge-large-en-v1.5)
        |
        v
Chroma vector store + BM25 index
        |
        v
RRF fusion -> cross-encoder reranker
        |
        v
LLM answer generation with source passages
        |
        v
FastAPI API + React frontend
```

## Retrieval Design

FinRAG uses a multi-stage retrieval stack rather than a single vector search.

1. Parse the 10-K into markdown with page-level structure.
2. Split by filing headers, then rechunk long sections while preserving tables.
3. Embed chunks with `BAAI/bge-large-en-v1.5`.
4. Retrieve candidates from:
   - Chroma dense similarity search
   - BM25 sparse keyword search
5. Merge candidate lists with Reciprocal Rank Fusion.
6. Rerank the merged shortlist with `BAAI/bge-reranker-base`.
7. Generate the final answer from retrieved evidence only.

This design improves recall on both semantic and keyword-heavy questions,
which matters for filings containing exact legal phrasing, section titles,
and numeric disclosures.

## Tech Stack

- Backend: FastAPI, Python
- Retrieval: ChromaDB, BM25, RRF fusion, cross-encoder reranking
- Embeddings: `BAAI/bge-large-en-v1.5`
- Parsing: LlamaCloud
- LLM serving: Groq via `langchain_groq`
- Frontend: React + Vite
- Packaging: Docker

## Repository Layout

```text
FinRAG/
├─ app/
│  ├─ app.py            # FastAPI application
│  └─ main.py           # CLI entrypoint for ingestion and query
├─ config/
│  └─ settings.py       # Central configuration
├─ rag/
│  ├─ ingestion.py      # Parse, normalize, rechunk, persist artifacts
│  ├─ embeddings.py     # Embedding model wrapper
│  ├─ retriever.py      # Dense + BM25 + RRF + reranker
│  ├─ pipeline.py       # Prompting and structured query output
│  └─ vectorstore.py    # Chroma collection management
├─ frontend/
│  └─ src/              # React client
├─ data/
│  ├─ raw/              # Uploaded / local 10-K PDFs
│  ├─ processed/        # Saved parse artifacts and chunk manifests
│  └─ vectorstore/      # Persistent Chroma data
├─ Dockerfile
└─ requirements.txt
```

## Local Development

### Prerequisites

- Python 3.10+
- Node.js 18+
- Groq API key
- LlamaCloud API key

Create a `.env` file:

```bash
GROQ_API_KEY=...
LLAMA_CLOUD_API_KEY=...
```

### Backend Setup

```bash
cd FinRAG
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Ingest a Filing

```bash
python -m app.main --ingest
```

This will:
- parse the most recent PDF in `data/raw/` if no explicit path is wired in,
- create processed parse artifacts under `data/processed/<doc_id>/`,
- build or extend the Chroma collection.

### Run the API

```bash
uvicorn app.app:app --reload --port 8000
```

### Run the Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:5173` and calls the backend on
`http://localhost:8000`.

## API Surface

- `GET /health`
  - liveness check plus current vector-store document count
- `POST /api/upload`
  - upload a PDF and trigger ingestion
- `POST /api/query`
  - submit a natural-language question against the indexed filings
- `POST /api/reset-index`
  - clear the vector index for a fresh dev session

## Example Query

```bash
python -m app.main --query "What are the main risk factors disclosed in this filing?"
```

Expected response shape:

```json
{
  "answer": "...",
  "top_retrieval_score": 0.61,
  "sources": [
    {
      "source": "Tesla 2025 10-K",
      "section": "Item 1A. Risk Factors",
      "score": 0.61,
      "preview": "..."
    }
  ]
}
```

## Engineering Notes

### Parsing and Chunking

The ingestion pipeline is intentionally two-stage:

- structural split by markdown headers to preserve filing hierarchy
- bounded rechunking for retrieval efficiency

Tables are preserved as standalone chunks where possible, and chunk metadata
includes section path, page span, chunk type, and quality flags.

### Why Hybrid Retrieval

10-K queries often mix semantic intent with exact language:
- "What changed in liquidity risk?"
- "Where does the filing mention material weakness?"
- "What is the company name?"

Dense retrieval handles paraphrase well. BM25 helps when wording is exact.
RRF fusion combines both, and the reranker improves final precision.

## Deployment

Build and run with Docker:

```bash
docker build -t finrag .
docker run -p 7860:7860 --env-file .env -v ${PWD}/data:/app/data finrag
```

Persisting `data/` as a volume preserves both the processed artifacts and the
vector store between container restarts.

## Benchmarks

Do not publish made-up metrics. Replace this section only after you have
measured the system.

Recommended metrics to track:

- p50 and p95 query latency
- ingestion latency per filing
- average prompt and completion tokens
- estimated cost per request
- retrieval hit rate at K
- answer accuracy on a labeled evaluation set
- failure rate for empty or low-confidence retrieval

Suggested reporting format:

```text
Query latency (p95): <fill after benchmark>
Cost per request: <fill after benchmark>
Retrieval hit@5: <fill after benchmark>
Answer accuracy: <fill after benchmark>
Test corpus: <number of filings / sectors / years>
```

## Operational Gaps

Current repository strengths:

- multi-stage retrieval stack
- persistent vector store
- upload and query API
- containerized app structure
- parse artifact persistence for debugging

Current gaps before calling this production-ready:

- no committed benchmark suite or evaluation harness
- limited automated test coverage
- no CI pipeline yet
- no auth, rate limiting, or audit logging
- weak query-time filtering for multi-filing collections

## Postmortem-Style Lessons

- Generic questions over a shared multi-filing index can retrieve evidence from
  multiple companies unless metadata filters are enforced.
- BM25 and dense similarity scores should not be shown as if they are the same
  metric.
- Raw uploaded filenames should be normalized into clean source labels to keep
  citations readable.

## Resume-Friendly Project Framing

Use quantified bullets only after measurement. A stronger framing looks like:

```text
Built a hybrid RAG system for SEC 10-K analysis using FastAPI, Chroma, BM25,
RRF fusion, and cross-encoder reranking. Implemented section-aware chunking,
source-grounded answers, Dockerized deployment, and persistent vector indexing
for multi-filing retrieval workflows.
```

Once benchmarks exist, append real numbers:

```text
Built a hybrid RAG system for SEC 10-K analysis.
- p95 latency: <measured>
- cost/request: <measured>
- retrieval hit@5: <measured>
- stack: FastAPI, Chroma, React, Docker, Groq, LlamaCloud
```
