---
title: FinRAG – 10‑K RAG
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# FinRAG – 10‑K Retrieval‑Augmented Generation

FinRAG is an end‑to‑end Retrieval‑Augmented Generation (RAG) system that lets you
ask natural‑language questions about a single SEC 10‑K filing. It parses the PDF
with Llama Cloud, builds a persistent ChromaDB vector store, and serves a FastAPI
backend plus a small React frontend for interactive analysis.

Core stack:
- Python RAG pipeline (LangChain‑style abstractions, custom retriever)
- ChromaDB persistent vector store
- BAAI `bge-large-en-v1.5` embeddings + cross‑encoder reranker
- Groq LLMs (`llama-3.3-70b-versatile` for answers, `openai/gpt-oss-120b` as judge)
- FastAPI backend + Vite/React frontend

---

## Project structure

- `app/` – CLI entrypoint (`app.main`) and FastAPI server (`app.api`).
- `rag/` – ingestion, embeddings, vector store, hybrid retriever, and RAG pipeline.
- `evaluation/` – LLM‑as‑judge evaluation utilities and default question set.
- `frontend/` – React SPA that calls the FastAPI backend.
- `data/raw/` – input 10‑K PDF (not tracked in git).
- `data/vectorstore/` – ChromaDB persistent index (not tracked in git).

---

## Prerequisites

- Python 3.10+ (recommended to use a virtualenv).
- A Groq API key with access to the models used here.
- A Llama Cloud API key for 10‑K parsing.
- Node.js 18+ if you want to run the React frontend in dev mode.

Required environment variables (e.g. in `.env`):

```bash
GROQ_API_KEY=...
LLAMA_CLOUD_API_KEY=...
```

---

## Setup (backend)

```bash
cd FinRAG
python -m venv venv
.\n+venv\Scripts\activate  # PowerShell / cmd on Windows
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place your 10‑K PDF under `data/raw/` and ensure the filename matches the
default in `rag/ingestion.py` (currently `tsla-20251231-gen.pdf`), or pass a
custom path when you adapt the ingestion step.

Build the vector store once:

```bash
python -m app.main --ingest
```

You should see logs showing Llama Cloud parsing the PDF and ChromaDB being
populated with chunks.

Run a one‑off CLI query:

```bash
python -m app.main --query "What are the main risk factors mentioned in this 10‑K?"
```

Optionally run the evaluation suite:

```bash
python -m app.main --evaluate
```

This writes a CSV of scores to `data/eval_results.csv`.

---

## Running the API server

After ingestion has completed at least once:

```bash
uvicorn app.api:app --reload --port 8000
```

Key endpoints:
- `GET /health` – liveness + current vector‑store document count.
- `POST /api/query` – main RAG endpoint for questions about the 10‑K.
- `GET /api/eval-results` – returns saved evaluation results as JSON.

The React frontend is served from `app/static` in production builds (see
Docker section) but can be run separately during development.

---

## Frontend (optional dev workflow)

```bash
cd frontend
npm install
npm run dev
```

By default this runs on `http://localhost:5173` and calls the FastAPI backend
on `http://localhost:8000` (CORS is already configured).

---

## Docker

The included `Dockerfile` builds a container image with the FastAPI backend and
static frontend bundled together. A typical build/run sequence:

```bash
docker build -t finrag .
docker run -p 7860:7860 --env-file .env \
	-v ${PWD}/data:/app/data finrag
```

Persisting `./data` as a volume ensures the ChromaDB index survives container
restarts. Run the ingestion command once inside the container (or as a
one‑off job that shares the same volume) before serving live traffic.

---

## Notes and limitations

- The system is designed for **one 10‑K at a time**; adapting it to multiple
	filings would require extending the metadata schema and ingestion logic.
- The quality of answers depends heavily on the Llama Cloud parse and the
	retrieval configuration (chunk size, BM25 fusion, reranker thresholds).
- The repo intentionally ignores `data/raw/`, `data/processed/`, and
	`data/vectorstore/`; only code and configuration are committed to git.

Contributions, issue reports, and suggestions are welcome.
