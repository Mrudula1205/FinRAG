FROM python:3.10-slim

# Prevents .pyc files and enables real-time log output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Build tools needed by some Python packages (e.g. tokenizers, numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app/       ./app/
COPY config/    ./config/
COPY rag/       ./rag/
#COPY evaluation/ ./evaluation/

# Copy pre-built vectorstore + processed data so the container is self-contained.
# To use a live/updated store instead, mount a volume at runtime:
#   docker run -v /host/data:/app/data ...
#COPY data/      ./data/

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Pass secrets via Space settings → Variables and secrets → GROQ_API_KEY
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "7860"]
