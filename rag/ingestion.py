"""10-K filing ingestion using LlamaCloud + Markdown-based chunking.

This module replaces the previous GDPR-focused ingestion. It is now
responsible only for turning a 10-K PDF into LangChain Documents that can
be embedded and indexed in the RAG pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from llama_cloud import AsyncLlamaCloud

from config.settings import RAW_DATA_DIR


# Default relative name of the 10-K PDF inside RAW_DATA_DIR
DEFAULT_10K_FILENAME = "tsla-20251231-gen.pdf"


def _build_markdown_splitter() -> MarkdownHeaderTextSplitter:
    """Create a MarkdownHeaderTextSplitter for 10-K style headings."""
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    return MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)


async def parse_10k_with_llamacloud(
    pdf_path: Path | None = None,
) -> List[Document]:
    """Parse a 10-K PDF with LlamaCloud and return markdown-based chunks.

    This mirrors the logic from the 10k_report_rag notebook:

    - Uses AsyncLlamaCloud (LLAMA_CLOUD_API_KEY must be set in the env).
    - Uploads the PDF and parses it with the agentic tier.
    - Concatenates all page markdown into a single markdown string.
    - Splits markdown into LangChain Documents using MarkdownHeaderTextSplitter.
    - Tags each Document with source metadata ("tsla-2025-10k" by default).
    """

    if pdf_path is None:
        pdf_path = RAW_DATA_DIR / DEFAULT_10K_FILENAME

    pdf_path = pdf_path.resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"10-K PDF not found at {pdf_path}")

    client = AsyncLlamaCloud()  # Uses LLAMA_CLOUD_API_KEY from environment

    file = await client.files.create(file=pdf_path, purpose="parse")
    result = await client.parsing.parse(
        file_id=file.id,
        tier="agentic",
        version="latest",
        expand=["markdown"],
    )

    all_markdown = "\n\n".join(page.markdown for page in result.markdown.pages)

    splitter = _build_markdown_splitter()
    docs = splitter.split_text(all_markdown)
    for d in docs:
        d.metadata["source"] = "tsla-2025-10k"

    print(f"✅ Parsed 10-K and created {len(docs)} markdown-based chunks")
    return docs


async def load_10k_documents(pdf_path: Path | None = None) -> List[Document]:
    """High-level entry point used by the RAG pipeline.

    Returns a list of LangChain Documents representing the chunked 10-K.
    """

    return await parse_10k_with_llamacloud(pdf_path=pdf_path)
