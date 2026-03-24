"""10-K filing ingestion using LlamaCloud plus section-aware chunking.

This module turns a 10-K PDF into LangChain Documents with richer metadata for
retrieval, filtering, and citation-friendly answers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from llama_cloud import AsyncLlamaCloud

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE, PROCESSED_DIR, RAW_DATA_DIR

TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
WHITESPACE_RE = re.compile(r"[ \t]+")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f]")
FISCAL_YEAR_RE = re.compile(
    r"For the fiscal year ended\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
    re.IGNORECASE,
)
COMPANY_RE = re.compile(
    r"([A-Z][A-Za-z0-9&.,()' /-]{2,}?)\s*\(Exact name of registrant",
    re.IGNORECASE,
)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "unknown"


def _source_from_pdf_path(pdf_path: Path) -> str:
    """Build a stable source label from the uploaded filename."""
    raw = pdf_path.stem.replace("_", " ").replace("-", " ").strip()
    return re.sub(r"\s+", " ", raw) or pdf_path.stem


def _doc_id_from_pdf_path(pdf_path: Path) -> str:
    return _slugify(pdf_path.stem)


def _build_markdown_splitter() -> MarkdownHeaderTextSplitter:
    """Create a MarkdownHeaderTextSplitter for 10-K style headings."""
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    return MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)


def _build_recursive_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _normalize_markdown(text: str) -> str:
    text = CONTROL_CHAR_RE.sub("", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _extract_pages(parse_result: Any) -> List[Dict[str, Any]]:
    pages: List[Dict[str, Any]] = []
    for idx, page in enumerate(parse_result.markdown.pages, start=1):
        page_number = getattr(page, "page", None) or getattr(page, "page_number", None) or idx
        markdown = _normalize_markdown(getattr(page, "markdown", "") or "")
        if markdown:
            pages.append({"page_number": int(page_number), "markdown": markdown})
    return pages


def _extract_filing_metadata(pages: List[Dict[str, Any]], pdf_path: Path) -> Dict[str, Any]:
    preview = "\n".join(page["markdown"] for page in pages[:3])
    company_match = COMPANY_RE.search(preview)
    fiscal_year_match = FISCAL_YEAR_RE.search(preview)

    source = _source_from_pdf_path(pdf_path)
    company = company_match.group(1).strip() if company_match else source
    filing_year = None
    fiscal_year_end = None

    if fiscal_year_match:
        fiscal_year_end = fiscal_year_match.group(1).strip()
        year_match = re.search(r"(\d{4})$", fiscal_year_end)
        filing_year = int(year_match.group(1)) if year_match else None

    return {
        "doc_id": _doc_id_from_pdf_path(pdf_path),
        "source": source,
        "file_name": pdf_path.name,
        "company": company,
        "form_type": "10-K",
        "filing_year": filing_year,
        "fiscal_year_end": fiscal_year_end,
    }


def _section_path(metadata: Dict[str, Any]) -> str:
    parts = [
        str(metadata.get("Header 1") or "").strip(),
        str(metadata.get("Header 2") or "").strip(),
        str(metadata.get("Header 3") or "").strip(),
    ]
    parts = [part for part in parts if part]
    return " > ".join(parts) if parts else "Unsectioned"


def _section_id(section_title: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", section_title.lower()).strip("_")
    return normalized or "unsectioned"


def _is_table_block(lines: List[str], start: int) -> bool:
    if start + 1 >= len(lines):
        return False
    if not TABLE_LINE_RE.match(lines[start]):
        return False
    return TABLE_LINE_RE.match(lines[start + 1]) is not None


def _extract_table_block(lines: List[str], start: int) -> Tuple[str, int]:
    i = start
    buf: List[str] = []
    while i < len(lines) and TABLE_LINE_RE.match(lines[i]):
        buf.append(lines[i])
        i += 1
    return "\n".join(buf).strip(), i


def _split_preserving_tables(text: str) -> List[str]:
    """Split text to target chunk size while preserving markdown tables."""
    lines = text.splitlines()
    blocks: List[Tuple[str, str]] = []
    current_text: List[str] = []
    index = 0

    while index < len(lines):
        if _is_table_block(lines, index):
            if current_text:
                blocks.append(("text", "\n".join(current_text).strip()))
                current_text = []
            table_text, next_index = _extract_table_block(lines, index)
            blocks.append(("table", table_text))
            index = next_index
            continue

        current_text.append(lines[index])
        index += 1

    if current_text:
        blocks.append(("text", "\n".join(current_text).strip()))

    splitter = _build_recursive_splitter()
    chunks: List[str] = []

    for block_type, block_text in blocks:
        if not block_text:
            continue
        if block_type == "table":
            chunks.append(block_text)
        else:
            chunks.extend(splitter.split_text(block_text))

    return [chunk.strip() for chunk in chunks if chunk and len(chunk.strip()) >= 120]


def _classify_chunk_type(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "empty"
    if "\n|" in stripped and "|" in stripped:
        return "table"
    if re.search(r"^\s*[-*]\s+", stripped, flags=re.MULTILINE):
        return "list"
    return "text"


def _quality_flags(text: str, section_title: str) -> List[str]:
    flags: List[str] = []
    compact = re.sub(r"\s+", " ", text).strip()
    alpha_ratio = (
        sum(char.isalpha() for char in compact) / max(len(compact), 1)
        if compact
        else 0.0
    )

    if len(compact) < 120:
        flags.append("too_short")
    if len(compact) > max(CHUNK_SIZE * 8, 4000):
        flags.append("too_long")
    if alpha_ratio < 0.25:
        flags.append("ocr_noisy")

    section_lower = section_title.lower()
    if "table of contents" in section_lower:
        flags.append("boilerplate_like")
    if section_lower.startswith("signatures"):
        flags.append("boilerplate_like")

    return flags


def _to_embedding_text(metadata: Dict[str, Any], content: str) -> str:
    section = metadata.get("section_title") or metadata.get("section_path") or "Unsectioned"
    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")
    page_text = f"{page_start}-{page_end}" if page_start and page_end and page_start != page_end else page_start
    prefix = (
        f"[Source: {metadata.get('source', 'Unknown')} | "
        f"Section: {section} | "
        f"Pages: {page_text or 'N/A'}]"
    )
    return f"{prefix}\n{content}".strip()


def _merge_adjacent_sections(docs: Iterable[Document]) -> List[Document]:
    """Merge consecutive per-page header chunks that belong to the same section."""
    merged: List[Document] = []

    for doc in docs:
        current_meta = dict(doc.metadata or {})
        current_path = _section_path(current_meta)
        current_page_start = current_meta.get("page_start")
        current_page_end = current_meta.get("page_end")

        if not merged:
            merged.append(
                Document(
                    page_content=doc.page_content,
                    metadata={**current_meta, "section_path": current_path},
                )
            )
            continue

        prev = merged[-1]
        prev_meta = dict(prev.metadata or {})
        same_path = prev_meta.get("section_path") == current_path
        adjacent_pages = (
            isinstance(prev_meta.get("page_end"), int)
            and isinstance(current_page_start, int)
            and current_page_start <= prev_meta["page_end"] + 1
        )

        if same_path and adjacent_pages:
            prev.page_content = f"{prev.page_content}\n\n{doc.page_content}".strip()
            if isinstance(current_page_end, int):
                prev.metadata["page_end"] = current_page_end
        else:
            merged.append(
                Document(
                    page_content=doc.page_content,
                    metadata={**current_meta, "section_path": current_path},
                )
            )

    return merged


def _page_level_header_docs(
    pages: List[Dict[str, Any]],
    base_metadata: Dict[str, Any],
) -> List[Document]:
    splitter = _build_markdown_splitter()
    docs: List[Document] = []

    for page in pages:
        page_number = page["page_number"]
        page_docs = splitter.split_text(page["markdown"])

        if not page_docs:
            continue

        for page_doc in page_docs:
            meta = {
                **base_metadata,
                **dict(page_doc.metadata or {}),
                "page_start": page_number,
                "page_end": page_number,
            }
            docs.append(Document(page_content=page_doc.page_content, metadata=meta))

    return docs


def _rechunk_sections(section_docs: List[Document], base_metadata: Dict[str, Any]) -> List[Document]:
    final_docs: List[Document] = []
    global_index = 0

    for section_doc in section_docs:
        meta = dict(section_doc.metadata or {})
        section_path = meta.get("section_path") or _section_path(meta)
        section_title = (
            meta.get("Header 3")
            or meta.get("Header 2")
            or meta.get("Header 1")
            or "Unsectioned"
        )
        section_chunks = _split_preserving_tables(section_doc.page_content)
        if not section_chunks:
            section_chunks = [section_doc.page_content.strip()]

        for local_index, chunk in enumerate(section_chunks):
            if not chunk.strip():
                continue

            global_index += 1
            chunk_meta = {
                **base_metadata,
                **meta,
                "section_id": _section_id(section_title),
                "section_title": section_title,
                "section_path": section_path,
                "subsection_title": meta.get("Header 3") or meta.get("Header 2") or None,
                "chunk_index": local_index,
                "global_chunk_index": global_index,
                "chunk_type": _classify_chunk_type(chunk),
                "contains_table": _classify_chunk_type(chunk) == "table",
                "char_count": len(chunk),
                "token_estimate": max(1, len(chunk) // 4),
            }
            chunk_meta["quality_flags"] = ",".join(_quality_flags(chunk, section_title))
            chunk_meta["chunk_id"] = (
                f"{base_metadata['doc_id']}__{chunk_meta['section_id']}__{global_index:04d}"
            )

            final_docs.append(
                Document(
                    page_content=_to_embedding_text(chunk_meta, chunk),
                    metadata=chunk_meta,
                )
            )

    return final_docs


def _persist_parse_artifacts(
    pdf_path: Path,
    base_metadata: Dict[str, Any],
    pages: List[Dict[str, Any]],
    docs: List[Document],
) -> None:
    processed_dir = PROCESSED_DIR / base_metadata["doc_id"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    (processed_dir / "parsed_markdown.md").write_text(
        "\n\n".join(page["markdown"] for page in pages),
        encoding="utf-8",
    )
    (processed_dir / "parsed_pages.json").write_text(
        json.dumps(
            {
                "pdf_path": str(pdf_path),
                "doc_id": base_metadata["doc_id"],
                "pages": pages,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (processed_dir / "chunk_manifest.json").write_text(
        json.dumps(
            [
                {
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "section_title": doc.metadata.get("section_title"),
                    "section_path": doc.metadata.get("section_path"),
                    "page_start": doc.metadata.get("page_start"),
                    "page_end": doc.metadata.get("page_end"),
                    "chunk_type": doc.metadata.get("chunk_type"),
                    "contains_table": doc.metadata.get("contains_table"),
                    "char_count": doc.metadata.get("char_count"),
                    "token_estimate": doc.metadata.get("token_estimate"),
                    "quality_flags": doc.metadata.get("quality_flags"),
                }
                for doc in docs
            ],
            indent=2,
        ),
        encoding="utf-8",
    )


async def parse_10k_with_llamacloud(pdf_path: Path | None = None) -> List[Document]:
    """Parse a 10-K PDF with LlamaCloud and return retrieval-ready chunks.

    Pipeline:
    - Parse the PDF into page-level markdown with LlamaCloud.
    - Persist the raw parse for inspection under ``data/processed/<doc_id>``.
    - Split each page by markdown headers to preserve filing structure.
    - Merge adjacent chunks from the same section across page boundaries.
    - Rechunk long sections to retrieval-sized chunks while preserving tables.
    - Attach metadata for citation, filtering, and future evaluation.
    """

    if pdf_path is None:
        candidates = sorted(
            RAW_DATA_DIR.glob("*.pdf"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No PDF found in {RAW_DATA_DIR}. Upload a file or provide pdf_path."
            )
        pdf_path = candidates[0]

    pdf_path = pdf_path.resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"10-K PDF not found at {pdf_path}")

    client = AsyncLlamaCloud()
    file = await client.files.create(file=pdf_path, purpose="parse")
    result = await client.parsing.parse(
        file_id=file.id,
        tier="agentic",
        version="latest",
        expand=["markdown"],
    )

    pages = _extract_pages(result)
    if not pages:
        raise ValueError(f"LlamaCloud returned no markdown pages for {pdf_path.name}")

    base_metadata = _extract_filing_metadata(pages, pdf_path)
    page_docs = _page_level_header_docs(pages, base_metadata)
    merged_sections = _merge_adjacent_sections(page_docs)
    final_docs = _rechunk_sections(merged_sections, base_metadata)
    _persist_parse_artifacts(pdf_path, base_metadata, pages, final_docs)

    print(
        f"Parsed 10-K '{pdf_path.name}' into {len(final_docs)} chunks "
        f"across {len(merged_sections)} merged sections."
    )
    return final_docs


async def load_10k_documents(pdf_path: Path | None = None) -> List[Document]:
    """High-level entry point used by the RAG pipeline."""

    return await parse_10k_with_llamacloud(pdf_path=pdf_path)
