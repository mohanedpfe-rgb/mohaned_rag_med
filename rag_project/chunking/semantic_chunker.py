from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from typing import List

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from rag_project.ingestion.document_models import Chunk, PageExtraction
from rag_project.intelligence.pdf_intelligence import enrich_text


class SemanticChunker:
    """Structure-aware child chunking with explicit parent/section and alternate search representations."""

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 120):
        self.chunk_size = max(1, int(chunk_size))
        self.chunk_overlap = max(0, int(chunk_overlap))
        self._headers = [("#", "chapter"), ("##", "section"), ("###", "subsection")]

    @staticmethod
    def _section_fallback(text: str) -> str | None:
        for line in (text or "").splitlines():
            line = line.strip()
            if re.match(r"^(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s+\S", line) and len(line) <= 180:
                return line
        return None

    def _parent_sections(self, text: str) -> list[tuple[str, dict[str, str]]]:
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self._headers)
        try:
            sections = splitter.split_text(text or "")
        except Exception:
            sections = []
        if not sections:
            fallback = self._section_fallback(text)
            return [(text or "", {"section": fallback} if fallback else {})]
        return [(s.page_content, {k: str(v) for k, v in s.metadata.items()}) for s in sections]

    def _child_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=min(self.chunk_overlap, max(0, self.chunk_size // 2)),
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        )

    def chunk_pages(self, pages: List[PageExtraction]) -> List[Chunk]:
        if not pages:
            return []
        child_splitter = self._child_splitter()
        chunks: List[Chunk] = []
        for page in pages:
            section_index = 0
            evidence: set[str] = {"text"}
            if page.has_images or page.figure_ids:
                evidence.add("figure")
            if page.table_count or page.table_ids:
                evidence.add("table")
            page_enriched = enrich_text(page.text or "")
            headings = page_enriched.get("headings") or []
            fallback_heading = headings[0] if headings else None
            for parent_text, parent_meta in self._parent_sections(page.text or ""):
                section_title = parent_meta.get("section") or parent_meta.get("subsection") or fallback_heading
                chapter_title = parent_meta.get("chapter")
                parent_id = f"{page.document_id}:p{page.page_number}:parent:{section_index}"
                section_id = f"{page.document_id}:p{page.page_number}:section:{section_index}"
                children = child_splitter.split_text(parent_text) or [parent_text]
                for child_index, child_text in enumerate(children):
                    child_text = child_text.strip()
                    if not child_text:
                        continue
                    enriched = enrich_text(child_text)
                    table_id = page.table_ids[section_index % len(page.table_ids)] if page.table_ids and "table" in evidence else None
                    figure_id = page.figure_ids[section_index % len(page.figure_ids)] if page.figure_ids and "figure" in evidence else None
                    prefix_parts = []
                    if chapter_title:
                        prefix_parts.append(f"Chapter: {chapter_title}")
                    if section_title:
                        prefix_parts.append(f"Section: {section_title}")
                    prefix = " - ".join(prefix_parts)
                    search_text = f"[{prefix}]\n{child_text}" if prefix else child_text
                    metadata = {
                        "source_pages": [page.page_number or 1],
                        "page_numbers": [page.page_number or 1],
                        "evidence_types": sorted(evidence),
                        "chapter": chapter_title,
                        "section": section_title,
                        "section_id": section_id,
                        "parent_id": parent_id,
                        "parent_text": parent_text,
                        "child_index": child_index,
                        "normalized_text": enriched["normalized_text"],
                        "entities": enriched["entities"],
                        "headings": enriched["headings"],
                        "number_forms": enriched["number_forms"],
                        "table_id": table_id,
                        "figure_id": figure_id,
                        "document_id": page.document_id,
                        "file_name": page.file_name,
                        "page_type": page.page_type,
                        "quality_score": page.quality_score,
                        "ocr_status": page.ocr_status,
                    }
                    chunks.append(Chunk(
                        doc_id=page.document_id,
                        file_name=page.file_name,
                        chunk_index=len(chunks),
                        text=search_text,
                        page_numbers=[page.page_number or 1],
                        metadata=metadata,
                        representation_type="canonical",
                        parent_id=parent_id,
                        section_id=section_id,
                        table_id=table_id,
                        figure_id=figure_id,
                        normalized_text=enriched["normalized_text"],
                    ))
                section_index += 1
        return chunks

    def chunk_page_batches(self, pages: Iterable[PageExtraction], batch_size: int = 16) -> Iterator[List[Chunk]]:
        """Yield true bounded page batches with globally sequential chunk indices."""
        limit = max(1, int(batch_size))
        buffer: list[PageExtraction] = []
        offset = 0
        for page in pages:
            buffer.append(page)
            if len(buffer) < limit:
                continue
            batch = self.chunk_pages(buffer)
            for chunk in batch:
                chunk.chunk_index = offset
                offset += 1
            if batch:
                yield batch
            buffer.clear()
        if buffer:
            batch = self.chunk_pages(buffer)
            for chunk in batch:
                chunk.chunk_index = offset
                offset += 1
            if batch:
                yield batch
