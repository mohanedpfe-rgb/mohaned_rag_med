from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator
from typing import List

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from rag_project.ingestion.document_models import Chunk, PageExtraction
from rag_project.intelligence.pdf_intelligence import enrich_text


class SemanticChunker:
    """Structure-aware child chunking with durable hierarchy context."""

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

    @staticmethod
    def _stable_id(value: str) -> str:
        return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]

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

    @staticmethod
    def _remove_table_blocks(text: str, table_texts: list[str]) -> str:
        prose = text or ""
        for table_text in table_texts:
            table_text = str(table_text or "").strip()
            if not table_text:
                continue
            prose = prose.replace(f"[TABLE]\n{table_text}", "")
        return prose

    def chunk_pages(self, pages: List[PageExtraction]) -> List[Chunk]:
        if not pages:
            return []

        child_splitter = self._child_splitter()
        chunks: List[Chunk] = []

        for page in pages:
            section_index = 0
            table_texts = list(getattr(page, "table_texts", []) or [])
            table_ids = list(getattr(page, "table_ids", []) or [])
            figure_ids = list(getattr(page, "figure_ids", []) or [])
            captions = list(getattr(page, "figure_captions", []) or [])

            evidence: set[str] = {"text"}
            if page.has_images or figure_ids:
                evidence.add("figure")
            if page.table_count or table_ids or table_texts:
                evidence.add("table")

            prose_text = self._remove_table_blocks(page.text or "", table_texts)
            page_enriched = enrich_text(prose_text)
            headings = page_enriched.get("headings") or []
            fallback_heading = headings[0] if headings else self._section_fallback(prose_text)

            for parent_text, parent_meta in self._parent_sections(prose_text):
                section_title = parent_meta.get("section") or parent_meta.get("subsection") or fallback_heading
                chapter_title = parent_meta.get("chapter")
                chapter_id = f"{page.document_id}:chapter:{self._stable_id(chapter_title or '__document__')}"
                global_section_key = f"{chapter_title or ''}|{section_title or '__page__'}"
                parent_id = f"{page.document_id}:parent:{self._stable_id(global_section_key)}"
                section_id = f"{page.document_id}:section:{self._stable_id(global_section_key)}"

                children = child_splitter.split_text(parent_text) or [parent_text]
                for child_index, child_text in enumerate(children):
                    child_text = child_text.strip()
                    if not child_text:
                        continue
                    enriched = enrich_text(child_text)
                    prefix_parts: list[str] = []
                    if chapter_title:
                        prefix_parts.append(f"Chapter: {chapter_title}")
                    if section_title:
                        prefix_parts.append(f"Section: {section_title}")
                    prefix = " - ".join(prefix_parts)
                    structure_header = (
                        f"[RAG-STRUCTURE chapter_id={chapter_id}; chapter={chapter_title or ''}; "
                        f"section_id={section_id}; section={section_title or ''}; parent_id={parent_id}; "
                        f"quality={float(page.quality_score or 0.0):.4f}; page_type={page.page_type or 'unknown'}; "
                        f"ocr_status={page.ocr_status or 'not_required'}; table_id=; figure_id=]"
                    )
                    search_text = f"{structure_header}\n"
                    if prefix:
                        search_text += f"[{prefix}]\n"
                    search_text += child_text

                    metadata = {
                        "source_pages": [page.page_number or 1],
                        "page_numbers": [page.page_number or 1],
                        "evidence_types": sorted(evidence),
                        "chapter": chapter_title,
                        "chapter_id": chapter_id,
                        "section": section_title,
                        "section_id": section_id,
                        "global_section_id": section_id,
                        "parent_id": parent_id,
                        "parent_text": parent_text,
                        "child_index": child_index,
                        "normalized_text": enriched["normalized_text"],
                        "entities": enriched["entities"],
                        "headings": enriched["headings"],
                        "number_forms": enriched["number_forms"],
                        "table_id": None,
                        "figure_id": None,
                        "document_id": page.document_id,
                        "file_name": page.file_name,
                        "page_type": page.page_type,
                        "quality_score": page.quality_score,
                        "ocr_status": page.ocr_status,
                        "ocr_confidence": page.ocr_confidence,
                        "routing_decision": page.routing_decision,
                    }
                    chunks.append(
                        Chunk(
                            doc_id=page.document_id,
                            file_name=page.file_name,
                            chunk_index=len(chunks),
                            text=search_text,
                            page_numbers=[page.page_number or 1],
                            metadata=metadata,
                            representation_type="canonical",
                            parent_id=parent_id,
                            section_id=section_id,
                            table_id=None,
                            figure_id=None,
                            normalized_text=enriched["normalized_text"],
                        )
                    )

                section_index += 1

            for table_index, table_text in enumerate(table_texts):
                table_text = str(table_text or "").strip()
                if not table_text:
                    continue
                table_id = table_ids[table_index] if table_index < len(table_ids) else f"{page.document_id}:p{page.page_number}:table:{table_index + 1}"
                table_enriched = enrich_text(table_text)
                parent_id = f"{table_id}:parent"
                section_id = f"{table_id}:section"
                prefix = f"Section: {fallback_heading}\n" if fallback_heading else ""
                searchable = f"[TABLE]\n{prefix}{table_text}".strip()
                metadata = {
                    "source_pages": [page.page_number or 1],
                    "page_numbers": [page.page_number or 1],
                    "evidence_types": ["table"],
                    "chapter": None,
                    "section": fallback_heading,
                    "section_id": section_id,
                    "parent_id": parent_id,
                    "parent_text": table_text,
                    "child_index": table_index,
                    "normalized_text": table_enriched["normalized_text"],
                    "entities": table_enriched["entities"],
                    "headings": table_enriched["headings"],
                    "number_forms": table_enriched["number_forms"],
                    "table_id": table_id,
                    "figure_id": None,
                    "document_id": page.document_id,
                    "file_name": page.file_name,
                    "page_type": page.page_type,
                    "quality_score": page.quality_score,
                    "ocr_status": page.ocr_status,
                    "routing_decision": page.routing_decision,
                }
                chunks.append(Chunk(doc_id=page.document_id, file_name=page.file_name, chunk_index=len(chunks), text=searchable, page_numbers=[page.page_number or 1], metadata=metadata, representation_type="table", parent_id=parent_id, section_id=section_id, table_id=table_id, figure_id=None, normalized_text=table_enriched["normalized_text"]))

            for figure_index, caption in enumerate(captions):
                caption = str(caption or "").strip()
                if not caption:
                    continue
                figure_id = figure_ids[figure_index] if figure_index < len(figure_ids) else f"{page.document_id}:p{page.page_number}:figure:{figure_index + 1}"
                figure_enriched = enrich_text(caption)
                parent_id = f"{figure_id}:parent"
                section_id = f"{figure_id}:section"
                metadata = {
                    "source_pages": [page.page_number or 1],
                    "page_numbers": [page.page_number or 1],
                    "evidence_types": ["figure"],
                    "chapter": None,
                    "section": fallback_heading,
                    "section_id": section_id,
                    "parent_id": parent_id,
                    "parent_text": caption,
                    "child_index": figure_index,
                    "normalized_text": figure_enriched["normalized_text"],
                    "entities": figure_enriched["entities"],
                    "headings": figure_enriched["headings"],
                    "number_forms": figure_enriched["number_forms"],
                    "table_id": None,
                    "figure_id": figure_id,
                    "document_id": page.document_id,
                    "file_name": page.file_name,
                    "page_type": page.page_type,
                    "quality_score": page.quality_score,
                    "ocr_status": page.ocr_status,
                    "routing_decision": page.routing_decision,
                }
                chunks.append(Chunk(doc_id=page.document_id, file_name=page.file_name, chunk_index=len(chunks), text=f"[FIGURE CAPTION]\n{caption}", page_numbers=[page.page_number or 1], metadata=metadata, representation_type="figure_caption", parent_id=parent_id, section_id=section_id, table_id=None, figure_id=figure_id, normalized_text=figure_enriched["normalized_text"]))

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
