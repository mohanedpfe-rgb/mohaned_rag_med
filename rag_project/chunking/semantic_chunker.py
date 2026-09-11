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
            exact = f"[TABLE]\n{table_text}"
            prose = prose.replace(exact, "")
            compact = re.sub(r"\s+", " ", table_text)
            if compact and compact != table_text:
                prose = prose.replace(f"[TABLE]\n{compact}", "")
        return prose

    @staticmethod
    def _compact_children(children: list[str]) -> list[str]:
        """Avoid standalone heading-only chunks that waste the searchable budget."""
        if not children:
            return []
        merged: list[str] = []
        pending = ""
        for index, child in enumerate(children):
            value = str(child or "").strip()
            if not value:
                continue
            tokens = re.findall(r"\w+", value, flags=re.UNICODE)
            heading_like = bool(re.match(r"^#{0,3}\s*(?:chapter|chapitre|\d+(?:\.\d+)*|[IVXLC]+)[\s.)]", value, re.I)) and len(tokens) <= 8
            if heading_like and index + 1 < len(children):
                pending = f"{pending}\n{value}".strip()
                continue
            if pending:
                value = f"{pending}\n{value}".strip()
                pending = ""
            merged.append(value)
        if pending:
            if merged:
                merged[-1] = f"{merged[-1]}\n{pending}".strip()
            else:
                merged.append(pending)
        return merged

    def chunk_pages(self, pages: List[PageExtraction]) -> List[Chunk]:
        pages = list(pages or [])
        if not pages:
            return []

        child_splitter = self._child_splitter()
        chunks: List[Chunk] = []

        for page in pages:
            table_texts = [str(x or "").strip() for x in list(getattr(page, "table_texts", []) or []) if str(x or "").strip()]
            table_ids = list(getattr(page, "table_ids", []) or [])
            figure_ids = list(getattr(page, "figure_ids", []) or [])
            captions = [str(x or "").strip() for x in list(getattr(page, "figure_captions", []) or []) if str(x or "").strip()]

            evidence: set[str] = {"text"}
            if bool(getattr(page, "has_images", False)) or figure_ids or captions:
                evidence.add("figure")
            if int(getattr(page, "table_count", 0) or 0) or table_ids or table_texts:
                evidence.add("table")
            evidence_types = [name for name in ("figure", "table", "text") if name in evidence]

            prose_text = self._remove_table_blocks(page.text or "", table_texts)
            page_enriched = enrich_text(prose_text)
            headings = page_enriched.get("headings") or []
            fallback_heading = headings[0] if headings else self._section_fallback(prose_text)

            canonical_rows: list[tuple[str, str | None, str | None, str, str, str]] = []
            for parent_text, parent_meta in self._parent_sections(prose_text):
                section_title = parent_meta.get("section") or parent_meta.get("subsection") or fallback_heading
                chapter_title = parent_meta.get("chapter")
                global_section_key = f"{page.page_number}|{chapter_title or ''}|{section_title or '__page__'}"
                parent_id = f"{page.document_id}:p{page.page_number}:parent:{self._stable_id(global_section_key)}"
                section_id = f"{page.document_id}:p{page.page_number}:section:{self._stable_id(global_section_key)}"
                chapter_id = f"{page.document_id}:p{page.page_number}:chapter:{self._stable_id(chapter_title or '__document__')}"
                canonical_rows.append((parent_text, chapter_title, section_title, parent_id, section_id, chapter_id))

            # Page-level specialized units use the first canonical hierarchy anchor.
            anchor = canonical_rows[0] if canonical_rows else (prose_text, None, fallback_heading, f"{page.document_id}:p{page.page_number}:parent:{self._stable_id(str(page.page_number))}", f"{page.document_id}:p{page.page_number}:section:{self._stable_id(str(page.page_number))}", f"{page.document_id}:p{page.page_number}:chapter:{self._stable_id('__document__')}")
            anchor_parent_id, anchor_section_id, anchor_chapter_id = anchor[3], anchor[4], anchor[5]
            anchor_chapter, anchor_section = anchor[1], anchor[2]

            for parent_text, chapter_title, section_title, parent_id, section_id, chapter_id in canonical_rows or [anchor]:
                children = self._compact_children(child_splitter.split_text(parent_text) or [parent_text])
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
                    # Keep the searchable text compact. Structural identity belongs
                    # in metadata; the compact schema marker is retained for index QA.
                    search_text = "[RAG-STRUCTURE schema=3]\n"
                    if prefix:
                        search_text += f"[{prefix}]\n"
                    search_text += child_text
                    metadata = {
                        "source_pages": [page.page_number or 1],
                        "page_numbers": [page.page_number or 1],
                        "evidence_types": evidence_types,
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
                        "table_id": table_ids[0] if table_ids else None,
                        "figure_id": figure_ids[0] if figure_ids else None,
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
                            table_id=metadata["table_id"],
                            figure_id=metadata["figure_id"],
                            normalized_text=enriched["normalized_text"],
                        )
                    )

            for table_index, table_text in enumerate(table_texts):
                table_id = table_ids[table_index] if table_index < len(table_ids) else f"{page.document_id}:p{page.page_number}:table:{table_index + 1}"
                table_enriched = enrich_text(table_text)
                metadata = {
                    "source_pages": [page.page_number or 1],
                    "page_numbers": [page.page_number or 1],
                    "evidence_types": evidence_types,
                    "chapter": anchor_chapter,
                    "chapter_id": anchor_chapter_id,
                    "section": anchor_section,
                    "section_id": anchor_section_id,
                    "global_section_id": anchor_section_id,
                    "parent_id": anchor_parent_id,
                    "parent_text": table_text,
                    "child_index": table_index,
                    "normalized_text": table_enriched["normalized_text"],
                    "entities": table_enriched["entities"],
                    "headings": table_enriched["headings"],
                    "number_forms": table_enriched["number_forms"],
                    "table_id": table_id,
                    "figure_id": figure_ids[0] if figure_ids else None,
                    "document_id": page.document_id,
                    "file_name": page.file_name,
                    "page_type": page.page_type,
                    "quality_score": page.quality_score,
                    "ocr_status": page.ocr_status,
                    "routing_decision": page.routing_decision,
                    "representation_type": "table",
                }
                chunks.append(Chunk(page.document_id, page.file_name, len(chunks), f"[TABLE]\nSection: {anchor_section}\n{table_text}" if anchor_section else f"[TABLE]\n{table_text}", [page.page_number or 1], metadata, "table", anchor_parent_id, anchor_section_id, table_id, metadata["figure_id"], table_enriched["normalized_text"]))

            for figure_index, caption in enumerate(captions):
                figure_id = figure_ids[figure_index] if figure_index < len(figure_ids) else f"{page.document_id}:p{page.page_number}:figure:{figure_index + 1}"
                figure_enriched = enrich_text(caption)
                metadata = {
                    "source_pages": [page.page_number or 1],
                    "page_numbers": [page.page_number or 1],
                    "evidence_types": evidence_types,
                    "chapter": anchor_chapter,
                    "chapter_id": anchor_chapter_id,
                    "section": anchor_section,
                    "section_id": anchor_section_id,
                    "global_section_id": anchor_section_id,
                    "parent_id": anchor_parent_id,
                    "parent_text": caption,
                    "child_index": figure_index,
                    "normalized_text": figure_enriched["normalized_text"],
                    "entities": figure_enriched["entities"],
                    "headings": figure_enriched["headings"],
                    "number_forms": figure_enriched["number_forms"],
                    "table_id": table_ids[0] if table_ids else None,
                    "figure_id": figure_id,
                    "document_id": page.document_id,
                    "file_name": page.file_name,
                    "page_type": page.page_type,
                    "quality_score": page.quality_score,
                    "ocr_status": page.ocr_status,
                    "routing_decision": page.routing_decision,
                    "representation_type": "figure_caption",
                }
                chunks.append(Chunk(page.document_id, page.file_name, len(chunks), f"[FIGURE CAPTION]\n{caption}", [page.page_number or 1], metadata, "figure_caption", anchor_parent_id, anchor_section_id, metadata["table_id"], figure_id, figure_enriched["normalized_text"]))

            # Normalize all units on a page to one canonical hierarchy anchor.
            page_chunks = [c for c in chunks if c.doc_id == page.document_id and int((c.page_numbers or [page.page_number])[0]) == int(page.page_number)]
            for chunk in page_chunks:
                meta = dict(chunk.metadata or {})
                for field in ("parent_id", "section_id", "chapter_id", "chapter", "section", "global_section_id"):
                    meta[field] = anchor[3] if field == "parent_id" else anchor[4] if field in {"section_id", "global_section_id"} else anchor[5] if field == "chapter_id" else anchor[1] if field == "chapter" else anchor[2]
                meta["evidence_types"] = evidence_types
                chunk.parent_id = anchor[3]
                chunk.section_id = anchor[4]
                chunk.metadata = meta
            
        for index, chunk in enumerate(chunks):
            chunk.chunk_index = index
        return chunks

    def chunk_page_batches(self, pages: Iterable[PageExtraction], batch_size: int = 16) -> Iterator[List[Chunk]]:
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
