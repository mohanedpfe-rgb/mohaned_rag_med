from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_project.ingestion.document_models import Chunk, PageExtraction
from rag_project.intelligence.pdf_intelligence import enrich_text


class SemanticChunker:
    """Structure-aware chunker that preserves hierarchy and specialized evidence."""

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 120):
        self.chunk_size = max(1, int(chunk_size))
        self.chunk_overlap = max(0, int(chunk_overlap))

    @staticmethod
    def _stable_id(value: str) -> str:
        return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _section_fallback(text: str) -> str | None:
        for line in str(text or "").splitlines():
            candidate = line.strip()
            if re.match(r"^(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s+\S", candidate) and len(candidate) <= 180:
                return candidate
        return None

    @staticmethod
    def _is_heading_line(line: str) -> bool:
        return bool(re.match(r"^\s*#{1,3}\s+\S", str(line or "")))

    def _child_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=min(self.chunk_overlap, max(0, self.chunk_size // 2)),
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        )

    def _parent_sections(self, text: str) -> list[tuple[str, dict[str, str]]]:
        value = str(text or "").replace("\r\n", "\n")
        lines = value.split("\n")
        sections: list[tuple[str, dict[str, str]]] = []
        chapter: str | None = None
        section: str | None = None
        subsection: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            nonlocal buffer
            content = "\n".join(buffer).strip()
            if content:
                meta: dict[str, str] = {}
                if chapter:
                    meta["chapter"] = chapter
                if section:
                    meta["section"] = section
                if subsection:
                    meta["subsection"] = subsection
                sections.append((content, meta))
            buffer = []

        for line in lines:
            match = re.match(r"^\s*(#{1,3})\s+(.+?)\s*$", line)
            if match:
                if buffer and any(not self._is_heading_line(item) and item.strip() for item in buffer):
                    flush()
                else:
                    buffer = []
                title = match.group(2).strip()
                level = len(match.group(1))
                if level == 1:
                    chapter, section, subsection = title, None, None
                elif level == 2:
                    section, subsection = title, None
                else:
                    subsection = title
                buffer.append(title)
            else:
                buffer.append(line)
        flush()
        if sections:
            return sections
        fallback = self._section_fallback(value)
        return [(value, {"section": fallback} if fallback else {})] if value.strip() else []

    @staticmethod
    def _remove_table_blocks(text: str, table_texts: list[str]) -> str:
        prose = str(text or "")
        for table_text in table_texts:
            if not table_text:
                continue
            prose = prose.replace(f"[TABLE]\n{table_text}", "")
            compact = re.sub(r"\s+", " ", table_text)
            if compact and compact != table_text:
                prose = prose.replace(f"[TABLE]\n{compact}", "")
        return prose

    @staticmethod
    def _compact_children(children: list[str]) -> list[str]:
        values = [str(v).strip() for v in children if str(v).strip()]
        out: list[str] = []
        pending = ""
        for value in values:
            words = re.findall(r"\w+", value, flags=re.UNICODE)
            heading_only = len(words) <= 8 and bool(re.match(r"^(?:#\s*)?(?:chapter|chapitre|section|\d+(?:\.\d+)*|[IVXLC]+)\b", value, re.I))
            if heading_only and not pending:
                pending = value
                continue
            if pending:
                value = f"{pending}\n{value}".strip()
                pending = ""
            out.append(value)
        if pending:
            if out:
                out[-1] = f"{out[-1]}\n{pending}".strip()
            else:
                out.append(pending)
        return out

    def chunk_pages(self, pages: List[PageExtraction]) -> List[Chunk]:
        pages = list(pages or [])
        if not pages:
            return []
        splitter = self._child_splitter()
        chunks: list[Chunk] = []
        document_chapter: str | None = None

        for page in pages:
            page_no = int(getattr(page, "page_number", getattr(page, "page_index", 0) + 1) or 1)
            table_texts = [str(v).strip() for v in (getattr(page, "table_texts", []) or []) if str(v).strip()]
            table_ids = list(getattr(page, "table_ids", []) or [])
            figure_ids = list(getattr(page, "figure_ids", []) or [])
            captions = [str(v).strip() for v in (getattr(page, "figure_captions", []) or []) if str(v).strip()]
            image_count = int(getattr(page, "image_count", 0) or 0)
            table_count = int(getattr(page, "table_count", 0) or 0)

            evidence = {"text"}
            if bool(getattr(page, "has_images", False)) or image_count > 0 or figure_ids or captions:
                evidence.add("figure")
            if table_count > 0 or table_ids or table_texts:
                evidence.add("table")
            evidence_types = [name for name in ("figure", "table", "text") if name in evidence]

            prose = self._remove_table_blocks(page.text or "", table_texts)
            page_info = enrich_text(prose)
            fallback = (page_info.get("headings") or [None])[0] or self._section_fallback(prose)
            rows = self._parent_sections(prose) or [(prose, {})]
            structured_page = any(bool(meta.get(key)) for _, meta in rows for key in ("chapter", "section", "subsection"))
            page_splitter = splitter if structured_page else RecursiveCharacterTextSplitter(
                chunk_size=max(1, self.chunk_size - 60),
                chunk_overlap=min(self.chunk_overlap, max(0, (self.chunk_size - 60) // 2)),
                separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
            )

            canonical_rows: list[tuple[str, str | None, str | None, str, str, str]] = []
            for parent_text, hierarchy in rows:
                chapter = hierarchy.get("chapter")
                if chapter:
                    document_chapter = chapter
                elif document_chapter:
                    chapter = document_chapter
                section = hierarchy.get("subsection") or hierarchy.get("section") or fallback
                key = f"{page_no}|{chapter or ''}|{section or '__page__'}"
                parent_id = f"{page.document_id}:p{page_no}:parent:{self._stable_id(key)}"
                section_id = f"{page.document_id}:p{page_no}:section:{self._stable_id(key)}"
                # A chapter spans pages; page number belongs to the section
                # and parent identities, not to the chapter identity.
                chapter_id = f"{page.document_id}:chapter:{self._stable_id(chapter or document_chapter or '__document__')}"
                canonical_rows.append((parent_text, chapter, section, parent_id, section_id, chapter_id))

            # Keep a standalone chapter heading attached to the first real
            # section so the first canonical chunk carries usable evidence.
            if len(canonical_rows) > 1:
                head = canonical_rows[0]
                if len(head[0].split()) <= 8 and len(head[0].splitlines()) == 1:
                    nxt = canonical_rows[1]
                    canonical_rows[1] = (f"{head[0]}\n{nxt[0]}", nxt[1] or head[1], nxt[2], nxt[3], nxt[4], nxt[5])
                    canonical_rows.pop(0)

            anchor = canonical_rows[0] if canonical_rows else (
                prose,
                None,
                fallback,
                f"{page.document_id}:p{page_no}:parent:{self._stable_id(str(page_no))}",
                f"{page.document_id}:p{page_no}:section:{self._stable_id(str(page_no))}",
                f"{page.document_id}:p{page_no}:chapter:{self._stable_id('__document__')}",
            )
            anchor_parent, anchor_section, anchor_chapter = anchor[3], anchor[4], anchor[5]
            anchor_chapter_title, anchor_section_title = anchor[1], anchor[2]
            page_chunk_start = len(chunks)

            for parent_text, chapter, section, parent_id, section_id, chapter_id in canonical_rows or [anchor]:
                children = self._compact_children(page_splitter.split_text(parent_text)) or ([parent_text.strip()] if parent_text.strip() else [])
                hierarchy_path = [anchor_chapter, anchor_section, anchor_parent]
                for child_index, child in enumerate(children):
                    enriched = enrich_text(child)
                    # A structural heading can be emitted as its own first
                    # splitter fragment.  Its searchable representation still
                    # belongs to the complete parent section, so retain the
                    # section's normalized evidence for entity/routing use.
                    if child_index == 0 and len(children) > 1 and self._is_heading_line(child):
                        enriched = dict(enriched)
                        enriched["normalized_text"] = enrich_text(parent_text).get("normalized_text", enriched["normalized_text"])
                    prefix = " - ".join(x for x in (f"Chapter: {chapter}" if chapter else "", f"Section: {section}" if section else "") if x)
                    marker = (
                        f"[RAG-STRUCTURE schema=3; page={page_no}]"
                        if not structured_page else
                        "[RAG-STRUCTURE schema=3; "
                        f"chapter_id={anchor_chapter}; chapter={chapter or ''}; "
                        f"section_id={anchor_section}; section={section or ''}; "
                        f"parent_id={anchor_parent}; path={json.dumps(hierarchy_path, ensure_ascii=False)}; "
                        f"page={page_no}; quality={float(getattr(page, 'quality_score', 0.0) or 0.0)}; "
                        f"ocr={getattr(page, 'ocr_status', 'not_required')}]"
                    )
                    searchable = marker + (f"\n[{prefix}]\n" if prefix else "\n") + child.strip()
                    metadata = {
                        "source_pages": [page_no], "page_numbers": [page_no], "evidence_types": evidence_types,
                        "chapter": chapter, "chapter_id": anchor_chapter, "section": section, "section_id": anchor_section,
                        "global_section_id": anchor_section, "parent_id": anchor_parent, "hierarchy_path": hierarchy_path,
                        "parent_text": parent_text, "child_index": child_index, "normalized_text": enriched["normalized_text"],
                        "entities": enriched["entities"], "headings": enriched["headings"], "number_forms": enriched["number_forms"],
                        "table_id": table_ids[0] if table_ids else None, "figure_id": figure_ids[0] if figure_ids else None,
                        "document_id": page.document_id, "file_name": page.file_name, "page_type": page.page_type,
                        "quality_score": page.quality_score, "ocr_status": page.ocr_status,
                        "ocr_confidence": getattr(page, "ocr_confidence", None), "routing_decision": getattr(page, "routing_decision", None),
                    }
                    chunks.append(Chunk(page.document_id, page.file_name, len(chunks), searchable, [page_no], metadata, "canonical", anchor_parent, anchor_section, metadata["table_id"], metadata["figure_id"], enriched["normalized_text"]))

            for table_index, table_text in enumerate(table_texts):
                table_id = table_ids[table_index] if table_index < len(table_ids) else f"{page.document_id}:p{page_no}:table:{table_index + 1}"
                enriched = enrich_text(table_text)
                hierarchy_path = [anchor_chapter, anchor_section, anchor_parent]
                metadata = {
                    "source_pages": [page_no], "page_numbers": [page_no], "evidence_types": evidence_types,
                    "chapter": anchor_chapter_title, "chapter_id": anchor_chapter, "section": anchor_section_title,
                    "section_id": anchor_section, "global_section_id": anchor_section, "parent_id": anchor_parent,
                    "hierarchy_path": hierarchy_path, "parent_text": table_text, "child_index": table_index,
                    "normalized_text": enriched["normalized_text"], "entities": enriched["entities"], "headings": enriched["headings"],
                    "number_forms": enriched["number_forms"], "table_id": table_id, "figure_id": figure_ids[0] if figure_ids else None,
                    "document_id": page.document_id, "file_name": page.file_name, "page_type": page.page_type,
                    "quality_score": page.quality_score, "ocr_status": page.ocr_status,
                    "routing_decision": getattr(page, "routing_decision", None), "representation_type": "table",
                }
                text = f"[TABLE]\nSection: {anchor_section_title}\n{table_text}" if anchor_section_title else f"[TABLE]\n{table_text}"
                chunks.append(Chunk(page.document_id, page.file_name, len(chunks), text, [page_no], metadata, "table", anchor_parent, anchor_section, table_id, metadata["figure_id"], enriched["normalized_text"]))

            for figure_index, caption in enumerate(captions):
                figure_id = figure_ids[figure_index] if figure_index < len(figure_ids) else f"{page.document_id}:p{page_no}:figure:{figure_index + 1}"
                enriched = enrich_text(caption)
                hierarchy_path = [anchor_chapter, anchor_section, anchor_parent]
                metadata = {
                    "source_pages": [page_no], "page_numbers": [page_no], "evidence_types": evidence_types,
                    "chapter": anchor_chapter_title, "chapter_id": anchor_chapter, "section": anchor_section_title,
                    "section_id": anchor_section, "global_section_id": anchor_section, "parent_id": anchor_parent,
                    "hierarchy_path": hierarchy_path, "parent_text": caption, "child_index": figure_index,
                    "normalized_text": enriched["normalized_text"], "entities": enriched["entities"], "headings": enriched["headings"],
                    "number_forms": enriched["number_forms"], "table_id": table_ids[0] if table_ids else None, "figure_id": figure_id,
                    "document_id": page.document_id, "file_name": page.file_name, "page_type": page.page_type,
                    "quality_score": page.quality_score, "ocr_status": page.ocr_status,
                    "routing_decision": getattr(page, "routing_decision", None), "representation_type": "figure_caption",
                }
                chunks.append(Chunk(page.document_id, page.file_name, len(chunks), f"[FIGURE CAPTION]\n{caption}", [page_no], metadata, "figure_caption", anchor_parent, anchor_section, metadata["table_id"], figure_id, enriched["normalized_text"]))

            for chunk in chunks[page_chunk_start:]:
                meta = dict(chunk.metadata or {})
                meta["chapter_id"] = anchor_chapter
                meta["section_id"] = anchor_section
                meta["global_section_id"] = anchor_section
                meta["parent_id"] = anchor_parent
                meta["hierarchy_path"] = [anchor_chapter, anchor_section, anchor_parent]
                chunk.metadata = meta
                chunk.parent_id = anchor_parent
                chunk.section_id = anchor_section

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
