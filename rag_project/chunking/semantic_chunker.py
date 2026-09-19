from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from typing import List, Tuple, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_project.ingestion.document_models import Chunk, PageExtraction
from rag_project.intelligence.pdf_intelligence import enrich_text


class SemanticChunker:
    """Structure-aware chunker that preserves hierarchy and specialized evidence."""

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 120):
        self.chunk_size = max(1, int(chunk_size))
        self.chunk_overlap = max(0, int(chunk_overlap))
        # Persist chapter state across batch calls to prevent chapters vanishing mid-document
        self._document_chapter: str | None = None
        self._current_document_id: str | None = None
        # Phase 33 fix: Medical-specific chunking patterns
        self._medical_boundary_patterns = [
            r'\b(?:diagnosis|treatment|management|guideline|protocol|algorithm)\b[:\s]',
            r'\b(?:contra-indication|side effect|adverse event|complication)\b[:\s]',
            r'\b(?:dosage|administration|monitoring|follow-up)\b[:\s]',
            r'\b(?:clinical trial|study|evidence|recommendation)\b[:\s]',
        ]

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
    
    @staticmethod
    def _detect_list_structure(text: str) -> Dict[str, Any]:
        """Phase 29 fix: Detect and preserve list structures."""
        lines = text.split('\n')
        list_items = []
        list_type = None
        
        for line in lines:
            stripped = line.strip()
            
            # Detect numbered lists
            numbered_match = re.match(r'^\s*(\d+[\.\)]\s+)', stripped)
            if numbered_match:
                list_type = "numbered"
                list_items.append({
                    'type': 'numbered',
                    'marker': numbered_match.group(1),
                    'content': stripped[numbered_match.end():],
                    'original': line
                })
                continue
            
            # Detect bulleted lists
            bullet_patterns = [r'^\s*[\-\*]\s+', r'^\s•\s+', r'^\s○\s+']
            for pattern in bullet_patterns:
                if re.match(pattern, stripped):
                    list_type = list_type or "bulleted"
                    list_items.append({
                        'type': 'bulleted',
                        'content': stripped,
                        'original': line
                    })
                    break
            else:
                # Non-list content
                if list_items:
                    # End of current list
                    break
        
        return {
            'has_list': len(list_items) > 1,
            'list_type': list_type,
            'list_items': list_items,
            'list_count': len(list_items)
        }
    
    @staticmethod
    def _preserve_formulas(text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """Phase 19 fix: Preserve mathematical formulas and equations."""
        # Protect mathematical expressions
        formula_patterns = [
            r'\$\$[^$]+\$\$',  # Display math: $$...$$ (must be checked first)
            r'\$[^$]+\$',  # LaTeX-style formulas: $...$
            r'\\[\[\{][^\\]*[\\\]\}]',  # LaTeX brackets: \[...\] or \{...\}
            r'[A-Za-z]\s*=\s*[^.]+?(?=[\n;]|$)',  # Simple equations: a = ...
        ]
        
        protected_text = text
        formula_placeholders = []
        
        for i, pattern in enumerate(formula_patterns):
            matches = re.finditer(pattern, protected_text)
            for match in reversed(list(matches)):
                formula = match.group()
                placeholder = f"__FORMULA_{i}_{len(formula_placeholders)}__"
                formula_placeholders.append((placeholder, formula))
                protected_text = protected_text[:match.start()] + placeholder + protected_text[match.end():]
        
        return protected_text, formula_placeholders
    
    @staticmethod
    def _restore_formulas(text: str, formula_placeholders: List[Tuple[str, str]]) -> str:
        """Restore preserved formulas."""
        restored = text
        for placeholder, formula in formula_placeholders:
            restored = restored.replace(placeholder, formula)
        return restored
    
    @staticmethod
    def _detect_column_layout(text: str) -> Dict[str, Any]:
        """Phase 8 fix: Enhanced multi-column layout detection."""
        lines = text.split('\n')
        
        if len(lines) < 3:
            return {"columns": 1, "confidence": 1.0}
        
        # Analyze line patterns to detect columns
        line_lengths = [len(line.strip()) for line in lines if line.strip()]
        if not line_lengths:
            return {"columns": 1, "confidence": 1.0}
        
        avg_length = sum(line_lengths) / len(line_lengths)
        max_length = max(line_lengths)
        
        # Look for patterns suggesting multiple columns
        # - Many short lines of similar length
        # - Alternating long/short patterns
        # - Gaps in horizontal spacing
        
        short_lines = [l for l in line_lengths if l < avg_length * 0.6]
        short_ratio = len(short_lines) / len(line_lengths)
        
        # Detect 2-column layout
        if short_ratio > 0.5 and avg_length < max_length * 0.7:
            return {
                "columns": 2,
                "confidence": min(0.9, short_ratio),
                "avg_length": avg_length,
                "max_length": max_length,
            }
        
        # Detect 3+ column layout (very structured)
        very_short_lines = [l for l in line_lengths if l < avg_length * 0.4]
        if len(very_short_lines) / len(line_lengths) > 0.6:
            return {
                "columns": 3,
                "confidence": min(0.8, len(very_short_lines) / len(line_lengths)),
                "avg_length": avg_length,
                "max_length": max_length,
            }
        
        return {"columns": 1, "confidence": 1.0 - short_ratio * 0.3}
    
    @staticmethod
    def _sort_by_column_reading_order(text: str, layout_info: Dict[str, Any]) -> str:
        """Phase 8 fix: Sort text by column reading order for multi-column layouts."""
        if layout_info.get("columns", 1) <= 1:
            return text
        
        lines = text.split('\n')
        if len(lines) < 3:
            return text
        
        # Simple column sorting: group lines by position/length pattern
        # This is a heuristic approach - actual column detection would need visual analysis
        
        columns = layout_info.get("columns", 2)
        sorted_lines = []
        
        # For 2-column: interleave lines (assume alternating columns)
        if columns == 2:
            for i in range(0, len(lines), 2):
                sorted_lines.append(lines[i])
                if i + 1 < len(lines):
                    sorted_lines.append(lines[i + 1])
        
        # For 3+ columns: simple sequential ordering (most common fallback)
        else:
            sorted_lines = lines
        
        return '\n'.join(sorted_lines)
    
    @staticmethod
    def _detect_mixed_language(text: str) -> Dict[str, Any]:
        """Phase 26 fix: Detect mixed-language documents."""
        # Language detection patterns
        french_patterns = [
            r'\b(?:le|la|les|un|une|des|et|ou|mais|où|qui|que|qu|dont|lui|leur|y|en)\b',
            r'\b(?:être|avoir|faire|aller|dire|prendre|venir|voir|savoir|pouvoir)\b',
            r'[àâäéèêëïîôùûüÿç]',
        ]
        
        arabic_patterns = [
            r'[\u0600-\u06FF]',  # Arabic script range
            r'\b(?:في|من|على|إلى|عن|مع|هذا|هذه|التي|الذي|الذين)\b',
        ]
        
        english_patterns = [
            r'\b(?:the|a|an|and|or|but|where|who|which|that|this|these|those)\b',
            r'\b(?:is|are|was|were|be|been|being|have|has|had|do|does|did)\b',
        ]
        
        text_lower = text.lower()
        
        # Count matches for each language
        french_score = sum(len(re.findall(pattern, text_lower)) for pattern in french_patterns)
        arabic_score = sum(len(re.findall(pattern, text)) for pattern in arabic_patterns)
        english_score = sum(len(re.findall(pattern, text_lower)) for pattern in english_patterns)
        
        total_score = french_score + arabic_score + english_score
        
        if total_score == 0:
            return {"primary_language": "unknown", "mixed": False, "scores": {}}
        
        # Determine if mixed (multiple languages with significant scores)
        language_scores = {
            "french": french_score,
            "arabic": arabic_score,
            "english": english_score,
        }
        
        significant_languages = [lang for lang, score in language_scores.items() if score > total_score * 0.2]
        
        primary_language = max(language_scores, key=language_scores.get)
        
        return {
            "primary_language": primary_language,
            "mixed": len(significant_languages) > 1,
            "significant_languages": significant_languages,
            "scores": language_scores,
        }

    def _child_splitter(self) -> RecursiveCharacterTextSplitter:
        # Phase 33 fix: Add medical-specific separators to avoid splitting clinical guidelines
        medical_separators = [
            "\n\n",  # Paragraph breaks
            "\n",   # Line breaks
            ". ",   # Sentence endings
            "; ",   # Semicolons
            ", ",   # Commas
            " ",    # Spaces
            "",     # Character level
        ]
        
        # Add medical-specific boundary patterns
        for pattern in self._medical_boundary_patterns:
            medical_separators.insert(-1, pattern)
        
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=min(self.chunk_overlap, max(0, self.chunk_size // 2)),
            separators=medical_separators,
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
        
        # Check if we're processing a new document and reset chapter state accordingly
        if pages and pages[0].document_id != self._current_document_id:
            self._document_chapter = None
            self._current_document_id = pages[0].document_id
        
        splitter = self._child_splitter()
        chunks: list[Chunk] = []
        document_chapter = self._document_chapter

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
                    self._document_chapter = chapter  # Persist across batch calls
                elif document_chapter:
                    chapter = document_chapter
                section = hierarchy.get("subsection") or hierarchy.get("section") or fallback
                
                # Use chapter/section hierarchy only, NOT page number
                # This ensures the same logical section gets the same ID across pages
                # Include document_id in hierarchy key to prevent collisions across documents
                section_hierarchy_key = f"{page.document_id}|{chapter or ''}|{section or '__page__'}"
                parent_id = f"{page.document_id}:parent:{self._stable_id(section_hierarchy_key)}"
                section_id = f"{page.document_id}:section:{self._stable_id(section_hierarchy_key)}"
                
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
                f"{page.document_id}:parent:{self._stable_id(fallback or '__page__')}",
                f"{page.document_id}:section:{self._stable_id(fallback or '__page__')}",
                f"{page.document_id}:chapter:{self._stable_id('__document__')}",
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
