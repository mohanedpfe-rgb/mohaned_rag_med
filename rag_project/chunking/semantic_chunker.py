from __future__ import annotations

import hashlib
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

    def _child_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=min(self.chunk_overlap, max(0, self.chunk_size // 2)), separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""])

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
                if chapter: meta["chapter"] = chapter
                if section: meta["section"] = section
                if subsection: meta["subsection"] = subsection
                sections.append((content, meta))
            buffer = []

        for line in lines:
            match = re.match(r"^\s*(#{1,3})\s+(.+?)\s*$", line)
            if not match:
                buffer.append(line)
                continue
            title = match.group(2).strip()
            level = len(match.group(1))
            if level == 1:
                if buffer: flush()
                chapter, section, subsection = title, None, None
                buffer.append(title)
            elif level == 2:
                if buffer: flush()
                section, subsection = title, None
                buffer.append(title)
            else:
                if buffer: flush()
                subsection = title
                buffer.append(title)
        if buffer: flush()
        if sections: return sections
        fallback = self._section_fallback(value)
        return [(value, {"section": fallback} if fallback else {})] if value.strip() else []

    @staticmethod
    def _remove_table_blocks(text: str, table_texts: list[str]) -> str:
        prose = str(text or "")
        for table_text in table_texts:
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
            if out: out[-1] = f"{out[-1]}\n{pending}".strip()
            else: out.append(pending)
        return out

    def chunk_pages(self, pages: List[PageExtraction]) -> List[Chunk]:
        pages = list(pages or [])
        if not pages: return []
        splitter = self._child_splitter()
        chunks: list[Chunk] = []
        for page in pages:
            page_no = int(getattr(page, "page_number", getattr(page, "page_index", 0) + 1) or 1)
            table_texts = [str(v).strip() for v in (getattr(page, "table_texts", []) or []) if str(v).strip()]
            table_ids = list(getattr(page, "table_ids", []) or [])
            figure_ids = list(getattr(page, "figure_ids", []) or [])
            captions = [str(v).strip() for v in (getattr(page, "figure_captions", []) or []) if str(v).strip()]
            evidence = {"text"}
            if bool(getattr(page, "has_images", False)) or figure_ids or captions: evidence.add("figure")
            if int(getattr(page, "table_count", 0) or 0) > 0 or table_ids or table_texts: evidence.add("table")
            evidence_types = [name for name in ("figure", "table", "text") if name in evidence]
            prose = self._remove_table_blocks(page.text or "", table_texts)
            page_info = enrich_text(prose)
            fallback = (page_info.get("headings") or [None])[0] or self._section_fallback(prose)
            rows = self._parent_sections(prose) or [(prose, {})]
            first_meta = rows[0][1]
            first_key = f"{page_no}|{first_meta.get('chapter', '')}|{first_meta.get('section', first_meta.get('subsection', fallback)) or '__page__'}"
            anchor_parent = f"{page.document_id}:p{page_no}:parent:{self._stable_id(first_key)}"
            anchor_section = f"{page.document_id}:p{page_no}:section:{self._stable_id(first_key)}"
            anchor_chapter = f"{page.document_id}:p{page_no}:chapter:{self._stable_id(first_meta.get('chapter', '__document__'))}"
            for parent_text, hierarchy in rows:
                chapter = hierarchy.get("chapter")
                section = hierarchy.get("subsection") or hierarchy.get("section") or fallback
                key = f"{page_no}|{chapter or ''}|{section or '__page__'}"
                parent_id = f"{page.document_id}:p{page_no}:parent:{self._stable_id(key)}"
                section_id = f"{page.document_id}:p{page_no}:section:{self._stable_id(key)}"
                chapter_id = f"{page.document_id}:p{page_no}:chapter:{self._stable_id(chapter or '__document__')}"
                children = self._compact_children(splitter.split_text(parent_text)) or ([parent_text.strip()] if parent_text.strip() else [])
                for child_index, child in enumerate(children):
                    enriched = enrich_text(child)
                    prefix = " - ".join(x for x in (f"Chapter: {chapter}" if chapter else "", f"Section: {section}" if section else "") if x)
                    searchable = "[RAG-STRUCTURE schema=3]\n" + (f"[{prefix}]\n" if prefix else "") + child.strip()
                    metadata = {"source_pages":[page_no],"page_numbers":[page_no],"evidence_types":evidence_types,"chapter":chapter,"chapter_id":chapter_id,"section":section,"section_id":section_id,"global_section_id":section_id,"parent_id":parent_id,"parent_text":parent_text,"child_index":child_index,"normalized_text":enriched["normalized_text"],"entities":enriched["entities"],"headings":enriched["headings"],"number_forms":enriched["number_forms"],"table_id":table_ids[0] if table_ids else None,"figure_id":figure_ids[0] if figure_ids else None,"document_id":page.document_id,"file_name":page.file_name,"page_type":page.page_type,"quality_score":page.quality_score,"ocr_status":page.ocr_status,"ocr_confidence":getattr(page,"ocr_confidence",None),"routing_decision":getattr(page,"routing_decision",None)}
                    chunks.append(Chunk(page.document_id,page.file_name,len(chunks),searchable,[page_no],metadata,"canonical",parent_id,section_id,metadata["table_id"],metadata["figure_id"],enriched["normalized_text"]))
            for i, table_text in enumerate(table_texts):
                table_id = table_ids[i] if i < len(table_ids) else f"{page.document_id}:p{page_no}:table:{i+1}"
                enriched = enrich_text(table_text)
                meta = {"source_pages":[page_no],"page_numbers":[page_no],"evidence_types":evidence_types,"chapter":first_meta.get("chapter"),"chapter_id":anchor_chapter,"section":first_meta.get("section") or fallback,"section_id":anchor_section,"global_section_id":anchor_section,"parent_id":anchor_parent,"parent_text":table_text,"child_index":i,"normalized_text":enriched["normalized_text"],"entities":enriched["entities"],"headings":enriched["headings"],"number_forms":enriched["number_forms"],"table_id":table_id,"figure_id":figure_ids[0] if figure_ids else None,"document_id":page.document_id,"file_name":page.file_name,"page_type":page.page_type,"quality_score":page.quality_score,"ocr_status":page.ocr_status,"routing_decision":getattr(page,"routing_decision",None),"representation_type":"table"}
                chunks.append(Chunk(page.document_id,page.file_name,len(chunks),f"[TABLE]\nSection: {meta['section']}\n{table_text}" if meta.get("section") else f"[TABLE]\n{table_text}",[page_no],meta,"table",anchor_parent,anchor_section,table_id,meta["figure_id"],enriched["normalized_text"]))
            for i, caption in enumerate(captions):
                figure_id = figure_ids[i] if i < len(figure_ids) else f"{page.document_id}:p{page_no}:figure:{i+1}"
                enriched = enrich_text(caption)
                meta = {"source_pages":[page_no],"page_numbers":[page_no],"evidence_types":evidence_types,"chapter":first_meta.get("chapter"),"chapter_id":anchor_chapter,"section":first_meta.get("section") or fallback,"section_id":anchor_section,"global_section_id":anchor_section,"parent_id":anchor_parent,"parent_text":caption,"child_index":i,"normalized_text":enriched["normalized_text"],"entities":enriched["entities"],"headings":enriched["headings"],"number_forms":enriched["number_forms"],"table_id":table_ids[0] if table_ids else None,"figure_id":figure_id,"document_id":page.document_id,"file_name":page.file_name,"page_type":page.page_type,"quality_score":page.quality_score,"ocr_status":page.ocr_status,"routing_decision":getattr(page,"routing_decision",None),"representation_type":"figure_caption"}
                chunks.append(Chunk(page.document_id,page.file_name,len(chunks),f"[FIGURE CAPTION]\n{caption}",[page_no],meta,"figure_caption",anchor_parent,anchor_section,meta["table_id"],figure_id,enriched["normalized_text"]))
        for index, chunk in enumerate(chunks): chunk.chunk_index=index
        return chunks

    def chunk_page_batches(self, pages: Iterable[PageExtraction], batch_size: int = 16) -> Iterator[List[Chunk]]:
        limit=max(1,int(batch_size)); buffer=[]; offset=0
        for page in pages:
            buffer.append(page)
            if len(buffer)<limit: continue
            batch=self.chunk_pages(buffer)
            for chunk in batch: chunk.chunk_index=offset; offset+=1
            if batch: yield batch
            buffer=[]
        if buffer:
            batch=self.chunk_pages(buffer)
            for chunk in batch: chunk.chunk_index=offset; offset+=1
            if batch: yield batch
