from __future__ import annotations

import contextlib
import contextvars
import json
import math
import re
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

from rag_project.chunking.semantic_chunker import SemanticChunker
from rag_project.intelligence.document_structure import (
    DocumentStructureStore,
    DocumentStructureTracker,
    STRUCTURE_SCHEMA_VERSION,
)
from rag_project.ingestion.robust_ingestor import robust_ingest_file as _ORIGINAL_ROBUST_INGEST
from rag_project.parsing.pdf_extractor import PDFExtractor
from rag_project.retrieval.context_builder import ContextBuilder
from rag_project.retrieval.hybrid_retriever import HybridRetriever
from rag_project.storage.vector_store import VectorStore

_INSTALL_LOCK = threading.RLock()
_INSTALLED = False
_ACTIVE_BUILD: contextvars.ContextVar[str | None] = contextvars.ContextVar("rag_active_build", default=None)
_ACTIVE_VERSION: contextvars.ContextVar[str | None] = contextvars.ContextVar("rag_active_version", default=None)
_ACTIVE_HASH: contextvars.ContextVar[str | None] = contextvars.ContextVar("rag_active_hash", default=None)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _clean_heading_candidates(text: str) -> list[str]:
    from rag_project.intelligence.document_structure import extract_heading_candidates

    return [candidate.title for candidate in extract_heading_candidates(text)]


def _infer_ocr_tables(text: str) -> list[str]:
    """Recover table-like OCR output when native PDF table detection found nothing."""
    lines = [re.sub(r"\s+", " ", line.strip()) for line in str(text or "").splitlines() if line.strip()]
    groups: list[list[str]] = []
    current: list[str] = []

    def row_signature(line: str) -> int:
        if "|" in line:
            return len([part for part in line.split("|") if part.strip()])
        if "\t" in line:
            return len([part for part in line.split("\t") if part.strip()])
        return len([part for part in re.split(r"\s{2,}", line) if part.strip()])

    for line in lines:
        signature = row_signature(line)
        if signature >= 2:
            if current and abs(row_signature(current[-1]) - signature) > 1:
                if len(current) >= 3:
                    groups.append(current)
                current = []
            current.append(line)
        else:
            if len(current) >= 3:
                groups.append(current)
            current = []
    if len(current) >= 3:
        groups.append(current)

    tables: list[str] = []
    for group in groups:
        # Require at least one explicit delimiter or repeated column spacing.
        if not any("|" in line or "\t" in line or re.search(r"\s{2,}", line) for line in group):
            continue
        table = "\n".join(group[:100]).strip()
        if table and table not in tables:
            tables.append(table)
    return tables[:20]


def _extract_captions(text: str) -> list[str]:
    captions: list[str] = []
    pattern = re.compile(
        r"^\s*(?:figure|fig\.?|illustration|image|photo|diagram|scheme|schéma|plate|panel)\b",
        re.I,
    )
    for raw in str(text or "").splitlines():
        line = re.sub(r"\s+", " ", raw.strip())
        if not line:
            continue
        if pattern.match(line) and line not in captions:
            captions.append(line[:1200])
        if len(captions) >= 20:
            break
    return captions


def _patch_pdf_extractor() -> None:
    if getattr(PDFExtractor, "_deep_pdf_patched", False):
        return
    original = PDFExtractor.extract_iter

    def extract_iter(self: PDFExtractor, pdf_path: str | Path, document_id: str | None = None):
        store = None
        if self.state_store is not None:
            database = Path(self.state_store.database_path)
            store = DocumentStructureStore(database.parent / "structure.sqlite3")
        for page in original(self, pdf_path, document_id):
            # OCR-derived caption recovery: parse the final merged page text, not only
            # the original PDF text layer.
            captions = list(page.figure_captions or [])
            for caption in _extract_captions(page.text):
                if caption not in captions:
                    captions.append(caption)
            page.figure_captions = captions[:20]

            # Scanned tables: native find_tables() can be empty before OCR. Infer
            # table-like OCR rows after the final page text is available.
            tables = list(page.table_texts or [])
            if not tables:
                tables = _infer_ocr_tables(page.text)
            page.table_texts = tables[:20]
            page.table_count = len(page.table_texts)
            page.table_ids = list(page.table_ids or [])[: len(page.table_texts)]
            while len(page.table_ids) < len(page.table_texts):
                page.table_ids.append(f"{page.document_id}:p{page.page_number}:table:{len(page.table_ids) + 1}")

            figure_count = max(int(page.image_count or 0), len(page.figure_captions))
            page.figure_ids = list(page.figure_ids or [])[:figure_count]
            while len(page.figure_ids) < figure_count:
                page.figure_ids.append(f"{page.document_id}:p{page.page_number}:figure:{len(page.figure_ids) + 1}")
            page.image_count = figure_count
            page.has_images = bool(page.has_images or figure_count)

            headings = _clean_heading_candidates(page.text)
            page.metadata.update(
                {
                    "structure_schema_version": STRUCTURE_SCHEMA_VERSION,
                    "table_texts": list(page.table_texts),
                    "figure_captions": list(page.figure_captions),
                    "table_ids": list(page.table_ids),
                    "figure_ids": list(page.figure_ids),
                    "headings": headings or list(page.headings),
                    "ocr_confidence": page.ocr_confidence,
                }
            )
            page.headings = headings or list(page.headings)

            if store is not None:
                # Persist the complete page representation needed to audit/rebuild
                # structure without losing the durable page text.
                payload = {
                    "document_id": page.document_id,
                    "page_number": page.page_number,
                    "page_type": page.page_type,
                    "quality_score": page.quality_score,
                    "routing_decision": page.routing_decision,
                    "ocr_status": page.ocr_status,
                    "ocr_confidence": page.ocr_confidence,
                    "image_count": page.image_count,
                    "has_images": page.has_images,
                    "table_ids": page.table_ids,
                    "table_texts": page.table_texts,
                    "figure_ids": page.figure_ids,
                    "figure_captions": page.figure_captions,
                    "headings": page.headings,
                    "text_checksum": __import__("hashlib").sha256((page.text or "").encode("utf-8")).hexdigest(),
                }
                store.put_page(page.document_id, page.page_number or 1, payload)
            yield page

    PDFExtractor.extract_iter = extract_iter
    PDFExtractor._deep_pdf_patched = True


def _patch_chunker() -> None:
    if getattr(SemanticChunker, "_deep_structure_patched", False):
        return

    def chunk_pages(self: SemanticChunker, pages):
        pages = list(pages or [])
        if not pages:
            return []
        child_splitter = self._child_splitter()
        tracker_by_document = getattr(self, "_deep_trackers", {})
        output = []

        for page in pages:
            document_id = str(page.document_id)
            tracker = tracker_by_document.get(document_id)
            if tracker is None:
                tracker = DocumentStructureTracker(document_id)
                tracker_by_document[document_id] = tracker

            table_texts = list(getattr(page, "table_texts", []) or [])
            prose = page.text or ""
            for table_text in table_texts:
                prose = prose.replace(f"[TABLE]\n{str(table_text).strip()}", "")
            prose = prose.strip()

            segments = tracker.split_page(page.page_number or 1, prose)
            if not segments and prose:
                segments = [(prose, tracker.snapshot(page.page_number or 1), False)]

            for segment_text, snapshot, is_heading in segments:
                if not segment_text.strip():
                    continue
                children = child_splitter.split_text(segment_text) or [segment_text]
                for child_index, child_text in enumerate(children):
                    child_text = child_text.strip()
                    if not child_text:
                        continue
                    enriched = __import__("rag_project.intelligence.pdf_intelligence", fromlist=["enrich_text"]).enrich_text(child_text)
                    chapter = snapshot.chapter
                    section = snapshot.section
                    hierarchy = list(snapshot.hierarchy_path)
                    structure_header = (
                        f"[RAG-STRUCTURE schema={STRUCTURE_SCHEMA_VERSION}; "
                        f"chapter_id={snapshot.chapter_id or ''}; chapter={chapter or ''}; "
                        f"section_id={snapshot.section_id or ''}; section={section or ''}; "
                        f"parent_id={snapshot.parent_id}; path={_json(hierarchy)}; "
                        f"page={page.page_number or 1}; quality={float(page.quality_score or 0.0):.4f}; "
                        f"ocr={page.ocr_status or 'not_required'}]"
                    )
                    searchable = f"{structure_header}\n{child_text}"
                    metadata = {
                        "source_pages": [page.page_number or 1],
                        "page_numbers": [page.page_number or 1],
                        "evidence_types": ["text"] + (["table"] if table_texts else []) + (["figure"] if page.figure_captions or page.figure_ids else []),
                        "chapter": chapter,
                        "chapter_id": snapshot.chapter_id,
                        "section": section,
                        "section_id": snapshot.section_id,
                        "global_section_id": snapshot.section_id,
                        "parent_id": snapshot.parent_id,
                        "hierarchy_path": list(hierarchy),
                        "child_index": child_index,
                        "normalized_text": enriched["normalized_text"],
                        "entities": enriched["entities"],
                        "headings": enriched["headings"],
                        "number_forms": enriched["number_forms"],
                        "table_id": None,
                        "figure_id": None,
                        "document_id": document_id,
                        "file_name": page.file_name,
                        "page_type": page.page_type,
                        "quality_score": page.quality_score,
                        "ocr_status": page.ocr_status,
                        "ocr_confidence": page.ocr_confidence,
                        "routing_decision": page.routing_decision,
                    }
                    output.append(__import__("rag_project.ingestion.document_models", fromlist=["Chunk"]).Chunk(
                        doc_id=document_id,
                        file_name=page.file_name,
                        chunk_index=len(output),
                        text=searchable,
                        page_numbers=[page.page_number or 1],
                        metadata=metadata,
                        representation_type="canonical",
                        parent_id=snapshot.parent_id,
                        section_id=snapshot.section_id,
                        table_id=None,
                        figure_id=None,
                        normalized_text=enriched["normalized_text"],
                    ))

            # Tables inherit the same document-global hierarchy as the surrounding page.
            page_snapshot = tracker.snapshot(page.page_number or 1)
            for table_index, table_text in enumerate(table_texts):
                value = str(table_text or "").strip()
                if not value:
                    continue
                table_id = page.table_ids[table_index] if table_index < len(page.table_ids) else f"{document_id}:p{page.page_number}:table:{table_index + 1}"
                enriched = __import__("rag_project.intelligence.pdf_intelligence", fromlist=["enrich_text"]).enrich_text(value)
                metadata = {
                    "source_pages": [page.page_number or 1],
                    "page_numbers": [page.page_number or 1],
                    "evidence_types": ["table"],
                    "chapter": page_snapshot.chapter,
                    "chapter_id": page_snapshot.chapter_id,
                    "section": page_snapshot.section,
                    "section_id": page_snapshot.section_id or page_snapshot.parent_id,
                    "global_section_id": page_snapshot.section_id or page_snapshot.parent_id,
                    "parent_id": page_snapshot.parent_id,
                    "hierarchy_path": list(page_snapshot.hierarchy_path),
                    "child_index": table_index,
                    "normalized_text": enriched["normalized_text"],
                    "entities": enriched["entities"],
                    "headings": enriched["headings"],
                    "number_forms": enriched["number_forms"],
                    "table_id": table_id,
                    "figure_id": None,
                    "document_id": document_id,
                    "file_name": page.file_name,
                    "page_type": page.page_type,
                    "quality_score": page.quality_score,
                    "ocr_status": page.ocr_status,
                    "ocr_confidence": page.ocr_confidence,
                    "routing_decision": page.routing_decision,
                }
                output.append(__import__("rag_project.ingestion.document_models", fromlist=["Chunk"]).Chunk(
                    doc_id=document_id,
                    file_name=page.file_name,
                    chunk_index=len(output),
                    text=f"[TABLE]\n{value}",
                    page_numbers=[page.page_number or 1],
                    metadata=metadata,
                    representation_type="table",
                    parent_id=page_snapshot.parent_id,
                    section_id=page_snapshot.section_id or page_snapshot.parent_id,
                    table_id=table_id,
                    figure_id=None,
                    normalized_text=enriched["normalized_text"],
                ))

            # Captions inherit the same hierarchy. Visual embedding is optional; the
            # caption representation remains useful on CPU-only laptops.
            figure_ids = list(page.figure_ids or [])
            for figure_index, caption in enumerate(list(page.figure_captions or [])):
                value = str(caption or "").strip()
                if not value:
                    continue
                figure_id = figure_ids[figure_index] if figure_index < len(figure_ids) else f"{document_id}:p{page.page_number}:figure:{figure_index + 1}"
                enriched = __import__("rag_project.intelligence.pdf_intelligence", fromlist=["enrich_text"]).enrich_text(value)
                metadata = {
                    "source_pages": [page.page_number or 1],
                    "page_numbers": [page.page_number or 1],
                    "evidence_types": ["figure"],
                    "chapter": page_snapshot.chapter,
                    "chapter_id": page_snapshot.chapter_id,
                    "section": page_snapshot.section,
                    "section_id": page_snapshot.section_id or page_snapshot.parent_id,
                    "global_section_id": page_snapshot.section_id or page_snapshot.parent_id,
                    "parent_id": page_snapshot.parent_id,
                    "hierarchy_path": list(page_snapshot.hierarchy_path),
                    "child_index": figure_index,
                    "normalized_text": enriched["normalized_text"],
                    "entities": enriched["entities"],
                    "headings": enriched["headings"],
                    "number_forms": enriched["number_forms"],
                    "table_id": None,
                    "figure_id": figure_id,
                    "document_id": document_id,
                    "file_name": page.file_name,
                    "page_type": page.page_type,
                    "quality_score": page.quality_score,
                    "ocr_status": page.ocr_status,
                    "ocr_confidence": page.ocr_confidence,
                    "routing_decision": page.routing_decision,
                }
                output.append(__import__("rag_project.ingestion.document_models", fromlist=["Chunk"]).Chunk(
                    doc_id=document_id,
                    file_name=page.file_name,
                    chunk_index=len(output),
                    text=f"[FIGURE CAPTION]\n{value}",
                    page_numbers=[page.page_number or 1],
                    metadata=metadata,
                    representation_type="figure_caption",
                    parent_id=page_snapshot.parent_id,
                    section_id=page_snapshot.section_id or page_snapshot.parent_id,
                    table_id=None,
                    figure_id=figure_id,
                    normalized_text=enriched["normalized_text"],
                ))

        self._deep_trackers = tracker_by_document
        return output

    def _init_wrapper(self, original_init):
        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._deep_trackers = {}
        return init

    original_init = SemanticChunker.__init__
    SemanticChunker.__init__ = self_init = _init_wrapper(None, original_init) if False else (lambda self, *args, **kwargs: None)
    # Replace with a real wrapper without lambda indirection so introspection remains useful.
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._deep_trackers = {}
    SemanticChunker.__init__ = init
    SemanticChunker.chunk_pages = chunk_pages
    SemanticChunker._deep_structure_patched = True


def _patch_vector_store() -> None:
    if getattr(VectorStore, "_deep_contract_patched", False):
        return

    original_delete_version = VectorStore.delete_version
    original_set_version_index_state = VectorStore.set_version_index_state

    def add_documents(self, documents, metadatas, embeddings, ids):
        documents_list = list(documents or [])
        metadata_list = [dict(item or {}) for item in list(metadatas or [])]
        embeddings_list = [list(vector) for vector in list(embeddings or [])]
        ids_list = [str(item) for item in list(ids or [])]
        if not documents_list:
            return
        if not (len(documents_list) == len(metadata_list) == len(embeddings_list) == len(ids_list)):
            raise ValueError("documents, metadatas, embeddings, and ids must have the same length")
        dim = self._resolve_dimension(embeddings_list)
        self._apply_collection_metadata(dim)
        build_id = _ACTIVE_BUILD.get() or uuid.uuid4().hex
        version_id = _ACTIVE_VERSION.get()
        content_hash = _ACTIVE_HASH.get()
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(documents_list):
            meta = self._coerce_metadata(metadata_list[index])
            meta["document_id"] = str(meta.get("document_id") or "unknown")
            meta["chunk_id"] = str(meta.get("chunk_id") or ids_list[index])
            meta["index_state"] = "BUILDING"
            meta["version_id"] = str(version_id or meta.get("version_id") or content_hash or meta["document_id"])
            if content_hash:
                meta["content_hash"] = str(content_hash)
            meta["build_id"] = build_id
            meta["structure_version"] = STRUCTURE_SCHEMA_VERSION
            meta.setdefault("representation_type", "canonical")
            meta.setdefault("parent_id", f"{meta['document_id']}:document")
            meta.setdefault("section_id", meta["parent_id"])
            meta.setdefault("hierarchy_path", [])
            meta.setdefault("page_numbers", [])
            meta.setdefault("quality_score", 0.0)
            meta.setdefault("ocr_status", "not_required")
            if isinstance(meta.get("hierarchy_path"), (dict, tuple)):
                meta["hierarchy_path"] = list(meta["hierarchy_path"])
            for key in ("entities", "headings", "number_forms", "evidence_types", "hierarchy_path"):
                if isinstance(meta.get(key), (dict, list, tuple)):
                    meta[key] = _json(meta[key]) if key in {"entities", "headings", "number_forms"} else meta[key]
            if meta.get("representation_type") == "table" and not meta.get("table_id"):
                raise ValueError(f"Structural table representation is missing table_id for {meta['chunk_id']}")
            if meta.get("representation_type") == "figure_caption" and not meta.get("figure_id"):
                raise ValueError(f"Structural figure representation is missing figure_id for {meta['chunk_id']}")
            if not self._valid_vector(embeddings_list[index], dim):
                raise ValueError(f"Invalid semantic embedding at index {index}.")
            normalized.append(meta)

        self.collection.upsert(
            ids=ids_list,
            documents=[str(item) for item in documents_list],
            metadatas=normalized,
            embeddings=[list(map(float, vector)) for vector in embeddings_list],
        )
        self._update_collection_identity(self.expected_identity)
        self._upsert_lexical_records(documents_list, normalized, ids_list)

    VectorStore.add_documents = add_documents

    def set_version_index_state(self, document_id: str, version_id: str, state: str) -> None:
        desired = str(state).upper()
        active_build = _ACTIVE_BUILD.get()
        records = self.collection.get(where={"document_id": document_id}, include=["metadatas"])
        promote: list[str] = []
        promoted_meta: list[dict[str, Any]] = []
        target_version = _ACTIVE_VERSION.get() or version_id
        for item_id, raw in zip(records.get("ids", []), records.get("metadatas", []), strict=False):
            meta = self._coerce_metadata(raw)
            same_version = str(meta.get("version_id") or "") in {str(target_version), str(version_id)} or str(meta.get("content_hash") or "") == str(version_id)
            same_build = active_build is None or str(meta.get("build_id") or "") == str(active_build)
            if same_version and (same_build or meta.get("index_state") == "BUILDING"):
                meta["index_state"] = desired
                promote.append(str(item_id))
                promoted_meta.append(meta)
        if promote:
            self.collection.update(ids=promote, metadatas=promoted_meta)
            with sqlite3.connect(self.lexical_database) as connection:
                for item_id, meta in zip(promote, promoted_meta, strict=True):
                    connection.execute("UPDATE lexical_documents SET index_state=?, metadata=? WHERE id=?", (desired, _json(meta), item_id))

        if desired == "READY" and active_build:
            stale_ids: list[str] = []
            stale_meta: list[dict[str, Any]] = []
            for item_id, raw in zip(records.get("ids", []), records.get("metadatas", []), strict=False):
                meta = self._coerce_metadata(raw)
                same_version = str(meta.get("version_id") or "") in {str(target_version), str(version_id)} or str(meta.get("content_hash") or "") == str(version_id)
                if same_version and str(meta.get("build_id") or "") != active_build:
                    stale_ids.append(str(item_id))
                    stale_meta.append(meta)
            if stale_ids:
                self.collection.delete(ids=stale_ids)
                with sqlite3.connect(self.lexical_database) as connection:
                    connection.executemany("DELETE FROM lexical_documents WHERE id=?", [(item_id,) for item_id in stale_ids])
        if not promote:
            return original_set_version_index_state(self, document_id, version_id, desired)

    VectorStore.set_version_index_state = set_version_index_state

    def delete_version(self, document_id: str, version_id: str) -> None:
        active_build = _ACTIVE_BUILD.get()
        records = self.collection.get(where={"document_id": document_id}, include=["metadatas"])
        remove_ids: list[str] = []
        for item_id, raw in zip(records.get("ids", []), records.get("metadatas", []), strict=False):
            meta = self._coerce_metadata(raw)
            same_version = str(meta.get("version_id") or "") == str(_ACTIVE_VERSION.get() or "") or str(meta.get("content_hash") or "") == str(version_id) or str(meta.get("version_id") or "") == str(version_id)
            if same_version and (active_build is None or str(meta.get("build_id") or "") == str(active_build) or str(meta.get("index_state") or "").upper() == "BUILDING"):
                remove_ids.append(str(item_id))
        if remove_ids:
            self.collection.delete(ids=remove_ids)
            with sqlite3.connect(self.lexical_database) as connection:
                connection.executemany("DELETE FROM lexical_documents WHERE id=?", [(item_id,) for item_id in remove_ids])
        else:
            # Preserve the original API's behavior for callers outside an active build.
            original_delete_version(self, document_id, version_id)

    VectorStore.delete_version = delete_version

    def validate_document_index(self, document_id: str, version_id: str | None = None):
        records = self.collection.get(where={"document_id": document_id}, include=["metadatas", "documents", "embeddings"])
        all_ids = list(records.get("ids", []))
        all_meta = list(records.get("metadatas", []))
        if version_id is not None:
            selected = []
            requested = str(version_id)
            for index, raw in enumerate(all_meta):
                meta = self._coerce_metadata(raw)
                if str(meta.get("version_id") or "") == requested or str(meta.get("content_hash") or "") == requested:
                    selected.append(index)
            # In an active ingestion the canonical version is known independently.
            active = _ACTIVE_VERSION.get()
            if not selected and active:
                selected = [index for index, raw in enumerate(all_meta) if str(self._coerce_metadata(raw).get("version_id") or "") == active]
        else:
            selected = list(range(len(all_ids)))

        issues: list[str] = []
        selected_ids = [str(all_ids[index]) for index in selected if index < len(all_ids)]
        seen: set[str] = set()
        allowed_representations = {"canonical", "table", "figure_caption", "section_anchor", "chapter_anchor"}
        expected_dim = int(self._collection_dim() or 0)
        embeddings = list(records.get("embeddings", []))
        documents = list(records.get("documents", []))
        for index in selected:
            if index >= len(all_meta):
                issues.append(f"missing metadata at {index}")
                continue
            meta = self._coerce_metadata(all_meta[index])
            chunk_id = str(meta.get("chunk_id") or meta.get("id") or "")
            if not chunk_id:
                issues.append("missing chunk_id")
            if chunk_id in seen:
                issues.append(f"duplicate chunk_id: {chunk_id}")
            seen.add(chunk_id)
            if meta.get("index_state") not in {"READY", "BUILDING"}:
                issues.append(f"unexpected index_state: {meta.get('index_state')}")
            if int(meta.get("structure_version") or 0) < STRUCTURE_SCHEMA_VERSION:
                issues.append(f"legacy structure_version: {chunk_id}")
            representation = str(meta.get("representation_type") or "")
            if representation not in allowed_representations:
                issues.append(f"invalid representation_type: {chunk_id}")
            for field in ("parent_id", "section_id"):
                if not meta.get(field):
                    issues.append(f"missing {field}: {chunk_id}")
            if not meta.get("page_numbers"):
                issues.append(f"missing page_numbers: {chunk_id}")
            if not isinstance(meta.get("quality_score", 0.0), (int, float)):
                issues.append(f"invalid quality_score: {chunk_id}")
            if representation == "table" and not meta.get("table_id"):
                issues.append(f"missing table_id: {chunk_id}")
            if representation == "figure_caption" and not meta.get("figure_id"):
                issues.append(f"missing figure_id: {chunk_id}")
            if index >= len(embeddings) or not self._valid_vector(embeddings[index], expected_dim):
                issues.append(f"invalid semantic embedding: {chunk_id}")
            if index >= len(documents) or not str(documents[index]).strip():
                issues.append(f"missing document text: {chunk_id}")

        valid = bool(selected_ids) and not issues
        lexical_ids = set()
        try:
            with sqlite3.connect(self.lexical_database) as connection:
                rows = connection.execute("SELECT id FROM lexical_documents WHERE index_state='READY'").fetchall()
            lexical_ids = {str(row[0]) for row in rows}
            if valid and any(item_id not in lexical_ids for item_id in selected_ids):
                issues.append("semantic/lexical index parity failure")
                valid = False
        except sqlite3.Error:
            issues.append("lexical index validation failed")
            valid = False

        return {"document_id": document_id, "count": len(selected_ids), "valid": valid, "issues": issues}

    VectorStore.validate_document_index = validate_document_index
    VectorStore._deep_contract_patched = True


def _patch_ingestion_versioning() -> None:
    from rag_project.ingestion import robust_ingestor

    if getattr(robust_ingestor.robust_ingest_file, "_deep_version_wrapped", False):
        return
    original = robust_ingestor.robust_ingest_file

    def wrapped(system, pdf_path):
        content_hash = system._hash_file(Path(pdf_path))
        chunking = json.dumps({"size": system.settings.chunk_size, "overlap": system.settings.chunk_overlap}, sort_keys=True)
        ocr_config = json.dumps({"engine": "rapidocr", "scale": 2, "structure_schema": STRUCTURE_SCHEMA_VERSION}, sort_keys=True)
        version_id = system._ingestion_version_id(
            content_hash=content_hash,
            parser_version="pdf-extractor-v3",
            ocr_config=ocr_config,
            chunking_config=chunking,
            embedding_model=system.settings.embedding_model,
            embedding_profile=None,
            embedding_dimension=None,
        )
        build_id = uuid.uuid4().hex
        tok_build = _ACTIVE_BUILD.set(build_id)
        tok_ver = _ACTIVE_VERSION.set(version_id)
        tok_hash = _ACTIVE_HASH.set(content_hash)
        try:
            return original(system, pdf_path)
        finally:
            _ACTIVE_BUILD.reset(tok_build)
            _ACTIVE_VERSION.reset(tok_ver)
            _ACTIVE_HASH.reset(tok_hash)

    wrapped._deep_version_wrapped = True
    wrapped._deep_version_original = original
    robust_ingestor.robust_ingest_file = wrapped


def _patch_retrieval() -> None:
    if getattr(HybridRetriever, "_deep_retrieval_patched", False):
        return
    original = HybridRetriever.retrieve

    def retrieve(self, query: str, top_k: int = 6, where=None):
        hits = list(original(self, query, top_k=max(int(top_k) * 3, int(top_k)), where=where) or [])
        q = str(query or "").casefold()
        table_intent = any(token in q for token in ("table", "tabular", "dose range", "reference range", "laboratory", "lab values", "normal values"))
        figure_intent = any(token in q for token in ("figure", "diagram", "illustration", "image", "anatomy diagram", "shown in"))
        structure_intent = any(token in q for token in ("chapter", "section", "subsection", "where in the book"))
        for hit in hits:
            meta = hit.metadata or {}
            representation = str(meta.get("representation_type") or "canonical")
            bonus = 0.0
            if table_intent and representation == "table":
                bonus += 0.18
            if figure_intent and representation == "figure_caption":
                bonus += 0.14
            if structure_intent and meta.get("section_id"):
                bonus += 0.08
            try:
                bonus += 0.06 * max(0.0, min(1.0, float(meta.get("quality_score", 0.0))))
            except (TypeError, ValueError):
                pass
            if meta.get("ocr_status") == "success":
                bonus += 0.02
            hit.score = float(hit.score) + bonus
            hit.metadata["structural_bonus"] = round(bonus, 6)
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[: max(1, int(top_k))]

    HybridRetriever.retrieve = retrieve
    HybridRetriever._deep_retrieval_patched = True


def _patch_context() -> None:
    if getattr(ContextBuilder, "_deep_context_patched", False):
        return
    original = ContextBuilder.build

    def build(self, hits):
        items = list(hits or [])
        # Prefer diverse structural representations before filling remaining slots.
        priority = {"table": 0, "figure_caption": 1, "section_anchor": 2, "chapter_anchor": 3, "canonical": 4}
        items.sort(key=lambda hit: (priority.get(str((hit.metadata or {}).get("representation_type") or "canonical"), 4), -float(getattr(hit, "score", 0.0))))
        context, selected = original(self, items)
        return context, selected

    ContextBuilder.build = build
    ContextBuilder._deep_context_patched = True


def install() -> None:
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        _patch_pdf_extractor()
        _patch_chunker()
        _patch_vector_store()
        _patch_ingestion_versioning()
        _patch_retrieval()
        _patch_context()
        # The repository already contains a standalone atomic-versioning hardening
        # module. Make its activation part of the production contract instead of an
        # optional unused file.
        with contextlib.suppress(Exception):
            from rag_project.intelligence.atomic_versioning import install as install_atomic_versioning
            install_atomic_versioning()
        _INSTALLED = True


__all__ = ["install", "STRUCTURE_SCHEMA_VERSION", "DocumentStructureStore", "DocumentStructureTracker"]
