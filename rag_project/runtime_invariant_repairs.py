from __future__ import annotations

from functools import wraps
from typing import Any

_INSTALLED = False
_ORIGINALS: dict[str, Any] = {}


def _safe_chroma_metadata(metadata: Any) -> dict[str, Any]:
    value = dict(metadata or {}) if isinstance(metadata, dict) else {}
    for key, item in list(value.items()):
        if isinstance(item, tuple):
            item = list(item)
            value[key] = item
        if isinstance(item, list) and not item:
            value.pop(key, None)
    return value


def _patch_collection_upsert() -> None:
    from rag_project.storage.vector_store import VectorStore
    original_init = VectorStore.__init__
    if getattr(original_init, "_invariant_collection_guard", False):
        return
    _ORIGINALS["vector_init"] = original_init
    @wraps(original_init)
    def guarded_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        collection = getattr(self, "collection", None)
        original_upsert = getattr(collection, "upsert", None)
        if collection is None or not callable(original_upsert):
            return
        if getattr(original_upsert, "_invariant_metadata_guard", False):
            return
        @wraps(original_upsert)
        def guarded_upsert(*u_args: Any, **u_kwargs: Any):
            if "metadatas" in u_kwargs and u_kwargs["metadatas"] is not None:
                u_kwargs["metadatas"] = [_safe_chroma_metadata(metadata) for metadata in (u_kwargs.get("metadatas") or [])]
            return original_upsert(*u_args, **u_kwargs)
        guarded_upsert._invariant_metadata_guard = True
        collection.upsert = guarded_upsert
    guarded_init._invariant_collection_guard = True
    VectorStore.__init__ = guarded_init


def _patch_lease_boundary() -> None:
    import rag_project.runtime_quality_gate as quality_gate
    current_guard = getattr(quality_gate, "_guard_transition", None)
    if not callable(current_guard) or getattr(current_guard, "_invariant_lease_semantics", False):
        return
    _ORIGINALS["guard_transition"] = current_guard
    @wraps(current_guard)
    def guard_transition(self: Any, document_id: str, new_stage: str, **values: Any) -> None:
        stage = str(new_stage).upper()
        if stage in getattr(quality_gate, "_TERMINAL_STAGES", set()) or stage in {"INDEXING", "VALIDATING_INDEX"}:
            record = self.get_document(document_id)
            if not record or not record.get("lease_owner"):
                original_transition = getattr(self, "_original_runtime_quality_transition", None)
                if callable(original_transition):
                    return original_transition(document_id, new_stage, **values)
        return current_guard(self, document_id, new_stage, **values)
    guard_transition._invariant_lease_semantics = True
    quality_gate._guard_transition = guard_transition


def _direct_table_extract(page: Any) -> str:
    try:
        finder = getattr(page, "find_tables", None)
        if not callable(finder):
            return ""
        result = finder()
        tables = list(getattr(result, "tables", []) or [])
        rendered: list[str] = []
        for table in tables[:20]:
            try:
                markdown = str(table.to_markdown() or "").strip()
            except Exception:
                continue
            if markdown:
                rendered.append(markdown)
        return "\n\n".join(rendered)
    except Exception:
        return ""


def _patch_table_extraction() -> None:
    from rag_project.parsing.pdf_extractor import PDFExtractor
    current = getattr(PDFExtractor, "_extract_tables", None)
    if not callable(current) or getattr(current, "_invariant_no_fork", False):
        return
    _ORIGINALS["extract_tables"] = current
    def extract_tables(self: Any, page: Any) -> str:
        return _direct_table_extract(page)
    extract_tables._invariant_no_fork = True
    PDFExtractor._extract_tables = extract_tables


def _patch_ingestion_failure_cleanup() -> None:
    from rag_project.ingestion import robust_ingestor
    current = robust_ingestor.robust_ingest_file
    if not callable(current) or getattr(current, "_invariant_failure_cleanup", False):
        return
    _ORIGINALS["robust_ingest_file"] = current
    @wraps(current)
    def guarded(system: Any, pdf_path: Any, *args: Any, **kwargs: Any):
        result = current(system, pdf_path, *args, **kwargs)
        status = str((result or {}).get("status") or "").upper() if isinstance(result, dict) else ""
        if status in {"FAILED", "FAILED_EMBEDDING", "FAILED_INDEXING", "FAILED_EXTRACTION", "FAILED_OCR"}:
            document_id = str((result or {}).get("document_id") or "")
            if document_id:
                try:
                    system.state_store.delete_pages(document_id)
                except Exception:
                    getattr(system, "logger", None) and system.logger.exception("Failed to purge page state for failed ingestion %s", document_id)
                try:
                    version_id = str((result or {}).get("version_id") or "")
                    if version_id:
                        system.vector_store.delete_version(document_id, version_id)
                except Exception:
                    pass
        return result
    guarded._invariant_failure_cleanup = True
    robust_ingestor.robust_ingest_file = guarded


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from rag_project import runtime_storage_contract_fix
    runtime_storage_contract_fix.install()
    _patch_lease_boundary()
    _patch_collection_upsert()
    _patch_table_extraction()
    _patch_ingestion_failure_cleanup()
    _INSTALLED = True


__all__ = ["install"]
