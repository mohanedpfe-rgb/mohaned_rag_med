from __future__ import annotations

from functools import wraps
from pathlib import Path
import threading
from typing import Any

from rag_project.ingestion import robust_ingestor
from rag_project.retrieval.hybrid_retriever import HybridRetriever

_LOCK = threading.RLock()
_INSTALLED = False


def _validate_pdf_path(pdf_path: str | Path) -> Path:
    path = Path(pdf_path)
    if path.suffix.casefold() != ".pdf" or not path.is_file():
        raise ValueError(f"Unsupported or missing PDF: {path}")
    return path


def _patch_ingestion_contract() -> None:
    current = robust_ingestor.robust_ingest_file
    if getattr(current, "_runtime_contract_compat", False):
        return

    @wraps(current)
    def guarded_ingest(system: Any, pdf_path: str | Path):
        path = _validate_pdf_path(pdf_path)

        state_store = getattr(system, "state_store", None)
        vector_store = getattr(system, "vector_store", None)
        previous: dict[str, Any] | None = None
        document_id: str | None = None
        previous_version: str | None = None
        try:
            if state_store is not None:
                previous = state_store.get_by_path(str(path.resolve())) or None
                if previous and hasattr(state_store, "is_ready_status") and state_store.is_ready_status(previous.get("status")):
                    document_id = str(previous.get("document_id") or "") or None
                    previous_version = str(previous.get("version_id") or previous.get("content_hash") or "") or None
        except Exception:
            previous = None

        result = current(system, path)
        status = str((result or {}).get("status") or "").casefold()
        if status == "failed" and previous and document_id and previous_version and vector_store is not None:
            try:
                vector_store.set_version_index_state(document_id, previous_version, "READY")
                repaired = dict(result)
                repaired["previous_version_restored"] = True
                return repaired
            except Exception as exc:
                repaired = dict(result)
                repaired["previous_version_restore_warning"] = f"{type(exc).__name__}: {exc}"
                return repaired
        return result

    guarded_ingest._runtime_contract_compat = True
    robust_ingestor.robust_ingest_file = guarded_ingest


def _patch_retrieval_contract() -> None:
    current = HybridRetriever.retrieve
    if getattr(current, "_runtime_contract_compat", False):
        return

    @wraps(current)
    def guarded_retrieve(self: HybridRetriever, query: str, top_k: int = 6, where=None):
        if str(query or "").strip():
            try:
                int(top_k)
            except (TypeError, ValueError) as exc:
                raise ValueError("top_k must be an integer") from exc
        return current(self, query, top_k, where)

    guarded_retrieve._runtime_contract_compat = True
    HybridRetriever.retrieve = guarded_retrieve


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        _patch_ingestion_contract()
        _patch_retrieval_contract()
        _INSTALLED = True
