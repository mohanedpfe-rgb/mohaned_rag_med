from __future__ import annotations

from pathlib import Path
import threading

from rag_project.ingestion import robust_ingestor
from rag_project.intelligence.document_structure import DocumentStructureStore

_LOCK = threading.RLock()
_INSTALLED = False


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        original = robust_ingestor.robust_ingest_file

        def wrapped(system, pdf_path):
            path = Path(pdf_path).resolve()
            state = getattr(system, "state_store", None)
            if state is not None:
                existing = state.get_by_path(str(path))
                if existing:
                    try:
                        sidecar = DocumentStructureStore(Path(state.database_path).parent / "structure.sqlite3")
                        sidecar.delete_document(str(existing.get("document_id")))
                    except Exception:
                        if getattr(system, "logger", None):
                            system.logger.exception("Failed to clear stale document structure sidecar for %s", path)
            return original(system, pdf_path)

        wrapped._structure_cleanup_wrapped = True
        wrapped._structure_cleanup_original = original
        robust_ingestor.robust_ingest_file = wrapped
        _INSTALLED = True


__all__ = ["install"]
