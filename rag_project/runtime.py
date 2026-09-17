from __future__ import annotations

import os
import threading
import time
from typing import Any

_INSTALL_LOCK = threading.RLock()
_INSTALL_APPLICATION_CONTRACT_LOCK = threading.RLock()
_INSTALLED = False
_INSTALL_PROVENANCE: list[dict[str, Any]] = []


def _load_installers():
    from rag_project.storage.vector_store_runtime import install as vector_store
    from rag_project.storage.atomic_index_transaction import install as atomic_index_transaction
    from rag_project.runtime_post_index_publication_contract_v2 import install as post_index_publication
    return (vector_store,)


def _install_vectorstore_compatibility_contract() -> None:
    """Expose the canonical helper methods required by runtime retrieval patches."""
    from rag_project.storage.vector_store import VectorStore

    if not hasattr(VectorStore, "_as_query_result"):
        def _as_query_result(
            ids: list[str],
            documents: list[str],
            metadatas: list[dict[str, Any]],
            distances: list[float] | None = None,
        ) -> dict[str, Any]:
            return {
                "ids": [list(ids)],
                "documents": [list(documents)],
                "metadatas": [list(metadatas)],
                "distances": [
                    [0.0] * len(ids) if distances is None else list(distances)
                ],
            }

        VectorStore._as_query_result = staticmethod(_as_query_result)

    if not hasattr(VectorStore, "_metadata_matches"):
        def _metadata_matches(
            metadata: dict[str, Any],
            where: dict[str, Any] | None,
        ) -> bool:
            if not where:
                return True
            if "$and" in where:
                return all(
                    _metadata_matches(metadata, clause)
                    for clause in (where.get("$and") or [])
                )
            if "$or" in where:
                return any(
                    _metadata_matches(metadata, clause)
                    for clause in (where.get("$or") or [])
                )
            return all(metadata.get(key) == value for key, value in where.items())

        VectorStore._metadata_matches = staticmethod(_metadata_matches)


def install() -> None:
    global _INSTALLED
    from rag_project.storage.atomic_index_transaction import install as atomic_index_transaction
    from rag_project.runtime_post_index_publication_contract_v2 import install as post_index_publication
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        _INSTALL_PROVENANCE.clear()
        for installer in _load_installers():
            name = getattr(installer, "__module__", "unknown") + "." + getattr(installer, "__name__", "install")
            started = time.perf_counter()
            entry: dict[str, Any] = {"installer": name, "status": "RUNNING"}
            try:
                installer()
            except Exception as exc:
                entry.update({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)})
                entry["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
                _INSTALL_PROVENANCE.append(entry)
                raise
            entry["status"] = "OK"
            entry["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
            _INSTALL_PROVENANCE.append(entry)
            if os.getenv("RAG_RUNTIME_TRACE", "").strip().lower() in {"1", "true", "yes"}:
                print(f"[runtime] {name}: OK ({entry['elapsed_ms']} ms)")
        # Secondary storage/publication installers are owned by the storage
        # bootstrap and remain explicit without expanding the canonical tuple.
        atomic_index_transaction()
        post_index_publication()
        _install_vectorstore_compatibility_contract()
        from rag_project.runtime_public_metadata import install as install_public_metadata
        install_public_metadata()
        from rag_project.runtime_stability_v5 import install as install_transition_guard
        install_transition_guard()
        _INSTALLED = True


def installation_report() -> dict[str, Any]:
    return {
        "installed": bool(_INSTALLED),
        "installer_count": len(_INSTALL_PROVENANCE),
        "failed": [dict(item) for item in _INSTALL_PROVENANCE if item.get("status") == "FAILED"],
        "installers": [dict(item) for item in _INSTALL_PROVENANCE],
    }


def install_application_contracts() -> dict[str, object]:
    from rag_project.intelligence.pipeline_integrity import install as pipeline_integrity
    from rag_project.ingestion.ingestion_contract import install as ingestion_contract
    from rag_project.canonical_runtime import install as canonical_runtime
    from rag_project.intelligence.production_contract_v2 import CONTRACT_VERSION
    with _INSTALL_APPLICATION_CONTRACT_LOCK:
        pipeline_integrity()
        ingestion_contract()
        canonical = canonical_runtime()
        return {
            "pipeline_integrity": True,
            "production_contract": {"version": CONTRACT_VERSION, "behavioral_patch": False},
            "ingestion_contract": True,
            "canonical_runtime": canonical,
        }
