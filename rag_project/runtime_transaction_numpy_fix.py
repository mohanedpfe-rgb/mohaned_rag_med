from __future__ import annotations

import threading
from typing import Any

_LOCK = threading.RLock()
_INSTALLED = False


def _to_plain_embedding(value: Any) -> Any:
    """Convert numpy/array-like embeddings to ordinary Python lists."""
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, tuple):
        return [_to_plain_embedding(item) for item in value]
    if isinstance(value, list):
        return [_to_plain_embedding(item) for item in value]
    return value


def _as_list(value: Any) -> list[Any]:
    """Convert list-like values without ever evaluating an array for truthiness."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        try:
            converted = value.tolist()
            if isinstance(converted, list):
                return converted
            if isinstance(converted, tuple):
                return list(converted)
            return [converted]
        except Exception:
            pass
    try:
        return list(value)
    except (TypeError, ValueError):
        return []


def _semantic_only_validation(self: Any, document_id: str, version_id: str | None = None) -> dict[str, Any]:
    records = self.collection.get(
        where={"document_id": document_id},
        include=["metadatas", "documents", "embeddings"],
    )
    ids = _as_list(records.get("ids"))
    metadatas_all = _as_list(records.get("metadatas"))
    raw_embeddings = _as_list(records.get("embeddings"))
    if version_id is not None:
        keep = [
            index
            for index, metadata in enumerate(metadatas_all)
            if str(self._coerce_metadata(metadata).get("version_id") or "") == str(version_id)
        ]
        normalized = {key: _as_list(value) for key, value in records.items()}
        ids = [normalized.get("ids", [])[index] for index in keep]
        metadatas_all = [normalized.get("metadatas", [])[index] for index in keep]
        embeddings = [normalized.get("embeddings", [])[index] for index in keep]
    else:
        embeddings = raw_embeddings

    issues: list[str] = []
    seen: set[str] = set()
    for metadata in metadatas_all:
        meta = self._coerce_metadata(metadata)
        chunk_id = str(meta.get("chunk_id") or meta.get("id") or "")
        if not chunk_id:
            issues.append("missing chunk_id")
        elif chunk_id in seen:
            issues.append(f"duplicate chunk_id: {chunk_id}")
        seen.add(chunk_id)
        if meta.get("index_state") not in {"READY", "BUILDING"}:
            issues.append(f"unexpected index_state: {meta.get('index_state')}")

    expected_dimension = int(self._collection_dim() or 0)
    for vector in embeddings:
        plain = _to_plain_embedding(vector)
        if not self._valid_vector(plain, expected_dimension):
            issues.append("invalid semantic embedding")

    valid = not issues and bool(ids)
    count = len(ids)
    return {
        "document_id": document_id,
        "count": count,
        "semantic_count": count,
        "lexical_count": count,
        "valid": valid,
        "issues": issues,
    }


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        from rag_project import runtime_version_transaction_fix as transaction
        from rag_project.storage.vector_store import VectorStore

        original = getattr(transaction, "_restore_snapshot", None)
        if callable(original) and not getattr(original, "_numpy_snapshot_fix", False):
            def safe_restore(system: Any, snapshot: dict[str, Any]) -> None:
                vector = snapshot.get("vector")
                if isinstance(vector, dict):
                    raw = vector.get("embeddings")
                    if raw is not None:
                        vector["embeddings"] = _to_plain_embedding(raw)
                    raw_ids = vector.get("ids")
                    if raw_ids is not None:
                        vector["ids"] = _as_list(raw_ids)
                    raw_docs = vector.get("documents")
                    if raw_docs is not None:
                        vector["documents"] = _as_list(raw_docs)
                    raw_meta = vector.get("metadatas")
                    if raw_meta is not None:
                        vector["metadatas"] = _as_list(raw_meta)
                return original(system, snapshot)
            safe_restore._numpy_snapshot_fix = True
            transaction._restore_snapshot = safe_restore

        original_validate = VectorStore.validate_document_index
        if not getattr(original_validate, "_numpy_validation_compat", False):
            def safe_validate(self: Any, document_id: str, version_id: str | None = None) -> dict[str, Any]:
                if not hasattr(self, "lexical_database"):
                    return _semantic_only_validation(self, document_id, version_id)
                return original_validate(self, document_id, version_id)
            safe_validate._numpy_validation_compat = True
            VectorStore.validate_document_index = safe_validate

        _INSTALLED = True


__all__ = ["install", "_as_list"]
