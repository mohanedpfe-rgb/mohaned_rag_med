from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from typing import Any

_LOCK = threading.RLock()
_INSTALLED = False


def _as_list(value: Any) -> list[Any]:
    """Normalize list/tuple/NumPy/Chroma sequence values without boolean coercion."""
    if value is None or isinstance(value, Mapping):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except (TypeError, ValueError):
        return []


def _safe_vector(value: Any) -> list[float] | None:
    """Return a finite Python vector for any array-like value."""
    values = _as_list(value)
    if not values:
        return None
    try:
        vector = [float(item) for item in values]
    except (TypeError, ValueError):
        return None
    if not vector or not all(math.isfinite(item) for item in vector):
        return None
    return vector


def _safe_validate_document_index(self: Any, document_id: str, version_id: str | None = None, *, sample_size: int = 8) -> dict[str, Any]:
    records = self.collection.get(where={"document_id": document_id}, include=["metadatas"])
    ids = [str(item_id) for item_id in _as_list(records.get("ids"))]
    metadatas = [self._coerce_metadata(item) for item in _as_list(records.get("metadatas"))]
    pairs = list(zip(ids, metadatas, strict=False))
    if version_id is not None:
        pairs = [(item_id, metadata) for item_id, metadata in pairs if str(metadata.get("version_id") or "") == str(version_id)]

    issues: list[str] = []
    seen_chunk_ids: set[str] = set()
    for item_id, metadata in pairs:
        chunk_id = str(metadata.get("chunk_id") or item_id or "")
        if not chunk_id:
            issues.append("missing chunk_id")
        elif chunk_id in seen_chunk_ids:
            issues.append(f"duplicate chunk_id: {chunk_id}")
        seen_chunk_ids.add(chunk_id)
        state = str(metadata.get("index_state") or "").upper()
        if state not in {"READY", "BUILDING"}:
            issues.append(f"unexpected index_state: {metadata.get('index_state')}")

    safe_size = max(1, int(sample_size))
    chosen_ids = [item_id for item_id, _ in pairs[:safe_size]]
    if len(pairs) > 1 and len(chosen_ids) < safe_size:
        chosen_ids.append(pairs[-1][0])
    if len(pairs) > 2:
        chosen_ids.append(pairs[len(pairs) // 2][0])
    chosen_ids = list(dict.fromkeys(chosen_ids))[:safe_size]

    if chosen_ids:
        try:
            sampled = self.collection.get(ids=chosen_ids, include=["embeddings"])
            embeddings = _as_list(sampled.get("embeddings"))
            expected_dim = self._collection_dim()
            if len(embeddings) != len(chosen_ids):
                issues.append("embedding validation sample returned an incomplete set")
            for vector in embeddings:
                normalized = _safe_vector(vector)
                norm_sq = sum(item * item for item in normalized) if normalized is not None else 0.0
                if normalized is None or (expected_dim and len(normalized) != expected_dim) or norm_sq <= 1e-24:
                    issues.append("invalid semantic embedding in validation sample")
        except Exception as exc:
            issues.append(f"embedding validation sample failed: {type(exc).__name__}: {exc}")

    return {
        "document_id": document_id,
        "count": len(pairs),
        "valid": bool(pairs) and not issues,
        "issues": issues,
        "sampled_embeddings": min(len(chosen_ids), len(pairs)),
    }


def _safe_resolve_dimension(self: Any, embeddings: Any = None) -> int:
    if embeddings is not None:
        try:
            if len(embeddings) > 0:
                first = _safe_vector(embeddings[0])
                if first is not None:
                    return len(first)
        except (TypeError, IndexError, ValueError):
            pass
    stored = self._collection_dim()
    if stored > 0:
        return stored
    return 0


def _safe_search(self: Any, embedding: Any, n_results: int = 5, where: dict[str, Any] | None = None) -> dict[str, Any]:
    if embedding is None:
        return self._runtime_v6_original_search([], n_results=n_results, where=where)
    normalized = _safe_vector(embedding)
    if normalized is None:
        raise ValueError("Query embedding is not a valid finite vector.")
    return self._runtime_v6_original_search(normalized, n_results=n_results, where=where)


def _ui_list_value(value: Any) -> list[Any]:
    """Normalize UI payload collections before any code can test their truthiness."""
    return _as_list(value)


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        from rag_project.storage.vector_store import VectorStore
        if not hasattr(VectorStore, "_runtime_v6_original_validate_document_index"):
            VectorStore._runtime_v6_original_validate_document_index = VectorStore.validate_document_index
            VectorStore.validate_document_index = _safe_validate_document_index
        if not hasattr(VectorStore, "_runtime_v6_original_resolve_dimension"):
            VectorStore._runtime_v6_original_resolve_dimension = VectorStore._resolve_dimension
            VectorStore._resolve_dimension = _safe_resolve_dimension
        if not hasattr(VectorStore, "_runtime_v6_original_search"):
            VectorStore._runtime_v6_original_search = VectorStore.search
            VectorStore.search = _safe_search
        try:
            from rag_project.app import bookrag_ui

            def safe_evidence(result: dict[str, Any]) -> list[Any]:
                for key in ("evidence", "hits", "citations"):
                    value = result.get(key)
                    if value is not None:
                        return _ui_list_value(value)
                return []

            bookrag_ui._evidence = safe_evidence
        except Exception:
            pass
        _INSTALLED = True
