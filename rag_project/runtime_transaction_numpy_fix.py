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


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        from rag_project import runtime_version_transaction_fix as transaction

        original = getattr(transaction, "_restore_snapshot", None)
        if not callable(original) or getattr(original, "_numpy_snapshot_fix", False):
            _INSTALLED = True
            return

        def safe_restore(system: Any, snapshot: dict[str, Any]) -> None:
            vector = snapshot.get("vector")
            if isinstance(vector, dict):
                raw = vector.get("embeddings")
                if raw is not None:
                    vector["embeddings"] = _to_plain_embedding(raw)
                raw_ids = vector.get("ids")
                if raw_ids is not None and not isinstance(raw_ids, list):
                    vector["ids"] = list(raw_ids)
                raw_docs = vector.get("documents")
                if raw_docs is not None and not isinstance(raw_docs, list):
                    vector["documents"] = list(raw_docs)
                raw_meta = vector.get("metadatas")
                if raw_meta is not None and not isinstance(raw_meta, list):
                    vector["metadatas"] = list(raw_meta)
            return original(system, snapshot)

        safe_restore._numpy_snapshot_fix = True
        transaction._restore_snapshot = safe_restore
        _INSTALLED = True


__all__ = ["install"]
