from __future__ import annotations

import threading
from typing import Any

_LOCK = threading.RLock()
_INSTALLED = False


def _normalize_embedding_result(self: Any, texts: Any):
    """Guarantee downstream callers receive a plain Python list, never NumPy/array truth values."""
    result = self._runtime_v7_original_embed_texts(texts)
    if result is None:
        return []
    if isinstance(result, list):
        return result
    if isinstance(result, tuple):
        return list(result)
    try:
        return list(result)
    except (TypeError, ValueError) as exc:
        raise TypeError("Embedding backend returned a non-iterable result.") from exc


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        from rag_project.embeddings.embedding_service import EmbeddingService
        if not hasattr(EmbeddingService, "_runtime_v7_original_embed_texts"):
            EmbeddingService._runtime_v7_original_embed_texts = EmbeddingService.embed_texts
            EmbeddingService.embed_texts = _normalize_embedding_result
        _INSTALLED = True


__all__ = ["install"]
