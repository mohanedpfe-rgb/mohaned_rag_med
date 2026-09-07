from __future__ import annotations

import logging
from typing import Iterable, List

from rag_project.retrieval.hybrid_retriever import RetrievalHit


class Reranker:
    """Rerank retrieval hits using a CrossEncoder model if available."""
    class CrossEncoderInitError(RuntimeError):
        """Raised when CrossEncoder model fails to initialize."""
        def __init__(self, message: str, original: Exception | None = None):
            super().__init__(message)
            self.original = original

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", *, enabled: bool = True):
        self.model_name = model_name
        self.model = None
        self.enabled = bool(enabled)
        self.error: str | None = None
        self.logger = logging.getLogger(__name__)

    def _ensure_model(self) -> bool:
        if self.model is not None:
            return True
        if not self.enabled:
            return False
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            return True
        except Exception as exc:
            self.error = f"Reranker unavailable: {exc}"
            self.logger.warning(self.error)
            self.model = None
            return False

    @staticmethod
    def confidence(hits: List[RetrievalHit]) -> dict[str, float | str]:
        if not hits:
            return {"level": "none", "top_score": 0.0, "margin": 0.0}
        top_score = float(hits[0].score)
        second_score = float(hits[1].score) if len(hits) > 1 else top_score
        margin = max(0.0, top_score - second_score)
        if top_score >= 0.6 and margin >= 0.08:
            level = "high"
        elif top_score >= 0.25:
            level = "medium"
        else:
            level = "low"
        return {"level": level, "top_score": top_score, "margin": margin}

    def rerank(self, query: str, hits: Iterable[RetrievalHit]) -> List[RetrievalHit]:
        hits_list = list(hits)
        if not hits_list:
            return []
        if not self._ensure_model():
            return sorted(hits_list, key=lambda item: item.score, reverse=True)
        try:
            pairs = [[query, hit.text] for hit in hits_list]
            scores = self.model.predict(pairs, show_progress_bar=False)
            for hit, score in zip(hits_list, scores, strict=True):
                hit.score = float(score)
            return sorted(hits_list, key=lambda item: item.score, reverse=True)
        except Exception as exc:
            self.error = f"Reranker inference failed: {exc}"
            self.logger.warning(self.error)
            self.model = None
            return sorted(hits_list, key=lambda item: item.score, reverse=True)

<<<<<<< HEAD
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", *, enabled: bool = True):
        self.model_name = model_name
        self.model = None
        self.enabled = bool(enabled)
        self.error: str | None = None
        self.logger = logging.getLogger(__name__)

    def _ensure_model(self) -> bool:
        if self.model is not None:
            return True
        if not self.enabled:
            return False
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)
            return True
        except Exception as exc:
            self.error = f"Reranker unavailable: {exc}"
            self.logger.warning(self.error)
            self.model = None
            return False
=======
    """Rerank retrieval hits using a CrossEncoder model if available."""

    class CrossEncoderInitError(RuntimeError):
        """Raised when CrossEncoder model fails to initialize."""
        def __init__(self, message: str, original: Exception | None = None):
            super().__init__(message)
            self.original = original

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            self.init_error = None
        except Exception as exc:
            self.model = None
            self.init_error = exc
>>>>>>> 9dd23f8 (Clean up duplicate rebuild_index implementation in VectorStore and ensure correct index activation)

    @staticmethod
    def confidence(hits: List[RetrievalHit]) -> dict[str, float | str]:
        if not hits:
            return {"level": "none", "top_score": 0.0, "margin": 0.0}
        top_score = float(hits[0].score)
        second_score = float(hits[1].score) if len(hits) > 1 else top_score
        margin = max(0.0, top_score - second_score)
        # These are retrieval-score heuristics, not probabilities.
        if top_score >= 0.6 and margin >= 0.08:
            level = "high"
        elif top_score >= 0.25:
            level = "medium"
        else:
            level = "low"
        return {"level": level, "top_score": top_score, "margin": margin}

    def rerank(self, query: str, hits: Iterable[RetrievalHit]) -> List[RetrievalHit]:
        hits_list = list(hits)
        if not hits_list:
            return []
<<<<<<< HEAD
        if not self._ensure_model():
            return sorted(hits_list, key=lambda item: item.score, reverse=True)
        try:
            pairs = [[query, hit.text] for hit in hits_list]
            scores = self.model.predict(pairs, show_progress_bar=False)
            for hit, score in zip(hits_list, scores, strict=True):
                hit.score = float(score)
            return sorted(hits_list, key=lambda item: item.score, reverse=True)
        except Exception as exc:
            self.error = f"Reranker inference failed: {exc}"
            self.logger.warning(self.error)
            self.model = None
            return sorted(hits_list, key=lambda item: item.score, reverse=True)
=======
        if self.model is None:
            # Fallback: keep original ordering by score
            return sorted(hits_list, key=lambda item: item.score, reverse=True)
        pairs = [[query, hit.text] for hit in hits_list]
        scores = self.model.predict(pairs)
        for hit, score in zip(hits_list, scores):
            hit.score = float(score)
        return sorted(hits_list, key=lambda item: item.score, reverse=True)
>>>>>>> 9dd23f8 (Clean up duplicate rebuild_index implementation in VectorStore and ensure correct index activation)
