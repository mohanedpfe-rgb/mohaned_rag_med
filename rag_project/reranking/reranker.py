from __future__ import annotations

import logging
from typing import Any, Iterable, List

from rag_project.retrieval.hybrid_retriever import RetrievalHit


class Reranker:
    """Memory-efficient reranker using a lightweight CrossEncoder with lazy loading."""

    class CrossEncoderInitError(RuntimeError):
        """Raised when CrossEncoder model fails to initialize."""

        def __init__(self, message: str, original: Exception | None = None):
            super().__init__(message)
            self.original = original

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        enabled: bool = True,
        max_batch_size: int = 8,
        max_text_length: int = 512,
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.model = None
        self.enabled = bool(enabled)
        self.error: str | None = None
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_text_length = max(64, int(max_text_length))
        self.logger = logging.getLogger(__name__)
        self._model_attempted = False

    def _ensure_model(self) -> bool:
        if self.model is not None:
            return True
        if not self.enabled:
            return False
        if self._model_attempted:
            return False
        self._model_attempted = True
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(
                self.model_name,
                max_length=self.max_text_length,
            )
            return True
        except Exception as exc:
            self.error = f"Reranker unavailable: {exc}"
            self.logger.warning(self.error)
            self.model = None
            return False

    def unload(self) -> None:
        """Release the underlying CrossEncoder model to reclaim memory."""
        self.model = None
        self._model_attempted = False

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

    def _truncate_text(self, text: str) -> str:
        if not text:
            return ""
        if len(text) <= self.max_text_length:
            return text
        head = text[: self.max_text_length // 2]
        tail = text[-self.max_text_length // 2 :]
        return f"{head}\n...\n{tail}"

    def rerank(self, query: str, hits: Iterable[RetrievalHit]) -> List[RetrievalHit]:
        hits_list = list(hits)
        if not hits_list:
            return []
        if not self._ensure_model():
            return sorted(hits_list, key=lambda item: item.score, reverse=True)
        try:
            pairs: list[list[str]] = []
            for hit in hits_list:
                truncated = self._truncate_text(hit.text or "")
                pairs.append([query or "", truncated])
            if len(pairs) <= self.max_batch_size:
                scores = self.model.predict(pairs, show_progress_bar=False)
            else:
                import numpy as np

                all_scores: list[Any] = []
                for start in range(0, len(pairs), self.max_batch_size):
                    batch = pairs[start : start + self.max_batch_size]
                    batch_scores = self.model.predict(batch, show_progress_bar=False)
                    all_scores.extend(
                        batch_scores.tolist() if isinstance(batch_scores, np.ndarray) else list(batch_scores)
                    )
                scores = all_scores
            for hit, score in zip(hits_list, scores, strict=True):
                hit.score = float(score)
            return sorted(hits_list, key=lambda item: item.score, reverse=True)
        except Exception as exc:
            self.error = f"Reranker inference failed: {exc}"
            self.logger.warning(self.error)
            self.model = None
            return sorted(hits_list, key=lambda item: item.score, reverse=True)
