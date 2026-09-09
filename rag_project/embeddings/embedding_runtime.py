from __future__ import annotations

import time
from typing import Any

import requests

_INSTALLED = False


def _ollama_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    attempt_texts = list(texts)
    last_error: Exception | None = None
    attempts = max(1, self.retries + 1)
    for attempt in range(attempts):
        if self._consecutive_timeouts >= 2:
            self._active_batch_size = 1
        if len(attempt_texts) > self._active_batch_size:
            return self._split_and_embed(attempt_texts)
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": attempt_texts},
                timeout=(5, self.timeout_seconds),
                allow_redirects=False,
            )
            status_code = getattr(response, "status_code", 200)
            if 300 <= status_code < 400:
                raise RuntimeError("Ollama redirect rejected")
            if status_code == 413 and len(attempt_texts) > 1:
                self._active_batch_size = max(1, len(attempt_texts) // 2)
                return self._split_and_embed(attempt_texts)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Embedding response was not an object.")
            if "embeddings" in payload:
                result = payload["embeddings"]
            elif isinstance(payload.get("embedding"), list):
                result = [payload["embedding"]]
            elif isinstance(payload.get("data"), list):
                result = [item["embedding"] for item in payload["data"]]
            else:
                raise ValueError("Embedding response did not contain embeddings.")
            result = list(result)
            self._validate(result, len(attempt_texts))
            self.provider = "ollama"
            self.last_error = None
            self._consecutive_timeouts = 0
            self._ollama_available = True
            self._active_batch_size = min(self.batch_size, self._active_batch_size + 1)
            return [list(vector) for vector in result]
        except requests.exceptions.Timeout as exc:
            last_error = exc
            self.last_error = "Embedding request timed out"
            self._consecutive_timeouts += 1
            reduced = max(1, self._active_batch_size // 2)
            self._active_batch_size = min(reduced, max(1, len(attempt_texts) // 2))
            if len(attempt_texts) > self._active_batch_size:
                return self._split_and_embed(attempt_texts)
            if attempt + 1 < attempts:
                time.sleep(min(2, 0.75 * (2**attempt)))
        except (requests.RequestException, ConnectionError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            last_error = exc
            self.last_error = type(exc).__name__
            self._ollama_available = False
            if attempt + 1 < attempts:
                time.sleep(min(1.5, 0.4 * (2**attempt)))
    raise RuntimeError(
        f"Ollama embedding service failed after {attempts} attempts for model {self.model!r}: {last_error}"
    ) from last_error


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from rag_project.embeddings.embedding_service import EmbeddingService

    _ollama_embed_batch.__wrapped__ = EmbeddingService._ollama_embed_batch
    EmbeddingService._ollama_embed_batch = _ollama_embed_batch
    _INSTALLED = True


install()

__all__ = ["install"]
