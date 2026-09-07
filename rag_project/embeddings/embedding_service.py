from __future__ import annotations

import hashlib
import json
import math
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import requests


class EmbeddingProfile:
    """Immutable identity for a specific embedding configuration and index profile."""

    def __init__(
        self,
        provider: str,
        model: str | None = None,
        dimension: int | None = None,
        *,
        model_name: str | None = None,
        model_version: str | None = None,
        normalization: str = "none",
        metric: str = "cosine",
        implementation_version: str = "embedding-v2",
        configuration: dict[str, Any] | None = None,
        creation_timestamp: str | None = None,
    ):
        if model is None:
            model = model_name
        if model is None:
            raise ValueError("Embedding profile requires a model name.")
        if dimension is None:
            raise ValueError("Embedding profile requires a dimension.")
        self.provider = provider
        self.model = str(model)
        self.model_name = self.model
        self.model_version = model_version
        self.dimension = int(dimension)
        self.normalization = normalization
        self.metric = metric
        self.implementation_version = implementation_version
        self.configuration = dict(configuration or {})
        self.creation_timestamp = creation_timestamp or datetime.now(timezone.utc).isoformat()
        self._configuration_fingerprint = self._compute_fingerprint()

    @property
    def normalized_model_identifier(self) -> str:
        version = self.model_version or "unknown"
        return f"{self.provider}:{self.model}:{version}"

    @property
    def configuration_fingerprint(self) -> str:
        return self._configuration_fingerprint

    @property
    def fingerprint(self) -> str:
        return self._configuration_fingerprint

    @property
    def embedding_id(self) -> str:
        return ":".join(
            (
                self.provider,
                self.model,
                self.model_version or "unknown",
                str(self.dimension),
                self.normalization,
                self.metric,
                self.implementation_version,
            )
        )

    def _compute_fingerprint(self) -> str:
        payload = {
            "provider": self.provider,
            "model": self.model,
            "model_version": self.model_version,
            "dimension": self.dimension,
            "normalization": self.normalization,
            "metric": self.metric,
            "implementation_version": self.implementation_version,
            **self.configuration,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "model_name": self.model,
            "model_version": self.model_version,
            "dimension": self.dimension,
            "normalization": self.normalization,
            "metric": self.metric,
            "implementation_version": self.implementation_version,
            "normalized_model_identifier": self.normalized_model_identifier,
            "configuration_fingerprint": self.configuration_fingerprint,
            "fingerprint": self.configuration_fingerprint,
            "creation_timestamp": self.creation_timestamp,
            "embedding_id": self.embedding_id,
        }


EmbeddingIdentity = EmbeddingProfile


class EmbeddingService:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        batch_size: int = 4,
        retries: int = 4,
        timeout_seconds: float = 180.0,
        test_mode: bool = False,
        cache_size: int = 10000,
        cache_ttl_seconds: float = 86400.0,
    ):
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.model = model
        self.batch_size = max(1, batch_size)
        self.retries = max(1, retries)
        self.timeout_seconds = max(5.0, float(timeout_seconds))
        self.test_mode = test_mode
        self.dimension: int | None = None
        self.provider = "deterministic-test" if test_mode else "sentence-transformers"
        self.cache_size = max(0, cache_size)
        self.cache_ttl_seconds = max(0.0, cache_ttl_seconds)
        self._query_cache: OrderedDict[str, tuple[float, list[float]]] = OrderedDict()
        self._embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._active_batch_size = self.batch_size
        self._consecutive_timeouts = 0
        self._sentence_transformer = None
        self.last_error: str | None = None

    def _get_sentence_transformer(self):
        if self._sentence_transformer is not None:
            return self._sentence_transformer
        from sentence_transformers import SentenceTransformer

        self._sentence_transformer = SentenceTransformer(self.model)
        return self._sentence_transformer

    @property
    def identity(self) -> EmbeddingProfile | None:
        if self.dimension is None:
            return None
        implementation = (
            "deterministic-test-v1"
            if self.test_mode
            else "sentence-transformers-v1" if self.provider == "sentence-transformers" else "ollama-api-v1"
        )
        return EmbeddingProfile(
            provider=self.provider,
            model=self.model,
            dimension=self.dimension,
            model_version="latest",
            normalization="none",
            metric="cosine",
            implementation_version=implementation,
        )

    def discover_dimension(self) -> int:
        self.embed_query("__rag_dimension_probe__")
        if self.dimension is None:
            raise RuntimeError("Embedding dimension discovery produced no dimension.")
        return self.dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.test_mode:
            return [self._test_embedding(text) for text in texts]

        ordered_results: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        for index, text in enumerate(texts):
            cached = self._embedding_cache.get(text)
            if cached is not None:
                ordered_results[index] = cached
            else:
                missing_indices.append(index)
                missing_texts.append(text)

        if missing_texts:
            new_vectors = self._embed_batch(missing_texts)
            if len(new_vectors) != len(missing_texts):
                raise RuntimeError(
                    f"Embedding backend returned {len(new_vectors)} vectors for {len(missing_texts)} texts."
                )
            for offset, (index, text) in enumerate(zip(missing_indices, missing_texts, strict=True)):
                vector = new_vectors[offset]
                self._embedding_cache[text] = vector
                if self.cache_size and len(self._embedding_cache) > self.cache_size:
                    self._embedding_cache.popitem(last=False)
                ordered_results[index] = vector
        return [vector for vector in ordered_results if vector is not None]

    def _fallback_http_embed_batch(self, texts: list[str]) -> list[list[float]]:
        attempt_texts = list(texts)
        last_error: Exception | None = None
        for attempt in range(self.retries):
            if self._consecutive_timeouts >= 2:
                self._active_batch_size = 1
            if len(attempt_texts) > self._active_batch_size:
                half = max(1, len(attempt_texts) // 2)
                left = self._fallback_http_embed_batch(attempt_texts[:half])
                right = self._fallback_http_embed_batch(attempt_texts[half:])
                return left + right
            try:
                response = requests.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": attempt_texts},
                    timeout=(5, self.timeout_seconds),
                )
                response.raise_for_status()
                payload = response.json()
                if "embeddings" in payload:
                    result = payload["embeddings"]
                elif isinstance(payload.get("embedding"), list):
                    result = [payload["embedding"]]
                elif isinstance(payload.get("data"), list):
                    result = [item["embedding"] for item in payload["data"]]
                else:
                    raise ValueError("Embedding response did not contain embeddings.")
                self._validate(result, len(attempt_texts))
                self.provider = "ollama"
                self.last_error = None
                self._consecutive_timeouts = 0
                if self._active_batch_size < self.batch_size:
                    self._active_batch_size = min(self.batch_size, self._active_batch_size + 1)
                return result
            except requests.exceptions.Timeout as exc:
                last_error = exc
                self.last_error = str(exc)
                self._consecutive_timeouts += 1
                self._active_batch_size = max(1, self._active_batch_size // 2)
                if attempt + 1 < self.retries:
                    time.sleep(1.5 * (2**attempt))
            except (requests.RequestException, ConnectionError, ValueError, KeyError, TypeError) as exc:
                last_error = exc
                self.last_error = str(exc)
                if attempt + 1 < self.retries:
                    time.sleep(0.5 * (2**attempt))
        raise RuntimeError(
            f"Embedding service failed after {self.retries} attempts for model {self.model!r}."
        ) from last_error

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        texts = [text for text in texts if text is not None]
        if not texts:
            return []
        try:
            model = self._get_sentence_transformer()
            vectors = model.encode(
                texts,
                batch_size=max(1, min(self.batch_size, len(texts))),
                show_progress_bar=False,
                normalize_embeddings=False,
                convert_to_numpy=True,
            )
            self.provider = "sentence-transformers"
            self.last_error = None
        except Exception as exc:
            self.last_error = str(exc)
            self._sentence_transformer = None
            return self._fallback_http_embed_batch(texts)
        if isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()
        if isinstance(vectors, list) and vectors and not isinstance(vectors[0], list):
            vectors = [list(vectors)]
        self._validate(vectors, len(texts))
        return [list(map(float, vector)) for vector in vectors]

    def _validate(self, vectors: list[list[float]], expected_count: int) -> None:
        if len(vectors) != expected_count or not vectors:
            raise ValueError("Embedding service returned an unexpected number of vectors.")
        dimension = len(vectors[0])
        if dimension == 0 or any(
            len(vector) != dimension
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector)
            for vector in vectors
        ):
            raise ValueError("Embedding vectors have inconsistent dimensions.")
        if self.dimension is None:
            self.dimension = dimension
        elif self.dimension != dimension:
            raise ValueError(f"Embedding dimension changed from {self.dimension} to {dimension}.")

    def embed_query(self, query: str) -> list[float]:
        cache_key = query.strip()
        now = time.monotonic()
        cached = self._query_cache.get(cache_key)
        if cached and now - cached[0] <= self.cache_ttl_seconds:
            self._query_cache.move_to_end(cache_key)
            return list(cached[1])
        vectors = self.embed_texts([query])
        if not vectors:
            raise RuntimeError("No embedding was produced for the query.")
        vector = list(vectors[0])
        if self.cache_size:
            self._query_cache[cache_key] = (now, vector)
            self._query_cache.move_to_end(cache_key)
            while len(self._query_cache) > self.cache_size:
                self._query_cache.popitem(last=False)
        return vector

    def _test_embedding(self, text: str) -> list[float]:
        vector = [byte / 255.0 for byte in hashlib.sha256(text.encode("utf-8")).digest()]
        self._validate([vector], 1)
        return vector
