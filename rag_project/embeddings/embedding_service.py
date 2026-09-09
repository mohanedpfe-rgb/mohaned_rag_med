from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import requests

from rag_project.security import sanitize_model_text, validate_ollama_url


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except (TypeError, ValueError):
        return []


class EmbeddingProfile:
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
    ) -> None:
        model = model if model is not None else model_name
        if model is None or dimension is None:
            raise ValueError("Embedding profile requires a model name and dimension.")
        self.provider = str(provider)
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
        return f"{self.provider}:{self.model}:{self.model_version or 'unknown'}"

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
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()

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
        batch_size: int = 16,
        retries: int = 2,
        timeout_seconds: float = 180.0,
        test_mode: bool = False,
        cache_size: int = 128,
        cache_ttl_seconds: float = 900.0,
        max_concurrency: int = 1,
        prefer_local_transformers: bool = False,
    ) -> None:
        self.base_url = (
            validate_ollama_url(base_url)
            if not test_mode
            else base_url.rstrip("/") if base_url else ""
        )
        self.model = sanitize_model_text(model, limit=200).strip()
        if not self.model:
            raise ValueError("Embedding model identifier cannot be empty.")
        self.batch_size = max(1, min(int(batch_size), 32))
        self.retries = max(0, min(int(retries), 3))
        self.timeout_seconds = max(30.0, min(float(timeout_seconds), 300.0))
        self.test_mode = test_mode
        self.dimension: int | None = None
        self.provider = "deterministic-test" if test_mode else "ollama"
        self.prefer_local_transformers = bool(prefer_local_transformers)
        self.cache_size = max(0, int(cache_size))
        self.cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self._profile_fingerprint: str | None = None
        self._query_cache: OrderedDict[str, tuple[float, list[float]]] = OrderedDict()
        self._embedding_cache: OrderedDict[str, tuple[float, list[float]]] = OrderedDict()
        self._cache_lock = threading.RLock()
        self._active_batch_size = self.batch_size
        self._consecutive_timeouts = 0
        self._inference_semaphore = threading.BoundedSemaphore(max(1, int(max_concurrency)))
        self._sentence_transformer = None
        self._ollama_available: bool | None = None
        self._ollama_last_check = 0.0
        self.last_error: str | None = None

    def _get_sentence_transformer(self) -> Any:
        if self._sentence_transformer is None:
            from sentence_transformers import SentenceTransformer

            self._sentence_transformer = SentenceTransformer(self.model)
        return self._sentence_transformer

    def _cache_namespace(self) -> str:
        if self.dimension is not None:
            identity = self.identity
            if identity is not None:
                return identity.fingerprint
        payload = f"{self.provider}|{self.model}|{self.dimension or 0}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _cache_value(cached: Any, now: float, ttl: float) -> list[float] | None:
        if not isinstance(cached, tuple) or len(cached) != 2:
            return None
        timestamp, vector = cached
        if not isinstance(timestamp, (int, float)) or not isinstance(vector, list):
            return None
        if ttl <= 0 or now - float(timestamp) > ttl:
            return None
        return list(vector)

    def _check_ollama_available(self, force: bool = False) -> bool:
        if self.test_mode:
            return False
        now = time.monotonic()
        if not force and self._ollama_available is not None and now - self._ollama_last_check < 10:
            return self._ollama_available
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=(1.5, 3.0),
                allow_redirects=False,
            )
            status_code = getattr(response, "status_code", 200)
            if 300 <= status_code < 400:
                raise requests.RequestException("redirect rejected")
            response.raise_for_status()
            self._ollama_available = True
            self.last_error = None
        except requests.RequestException as exc:
            self._ollama_available = False
            self.last_error = f"Ollama unavailable: {exc}"
        self._ollama_last_check = now
        return self._ollama_available

    @property
    def identity(self) -> EmbeddingProfile | None:
        if self.dimension is None:
            return None
        implementation = (
            "deterministic-test-v1"
            if self.test_mode
            else "sentence-transformers-v1"
            if self.provider == "sentence-transformers"
            else "ollama-api-v1"
        )
        profile = EmbeddingProfile(
            provider=self.provider,
            model=self.model,
            dimension=self.dimension,
            model_version="latest",
            normalization="none",
            metric="cosine",
            implementation_version=implementation,
        )
        if self._profile_fingerprint != profile.fingerprint:
            self._profile_fingerprint = profile.fingerprint
        return profile

    def discover_dimension(self) -> int:
        self.embed_query("__rag_dimension_probe__")
        if self.dimension is None:
            raise RuntimeError("Embedding dimension discovery produced no dimension.")
        return self.dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        clean_texts = [sanitize_model_text(text, limit=12000) for text in texts]
        if self.test_mode:
            return [self._test_embedding(text) for text in clean_texts]

        ordered: list[list[float] | None] = [None] * len(clean_texts)
        missing: list[str] = []
        missing_indices: list[int] = []
        now = time.monotonic()
        namespace = self._cache_namespace()

        with self._cache_lock:
            for index, text in enumerate(clean_texts):
                key = f"{namespace}::{text}"
                cached = self._cache_value(self._embedding_cache.get(key), now, self.cache_ttl_seconds)
                if cached is not None:
                    ordered[index] = cached
                    self._embedding_cache.move_to_end(key)
                elif key in self._embedding_cache:
                    self._embedding_cache.pop(key, None)
                else:
                    missing_indices.append(index)
                    missing.append(text)

        if missing:
            with self._inference_semaphore:
                new_vectors = _as_list(self._embed_batch(missing))
            if len(new_vectors) != len(missing):
                raise RuntimeError("Embedding backend returned an unexpected result count.")
            store_namespace = self._cache_namespace()
            with self._cache_lock:
                stored_at = time.monotonic()
                for offset, (index, text) in enumerate(zip(missing_indices, missing, strict=True)):
                    vector = list(_as_list(new_vectors[offset]))
                    key = f"{store_namespace}::{text}"
                    self._embedding_cache[key] = (stored_at, vector)
                    self._embedding_cache.move_to_end(key)
                    if self.cache_size and len(self._embedding_cache) > self.cache_size:
                        self._embedding_cache.popitem(last=False)
                    ordered[index] = vector

        if any(vector is None for vector in ordered):
            raise RuntimeError("Embedding backend returned incomplete results.")
        return [list(vector) for vector in ordered if vector is not None]

    def _split_and_embed(self, texts: list[str]) -> list[list[float]]:
        if len(texts) <= 1:
            raise RuntimeError("Adaptive embedding split cannot reduce a single-item batch further.")
        half = max(1, len(texts) // 2)
        left = self._ollama_embed_batch(texts[:half])
        right = self._ollama_embed_batch(texts[half:])
        return [*left, *right]

    def _ollama_embed_batch(self, texts: list[str]) -> list[list[float]]:
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
                result = _as_list(result)
                self._validate(result, len(attempt_texts))
                self.provider = "ollama"
                self.last_error = None
                self._consecutive_timeouts = 0
                self._ollama_available = True
                self._active_batch_size = min(self.batch_size, self._active_batch_size + 1)
                return [list(_as_list(vector)) for vector in result]
            except requests.exceptions.Timeout as exc:
                last_error = exc
                self.last_error = "Embedding request timed out"
                self._consecutive_timeouts += 1
                self._active_batch_size = max(1, self._active_batch_size // 2)
                if len(attempt_texts) > 1 and self._active_batch_size < len(attempt_texts):
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

    def _transformers_embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not self.prefer_local_transformers:
            raise RuntimeError(
                "Local SentenceTransformers fallback is disabled; use the configured Ollama embedding service."
            )
        try:
            vectors = self._get_sentence_transformer().encode(
                texts,
                batch_size=max(1, min(self.batch_size, len(texts))),
                show_progress_bar=False,
                normalize_embeddings=False,
                convert_to_numpy=True,
            )
            self.provider = "sentence-transformers"
            self.last_error = None
        except Exception as exc:
            self.last_error = type(exc).__name__
            self._sentence_transformer = None
            raise RuntimeError("SentenceTransformers fallback failed.") from exc
        if isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()
        vectors = _as_list(vectors)
        if vectors and not isinstance(vectors[0], list):
            vectors = [list(vectors)]
        self._validate(vectors, len(texts))
        return [list(map(float, vector)) for vector in vectors]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.prefer_local_transformers:
            try:
                return self._transformers_embed_batch(texts)
            except Exception:
                if not self._check_ollama_available():
                    raise
        if not self._check_ollama_available():
            raise RuntimeError(
                f"Embedding backend unavailable: Ollama at {self.base_url!r} did not respond to its health check. "
                f"Start Ollama and ensure model {self.model!r} is installed."
            )
        return self._ollama_embed_batch(texts)

    def _validate(self, vectors: list[list[float]], expected_count: int) -> None:
        normalized = [_as_list(vector) for vector in _as_list(vectors)]
        if len(normalized) != expected_count or not normalized:
            raise ValueError("Embedding service returned an unexpected number of vectors.")
        dimension = len(normalized[0])
        if dimension == 0:
            raise ValueError("Embedding vectors must not be empty.")
        for vector in normalized:
            if len(vector) != dimension:
                raise ValueError("Embedding vectors have inconsistent dimensions.")
            if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in vector):
                raise ValueError("Embedding vectors contain non-finite values.")
            if math.sqrt(sum(float(value) ** 2 for value in vector)) <= 1e-12:
                raise ValueError("Embedding vectors must have a non-zero norm.")
        if self.dimension is None:
            self.dimension = dimension
        elif self.dimension != dimension:
            raise ValueError(f"Embedding dimension changed from {self.dimension} to {dimension}.")

    def embed_query(self, query: str) -> list[float]:
        clean_query = sanitize_model_text(query, limit=4000).strip()
        if not clean_query:
            raise ValueError("Query cannot be empty after sanitization.")
        now = time.monotonic()
        namespace = self._cache_namespace()
        key = f"{namespace}::{clean_query}"
        with self._cache_lock:
            cached = self._cache_value(self._query_cache.get(key), now, self.cache_ttl_seconds)
            if cached is not None:
                self._query_cache.move_to_end(key)
                return cached
            self._query_cache.pop(key, None)

        vectors = self.embed_texts([clean_query])
        if not vectors:
            raise RuntimeError("No embedding was produced for the query.")
        vector = list(vectors[0])
        resolved_key = f"{self._cache_namespace()}::{clean_query}"
        with self._cache_lock:
            if self.cache_size:
                self._query_cache[resolved_key] = (time.monotonic(), vector)
                self._query_cache.move_to_end(resolved_key)
                while len(self._query_cache) > self.cache_size:
                    self._query_cache.popitem(last=False)
        return vector

    def validate_embedding(self, vector: list[float] | tuple[float, ...]) -> None:
        self._validate([list(vector)], 1)

    def validate_batch(self, vectors: list[list[float]]) -> None:
        self._validate(vectors, len(_as_list(vectors)))

    def _test_embedding(self, text: str) -> list[float]:
        vector = [byte / 255.0 for byte in hashlib.sha256(text.encode("utf-8")).digest()]
        self._validate([vector], 1)
        return vector
