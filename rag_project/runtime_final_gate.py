from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

import requests


_LEASE_CONTEXT = threading.local()


def _set_lease(state_store: Any, document_id: str, worker_id: str) -> None:
    _LEASE_CONTEXT.state_store = state_store
    _LEASE_CONTEXT.document_id = str(document_id)
    _LEASE_CONTEXT.worker_id = str(worker_id)


def _clear_lease(document_id: str | None = None, worker_id: str | None = None) -> None:
    current_document = getattr(_LEASE_CONTEXT, "document_id", None)
    current_worker = getattr(_LEASE_CONTEXT, "worker_id", None)
    if document_id is not None and current_document != str(document_id):
        return
    if worker_id is not None and current_worker != str(worker_id):
        return
    for name in ("state_store", "document_id", "worker_id"):
        try:
            delattr(_LEASE_CONTEXT, name)
        except AttributeError:
            pass


def _current_lease() -> tuple[Any | None, str | None, str | None]:
    return (
        getattr(_LEASE_CONTEXT, "state_store", None),
        getattr(_LEASE_CONTEXT, "document_id", None),
        getattr(_LEASE_CONTEXT, "worker_id", None),
    )


def _lease_matches(record: dict[str, Any] | None, document_id: str) -> bool:
    _, expected_document, expected_worker = _current_lease()
    if expected_document != str(document_id) or not expected_worker or not record:
        return False
    if str(record.get("lease_owner") or "") != str(expected_worker):
        return False
    expires = record.get("lease_expires_at")
    if not expires:
        return False
    try:
        return datetime.fromisoformat(str(expires).replace("Z", "+00:00")) > datetime.now(timezone.utc)
    except (TypeError, ValueError):
        return False


def _safe_claim(self: Any, document_id: str, worker_id: str, lease_seconds: int = 900) -> bool:
    result = self._original_runtime_final_claim(document_id, worker_id, lease_seconds)
    if result:
        _set_lease(self, document_id, worker_id)
    return result


def _safe_release(self: Any, document_id: str, worker_id: str) -> bool:
    try:
        return self._original_runtime_final_release(document_id, worker_id)
    finally:
        _clear_lease(document_id, worker_id)


def _safe_transition(self: Any, document_id: str, new_stage: str, **values: Any) -> None:
    stage = str(new_stage).upper()
    if stage in {
        "INDEXING",
        "VALIDATING_INDEX",
        "READY",
        "COMPLETED",
        "DEGRADED_LEXICAL",
        "FAILED",
        "FAILED_EXTRACTION",
        "FAILED_OCR",
        "FAILED_EMBEDDING",
        "FAILED_INDEXING",
        "QUARANTINED",
    }:
        _, expected_document, expected_worker = _current_lease()
        if expected_worker is not None and expected_document == str(document_id):
            record = self.get_document(document_id)
            if not _lease_matches(record, document_id):
                raise RuntimeError(
                    f"Lease ownership changed or expired for {document_id}; refusing {stage} transition."
                )
    return self._original_runtime_final_transition(document_id, new_stage, **values)


def _safe_version_state(self: Any, document_id: str, version_id: str, state: str) -> None:
    store, expected_document, expected_worker = _current_lease()
    if expected_worker is not None and expected_document == str(document_id):
        record = store.get_document(document_id) if store is not None else None
        if not _lease_matches(record, document_id):
            raise RuntimeError(
                f"Lease ownership changed or expired for {document_id}; refusing index visibility change."
            )
    return self._original_runtime_final_version_state(document_id, version_id, state)


def _model_digest(self: Any) -> str | None:
    cached = getattr(self, "_runtime_final_model_digest", None)
    checked_at = float(getattr(self, "_runtime_final_model_digest_checked_at", 0.0) or 0.0)
    if cached and time.monotonic() - checked_at < 300.0:
        return str(cached)
    base_url = str(getattr(self, "base_url", "") or "").rstrip("/")
    model = str(getattr(self, "model", "") or "")
    if not base_url or not model or getattr(self, "test_mode", False):
        return None
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=(1.5, 4.0))
        response.raise_for_status()
        payload = response.json()
        for item in payload.get("models", []) or []:
            name = str(item.get("name") or item.get("model") or "")
            if name == model or name.split(":", 1)[0] == model.split(":", 1)[0]:
                digest = str(item.get("digest") or "").strip()
                if digest:
                    self._runtime_final_model_digest = digest
                    self._runtime_final_model_digest_checked_at = time.monotonic()
                    return digest
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        pass
    self._runtime_final_model_digest_checked_at = time.monotonic()
    return None


def install() -> None:
    from rag_project.app.rag_system import RAGSystem
    from rag_project.embeddings.embedding_service import EmbeddingProfile, EmbeddingService
    from rag_project.ingestion.state_store import IngestionStateStore
    from rag_project.storage.vector_store import VectorStore

    if not hasattr(IngestionStateStore, "_original_runtime_final_claim"):
        IngestionStateStore._original_runtime_final_claim = IngestionStateStore.claim_document
        IngestionStateStore.claim_document = _safe_claim
    if not hasattr(IngestionStateStore, "_original_runtime_final_release"):
        IngestionStateStore._original_runtime_final_release = IngestionStateStore.release_document
        IngestionStateStore.release_document = _safe_release
    if not hasattr(IngestionStateStore, "_original_runtime_final_transition"):
        IngestionStateStore._original_runtime_final_transition = IngestionStateStore.transition_document_state
        IngestionStateStore.transition_document_state = _safe_transition

    if not hasattr(VectorStore, "_original_runtime_final_version_state"):
        VectorStore._original_runtime_final_version_state = VectorStore.set_version_index_state
        VectorStore.set_version_index_state = _safe_version_state

    current_property = EmbeddingService.identity
    if not getattr(current_property.fget, "_runtime_final_wrapped", False):
        original_getter = current_property.fget

        def identity_with_digest(self: Any):
            profile = original_getter(self)
            if profile is None or getattr(self, "test_mode", False) or getattr(self, "provider", "") != "ollama":
                return profile
            digest = _model_digest(self)
            if not digest:
                return profile
            return EmbeddingProfile(
                provider=profile.provider,
                model=profile.model,
                dimension=profile.dimension,
                model_version=digest,
                normalization=profile.normalization,
                metric=profile.metric,
                implementation_version=profile.implementation_version,
                configuration=profile.configuration,
            )

        identity_with_digest._runtime_final_wrapped = True
        EmbeddingService.identity = property(identity_with_digest)
