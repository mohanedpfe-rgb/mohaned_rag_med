from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from rag_project.application import _normalize_runtime_settings, runtime_contract
from rag_project.configuration.config_i5_16gb import get_preset, get_profile
from rag_project.configuration.settings import Settings
from rag_project.embeddings.embedding_service import EmbeddingProfile, EmbeddingService
from rag_project.security import (
    validate_ollama_url,
    validate_pdf_payload,
    validate_storage_path,
)
from rag_project.storage.vector_store import VectorStore


def test_runtime_contract_declares_single_composition_root():
    contract = runtime_contract()
    assert contract["composition_root"] == "rag_project.application.create_rag_system"
    assert contract["canonical_service"].endswith("ProductionRAGSystem")
    assert contract["canonical_ingestion"].endswith("robust_ingest_file")
    assert contract["answer_monkey_patch"] is False
    assert contract["medical_safety_gate"] is True


def test_runtime_settings_normalization_is_idempotent():
    original = Settings(
        embedding_batch_size=1,
        embedding_retries=0,
        embedding_timeout_seconds=1,
        max_workers=99,
        ollama_concurrency=99,
    )
    first = _normalize_runtime_settings(original)
    snapshot = (
        first.embedding_batch_size,
        first.embedding_retries,
        first.embedding_timeout_seconds,
        first.max_workers,
        first.ollama_concurrency,
    )
    second = _normalize_runtime_settings(first)
    assert snapshot == (
        second.embedding_batch_size,
        second.embedding_retries,
        second.embedding_timeout_seconds,
        second.max_workers,
        second.ollama_concurrency,
    )
    assert snapshot == (16, 1, 30.0, 4, 2)


def test_runtime_settings_normalization_keeps_safe_values():
    settings = Settings(
        embedding_batch_size=24,
        embedding_retries=2,
        embedding_timeout_seconds=90,
        max_workers=3,
        ollama_concurrency=2,
    )
    normalized = _normalize_runtime_settings(settings)
    assert normalized.embedding_batch_size == 24
    assert normalized.embedding_retries == 2
    assert normalized.embedding_timeout_seconds == 90.0
    assert normalized.max_workers == 3
    assert normalized.ollama_concurrency == 2


def test_settings_parsers_handle_invalid_values():
    assert Settings._parse_bool("definitely-not-true", True) is False
    assert Settings._parse_bool(None, True) is True
    assert Settings._parse_float("not-a-number", 3.5) == 3.5
    assert Settings._parse_int("not-a-number", 7) == 7
    assert Settings._parse_int(True, 7) is True


def test_settings_post_init_clamps_core_bounds():
    settings = Settings(
        embedding_batch_size=999,
        embedding_retries=-4,
        embedding_timeout_seconds=1,
        top_k=999,
        temperature=9,
        vector_weight=-4,
        chunk_size=10,
        chunk_overlap=999,
        max_workers=0,
        ollama_concurrency=0,
    )
    assert settings.embedding_batch_size == 32
    assert settings.embedding_retries == 0
    assert settings.embedding_timeout_seconds == 30.0
    assert settings.top_k == 50
    assert settings.temperature == 1.0
    assert settings.vector_weight == 0.0
    assert settings.chunk_size == 200
    assert settings.chunk_overlap == 199
    assert settings.max_workers == 1
    assert settings.ollama_concurrency == 1


def test_device_presets_and_profiles_are_consistent():
    i5 = get_preset("i5_16gb")
    full = get_preset("full")
    assert i5["embedding_batch_size"] == 16
    assert full["embedding_batch_size"] == 32
    assert get_profile("i5_16gb").matches_mode("i5_16gb")
    assert get_profile("unknown").name == "i5_16gb"


def test_embedding_profile_identity_is_stable_and_complete():
    profile = EmbeddingProfile(
        provider="ollama",
        model="nomic-embed-text",
        dimension=768,
        model_version="latest",
        normalization="none",
        metric="cosine",
        implementation_version="ollama-api-v1",
        configuration={"batch_size": 16},
    )
    data = profile.to_dict()
    assert data["provider"] == "ollama"
    assert data["model"] == "nomic-embed-text"
    assert data["dimension"] == 768
    assert data["embedding_id"].startswith("ollama:nomic-embed-text:latest:768")
    assert len(data["fingerprint"]) == 64
    assert data["fingerprint"] == data["configuration_fingerprint"]


def test_embedding_profile_fingerprint_changes_with_configuration():
    a = EmbeddingProfile("ollama", "model", 3, configuration={"x": 1})
    b = EmbeddingProfile("ollama", "model", 3, configuration={"x": 2})
    assert a.fingerprint != b.fingerprint


def test_embedding_service_rejects_empty_model():
    with pytest.raises(ValueError, match="cannot be empty"):
        EmbeddingService("http://127.0.0.1:11434", "", test_mode=True)


def test_embedding_service_bounds_constructor_values():
    service = EmbeddingService(
        "http://127.0.0.1:11434",
        "model",
        batch_size=999,
        retries=999,
        timeout_seconds=1,
        cache_size=-1,
        cache_ttl_seconds=-1,
        test_mode=True,
    )
    assert service.batch_size == 32
    assert service.retries == 3
    assert service.timeout_seconds == 30.0
    assert service.cache_size == 0
    assert service.cache_ttl_seconds == 0.0


def test_embedding_service_test_mode_is_deterministic():
    service = EmbeddingService("unused", "test", test_mode=True)
    first = service.embed_texts(["same text", "different text"])
    second = service.embed_texts(["same text", "different text"])
    assert first == second
    assert len(first) == 2
    assert all(vector for vector in first)
    assert service.dimension is None


def test_embedding_service_empty_input_does_not_call_backend(monkeypatch):
    service = EmbeddingService("unused", "test", test_mode=True)
    monkeypatch.setattr(service, "_test_embedding", lambda _text: (_ for _ in ()).throw(AssertionError()))
    assert service.embed_texts([]) == []


def test_embedding_vector_validation_rejects_bad_shapes_and_values():
    service = EmbeddingService("unused", "test", test_mode=True)
    with pytest.raises(ValueError):
        service._validate([], 1)
    with pytest.raises(ValueError):
        service._validate([[]], 1)
    with pytest.raises(ValueError):
        service._validate([[1.0], [2.0, 3.0]], 2)
    with pytest.raises(ValueError):
        service._validate([[math.inf, 0.0]], 1)


def test_cache_value_rejects_expired_or_malformed_entries():
    assert EmbeddingService._cache_value(None, 10, 100) is None
    assert EmbeddingService._cache_value((0, [1.0]), 10, 0) is None
    assert EmbeddingService._cache_value((0, [1.0]), 101, 100) is None
    assert EmbeddingService._cache_value((0, (1.0,)), 10, 100) is None
    assert EmbeddingService._cache_value((0, [1.0]), 10, 100) == [1.0]


def test_security_rejects_non_local_ollama_targets():
    with pytest.raises(Exception):
        validate_ollama_url("http://example.com:11434")
    assert validate_ollama_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434"
    assert validate_ollama_url("http://localhost:11434/") == "http://localhost:11434"


def test_security_rejects_unsafe_storage_path(tmp_path: Path):
    inside = tmp_path / "data"
    outside = tmp_path.parent / "outside"
    inside.mkdir()
    with pytest.raises(Exception):
        validate_storage_path(tmp_path, outside, "path")
    assert validate_storage_path(tmp_path, inside, "path") == inside.resolve()


def test_security_accepts_real_pdf_payload_and_rejects_obvious_invalid_payload():
    good = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n"
    assert validate_pdf_payload("file.pdf", good) is None
    with pytest.raises(Exception):
        validate_pdf_payload("file.txt", good)
    with pytest.raises(Exception):
        validate_pdf_payload("file.pdf", b"plain text")


def test_vector_store_helpers_handle_numpy_without_truthiness_errors():
    assert VectorStore._valid_vector([1.0, 2.0], 2)
    assert VectorStore._valid_vector([1, 2], 2)
    assert not VectorStore._valid_vector([], 2)
    assert not VectorStore._valid_vector([math.nan, 1.0], 2)


def test_vector_store_normalizes_records_with_missing_metadata():
    store = VectorStore.__new__(VectorStore)
    normalized = store._normalize_records(
        {
            "ids": ["a"],
            "documents": ["text"],
            "metadatas": [{}],
        }
    )
    assert normalized[0]["id"] == "a"
    assert normalized[0]["document"] == "text"
    assert normalized[0]["metadata"]["index_state"] == "READY"
    assert normalized[0]["metadata"]["chunk_id"] == "a"
    assert normalized[0]["metadata"]["version_id"] == "legacy"


def test_vector_store_dimension_resolution_prefers_embedding_dimension():
    store = VectorStore.__new__(VectorStore)
    class Collection:
        metadata = {"dimension": 7}
    store.collection = Collection()
    assert store._resolve_dimension([[0.1, 0.2, 0.3]]) == 3
    assert store._resolve_dimension([]) == 7
    assert store._resolve_dimension(None) == 7


def test_vector_store_clear_and_count_are_safe_on_empty_persistent_store(tmp_path: Path):
    store = VectorStore(tmp_path / "vectors")
    assert store.count() == 0
    assert store.lexical_count() == 0
    store.clear_all()
    assert store.count() == 0
    assert store.lexical_count() == 0


def test_vector_store_index_state_transition_on_empty_document_is_noop(tmp_path: Path):
    store = VectorStore(tmp_path / "vectors")
    store.set_document_index_state("missing-doc", "READY")
    store.set_version_index_state("missing-doc", "missing-version", "READY")
    store.delete_version("missing-doc", "missing-version")
    assert store.count() == 0


def test_vector_store_reconcile_empty_index_reports_consistent_state(tmp_path: Path):
    store = VectorStore(tmp_path / "vectors")
    result = store.reconcile_index("missing-doc")
    assert result["valid"] is True
    assert result["removed"] == 0
    assert result["count"] == 0
    assert result["document_id"] == "missing-doc"


def test_settings_from_env_reads_project_scoped_paths(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("INCOMING_DIR", "incoming")
    monkeypatch.setenv("PROCESSED_DIR", "processed")
    monkeypatch.setenv("VECTOR_DB_DIR", "vectors")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "24")
    monkeypatch.setenv("EMBEDDING_RETRIES", "2")
    monkeypatch.setenv("EMBEDDING_TIMEOUT_SECONDS", "90")
    settings = Settings.from_env()
    assert settings.incoming_dir == (tmp_path / "incoming").resolve()
    assert settings.processed_dir == (tmp_path / "processed").resolve()
    assert settings.vector_db_dir == (tmp_path / "vectors").resolve()
    assert settings.embedding_batch_size == 24
    assert settings.embedding_retries == 2
    assert settings.embedding_timeout_seconds == 90.0


def test_settings_from_env_invalid_device_falls_back(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("DEVICE_MODE", "not-a-real-profile")
    settings = Settings.from_env()
    assert settings.device_mode == "i5_16gb"


def test_settings_override_from_dict_reports_unknown_keys(tmp_path: Path):
    settings = Settings(project_root=tmp_path)
    warnings = settings.override_from_dict(
        {"embedding_batch_size": 20, "does_not_exist": 123}
    )
    assert settings.embedding_batch_size == 20
    assert warnings == ["Unknown setting 'does_not_exist'; skipped."]
