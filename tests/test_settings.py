from __future__ import annotations

from pathlib import Path

from rag_project.configuration.settings import Settings


def test_settings_clamp_unsafe_direct_values(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        chunk_size=20,
        chunk_overlap=999,
        top_k=0,
        temperature=99,
        vector_weight=-5,
        max_workers=0,
        ollama_concurrency=0,
        context_token_budget=1,
    )

    assert settings.chunk_size == 200
    assert 0 <= settings.chunk_overlap < settings.chunk_size
    assert settings.top_k == 1
    assert settings.temperature == 1.0
    assert settings.vector_weight == 0.0
    assert settings.max_workers == 1
    assert settings.ollama_concurrency == 1
    assert settings.context_token_budget == 256


def test_invalid_environment_numbers_fall_back(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("CHUNK_SIZE", "not-a-number")
    monkeypatch.setenv("TOP_K", "bad")
    monkeypatch.setenv("TEMPERATURE", "bad")
    monkeypatch.setenv("EMBEDDING_TIMEOUT_SECONDS", "bad")

    settings = Settings.from_env()

    assert settings.chunk_size == 600
    assert settings.top_k == 6
    assert settings.temperature == 0.2
    assert settings.embedding_timeout_seconds == 180.0


def test_unknown_device_mode_is_safe(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("DEVICE_MODE", "does-not-exist")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    settings = Settings.from_env()

    assert settings.device_mode == "i5_16gb"
    assert settings.embedding_model == "nomic-embed-text"


def test_universal_mode_defaults_to_safe_coverage(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    for name in ("OCR_ENABLED", "AUTO_OCR", "UNIVERSAL_PDF_MODE"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env()
    assert settings.ocr_enabled is True
    assert settings.auto_ocr is True
    assert settings.universal_pdf_mode is True
    assert settings.answer_verification_enabled is True
