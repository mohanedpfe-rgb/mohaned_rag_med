from __future__ import annotations

from rag_project.application import _normalize_runtime_settings
from rag_project.configuration.settings import Settings


def test_normalize_runtime_settings_repairs_unsafe_local_profile(tmp_path):
    settings = Settings(project_root=tmp_path, embedding_batch_size=1, embedding_retries=0, embedding_timeout_seconds=1, max_workers=99, ollama_concurrency=99)
    normalized = _normalize_runtime_settings(settings)
    assert normalized.embedding_batch_size == 16
    assert normalized.embedding_retries == 1
    assert normalized.embedding_timeout_seconds == 30.0
    assert normalized.max_workers == 4
    assert normalized.ollama_concurrency == 2
