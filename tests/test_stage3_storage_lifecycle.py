from __future__ import annotations

import importlib
from pathlib import Path


def test_storage_runtime_lifecycle_is_idempotent_and_closes_store():
    runtime = importlib.import_module("rag_project.runtime")
    lifecycle = importlib.import_module("rag_project.runtime_chroma_lifecycle_fix")

    assert callable(runtime.install)
    assert callable(lifecycle.install)


def test_cross_process_lock_is_index_scoped(tmp_path: Path):
    concurrency = importlib.import_module("rag_project.storage.concurrency_index_fix")

    class Store:
        persist_directory = str(tmp_path / "index-a")

    first = concurrency._database_lock(Store())
    second = concurrency._database_lock(Store())
    assert first is second

    class OtherStore:
        persist_directory = str(tmp_path / "index-b")

    assert concurrency._database_lock(OtherStore()) is not first
