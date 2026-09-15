from __future__ import annotations

import importlib


def test_storage_package_import_has_no_runtime_install_side_effect(monkeypatch):
    import rag_project.storage

    called: list[str] = []

    monkeypatch.setattr("rag_project.storage.vector_store_runtime.install", lambda: called.append("vector_store"))
    monkeypatch.setattr("rag_project.runtime_chroma_metadata_fix.install", lambda: called.append("metadata"))
    monkeypatch.setattr("rag_project.storage.atomic_index_transaction.install", lambda: called.append("atomic"))
    monkeypatch.setattr("rag_project.storage.concurrency_index_fix.install", lambda: called.append("concurrency"))
    monkeypatch.setattr("rag_project.storage.reconcile_parity_fix.install", lambda: called.append("reconcile"))

    importlib.reload(rag_project.storage)

    assert called == []


def test_runtime_loader_owns_all_hardening_installers():
    runtime = importlib.import_module("rag_project.runtime")
    installers = runtime._load_installers()
    modules = {getattr(installer, "__module__", "") for installer in installers}

    expected = {
        "rag_project.runtime_chroma_lifecycle_fix",
        "rag_project.runtime_chroma_metadata_fix",
        "rag_project.storage.vector_store_runtime",
        "rag_project.storage.atomic_index_transaction",
        "rag_project.storage.concurrency_index_fix",
        "rag_project.storage.reconcile_parity_fix",
        "rag_project.runtime_post_index_publication_contract_v2",
    }
    assert modules == expected
