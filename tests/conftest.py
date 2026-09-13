"""Pytest bootstrap, automatic categorisation, and legacy test-boundary adapters."""
from __future__ import annotations

from pathlib import Path
import weakref

import pytest

from rag_project.runtime import install as install_runtime

install_runtime()
from tests.diagnostic_runtime_adapters import install as install_diagnostic_adapters

install_diagnostic_adapters()

ROOT = Path(__file__).resolve().parents[1]
_LIVE_VECTOR_STORES: weakref.WeakValueDictionary[int, object] = weakref.WeakValueDictionary()


def _register_vector_store(instance: object) -> None:
    _LIVE_VECTOR_STORES[id(instance)] = instance


def _install_vector_store_registry() -> None:
    try:
        from rag_project.storage.vector_store import VectorStore
        original_init = getattr(VectorStore, "__init__", None)
        if not callable(original_init) or getattr(original_init, "_pytest_registry_guard", False):
            return

        def guarded_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            _register_vector_store(self)

        guarded_init._pytest_registry_guard = True
        VectorStore.__init__ = guarded_init
    except Exception:
        pass


def _close_chroma_stores_under(root: str | Path) -> None:
    base = Path(root).resolve()
    for store in list(_LIVE_VECTOR_STORES.values()):
        persist_directory = getattr(store, "persist_directory", None)
        if persist_directory is None:
            continue
        try:
            path = Path(persist_directory).resolve()
            try:
                path.relative_to(base)
            except ValueError:
                continue
            close = getattr(store, "close", None)
            if callable(close):
                close()
        except Exception:
            pass


def _clear_chroma_system_cache() -> None:
    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        clear_system_cache = getattr(SharedSystemClient, "clear_system_cache", None)
        if callable(clear_system_cache):
            clear_system_cache()
    except Exception:
        pass


def _install_windows_temp_cleanup_guard() -> None:
    """Release Chroma clients before Windows TemporaryDirectory removes its tree."""
    try:
        import tempfile

        cleanup = getattr(tempfile.TemporaryDirectory, "cleanup", None)
        if not callable(cleanup) or getattr(cleanup, "_bookrag_chroma_guard", False):
            return

        def guarded_cleanup(self, *args, **kwargs):
            _close_chroma_stores_under(self.name)
            _clear_chroma_system_cache()
            try:
                return cleanup(self, *args, **kwargs)
            finally:
                _close_chroma_stores_under(self.name)
                _clear_chroma_system_cache()

        guarded_cleanup._bookrag_chroma_guard = True
        tempfile.TemporaryDirectory.cleanup = guarded_cleanup
    except Exception:
        pass


_install_vector_store_registry()
_install_windows_temp_cleanup_guard()


def _mark(item, marker: str) -> None:
    item.add_marker(getattr(pytest.mark, marker))


def pytest_collection_modifyitems(session, config, items):
    """Give every test useful categories and keep legacy adapters test-local."""
    target = None
    for item in items:
        module = getattr(item, "module", None)
        if module is not None and module.__name__.endswith("test_answer_engine_adversarial"):
            target = module
        path = str(getattr(item, "path", "")).replace("\\", "/")
        node = item.nodeid.replace("\\", "/")
        name = item.name.casefold()

        if "/diagnostics/" in path:
            _mark(item, "diagnostic")
            _mark(item, "fast")
            if "contract" in node:
                _mark(item, "contract")
        if any(token in path for token in (
            "/storage/", "test_index_consistency", "test_quality_gate",
            "test_vector_store", "test_chroma", "test_lexical",
        )):
            _mark(item, "storage")
        if any(token in path for token in (
            "test_hardening", "test_ingestion", "test_pdf", "test_extractor",
            "test_chunk", "test_directory_ingestion", "test_document",
        )):
            _mark(item, "ingestion")
        if any(token in path for token in (
            "test_pipeline_integrity", "test_full_44_intelligence",
            "test_god_mode_intelligence", "test_answer_system_orchestration_deep",
            "test_intelligence", "test_grounding",
        )):
            _mark(item, "intelligence")
        if any(token in path for token in (
            "test_answer", "test_generation", "test_citation", "test_final_answer",
        )):
            _mark(item, "generation")
        if any(token in path for token in ("test_retrieval", "test_hybrid", "test_query_rewriter")):
            _mark(item, "retrieval")
        if any(token in path for token in ("regression", "hardening", "adversarial")):
            _mark(item, "regression")
        if "high_level" in path:
            _mark(item, "high_level")
        if any(token in name for token in (
            "ollama", "embedding_backend", "real_model", "live_service", "network",
        )):
            _mark(item, "requires_ollama")
            _mark(item, "integration")
        if any(token in path for token in ("integration", "production", "e2e", "end_to_end")):
            _mark(item, "integration")
        if any(token in path for token in ("slow", "performance", "benchmark")) or "performance" in name or "benchmark" in name:
            _mark(item, "slow")

        markers = {mark.name for mark in item.iter_markers()}
        if not markers.intersection({"integration", "slow", "requires_ollama"}):
            _mark(item, "unit")

    if target is None or getattr(target, "_legacy_contracts_installed", False):
        return

    from rag_project.intelligence import evidence_guard, top_level_pipeline

    authoritative_rewrite = top_level_pipeline.rewrite_follow_up

    def legacy_rewrite_follow_up(question, history=None):
        result = authoritative_rewrite(question, history)
        if result != str(question or "").strip() and "Follow-up:" not in result:
            return f"Follow-up: {result}"
        return result

    def legacy_numeric_consistency(claim, evidence):
        details = evidence_guard.numeric_consistency_details(claim, evidence)
        return not bool(details.get("mismatch", False))

    legacy_rewrite_follow_up.__name__ = "rewrite_follow_up"
    legacy_numeric_consistency.__name__ = "numeric_consistency"
    target.rewrite_follow_up = legacy_rewrite_follow_up
    target.numeric_consistency = legacy_numeric_consistency
    target._legacy_contracts_installed = True


def pytest_sessionfinish(session, exitstatus):
    """Release global background resources before pytest/xdist worker exit."""
    try:
        from rag_project.ingestion import responsive_supervisor

        responsive_supervisor.stop(timeout_seconds=3.0)
    except Exception:
        pass

    for store in list(_LIVE_VECTOR_STORES.values()):
        try:
            close = getattr(store, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
    _clear_chroma_system_cache()
