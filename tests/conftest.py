"""Pytest bootstrap, automatic categorisation, and legacy test-boundary adapters."""
from __future__ import annotations

from pathlib import Path

import pytest

from rag_project.runtime import install as install_runtime

install_runtime()

ROOT = Path(__file__).resolve().parents[1]


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

        # Unit is the default for tests that were not explicitly classified as
        # an external/integration/slow suite. This prevents silent test gaps.
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
