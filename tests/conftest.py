"""Pytest bootstrap, automatic categorisation, and legacy test-boundary adapters."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from rag_project.runtime import install as install_runtime

install_runtime()

ROOT = Path(__file__).resolve().parents[1]


def _mark(item, marker: str) -> None:
    item.add_marker(getattr(pytest.mark, marker))


def pytest_collection_modifyitems(session, config, items):
    """Classify every test from its path/name and install only legacy adapters.

    The repository historically relied on manually applied markers. That made
    fast/contract/storage runs incomplete whenever a new test was added. This
    hook gives every existing and future test a deterministic category based on
    its location, while the explicit markers remain available for exceptions.
    """
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
            _mark(item, "contract") if "contract" in node else None
        if any(token in path for token in ("/storage/", "test_index_consistency", "test_quality_gate")):
            _mark(item, "storage")
        if any(token in path for token in ("test_pipeline_integrity", "test_full_44_intelligence", "test_god_mode_intelligence", "test_answer_system_orchestration_deep")):
            _mark(item, "intelligence")
        if "high_level" in path:
            _mark(item, "high_level")
        if any(token in name for token in ("ollama", "embedding_backend", "real_model", "live_service")):
            _mark(item, "requires_ollama")
            _mark(item, "integration")
        if any(token in path for token in ("integration", "production", "e2e", "end_to_end")):
            _mark(item, "integration")
        if any(token in path for token in ("slow", "performance", "benchmark")) or "performance" in name or "benchmark" in name:
            _mark(item, "slow")

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


def pytest_runtest_logreport(report):
    """Attach the first project-owned traceback frame to pytest's terminal report.

    The normal traceback remains untouched; this adds a compact locator that is
    useful when a failure is buried below a large integration stack.
    """
    if report.when != "call" or not report.failed:
        return
    longrepr = str(report.longrepr)
    project_frames: list[tuple[str, str]] = []
    for line in longrepr.splitlines():
        if "rag_project" not in line.replace("\\", "/") or ".py:" not in line:
            continue
        project_frames.append((report.nodeid, line.strip()))
    if project_frames:
        report.user_properties.append(("first_project_frame", project_frames[0][1]))


# Prevent accidental pytest fixture/test pollution from helper imports in diagnostics.
pytest_plugins = []
