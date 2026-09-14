from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_runtime_composition_has_no_behavioral_patch_stack() -> None:
    tree = ast.parse(_source("rag_project/runtime.py"))
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert "runtime_stability_v8" not in names
    assert "runtime_final_contracts_v8" not in names
    assert "runtime_deep_contract_fix" not in names
    assert "runtime_chroma_lifecycle_fix" not in names
    assert len([node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "install"]) <= 2


def test_semantic_cache_has_no_process_global_system_binding() -> None:
    source = _source("rag_project/intelligence/semantic_cache.py")
    tree = ast.parse(source)
    assert "_CURRENT_SYSTEM" not in source
    assert not any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_CURRENT_SYSTEM" for target in node.targets)
        for node in ast.walk(tree)
    )
    assert "med_evidence_pro.SemanticCache" not in source


def test_request_context_is_created_at_the_application_answer_boundary() -> None:
    source = _source("rag_project/application_answer_service.py")
    assert "build_request_context(" in source
    assert "execute_canonical_answer(" in source
    assert "install_semantic_cache" not in source


def test_publication_state_machine_is_explicit() -> None:
    source = _source("rag_project/ingestion/publication_coordinator.py")
    for state in ("BUILDING", "VALIDATED", "PUBLISHED", "RETIRED", "FAILED"):
        assert state in source
    assert "PublicationTransaction" in source
