from __future__ import annotations

import ast
from pathlib import Path

from rag_project.intelligence.semantic_cache import SemanticRetrievalCache
from rag_project.retrieval.hybrid_retriever import RetrievalHit

ROOT = Path(__file__).resolve().parents[1]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_runtime_composition_has_no_behavioral_patch_stack() -> None:
    source = _source("rag_project/runtime.py")
    tree = ast.parse(source)
    forbidden = (
        "runtime_stability_v8",
        "runtime_final_contracts_v8",
        "runtime_deep_contract_fix",
        "runtime_chroma_lifecycle_fix",
    )
    assert not any(name in source for name in forbidden)
    installer_tuple = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Tuple)
        and any(isinstance(item, ast.Name) and item.id == "vector_store" for item in node.value.elts)
    )
    assert len(installer_tuple.value.elts) == 1


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
    assert "cache_namespace" in source
    assert "_identity_namespace" in source


def test_semantic_cache_isolated_by_embedding_identity(tmp_path: Path) -> None:
    embed = lambda _query: [1.0, 0.0, 0.0]
    hit = RetrievalHit("doc-1", "evidence", {"document_id": "doc-1"}, 0.9, 0.9, 0.0)
    cache_a = SemanticRetrievalCache(tmp_path / "cache.sqlite3", embed_query=embed, expected_dimension=3, cache_namespace="model-a")
    cache_b = SemanticRetrievalCache(tmp_path / "cache.sqlite3", embed_query=embed, expected_dimension=3, cache_namespace="model-b")

    assert cache_a.put("same question", [hit]) is True
    assert cache_a.get("same question") is not None
    assert cache_b.get("same question") is None


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
