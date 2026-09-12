"""Fault-sensitive authoritative implementations for diagnostic Phases 3-5."""
from __future__ import annotations

from typing import Any

from rag_project.testing.deep_diagnostics import PhaseResult


def _wrap_result(base: PhaseResult, name: str, probes: list[dict[str, Any]], evidence: str) -> PhaseResult:
    base.details = {
        **(base.details or {}),
        "authoritative_implementation": name,
        "fault_probes": probes,
        "fault_sensitivity_verified": all(bool(p.get("detector_failed")) for p in probes),
        "fault_probe_count": len(probes),
        "evidence_level": evidence,
    }
    base.status = "PASS" if base.status == "PASS" and base.details["fault_sensitivity_verified"] else "FAIL"
    base.score = 1.0 if base.status == "PASS" else 0.0
    if base.status == "FAIL":
        base.failures.append({"location": name, "exception": "FaultSensitivityFailure", "message": str(base.details)})
    return base


def strict_diagnostic_chain(phase: Any) -> PhaseResult:
    from rag_project.testing import advanced_phases
    from rag_project.chunking.semantic_chunker import SemanticChunker

    baseline = advanced_phases.diagnostic_chain(phase)
    original = SemanticChunker.chunk_pages

    def broken(self, pages, *args, **kwargs):
        return []

    SemanticChunker.chunk_pages = broken
    try:
        observed = advanced_phases.diagnostic_chain(phase)
    finally:
        SemanticChunker.chunk_pages = original
    return _wrap_result(
        baseline,
        "strict_diagnostic_chain",
        [{"probe": "empty_production_chunk_output", "detector_failed": observed.status == "FAIL", "observed_failures": observed.failures[:3]}],
        "fault_sensitive_page_to_chunk_to_context_chain",
    )


def strict_contract_triangulation(phase: Any) -> PhaseResult:
    from rag_project.testing import advanced_phases
    from rag_project.retrieval.context_builder import ContextBuilder

    baseline = advanced_phases.contract_triangulation(phase)
    original = ContextBuilder.build

    def broken(self, hits, *args, **kwargs):
        return "", []

    ContextBuilder.build = broken
    try:
        observed = advanced_phases.contract_triangulation(phase)
    finally:
        ContextBuilder.build = original
    return _wrap_result(
        baseline,
        "strict_contract_triangulation",
        [{"probe": "erase_context_builder_output", "detector_failed": observed.status == "FAIL", "observed_failures": observed.failures[:3]}],
        "fault_sensitive_input_transform_output_contracts",
    )


def strict_cross_layer_invariants(phase: Any) -> PhaseResult:
    from rag_project.testing import advanced_phases
    from rag_project.storage.vector_store import VectorStore

    baseline = advanced_phases.cross_layer_invariants(phase)
    original = VectorStore.add_documents

    def broken(self, documents, metadatas, embeddings, ids, *args, **kwargs):
        mutated = [dict(value or {}) for value in metadatas]
        for meta in mutated:
            meta.pop("chunk_id", None)
        return original(self, documents, mutated, embeddings, ids, *args, **kwargs)

    VectorStore.add_documents = broken
    try:
        observed = advanced_phases.cross_layer_invariants(phase)
    finally:
        VectorStore.add_documents = original
    return _wrap_result(
        baseline,
        "strict_cross_layer_invariants",
        [{"probe": "drop_chunk_identity_before_storage", "detector_failed": observed.status == "FAIL", "observed_failures": observed.failures[:3]}],
        "fault_sensitive_page_chunk_storage_identity_conservation",
    )


__all__ = ["strict_diagnostic_chain", "strict_contract_triangulation", "strict_cross_layer_invariants"]
