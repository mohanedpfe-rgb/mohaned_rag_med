"""Fault-sensitive authoritative implementation for diagnostic Phase 6.

Phase 6 must prove that its information-loss detector notices deliberate corruption in
multiple production representations, not merely report that a healthy fixture looks intact.
"""
from __future__ import annotations

from typing import Any

from rag_project.testing.deep_diagnostics import PhaseResult


def strict_information_loss(phase: Any) -> PhaseResult:
    from rag_project.testing import advanced_phases
    from rag_project.storage.vector_store import VectorStore

    baseline_fn = advanced_phases.information_loss
    baseline = baseline_fn(phase)
    result = PhaseResult(phase.number, phase.key, phase.name, status="FAIL", started_at=baseline.started_at)

    probes: list[dict[str, Any]] = []

    original_add = VectorStore.add_documents

    def run_probe(name: str, mutator):
        def corrupted(self, documents, metadatas, embeddings, ids, *args, **kwargs):
            mutated_documents = [str(value) for value in documents]
            mutated_metadatas = [dict(value or {}) for value in metadatas]
            mutator(mutated_documents, mutated_metadatas)
            return original_add(self, mutated_documents, mutated_metadatas, embeddings, ids, *args, **kwargs)

        VectorStore.add_documents = corrupted
        try:
            observed = baseline_fn(phase)
        finally:
            VectorStore.add_documents = original_add
        probes.append(
            {
                "probe": name,
                "detector_status": observed.status,
                "detector_failed": bool(observed.status == "FAIL" and observed.failures),
                "detector_failures": observed.failures[:3],
                "measurements": (observed.details or {}).get("measurements", {}),
            }
        )

    try:
        def drop_structure(documents, metadatas):
            if metadatas:
                metadatas[0].pop("section_id", None)
                metadatas[0].pop("parent_id", None)

        def erase_document_text(documents, metadatas):
            if documents:
                documents[0] = ""

        run_probe("drop_structural_identity", drop_structure)
        run_probe("erase_vector_document_text", erase_document_text)

        probes_pass = all(item["detector_failed"] for item in probes)
        baseline_pass = baseline.status == "PASS" and not baseline.failures
        result.details = {
            **(baseline.details or {}),
            "authoritative_implementation": "strict_information_loss",
            "baseline_healthy_fixture_pass": baseline_pass,
            "fault_probes": probes,
            "fault_sensitivity_verified": probes_pass,
            "fault_probe_count": len(probes),
            "evidence_level": "fault_sensitive_production_information_loss",
            "evidence_level_extended": "real_representation_chain_plus_controlled_metadata_and_text_corruption",
        }
        result.score = 1.0 if baseline_pass and probes_pass else 0.0
        result.status = "PASS" if result.score == 1.0 else "FAIL"
        if result.status == "FAIL":
            result.failures.append(
                {
                    "location": "phase 6 fault-sensitive representation contract",
                    "exception": "InformationLossFaultSensitivityFailure",
                    "message": str(result.details),
                }
            )
    except Exception as exc:
        result.status = "FAIL"
        result.score = 0.0
        result.failures.append(
            {
                "location": "phase 6 fault-sensitive production probe",
                "exception": type(exc).__name__,
                "message": str(exc),
            }
        )
    result.duration_s = baseline.duration_s
    return result


__all__ = ["strict_information_loss"]
