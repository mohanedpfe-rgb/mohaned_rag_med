from __future__ import annotations

from rag_project.testing.production_answer_probes import phase10_canonical_answer_engine
from rag_project.testing.runner import PHASES


def test_phase_ten_executes_canonical_med_evidence_engine() -> None:
    result = phase10_canonical_answer_engine(PHASES[9])
    assert result.status == "PASS", result.failures
    assert result.details["evidence_level"] == "canonical_med_evidence_pro_engine"
    assert result.details["production_entrypoint"].endswith("MedEvidenceProEngine.answer")
    assert result.details["canonical_engine_executed"] is True
    assert result.details["answer_generated"] is True
    assert result.details["citations_present"] is True
    assert result.details["citation_ids_valid"] is True
    assert result.details["verification_allow"] is True
