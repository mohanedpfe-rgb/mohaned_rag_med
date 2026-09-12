from __future__ import annotations

from rag_project.testing.production_path_probes import phase16_production_ingestion_benchmark
from rag_project.testing.runner import PHASES


def test_phase_sixteen_executes_canonical_robust_ingestion() -> None:
    phase = PHASES[15]
    result = phase16_production_ingestion_benchmark(phase)
    assert result.status == "PASS", result.failures
    assert result.details["evidence_level"] == "real_production_robust_ingestion_to_storage_retrieval"
    assert result.details["production_entrypoint"].endswith("robust_ingest_file")
    assert result.details["durable_state_verified"] is True
    assert result.details["index_integrity_verified"] is True
    assert result.details["gold_labels_independent_of_corpus_text"] is True
    assert result.details["retrieval_recall"] >= 0.8
