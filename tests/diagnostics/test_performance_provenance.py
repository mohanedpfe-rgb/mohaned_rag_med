from __future__ import annotations

from rag_project.testing.production_benchmark_probes import phase14_production_benchmark
from rag_project.testing.runner import PHASES


def test_phase14_records_benchmark_and_baseline_provenance() -> None:
    result = phase14_production_benchmark(PHASES[13])
    assert result.status == "PASS", result.failures
    assert result.details["evidence_level"] == "real_pdf_to_retrieval_benchmark"
    assert result.details["baseline_type"] == "certification_ceiling"
    assert result.details["historical_baseline_available"] is False
    assert result.details["benchmark_environment"]["git_head_sha"]
    assert result.details["benchmark_environment"]["python_version"]
    assert result.details["benchmark_environment"]["requirements_lock_sha256"]
    assert result.details["regression_pass"] is True
