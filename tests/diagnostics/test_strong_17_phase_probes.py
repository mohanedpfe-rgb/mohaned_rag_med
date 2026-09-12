from rag_project.testing.implementation_contracts import validate_runtime_ownership
from rag_project.testing.runner import PHASES
from rag_project.testing.full_mutation_probes import run_full_mutation_suite
from rag_project.testing.full_metamorphic_probes import run_full_metamorphic_suite
from rag_project.testing.production_retrieval_probes import phase9_independent_retrieval
from rag_project.testing.production_document_probes import phase7_production_pdf_lab
from rag_project.testing.strict_phase16_production import strict_phase16_production_ingestion_benchmark, validate_phase16_benchmark_data
from rag_project.testing.production_benchmark_probes import phase14_production_benchmark
from rag_project.testing.strict_runtime_contracts import strict_resource_stability


def test_strong_phase_wiring_is_authoritative() -> None:
    report = validate_runtime_ownership()
    assert report["pass"], report["failures"]
    rows = {row["phase"]: row for row in report["ownership_rows"]}
    expected = {
        3: "strict_diagnostic_chain",
        4: "strict_contract_triangulation",
        5: "strict_cross_layer_invariants",
        6: "strict_information_loss",
        8: "run_full_metamorphic_suite",
        11: "run_full_mutation_suite",
        13: "strict_causal_phase",
        15: "strict_resource_stability",
        16: "strict_phase16_production_ingestion_benchmark",
        17: "phase17_strict_completion",
    }
    expected_modules = {16: "rag_project.testing.strict_phase16_production", 17: "rag_project.testing.strict_phase17_final"}
    for phase, qualname in expected.items():
        assert rows[phase]["actual_qualname"] == qualname
        assert rows[phase]["status"] == "PASS"
    for phase, module in expected_modules.items():
        assert rows[phase]["actual_module"] == module


def test_phase_registry_remains_exactly_seventeen() -> None:
    assert [phase.number for phase in PHASES] == list(range(1, 18))


def test_phase7_production_document_contract_is_adversarial() -> None:
    result = phase7_production_pdf_lab(PHASES[6])
    assert result.status == "PASS", result.failures
    assert result.details["variant_count"] >= 5
    assert all(bool(v) for v in result.details["variant_results"].values())
    assert result.details["malformed_pdf_rejected"] is True


def test_phase8_authoritative_metamorphic_contract_is_end_to_end() -> None:
    result = run_full_metamorphic_suite(PHASES[7])
    assert result.status == "PASS", result.failures
    assert len(result.details["checks"]) >= 8
    assert all(bool(v) for v in result.details["checks"].values())
    assert result.details["end_to_end_answer_path_executed"] is True
    assert result.details["ollama_protocol_path_executed"] is True


def test_phase9_authoritative_retrieval_contract_uses_external_gold() -> None:
    result = phase9_independent_retrieval(PHASES[8])
    assert result.status == "PASS", result.failures
    assert result.details["gold_labels_independent_of_corpus_text"] is True
    assert result.details["metadata_filter_correct"] is True
    assert result.details["lexical_recall_at_3"] >= 0.8
    assert result.details["semantic_recall_at_3"] >= 0.8


def test_phase10_protocol_fixture_rejects_invalid_chat_requests() -> None:
    import requests
    from rag_project.testing.production_answer_probes import _LocalOllamaServer

    server = _LocalOllamaServer()
    try:
        response = requests.post(f"{server.base_url}/api/chat", json={"model": "wrong-model", "messages": [{"role": "user", "content": "hello"}], "stream": False}, timeout=5)
        assert response.status_code == 400
        valid_shape = requests.post(f"{server.base_url}/api/chat", json={"model": "diagnostic-protocol:latest", "messages": [{"role": "user", "content": "hello"}], "stream": False}, timeout=5)
        assert valid_shape.status_code == 200
    finally:
        server.close()


def test_phase11_authoritative_mutation_contract_is_multi_module() -> None:
    result = run_full_mutation_suite(PHASES[10])
    assert result.status == "PASS", result.failures
    assert result.details["mutants_applicable"] >= 12
    assert result.details["mutants_killed"] == result.details["mutants_applicable"]
    assert result.details["kill_score"] == 1.0
    assert result.details["target_module_count"] >= 3
    assert result.details["real_pytest_subprocess"] is True


def test_phase14_authoritative_benchmark_contract_has_regression_budget() -> None:
    result = phase14_production_benchmark(PHASES[13])
    assert result.status == "PASS", result.failures
    assert result.details["evidence_level"] == "real_pdf_to_retrieval_benchmark"
    assert result.details["minimum_samples_per_stage"] >= 7
    assert result.details["regression_pass"] is True


def test_phase15_authoritative_resource_contract_is_trend_aware() -> None:
    result = strict_resource_stability(PHASES[14])
    assert result.status == "PASS", result.failures
    assert result.details["evidence_level"] == "real_subprocess_resource_observation"
    assert result.details["sample_count"] >= 3
    assert result.details["rss_trend_ok"] is True
    assert result.details["fd_leak_ok"] is True


def test_phase16_benchmark_integrity_negative_controls() -> None:
    corpus = [{"doc_id": chr(97 + index), "text": "text"} for index in range(8)]
    gold = [{"id": f"case-{index}", "question": "question", "expected_doc_ids": ["missing"], "expected_evidence_terms": ["term"]} for index in range(8)]
    try:
        validate_phase16_benchmark_data(corpus, gold)
    except RuntimeError as exc:
        assert "missing corpus ids" in str(exc)
    else:
        raise AssertionError("Phase 16 must reject gold references to missing corpus documents")


def test_phase16_authoritative_ingestion_contract_is_independent_and_grounded() -> None:
    result = strict_phase16_production_ingestion_benchmark(PHASES[15])
    assert result.status == "PASS", result.failures
    assert result.details["evidence_level"] == "real_production_robust_ingestion_to_storage_retrieval"
    assert result.details["dataset_id"] == "phase16_production_independent_v2"
    assert result.details["gold_labels_independent_of_corpus_text"] is True
    assert result.details["gold_integrity_contract_verified"] is True
    assert result.details["gold_references_resolved"] is True
    assert result.details["independent_from_phase9_dataset"] is True
    assert result.details["retrieval_recall"] >= 0.8
    assert result.details["evidence_grounding_case_rate"] >= 0.8
    assert result.details["evidence_term_recall"] >= 0.8
