from __future__ import annotations

from rag_project.testing.runner import PHASES, _hardened_mutation_phase
from rag_project.testing.robust_probes import retrieval_microscope
from rag_project.testing.production_document_probes import phase7_production_pdf_lab


def test_authoritative_phase7_uses_production_document_probe() -> None:
    assert PHASES[6].number == 7
    result = phase7_production_pdf_lab(PHASES[6])
    assert result.details["real_scanned_pdf"] is True
    assert result.details["real_ocr_attempted"] is True
    assert result.details["checks"]["scanned_page_detected"] is True
    assert result.details["checks"]["ocr_branch_reached"] is True
    assert result.details["checks"]["malformed_pdf_rejected"] is True


def test_phase9_uses_independent_gold_labels() -> None:
    result = retrieval_microscope(PHASES[8])
    assert result.details["gold_independent_of_corpus_text"] is True
    assert result.details["relevance_derived_from_fixture_text"] is False
    assert result.details["gold_cases"] >= 3
    assert result.details["metadata_filter_correct"] is True


def test_phase11_has_eight_executable_mutants() -> None:
    result = _hardened_mutation_phase(PHASES[10])
    assert result.status == "PASS", result.failures
    assert result.details["mutants_applicable"] == 8
    assert result.details["mutants_killed"] == 8
    assert result.details["kill_score"] == 1.0
    assert result.details["real_pytest_subprocess"] is True
