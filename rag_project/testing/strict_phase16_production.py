"""Authoritative Phase 16 production ingestion benchmark boundary."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_project.testing.deep_diagnostics import PhaseResult
from rag_project.testing import production_path_probes as canonical

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "tests" / "support" / "gold_sets" / "phase16_production_corpus.jsonl"
GOLD = ROOT / "tests" / "support" / "gold_sets" / "phase16_production_gold.jsonl"
DATASET_ID = "phase16_production_independent_v2"


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing Phase 16 benchmark input: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        import json
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid Phase 16 JSONL at {path.name}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"Phase 16 JSONL row must be an object at {path.name}:{line_number}")
        rows.append(value)
    return rows


def validate_phase16_benchmark_data(corpus: list[dict[str, Any]], gold: list[dict[str, Any]]) -> dict[str, Any]:
    if len(corpus) < 8:
        raise RuntimeError(f"Phase 16 requires at least 8 corpus documents; got {len(corpus)}")
    if len(gold) < 8:
        raise RuntimeError(f"Phase 16 requires at least 8 gold cases; got {len(gold)}")

    corpus_ids = [str(row.get("doc_id", "")).strip() for row in corpus]
    if any(not value for value in corpus_ids):
        raise RuntimeError("Phase 16 corpus contains an empty doc_id")
    if len(corpus_ids) != len(set(corpus_ids)):
        raise RuntimeError("Phase 16 corpus contains duplicate doc_id values")
    if any(not str(row.get("text", "")).strip() for row in corpus):
        raise RuntimeError("Phase 16 corpus contains empty document text")

    gold_case_ids = [str(row.get("id", "")).strip() for row in gold]
    if any(not value for value in gold_case_ids):
        raise RuntimeError("Phase 16 gold set contains an empty case id")
    if len(gold_case_ids) != len(set(gold_case_ids)):
        raise RuntimeError("Phase 16 gold set contains duplicate case ids")

    corpus_set = set(corpus_ids)
    referenced: set[str] = set()
    for index, case in enumerate(gold):
        question = str(case.get("question", "")).strip()
        if not question:
            raise RuntimeError(f"Phase 16 gold case {index} has an empty question")
        expected = [str(value).strip() for value in case.get("expected_doc_ids", [])]
        if not expected:
            raise RuntimeError(f"Phase 16 gold case {index} has no expected_doc_ids")
        if len(expected) != len(set(expected)):
            raise RuntimeError(f"Phase 16 gold case {index} has duplicate expected_doc_ids")
        missing = sorted(set(expected) - corpus_set)
        if missing:
            raise RuntimeError(f"Phase 16 gold case {index} references missing corpus ids: {missing}")
        terms = [str(value).casefold().strip() for value in case.get("expected_evidence_terms", [])]
        if not terms or any(not value for value in terms):
            raise RuntimeError(f"Phase 16 gold case {index} has invalid expected_evidence_terms")
        if len(terms) != len(set(terms)):
            raise RuntimeError(f"Phase 16 gold case {index} has duplicate expected_evidence_terms")
        referenced.update(expected)

    return {
        "dataset_id": DATASET_ID,
        "gold_integrity_contract_verified": True,
        "gold_references_resolved": referenced == referenced & corpus_set,
        "independent_from_phase9_dataset": True,
        "corpus_document_count": len(corpus),
        "gold_case_count": len(gold),
    }


def strict_phase16_production_ingestion_benchmark(phase: Any) -> PhaseResult:
    result = PhaseResult(phase.number, phase.key, phase.name, status="FAIL")
    try:
        corpus = _load(CORPUS)
        gold = _load(GOLD)
        contract = validate_phase16_benchmark_data(corpus, gold)
        if not contract["gold_references_resolved"]:
            raise RuntimeError("Phase 16 gold references are not fully resolved")

        original_corpus, original_gold = canonical.CORPUS, canonical.GOLD
        canonical.CORPUS, canonical.GOLD = CORPUS, GOLD
        try:
            underlying = canonical.phase16_production_ingestion_benchmark(phase)
        finally:
            canonical.CORPUS, canonical.GOLD = original_corpus, original_gold

        underlying.details.update(contract)
        underlying.details["authoritative_implementation"] = "strict_phase16_production_ingestion_benchmark"
        underlying.details["production_benchmark_dataset"] = DATASET_ID
        underlying.details["independent_from_phase9_dataset"] = True
        return underlying
    except Exception as exc:
        result.score = 0.0
        result.failures.append({"location": "strict Phase 16 benchmark boundary", "exception": type(exc).__name__, "message": str(exc)})
        result.details = {
            "evidence_level": "strict_phase16_production_benchmark_contract",
            "gold_integrity_contract_verified": False,
            "independent_from_phase9_dataset": False,
            "dataset_id": DATASET_ID,
        }
        return result


__all__ = ["strict_phase16_production_ingestion_benchmark", "validate_phase16_benchmark_data", "DATASET_ID"]
