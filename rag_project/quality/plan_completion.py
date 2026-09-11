"""Fail-closed completion audit for the two requested project plans.

This module distinguishes implementation from evidence. A feature is not marked
complete merely because a module or placeholder exists: required datasets,
measured coverage/benchmarks, and clinical sign-off artifacts must exist too.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CompletionCheck:
    name: str
    passed: bool
    detail: str


HIGH_LEVEL_DIRS = [f"tests/high_level/{i:02d}_{name}" for i, name in [
    (1, "ingestion"), (2, "retrieval"), (3, "query_intelligence"), (4, "answer_paths"),
    (5, "grounding_safety"), (6, "latency_performance"), (7, "multilingual"), (8, "storage_index"),
    (9, "conversation"), (10, "security_privacy"), (11, "resilience"), (12, "production_contracts"), (13, "end_to_end")
]]
CONTROLLED_FIXTURES = [
    "clean_diabetes_en.pdf", "clean_diabetes_fr.pdf", "clean_diabetes_ar.pdf", "numeric_doses.pdf",
    "scanned_mixed.pdf", "adversarial_injection.pdf", "large_100pages.pdf", "empty_or_low_content.pdf",
]


class TwoPlanCompletionAudit:
    def __init__(self, project_root: str | Path):
        self.root = Path(project_root).resolve()

    def _exists(self, rel: str) -> bool:
        return (self.root / rel).exists()

    def evaluate(self) -> dict[str, Any]:
        checks: list[CompletionCheck] = []
        checks.extend(CompletionCheck(f"high_level_dir:{p}", self._exists(p), "present" if self._exists(p) else "missing") for p in HIGH_LEVEL_DIRS)
        checks.append(CompletionCheck("high_level_conftest", self._exists("tests/high_level/conftest.py"), "present" if self._exists("tests/high_level/conftest.py") else "missing"))
        checks.append(CompletionCheck("high_level_helpers", self._exists("tests/high_level/helpers.py"), "present" if self._exists("tests/high_level/helpers.py") else "missing"))
        checks.append(CompletionCheck("high_level_gold_set", self._exists("tests/support/gold_sets/core.jsonl"), "present" if self._exists("tests/support/gold_sets/core.jsonl") else "missing"))
        missing_fixtures = [p for p in CONTROLLED_FIXTURES if not self._exists(f"tests/support/pdfs/{p}")]
        checks.append(CompletionCheck("controlled_pdf_fixtures", not missing_fixtures, "all 8 present" if not missing_fixtures else f"missing={missing_fixtures}"))

        semantic = self.root / "rag_project" / "intelligence" / "semantic_cache.py"
        semantic_text = semantic.read_text(encoding="utf-8") if semantic.exists() else ""
        checks.append(CompletionCheck("semantic_cache_implementation", all(x in semantic_text for x in ("similarity_threshold: float = 0.95", "7 * 24 * 60 * 60", "max_entries: int = 10_000", "expected_dimension: int = 768")), "contract constants present" if semantic_text else "missing"))

        required_data = {
            "medical_kb": "data/medical_knowledge.sqlite3",
            "clinical_corpus_manifest": "data/med_evidence_corpus/manifest.json",
            "intent_dataset": "data/evaluation/intent_dataset.jsonl",
            "complexity_dataset": "data/evaluation/complexity_dataset.jsonl",
            "harm_dataset": "data/evaluation/harm_dataset.jsonl",
            "ner_dataset": "data/evaluation/ner_dataset.jsonl",
        }
        for name, rel in required_data.items():
            checks.append(CompletionCheck(name, self._exists(rel), "present" if self._exists(rel) else "missing; external/licensed data still required"))

        evidence_paths = {
            "coverage_report": "artifacts/coverage.xml",
            "kpi_benchmark": "artifacts/med_evidence_kpi.json",
            "load_benchmark": "artifacts/load_test_1000qpm.json",
            "memory_24h": "artifacts/memory_24h.json",
            "clinical_signoff": "artifacts/clinical_validation_signoff.json",
            "backup_restore_validation": "artifacts/backup_restore_validation.json",
        }
        for name, rel in evidence_paths.items():
            checks.append(CompletionCheck(name, self._exists(rel), "validated artifact present" if self._exists(rel) else "measured/sign-off artifact missing"))

        checks.append(CompletionCheck("plan_completion_manifest", self._exists("docs/PLAN_COMPLETION_MATRIX.md"), "present" if self._exists("docs/PLAN_COMPLETION_MATRIX.md") else "missing"))
        passed = all(c.passed for c in checks)
        return {"ready": passed, "checks": [asdict(c) for c in checks], "failed": [asdict(c) for c in checks if not c.passed]}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Fail-closed audit for the MedEvidence Pro and High-Level Test Suite plans")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args()
    report = TwoPlanCompletionAudit(args.root).evaluate()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ready"] else 2


__all__ = ["TwoPlanCompletionAudit", "CompletionCheck"]

if __name__ == "__main__":
    raise SystemExit(main())
