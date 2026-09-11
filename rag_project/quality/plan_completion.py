"""Fail-closed completion audit for the two requested project plans."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
    (9, "conversation"), (10, "security_privacy"), (11, "resilience"), (12, "production_contracts"), (13, "end_to_end"),
]]


class TwoPlanCompletionAudit:
    def __init__(self, project_root: str | Path):
        self.root = Path(project_root).resolve()

    def _exists(self, rel: str) -> bool:
        return (self.root / rel).exists()

    def _pdf_fixture_generator_ready(self) -> bool:
        conftest = self.root / "tests/high_level/conftest.py"
        if not conftest.exists():
            return False
        text = conftest.read_text(encoding="utf-8")
        required = ("def write_minimal_pdf", "def write_scanned_pdf", "def write_empty_pdf", "def write_large_pdf")
        return all(x in text for x in required)

    def _evidence_harnesses_ready(self) -> bool:
        required = (
            "scripts/run_medevidence_load_test.py",
            "scripts/run_memory_stability.py",
            "scripts/validate_backup_restore.py",
        )
        return all(self._exists(path) for path in required)

    def evaluate(self) -> dict[str, Any]:
        checks: list[CompletionCheck] = []
        for path in HIGH_LEVEL_DIRS:
            checks.append(CompletionCheck(f"high_level_dir:{path}", self._exists(path), "present" if self._exists(path) else "missing"))
        checks.extend([
            CompletionCheck("high_level_conftest", self._exists("tests/high_level/conftest.py"), "present" if self._exists("tests/high_level/conftest.py") else "missing"),
            CompletionCheck("high_level_helpers", self._exists("tests/high_level/helpers.py"), "present" if self._exists("tests/high_level/helpers.py") else "missing"),
            CompletionCheck("high_level_gold_set", self._exists("tests/support/gold_sets/core.jsonl"), "present" if self._exists("tests/support/gold_sets/core.jsonl") else "missing"),
            CompletionCheck("controlled_pdf_generator", self._pdf_fixture_generator_ready(), "text/scanned/empty/large deterministic fixtures present" if self._pdf_fixture_generator_ready() else "controlled fixture generators incomplete"),
            CompletionCheck("test_inventory_checker", self._exists("scripts/validate_test_inventory.py"), "present" if self._exists("scripts/validate_test_inventory.py") else "missing"),
            CompletionCheck("evidence_harnesses", self._evidence_harnesses_ready(), "load/memory/backup evidence runners present" if self._evidence_harnesses_ready() else "one or more evidence runners missing"),
        ])

        semantic = self.root / "rag_project/intelligence/semantic_cache.py"
        semantic_text = semantic.read_text(encoding="utf-8") if semantic.exists() else ""
        semantic_contract = all(x in semantic_text for x in (
            "similarity_threshold: float = 0.95", "7 * 24 * 60 * 60", "max_entries: int = 10_000", "expected_dimension: int = 768", "embed_query",
        ))
        checks.extend([
            CompletionCheck("semantic_cache_implementation", semantic_contract, "embedding cosine cache contract present" if semantic_contract else "semantic cache contract incomplete"),
            CompletionCheck("semantic_cache_tests", self._exists("tests/test_semantic_retrieval_cache.py"), "present" if self._exists("tests/test_semantic_retrieval_cache.py") else "missing"),
            CompletionCheck("retraining_implementation", self._exists("rag_project/intelligence/retraining_pipeline.py"), "present" if self._exists("rag_project/intelligence/retraining_pipeline.py") else "missing"),
            CompletionCheck("retraining_executable_manager", self._exists("tests/test_retraining_pipeline.py"), "present" if self._exists("tests/test_retraining_pipeline.py") else "missing"),
        ])

        required_data = {
            "medical_kb": "data/medical_knowledge.sqlite3",
            "clinical_corpus_manifest": "data/med_evidence_corpus/manifest.json",
            "intent_dataset": "data/evaluation/intent_dataset.jsonl",
            "complexity_dataset": "data/evaluation/complexity_dataset.jsonl",
            "harm_dataset": "data/evaluation/harm_dataset.jsonl",
            "ner_dataset": "data/evaluation/ner_dataset.jsonl",
        }
        for name, rel in required_data.items():
            checks.append(CompletionCheck(name, self._exists(rel), "present" if self._exists(rel) else "missing; required external/licensed data is not supplied"))

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
