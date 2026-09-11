"""Strict pre-human-test certification gate for MedEvidence Pro.

The gate is fail-closed. Human testing cannot be certified from architecture
alone: the runtime contracts, safety layer, production operations, medical KB,
evaluation assets, CI/deployment contracts, and behavior-driven test suite
must all be present.
"""
from __future__ import annotations

import importlib
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REQUIRED_MODULES = (
    "rag_project.intelligence.med_evidence_pro",
    "rag_project.intelligence.production_ops",
    "rag_project.intelligence.production_ops_strict",
    "rag_project.intelligence.cloud_hybrid",
    "rag_project.knowledge.medical_kb",
    "rag_project.api.med_evidence_api",
    "rag_project.quality_gate",
)

KB_THRESHOLDS = {
    "drugs": 10_000,
    "interactions": 50_000,
    "guidelines": 100_000,
    "contraindications": 5_000,
    "disease_graph": 5_000,
}

REQUIRED_PATHS = (
    "requirements.in",
    "requirements.lock",
    ".github/workflows/ci.yml",
    ".github/workflows/high-level-nightly.yml",
    "ARCHITECTURE.md",
    "SECURITY.md",
    "docs/MEDEVIDENCE_PRO_IMPLEMENTATION_MAP.md",
    "docs/PRE_HUMAN_TEST_CERTIFICATION.md",
    "scripts/run_medevidence_benchmarks.py",
    "scripts/run_medevidence_load_test.py",
    "scripts/run_medevidence_ops.py",
    "scripts/run_medevidence_maintenance.py",
    "scripts/setup_medevidence_db.py",
    "scripts/validate_medevidence_kb.py",
    "scripts/run_medevidence_api.py",
    "tests/high_level/conftest.py",
    "tests/high_level/helpers.py",
    "tests/high_level/test_suite_structure.py",
    "tests/support/gold_sets/core.jsonl",
    "tests/support/pdfs/README.md",
    "tests/support/fixtures/README.md",
)

REQUIRED_HIGH_LEVEL_DIRS = (
    "01_ingestion",
    "02_retrieval",
    "03_query_intelligence",
    "04_answer_paths",
    "05_grounding_safety",
    "06_latency_performance",
    "07_multilingual",
    "08_storage_index",
    "09_conversation",
    "10_security_privacy",
    "11_resilience",
    "12_production_contracts",
    "13_end_to_end",
)


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ReadinessReport:
    ready: bool
    checks: tuple[GateCheck, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"ready": self.ready, "checks": [asdict(item) for item in self.checks]}


class HumanTestReadinessGate:
    def __init__(self, project_root: str | Path):
        self.root = Path(project_root).resolve()

    def _module_check(self) -> list[GateCheck]:
        checks: list[GateCheck] = []
        for module in REQUIRED_MODULES:
            try:
                importlib.import_module(module)
                checks.append(GateCheck(f"module:{module}", True, "importable"))
            except Exception as exc:
                checks.append(GateCheck(f"module:{module}", False, f"{type(exc).__name__}: {exc}"))
        return checks

    def _path_check(self) -> list[GateCheck]:
        checks: list[GateCheck] = []
        for path in REQUIRED_PATHS:
            target = self.root / path
            checks.append(GateCheck(f"path:{path}", target.exists(), "present" if target.exists() else "missing"))
        for directory in REQUIRED_HIGH_LEVEL_DIRS:
            target = self.root / "tests" / "high_level" / directory
            checks.append(GateCheck(f"high_level_dir:{directory}", target.is_dir(), "present" if target.is_dir() else "missing"))
        return checks

    def _kb_check(self) -> GateCheck:
        db_path = Path(os.getenv("MEDEVIDENCE_KB_PATH", str(self.root / "data" / "medical_knowledge.sqlite3")))
        if not db_path.exists():
            return GateCheck("medical_kb", False, f"database missing: {db_path}")
        try:
            from rag_project.knowledge.medical_kb import counts
            values = counts(db_path)
        except Exception as exc:
            return GateCheck("medical_kb", False, f"cannot inspect DB: {type(exc).__name__}: {exc}")
        missing = {name: (values.get(name, 0), minimum) for name, minimum in KB_THRESHOLDS.items() if values.get(name, 0) < minimum}
        if missing:
            return GateCheck("medical_kb", False, json.dumps({"counts": values, "below_minimum": missing}, sort_keys=True))
        return GateCheck("medical_kb", True, json.dumps(values, sort_keys=True))

    def _sqlite_integrity_check(self) -> GateCheck:
        candidates = [self.root / "data" / "medical_knowledge.sqlite3", self.root / "data" / "med_evidence_ops.sqlite3"]
        failures: list[str] = []
        checked = 0
        for db_path in candidates:
            if not db_path.exists():
                continue
            checked += 1
            try:
                with sqlite3.connect(db_path) as db:
                    result = str(db.execute("PRAGMA integrity_check").fetchone()[0])
                if result.lower() != "ok":
                    failures.append(f"{db_path}: {result}")
            except Exception as exc:
                failures.append(f"{db_path}: {type(exc).__name__}: {exc}")
        if failures:
            return GateCheck("sqlite_integrity", False, "; ".join(failures))
        return GateCheck("sqlite_integrity", checked > 0, f"checked={checked}")

    def _test_inventory_check(self) -> GateCheck:
        tests_dir = self.root / "tests"
        if not tests_dir.exists():
            return GateCheck("test_inventory", False, "tests directory missing")
        files = list(tests_dir.rglob("test_*.py"))
        test_functions = 0
        for path in files:
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            test_functions += sum(1 for line in text.splitlines() if line.lstrip().startswith("def test_"))
        high_level_files = list((tests_dir / "high_level").rglob("test_*.py")) if (tests_dir / "high_level").exists() else []
        high_level_functions = 0
        for path in high_level_files:
            try:
                high_level_functions += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.lstrip().startswith("def test_"))
            except OSError:
                continue
        passed = len(files) >= 40 and test_functions >= 150 and len(high_level_files) >= 30 and high_level_functions >= 50
        return GateCheck(
            "test_inventory",
            passed,
            f"files={len(files)}, test_functions={test_functions}, high_level_files={len(high_level_files)}, high_level_functions={high_level_functions}, floor=40/150 plus 30/50",
        )

    def evaluate(self) -> ReadinessReport:
        checks: list[GateCheck] = []
        checks.extend(self._module_check())
        checks.extend(self._path_check())
        checks.append(self._kb_check())
        checks.append(self._sqlite_integrity_check())
        checks.append(self._test_inventory_check())
        ci_path = self.root / ".github" / "workflows" / "ci.yml"
        checks.append(GateCheck("ci_configuration", ci_path.exists(), "CI workflow present" if ci_path.exists() else "CI workflow missing"))
        checks.append(GateCheck("local_first_default", os.getenv("MEDEVIDENCE_CLOUD_ENABLED", "false").casefold() != "true", "cloud disabled by default"))
        return ReadinessReport(all(item.passed for item in checks), tuple(checks))


def assert_human_test_ready(project_root: str | Path) -> ReadinessReport:
    report = HumanTestReadinessGate(project_root).evaluate()
    if not report.ready:
        failed = [item for item in report.checks if not item.passed]
        raise RuntimeError("human_test_not_ready: " + " | ".join(f"{x.name}: {x.detail}" for x in failed))
    return report


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Strict MedEvidence Pro pre-human-test certification gate")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = HumanTestReadinessGate(args.root).evaluate()
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    return 0 if report.ready else 2


__all__ = ["HumanTestReadinessGate", "ReadinessReport", "GateCheck", "assert_human_test_ready"]

if __name__ == "__main__":
    raise SystemExit(main())
