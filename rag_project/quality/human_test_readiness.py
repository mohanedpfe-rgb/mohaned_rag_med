"""Strict pre-human-test certification gate for MedEvidence Pro.

The gate is intentionally fail-closed. A human test run must not start until the
software architecture, safety controls, medical KB minimums, evaluation suite,
observability, backup/recovery and deployment contracts are all present.
"""
from __future__ import annotations

import importlib
import json
import os
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


REQUIRED_MODULES = (
    "rag_project.intelligence.med_evidence_pro",
    "rag_project.intelligence.production_ops",
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
    "ARCHITECTURE.md",
    "SECURITY.md",
    "docs/MEDEVIDENCE_PRO_IMPLEMENTATION_MAP.md",
    "scripts/run_medevidence_benchmarks.py",
    "scripts/run_medevidence_ops.py",
    "scripts/setup_medevidence_db.py",
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
        return {
            "ready": self.ready,
            "checks": [asdict(item) for item in self.checks],
        }


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
        return [
            GateCheck(f"path:{path}", (self.root / path).exists(), "present" if (self.root / path).exists() else "missing")
            for path in REQUIRED_PATHS
        ]

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
        candidates = [
            self.root / "data" / "medical_knowledge.sqlite3",
            self.root / "data" / "med_evidence_ops.sqlite3",
        ]
        failures = []
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
        files = list(tests_dir.glob("test_medevidence*.py")) + list(tests_dir.glob("test_*evidence*.py"))
        line_count = 0
        for path in files:
            try:
                line_count += len(path.read_text(encoding="utf-8").splitlines())
            except OSError:
                pass
        # This is a structural floor, not a claim of semantic coverage. CI remains authoritative.
        passed = len(files) >= 4 and line_count >= 500
        return GateCheck("test_inventory", passed, f"files={len(files)}, lines={line_count}, floor=4 files/500 lines")

    def evaluate(self) -> ReadinessReport:
        checks = []
        checks.extend(self._module_check())
        checks.extend(self._path_check())
        checks.append(self._kb_check())
        checks.append(self._sqlite_integrity_check())
        checks.append(self._test_inventory_check())
        checks.append(GateCheck(
            "ci_configuration",
            (self.root / ".github" / "workflows" / "ci.yml").exists(),
            "CI workflow present" if (self.root / ".github" / "workflows" / "ci.yml").exists() else "CI workflow missing",
        ))
        checks.append(GateCheck(
            "local_first_default",
            os.getenv("MEDEVIDENCE_CLOUD_ENABLED", "false").casefold() != "true",
            "cloud disabled by default" if os.getenv("MEDEVIDENCE_CLOUD_ENABLED", "false").casefold() != "true" else "cloud explicitly enabled in environment",
        ))
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
    payload = json.dumps(report.as_dict(), ensure_ascii=False, indent=2)
    print(payload)
    return 0 if report.ready else 2


__all__ = ["HumanTestReadinessGate", "ReadinessReport", "GateCheck", "assert_human_test_ready"]

if __name__ == "__main__":
    raise SystemExit(main())
