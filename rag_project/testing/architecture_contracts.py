from __future__ import annotations

import ast
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from rag_project.testing.deep_diagnostics import PhaseResult

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "rag_project"

FORBIDDEN_EDGES: tuple[tuple[str, str], ...] = (
    ("parsing", "intelligence"), ("parsing", "generation"),
    ("chunking", "intelligence"), ("chunking", "generation"),
    ("embeddings", "intelligence"), ("embeddings", "generation"),
    ("storage", "intelligence"), ("storage", "generation"),
    ("retrieval", "generation"), ("retrieval", "intelligence"),
    ("ingestion", "generation"), ("ingestion", "intelligence"),
)

DOMAIN_NAMES = ("ingestion", "chunking", "embeddings", "retrieval", "intelligence", "generation", "storage", "evaluation")


def _domain(path: str) -> str | None:
    parts = Path(path).parts
    if len(parts) >= 2 and parts[0] == "rag_project" and parts[1] in DOMAIN_NAMES:
        return parts[1]
    return None


def analyze() -> dict[str, Any]:
    modules: dict[str, str] = {}
    graph: dict[str, set[str]] = defaultdict(set)
    parse_failures: list[str] = []
    forbidden: list[dict[str, str]] = []

    for path in PROJECT.rglob("*.py"):
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        modules[rel[:-3].replace("/", ".").replace(".__init__", "")] = rel
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except (OSError, SyntaxError) as exc:
            parse_failures.append(f"{rel}:{type(exc).__name__}")
            continue
        source_domain = _domain(rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports = [node.module or ""]
            else:
                continue
            for imported in imports:
                if not imported.startswith("rag_project."):
                    continue
                target_path = imported.removeprefix("rag_project.").replace(".", "/") + ".py"
                target_domain = _domain("rag_project/" + target_path)
                if source_domain and target_domain:
                    graph[source_domain].add(target_domain)
                    if (source_domain, target_domain) in FORBIDDEN_EDGES:
                        forbidden.append({"from": source_domain, "to": target_domain, "module": rel})

    test_imports: list[str] = []
    for path in PROJECT.rglob("*.py"):
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        if rel.startswith("rag_project/testing/") or rel == "rag_project/testing.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import): names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module: names = [node.module]
            if any(name == "rag_project.testing" or name.startswith("rag_project.testing.") for name in names):
                test_imports.append(rel); break

    ownership_failures: list[str] = []
    application = PROJECT / "application.py"
    if not application.exists(): ownership_failures.append("missing authoritative application.py")
    else:
        text = application.read_text(encoding="utf-8")
        if "MedEvidenceProEngine" not in text: ownership_failures.append("application.py does not reference MedEvidenceProEngine")

    cycles: list[list[str]] = []; visiting: set[str] = set(); visited: set[str] = set()
    def visit(node: str, stack: list[str]) -> None:
        if node in visiting:
            cycle = stack[stack.index(node):] + [node]
            if cycle not in cycles: cycles.append(cycle)
            return
        if node in visited: return
        visiting.add(node); stack.append(node)
        for child in sorted(graph.get(node, ())): visit(child, stack)
        stack.pop(); visiting.remove(node); visited.add(node)
    for domain in DOMAIN_NAMES: visit(domain, [])

    return {
        "domain_dependency_edges": {key: sorted(value) for key, value in sorted(graph.items())},
        "forbidden_edges": forbidden, "dependency_cycles": cycles,
        "production_test_harness_imports": sorted(test_imports), "ownership_failures": ownership_failures,
        "parse_failures": parse_failures, "internal_module_count": len(modules),
        "contract_pass": not forbidden and not cycles and not test_imports and not ownership_failures and not parse_failures,
        "forbidden_edge_policy_count": len(FORBIDDEN_EDGES),
    }


def strict_fast_health(phase: Any) -> PhaseResult:
    result = PhaseResult(phase.number, phase.key, phase.name, started_at=time.time())
    checks: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    try:
        collect = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "--disable-warnings"], cwd=ROOT, text=True, capture_output=True, timeout=90)
        collected_match = re.search(r"(\d+) tests? collected", collect.stdout + "\n" + collect.stderr)
        collected = int(collected_match.group(1)) if collected_match else 0
        checks["pytest_collection_exit_zero"] = collect.returncode == 0
        checks["collected_tests_positive"] = collected > 0
        checks["collected_tests"] = collected
        if collect.returncode != 0: failures.append({"location": "pytest --collect-only", "exception": "CollectionFailure", "message": (collect.stdout + collect.stderr)[-1200:]})

        contracts = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "fast and contract", "--tb=short", "--maxfail=8"], cwd=ROOT, text=True, capture_output=True, timeout=120)
        passed_match = re.search(r"(\d+) passed", contracts.stdout + "\n" + contracts.stderr)
        passed = int(passed_match.group(1)) if passed_match else 0
        checks["fast_contract_exit_zero"] = contracts.returncode == 0
        checks["fast_contract_passed"] = passed > 0
        checks["fast_contract_passed_count"] = passed
        if contracts.returncode != 0: failures.append({"location": "pytest -m 'fast and contract'", "exception": "FastContractFailure", "message": (contracts.stdout + contracts.stderr)[-1600:]})

        compile_run = subprocess.run([sys.executable, "-m", "compileall", "-q", "rag_project"], cwd=ROOT, text=True, capture_output=True, timeout=90)
        checks["compileall_exit_zero"] = compile_run.returncode == 0
        if compile_run.returncode != 0: failures.append({"location": "python -m compileall -q rag_project", "exception": "CompileFailure", "message": (compile_run.stdout + compile_run.stderr)[-1200:]})

        smoke = subprocess.run([sys.executable, "-c", "from rag_project.application import *; from rag_project.intelligence.med_evidence_pro import MedEvidenceProEngine; from rag_project.parsing.pdf_extractor import PDFExtractor; from rag_project.storage.vector_store import VectorStore; from rag_project.generation.llm_client import OllamaLLMClient; print('PRODUCTION_IMPORT_SMOKE_OK')"], cwd=ROOT, text=True, capture_output=True, timeout=60)
        checks["production_import_smoke_exit_zero"] = smoke.returncode == 0
        checks["production_import_smoke_marker"] = "PRODUCTION_IMPORT_SMOKE_OK" in smoke.stdout
        if smoke.returncode != 0 or "PRODUCTION_IMPORT_SMOKE_OK" not in smoke.stdout: failures.append({"location": "authoritative production import smoke", "exception": "ProductionImportFailure", "message": (smoke.stdout + smoke.stderr)[-1600:]})

        result.details = {"evidence_level": "strict_fast_runtime_health", "checks": checks, "collected_tests": collected, "fast_contract_passed": passed, "production_imports": ["rag_project.application", "MedEvidenceProEngine", "PDFExtractor", "VectorStore", "OllamaLLMClient"]}
        result.score = sum(bool(value) for key, value in checks.items() if isinstance(value, bool)) / max(1, sum(isinstance(value, bool) for value in checks.values()))
        result.status = "PASS" if not failures and all(value for key, value in checks.items() if isinstance(value, bool)) else "FAIL"
        result.failures.extend(failures)
    except Exception as exc:
        result.status = "FAIL"; result.score = 0.0; result.failures.append({"location": "phase 2 strict fast health", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


def install() -> None:
    from rag_project.testing import production_diagnostic_probes
    from rag_project.testing import deep_diagnostics
    original = production_diagnostic_probes._phase1_semantics

    def hardened(details: dict[str, Any]) -> list[dict[str, Any]]:
        failures = list(original(details)); report = analyze()
        for edge in report["forbidden_edges"]: failures.append({"phase": 1, "reason": f"forbidden dependency edge: {edge['from']} -> {edge['to']} in {edge['module']}"})
        for cycle in report["dependency_cycles"]: failures.append({"phase": 1, "reason": f"architecture dependency cycle: {' -> '.join(cycle)}"})
        for module in report["production_test_harness_imports"]: failures.append({"phase": 1, "reason": f"production module imports diagnostic/test harness: {module}"})
        for problem in report["ownership_failures"]: failures.append({"phase": 1, "reason": problem})
        for problem in report["parse_failures"]: failures.append({"phase": 1, "reason": problem})
        return failures

    production_diagnostic_probes._phase1_semantics = hardened
    deep_diagnostics._fast_health = strict_fast_health


__all__ = ["analyze", "install", "strict_fast_health"]
