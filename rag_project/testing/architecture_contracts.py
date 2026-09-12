from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "rag_project"

FORBIDDEN_EDGES: tuple[tuple[str, str], ...] = (
    ("parsing", "intelligence"),
    ("parsing", "generation"),
    ("chunking", "intelligence"),
    ("chunking", "generation"),
    ("embeddings", "intelligence"),
    ("embeddings", "generation"),
    ("storage", "intelligence"),
    ("storage", "generation"),
    ("retrieval", "generation"),
    ("retrieval", "intelligence"),
    ("ingestion", "generation"),
    ("ingestion", "intelligence"),
)

DOMAIN_NAMES = (
    "ingestion",
    "chunking",
    "embeddings",
    "retrieval",
    "intelligence",
    "generation",
    "storage",
    "evaluation",
)


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
        module = rel[:-3].replace("/", ".").replace(".__init__", "")
        modules[module] = rel
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

    # Only production code is subject to this ownership rule. The diagnostic
    # harness is intentionally allowed to import itself and other testing code.
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
            imported_names: list[str] = []
            if isinstance(node, ast.Import):
                imported_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names = [node.module]
            if any(name == "rag_project.testing" or name.startswith("rag_project.testing.") for name in imported_names):
                test_imports.append(rel)
                break

    ownership_failures: list[str] = []
    application = PROJECT / "application.py"
    if not application.exists():
        ownership_failures.append("missing authoritative application.py")
    else:
        text = application.read_text(encoding="utf-8")
        if "MedEvidenceProEngine" not in text:
            ownership_failures.append("application.py does not reference MedEvidenceProEngine")

    cycles: list[list[str]] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str, stack: list[str]) -> None:
        if node in visiting:
            cycle = stack[stack.index(node):] + [node]
            if cycle not in cycles:
                cycles.append(cycle)
            return
        if node in visited:
            return
        visiting.add(node)
        stack.append(node)
        for child in sorted(graph.get(node, ())):
            visit(child, stack)
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    for domain in DOMAIN_NAMES:
        visit(domain, [])

    return {
        "domain_dependency_edges": {key: sorted(value) for key, value in sorted(graph.items())},
        "forbidden_edges": forbidden,
        "dependency_cycles": cycles,
        "production_test_harness_imports": sorted(test_imports),
        "ownership_failures": ownership_failures,
        "parse_failures": parse_failures,
        "internal_module_count": len(modules),
        "contract_pass": not forbidden and not cycles and not test_imports and not ownership_failures and not parse_failures,
        "forbidden_edge_policy_count": len(FORBIDDEN_EDGES),
    }


def install() -> None:
    from rag_project.testing import production_diagnostic_probes
    original = production_diagnostic_probes._phase1_semantics

    def hardened(details: dict[str, Any]) -> list[dict[str, Any]]:
        failures = list(original(details))
        report = analyze()
        for edge in report["forbidden_edges"]:
            failures.append({"phase": 1, "reason": f"forbidden dependency edge: {edge['from']} -> {edge['to']} in {edge['module']}"})
        for cycle in report["dependency_cycles"]:
            failures.append({"phase": 1, "reason": f"architecture dependency cycle: {' -> '.join(cycle)}"})
        for module in report["production_test_harness_imports"]:
            failures.append({"phase": 1, "reason": f"production module imports diagnostic/test harness: {module}"})
        for problem in report["ownership_failures"]:
            failures.append({"phase": 1, "reason": problem})
        for problem in report["parse_failures"]:
            failures.append({"phase": 1, "reason": problem})
        return failures

    production_diagnostic_probes._phase1_semantics = hardened


__all__ = ["analyze", "install"]
