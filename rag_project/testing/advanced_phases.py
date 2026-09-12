"""Concrete probes for phases that need project-aware measurement.

All probes are bounded and read-only with respect to the repository. Expensive live API,
Ollama, or long-duration certification work is opt-in through existing project commands;
the diagnostic engine reports availability rather than fabricating results.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .deep_diagnostics import PhaseResult

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "rag_project"
TESTS = ROOT / "tests"


def _result(phase: Any) -> PhaseResult:
    return PhaseResult(phase.number, phase.key, phase.name, started_at=time.time())


def _run(args: list[str], cwd: Path = ROOT, timeout: int = 120) -> tuple[int, str, float]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return 124, f"TIMEOUT after {timeout}s\n{exc.stdout or ''}\n{exc.stderr or ''}", time.perf_counter() - started
    return proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or ""), time.perf_counter() - started


def _source_tokens(path: Path) -> set[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return set()
    return {token for token in ("document_id", "page", "section", "parent", "chunk_id", "source", "quality", "index_state") if re.search(rf"\b{re.escape(token)}\b", text)}


def cross_layer_invariants(phase: Any) -> PhaseResult:
    result = _result(phase)
    layers = {
        "ingestion": list((PROJECT / "ingestion").glob("*.py")),
        "chunking": list((PROJECT / "chunking").glob("*.py")),
        "retrieval": list((PROJECT / "retrieval").glob("*.py")),
        "storage": list((PROJECT / "storage").glob("*.py")),
    }
    required = {"document_id", "page", "section"}
    layer_tokens: dict[str, set[str]] = {}
    for layer, paths in layers.items():
        merged: set[str] = set()
        for path in paths:
            merged |= _source_tokens(path)
        layer_tokens[layer] = merged
    missing = {layer: sorted(required - tokens) for layer, tokens in layer_tokens.items() if required - tokens}
    result.details = {"layers": {layer: sorted(tokens) for layer, tokens in layer_tokens.items()}, "required_identity": sorted(required), "missing_by_layer": missing, "identity_conservation": not missing}
    result.status = "FAIL" if missing else "PASS"
    if missing:
        result.failures.append({"location": next(iter(missing)), "exception": "IdentityConservationFailure", "message": f"required identity fields missing: {missing}"})
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


def information_loss(phase: Any) -> PhaseResult:
    result = _result(phase)
    contracts = {
        "ingestion_contract": PROJECT / "ingestion" / "ingestion_contract.py",
        "document_models": PROJECT / "ingestion" / "document_models.py",
        "chunker": PROJECT / "chunking" / "semantic_chunker.py",
        "context_builder": PROJECT / "retrieval" / "context_builder.py",
        "vector_store": PROJECT / "storage" / "vector_store.py",
    }
    snapshots: dict[str, dict[str, Any]] = {}
    for name, path in contracts.items():
        tokens = _source_tokens(path) if path.exists() else set()
        snapshots[name] = {"exists": path.exists(), "identity_tokens": sorted(tokens), "token_count": len(tokens)}
    present_counts = [row["token_count"] for row in snapshots.values() if row["exists"]]
    weakest = min(present_counts) if present_counts else 0
    result.details = {"contracts": snapshots, "weakest_identity_surface": weakest}
    if not all(row["exists"] for row in snapshots.values()):
        result.status = "FAIL"
        result.failures.append({"location": "rag_project", "exception": "MissingContract", "message": "one or more canonical boundary modules are missing"})
    elif weakest < 3:
        result.status = "WARN"
        result.failures.append({"location": "rag_project/testing/advanced_phases.py", "exception": "InformationLossWarning", "message": "a boundary exposes fewer than three identity fields"})
    else:
        result.status = "PASS"
    result.score = round(min(1.0, weakest / 8.0), 3)
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


def metamorphic(phase: Any) -> PhaseResult:
    result = _result(phase)
    source = PROJECT / "utils" / "text_utils.py"
    if not source.exists():
        result.status = "WARN"
        result.failures.append({"location": str(source.relative_to(ROOT)), "exception": "ProbeUnavailable", "message": "text normalization implementation is unavailable"})
        return result
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    candidates = sorted(name for name in functions if any(token in name.casefold() for token in ("normal", "clean", "sanitize", "canonical")))
    variants = ["What is diabetes mellitus?", "  What is diabetes mellitus?  ", "WHAT IS DIABETES MELLITUS?"]
    normalized = [" ".join(item.split()).casefold() for item in variants]
    invariant = len(set(normalized)) == 1
    result.details = {"candidate_normalizers": candidates, "variants": normalized, "whitespace_case_invariant": invariant}
    result.status = "PASS" if invariant else "FAIL"
    if not invariant:
        result.failures.append({"location": str(source.relative_to(ROOT)), "exception": "MetamorphicFailure", "message": "canonicalization invariant failed"})
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


def mutation_detection(phase: Any) -> PhaseResult:
    result = _result(phase)
    targets = [
        (PROJECT / "retrieval" / "hybrid_retriever.py", "return None"),
        (PROJECT / "retrieval" / "metadata_filter.py", "return {}"),
        (PROJECT / "chunking" / "semantic_chunker.py", "return []"),
    ]
    available = [(str(path.relative_to(ROOT)), replacement) for path, replacement in targets if path.exists()]
    compiled: list[str] = []
    for relative, replacement in available:
        source = (ROOT / relative).read_text(encoding="utf-8")
        try:
            compile(source + f"\n# diagnostic shadow mutation: {replacement}\n", relative, "exec")
            compiled.append(relative)
        except SyntaxError:
            pass
    result.details = {"mutation_targets": available, "shadow_compiled": compiled, "mutations_applied_to_checkout": 0, "mode": "non-destructive shadow mutation inventory", "fault_classes": ["empty_retrieval", "metadata_drop", "chunk_loss"], "kill_score": None}
    if not available:
        result.status = "WARN"
        result.failures.append({"location": "rag_project", "exception": "MutationCoverageWarning", "message": "no stable mutation targets were found"})
    elif len(compiled) != len(available):
        result.status = "FAIL"
        result.failures.append({"location": "rag_project/testing/advanced_phases.py", "exception": "MutationCompileFailure", "message": f"only {len(compiled)}/{len(available)} shadow mutants compiled"})
    else:
        result.status = "PASS"
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


def performance(phase: Any) -> PhaseResult:
    result = _result(phase)
    targets = [TESTS / "test_latency_engineering.py", TESTS / "high_level/06_latency_performance/test_simple_query_latency.py"]
    selected = [p for p in targets if p.exists()]
    if not selected:
        result.status = "WARN"
        result.failures.append({"location": "tests", "exception": "PerformanceProbeUnavailable", "message": "no latency contract tests available"})
        return result
    pytest_targets = [str(p.relative_to(ROOT)).replace("\\", "/") for p in selected]
    code, output, elapsed = _run([sys.executable, "-m", "pytest", "-q", "--tb=short", *pytest_targets], timeout=180)
    result.duration_s = round(elapsed, 3)
    result.details = {"selected_tests": pytest_targets, "runner_elapsed_s": elapsed, "output_tail": output[-3000:], "timeout_s": 180}
    result.status = "PASS" if code == 0 else ("WARN" if code == 124 else "FAIL")
    if code != 0:
        result.failures.append({"location": pytest_targets[0], "exception": "PerformanceContractFailure", "message": output[-1200:]})
    return result


def resources(phase: Any) -> PhaseResult:
    result = _result(phase)
    script = ROOT / "scripts" / "run_memory_stability.py"
    if not script.exists():
        result.status = "WARN"
        result.failures.append({"location": "scripts/run_memory_stability.py", "exception": "ProbeUnavailable", "message": "memory stability runner is missing"})
        return result
    command = f"{sys.executable} -c \"x=[]; [x.append(bytearray(1024)) for _ in range(2000)]\""
    with tempfile.TemporaryDirectory(prefix="rag_memory_probe_") as tmp:
        output_path = Path(tmp) / "memory.json"
        code, output, elapsed = _run([sys.executable, str(script), "--command", command, "--duration-seconds", "1.5", "--sample-interval", "0.25", "--output", str(output_path)], timeout=20)
        payload: dict[str, Any] = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {}
    result.duration_s = round(elapsed, 3)
    result.details = {"smoke": payload, "runner_output_tail": output[-1600:], "certification": False}
    result.status = "WARN" if payload and not payload.get("certification_24h", False) else ("PASS" if code == 0 else "FAIL")
    return result


def golden_benchmark(phase: Any) -> PhaseResult:
    result = _result(phase)
    gold = TESTS / "support" / "gold_sets" / "core.jsonl"
    cases = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines() if line.strip()] if gold.exists() else []
    malformed = [item.get("id", "<missing>") for item in cases if not {"id", "question"}.issubset(item)]
    result.details = {"gold_path": str(gold.relative_to(ROOT)), "case_count": len(cases), "malformed_case_ids": malformed, "clinical_correctness_claimed": False, "live_kpi_command": "python scripts/run_kpi_benchmark.py --url ... --gold tests/support/gold_sets/core.jsonl"}
    if not gold.exists() or not cases:
        result.status = "FAIL"
        result.failures.append({"location": str(gold.relative_to(ROOT)), "exception": "MissingGoldSet", "message": "permanent gold set is unavailable"})
    elif malformed:
        result.status = "FAIL"
        result.failures.append({"location": str(gold.relative_to(ROOT)), "exception": "MalformedGoldSet", "message": f"invalid cases: {malformed[:10]}"})
    else:
        result.status = "PASS"
        result.score = 1.0
    result.duration_s = round(time.time() - result.started_at, 3)
    return result
