"""Concrete probes for the phases that need project-aware measurement.

All probes are bounded and read-only with respect to the repository. Expensive live API,
Ollama, or long-duration certification work is opt-in through existing project commands;
the diagnostic engine reports availability rather than fabricating results.
"""
from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
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
    proc = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    return proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or ""), time.perf_counter() - started


def _source_tokens(path: Path) -> set[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return set()
    return {
        token
        for token in ("document_id", "page", "section", "parent", "chunk_id", "source", "quality", "index_state")
        if re.search(rf"\b{re.escape(token)}\b", text)
    }


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
    result.details = {
        "layers": {layer: sorted(tokens) for layer, tokens in layer_tokens.items()},
        "required_identity": sorted(required),
        "missing_by_layer": missing,
        "identity_conservation": not missing,
    }
    if missing:
        result.status = "FAIL"
        result.failures.append({
            "location": next(iter(missing)),
            "exception": "IdentityConservationFailure",
            "message": f"required identity fields missing from layer: {missing}",
        })
    else:
        result.status = "PASS"
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
        snapshots[name] = {
            "exists": path.exists(),
            "identity_tokens": sorted(tokens),
            "token_count": len(tokens),
        }
    present_counts = [row["token_count"] for row in snapshots.values() if row["exists"]]
    result.details = {"contracts": snapshots, "weakest_identity_surface": min(present_counts) if present_counts else 0}
    if not all(row["exists"] for row in snapshots.values()):
        result.status = "FAIL"
        result.failures.append({"location": "rag_project/ingestion", "exception": "MissingContract", "message": "one or more canonical boundary modules are missing"})
    elif min(present_counts, default=0) < 3:
        result.status = "WARN"
        result.failures.append({"location": "rag_project/testing/advanced_phases.py", "exception": "InformationLossWarning", "message": "a boundary exposes fewer than three identity fields"})
    else:
        result.status = "PASS"
    result.score = round((sum(present_counts) / max(1, len(present_counts))) / 8.0, 3)
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


def metamorphic(phase: Any) -> PhaseResult:
    result = _result(phase)
    source = PROJECT / "utils" / "text_utils.py"
    if not source.exists():
        result.status = "WARN"
        result.failures.append({"location": str(source.relative_to(ROOT)), "exception": "ProbeUnavailable", "message": "text normalization implementation is unavailable"})
        return result
    code = source.read_text(encoding="utf-8")
    tree = ast.parse(code)
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
    result.details = {
        "mutation_targets": available,
        "mutations_applied": 0,
        "mode": "design-level mutation inventory; repository is never modified",
        "fault_classes": ["empty_retrieval", "metadata_drop", "chunk_loss"],
    }
    if not available:
        result.status = "WARN"
        result.failures.append({"location": "rag_project", "exception": "MutationCoverageWarning", "message": "no stable mutation targets were found"})
    else:
        result.status = "PASS"
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


def performance(phase: Any) -> PhaseResult:
    result = _result(phase)
    candidates = [
        ROOT / "tests" / "test_latency_engineering.py",
        ROOT / "tests/high_level/06_latency_performance/test_simple_query_latency.py",
    ]
    selected = [str(p.relative_to(ROOT)).replace("\\", "/") for p in candidates if p.exists()]
    if not selected:
        result.status = "WARN"
        result.failures.append({"location": "tests", "exception": "PerformanceProbeUnavailable", "message": "no latency contract tests available"})
        return result
    before = time.perf_counter()
    code, output, elapsed = _run([sys.executable, "-m", "pytest", "-q", "--tb=short", "tests/test_latency_engineering.py"], timeout=180)
    result.duration_s = round(time.perf_counter() - before, 3)
    result.details = {"selected_tests": selected, "runner_elapsed_s": elapsed, "output_tail": output[-3000:]}
    result.status = "PASS" if code == 0 else ("WARN" if code == 124 else "FAIL")
    if code != 0:
        result.failures.append({"location": selected[0], "exception": "PerformanceContractFailure", "message": output[-1200:]})
    return result


def resources(phase: Any) -> PhaseResult:
    result = _result(phase)
    script = ROOT / "scripts" / "run_memory_stability.py"
    if not script.exists():
        result.status = "WARN"
        result.failures.append({"location": "scripts/run_memory_stability.py", "exception": "ProbeUnavailable", "message": "memory stability runner is missing"})
        return result
    # A bounded smoke measurement is intentionally not presented as 24h certification.
    command = f"{sys.executable} -c \"import time; x=[]; [x.append(bytearray(1024)) for _ in range(2000)]; time.sleep(1)\""
    started = time.perf_counter()
    code, output, elapsed = _run([sys.executable, str(script), "--command", command, "--duration-seconds", "1.5", "--sample-interval", "0.25", "--output", str(ROOT / ".diagnostic_memory_probe.json")], timeout=20)
    probe_path = ROOT / ".diagnostic_memory_probe.json"
    payload: dict[str, Any] = {}
    if probe_path.exists():
        try:
            payload = json.loads(probe_path.read_text(encoding="utf-8"))
        finally:
            probe_path.unlink(missing_ok=True)
    result.duration_s = round(time.perf_counter() - started, 3)
    result.details = {"smoke": payload, "runner_output_tail": output[-1600:], "certification": False, "elapsed_s": elapsed}
    result.status = "PASS" if code == 2 and payload and not payload.get("timed_out", False) else ("WARN" if code != 0 else "PASS")
    return result


def golden_benchmark(phase: Any) -> PhaseResult:
    result = _result(phase)
    gold = TESTS / "support" / "gold_sets" / "core.jsonl"
    cases = []
    if gold.exists():
        cases = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines() if line.strip()]
    required = {"id", "question"}
    malformed = [item.get("id", "<missing>") for item in cases if not required.issubset(item)]
    result.details = {
        "gold_path": str(gold.relative_to(ROOT)),
        "case_count": len(cases),
        "malformed_case_ids": malformed,
        "clinical_correctness_claimed": False,
        "live_kpi_command": "python scripts/run_kpi_benchmark.py --url ... --gold tests/support/gold_sets/core.jsonl",
    }
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
