"""Authoritative entry point for the unified 17-phase RAG diagnostic system."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable
import hashlib
import re
import subprocess
import sys
import tempfile

from . import deep_diagnostics as core
from .advanced_phases import metamorphic
from .production_answer_probes import phase10_canonical_answer_engine
from .production_benchmark_probes import phase14_production_benchmark
from .production_diagnostic_probes import (
    phase12_stable_fingerprinting,
    phase17_strict_completion,
)
from .production_document_probes import phase7_production_pdf_lab
from .production_path_probes import phase16_production_ingestion_benchmark
from .production_retrieval_probes import phase9_independent_retrieval
from .strict_foundation_phases import (
    strict_contract_triangulation,
    strict_cross_layer_invariants,
    strict_diagnostic_chain,
)
from .strict_information_loss import strict_information_loss
from .strict_runtime_contracts import strict_resource_stability
from .full_metamorphic_probes import run_full_metamorphic_suite
from .full_mutation_probes import run_full_mutation_suite
from .strict_causal_phase import strict_causal_phase

ROOT = Path(__file__).resolve().parents[2]

PHASES = tuple(
    replace(p, markers=("generation", "intelligence"))
    if p.number == 10
    else replace(p, markers=("slow",))
    if p.number == 14
    else replace(p, markers=("high_level",))
    if p.number == 16
    else p
    for p in core.PHASES
)


def _hardened_mutation_phase(phase: core.PhaseSpec) -> core.PhaseResult:
    return run_full_mutation_suite(phase)


def _hardened_fingerprinting(
    spec: core.PhaseSpec, results: dict[int, core.PhaseResult]
) -> core.PhaseResult:
    result = core.PhaseResult(spec.number, spec.key, spec.name, status="PASS", started_at=core.time.time())
    fingerprints = []
    for number, phase_result in sorted(results.items()):
        for index, failure in enumerate(phase_result.failures):
            location = str(failure.get("location") or "unknown").replace("\\", "/")
            exception = str(failure.get("exception") or "UnknownFailure")
            message = re.sub(
                r"0x[0-9a-fA-F]+|\b\d+(?:\.\d+)?\b",
                "#",
                str(failure.get("message") or failure.get("detail") or ""),
            )
            normalized = "|".join(
                (location.split(":", 1)[0], exception, " ".join(message.casefold().split()))
            )
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
            fingerprints.append(
                {
                    "id": f"p{number}f{index}",
                    "phase": number,
                    "fingerprint": digest,
                    "location": location,
                    "exception": exception,
                    "normalized_message": message,
                }
            )

    multiplicity: dict[str, int] = {}
    for row in fingerprints:
        multiplicity[row["fingerprint"]] = multiplicity.get(row["fingerprint"], 0) + 1

    result.details = {
        "evidence_level": "structured_runtime_failure_fingerprint",
        "algorithm": "normalized project frame + exception + normalized message + SHA-256 digest",
        "unique_fingerprints": len(multiplicity),
        "failure_count": len(fingerprints),
        "fingerprints": fingerprints[:100],
        "multiplicity": multiplicity,
        "evidence_phases": sorted(results),
    }
    result.score = 1.0
    result.duration_s = round(core.time.time() - result.started_at, 3)
    return result


def _hardened_causal_graph(
    spec: core.PhaseSpec, results: dict[int, core.PhaseResult]
) -> core.PhaseResult:
    result = core.PhaseResult(spec.number, spec.key, spec.name, started_at=core.time.time())
    try:
        spec_map = {item.number: item for item in PHASES}
        nodes = []
        edges = []
        for number, phase_result in sorted(results.items()):
            for index, failure in enumerate(phase_result.failures):
                nodes.append(
                    {
                        "id": f"p{number}f{index}",
                        "phase": number,
                        "location": failure.get("location"),
                        "exception": failure.get("exception"),
                        "message": failure.get("message"),
                    }
                )

        for left in nodes:
            for right in nodes:
                if left["id"] == right["id"] or left["phase"] >= right["phase"]:
                    continue
                same_exception = bool(left["exception"]) and left["exception"] == right["exception"]
                left_module = str(left.get("location") or "").split(":", 1)[0]
                right_module = str(right.get("location") or "").split(":", 1)[0]
                shared_module = bool(left_module) and left_module == right_module
                dependency = left["phase"] in set(spec_map[right["phase"]].dependencies)
                shared_terms = set(re.findall(r"[a-z_]{5,}", str(left.get("message") or "").casefold())) & set(
                    re.findall(r"[a-z_]{5,}", str(right.get("message") or "").casefold())
                )
                if dependency or (same_exception and shared_module) or (shared_terms and shared_module):
                    reasons = []
                    if dependency:
                        reasons.append("declared_phase_dependency")
                    if same_exception:
                        reasons.append("same_exception")
                    if shared_module:
                        reasons.append("shared_project_module")
                    if shared_terms:
                        reasons.append("shared_failure_terms")
                    confidence = 0.95 if dependency and same_exception else 0.85 if dependency or (same_exception and shared_module) else 0.70
                    edges.append({"from": left["id"], "to": right["id"], "reason": reasons, "confidence": confidence})

        roots = [node["id"] for node in nodes if not any(edge["to"] == node["id"] for edge in edges)]
        result.details = {
            "evidence_level": "graph_causal_hypothesis",
            "algorithm": "declared dependency + shared module + exception/message overlap",
            "nodes": nodes,
            "edges": edges,
            "candidate_roots": roots,
            "root_count": len(roots),
            "independent_failure_count": max(0, len(nodes) - len(edges)),
        }
        result.score = 1.0 if all(0.0 < edge["confidence"] <= 1.0 for edge in edges) and (not nodes or roots) else 0.0
        result.status = "PASS" if result.score == 1.0 else "FAIL"
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "phase 13 hardened causal graph", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(core.time.time() - result.started_at, 3)
    return result


def _base_phase17_strict(spec: core.PhaseSpec, results: dict[int, core.PhaseResult]) -> core.PhaseResult:
    return core.PhaseResult(spec.number, spec.key, spec.name, status="PASS", started_at=core.time.time())


class UnifiedDiagnosticEngine:
    def __init__(self, mode: str = "all", timeout_scale: float = 1.0, fail_fast: bool = False):
        self.mode = mode
        self.timeout_scale = timeout_scale
        self.fail_fast = fail_fast
        self.results: dict[int, core.PhaseResult] = {}
        self.architecture: dict[str, object] = {}

    def _phase1(self, spec: core.PhaseSpec) -> core.PhaseResult:
        result = core.PhaseResult(spec.number, spec.key, spec.name, status="PASS", started_at=core.time.time())
        details = core.build_architecture()
        result.details.update(details)
        result.details["evidence_level"] = "architecture_runtime_inventory"
        result.duration_s = round(core.time.time() - result.started_at, 3)
        return result

    def _blocked(self, spec: core.PhaseSpec) -> core.PhaseResult | None:
        missing = [dep for dep in spec.dependencies if dep not in self.results]
        failed = [dep for dep in spec.dependencies if self.results.get(dep) and self.results[dep].status == "FAIL"]
        if missing:
            return core.PhaseResult(spec.number, spec.key, spec.name, status="BLOCKED", blocked_by=missing, details={"evidence_level": "dependency_block"}, failures=[{"location": f"phase:{missing[0]}", "exception": "DependencyMissing", "message": "prerequisite phase result was not produced"}])
        if failed:
            return core.PhaseResult(spec.number, spec.key, spec.name, status="BLOCKED", blocked_by=failed, details={"evidence_level": "dependency_block"}, failures=[{"location": f"phase:{failed[0]}", "exception": "DependencyFailed", "message": "prerequisite phase failed"}])
        return None

    def _upstream_failure_context(self, spec: core.PhaseSpec, result: core.PhaseResult) -> core.PhaseResult:
        failed = [dep for dep in spec.dependencies if self.results.get(dep) and self.results[dep].status == "FAIL"]
        if failed:
            result.details.setdefault("upstream_failed_phases", failed)
        return result

    def _execute(self, spec: core.PhaseSpec) -> core.PhaseResult:
        blocked = self._blocked(spec)
        if blocked:
            return blocked

        if spec.number == 1:
            result = self._phase1(spec)
        elif spec.number == 2:
            result = core._fast_health(spec)
        elif spec.number == 3:
            result = strict_diagnostic_chain(spec)
        elif spec.number == 4:
            result = strict_contract_triangulation(spec)
        elif spec.number == 5:
            result = strict_cross_layer_invariants(spec)
        elif spec.number == 6:
            result = strict_information_loss(spec)
        elif spec.number == 7:
            result = phase7_production_pdf_lab(spec)
        elif spec.number == 8:
            result = run_full_metamorphic_suite(spec)
        elif spec.number == 9:
            result = phase9_independent_retrieval(spec)
        elif spec.number == 10:
            result = phase10_canonical_answer_engine(spec)
        elif spec.number == 11:
            result = run_full_mutation_suite(spec)
        elif spec.number == 12:
            result = phase12_stable_fingerprinting(spec, self.results)
        elif spec.number == 13:
            result = strict_causal_phase(spec, self.results)
        elif spec.number == 14:
            result = phase14_production_benchmark(spec)
        elif spec.number == 15:
            result = strict_resource_stability(spec)
        elif spec.number == 16:
            result = phase16_production_ingestion_benchmark(spec)
        elif spec.number == 17:
            result = phase17_strict_completion(spec, self.results)
        else:
            raise RuntimeError(f"unimplemented diagnostic phase: {spec.number}")

        if spec.number == 15:
            result.details["pipeline_exercised"] = [
                "robust_ingest_file", "PDFExtractor", "SemanticChunker", "EmbeddingService(test_mode)",
                "VectorStore", "IngestionStateStore", "RSS sampling", "FD sampling",
            ]
            result.details["production_path_strict"] = True
        return self._upstream_failure_context(spec, result)

    def run(self, phases: Iterable[int] | None = None) -> core.DiagnosticReport:
        started = core.time.time()
        wanted = set(phases or range(1, 18))
        if self.mode == "fast":
            wanted &= {1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 17}
        elif self.mode == "deep":
            wanted &= set(range(1, 18))

        original = core.PHASES
        try:
            core.PHASES = PHASES
            self.results = {}
            self.architecture = core.build_architecture()
            for spec in PHASES:
                if spec.number not in wanted:
                    continue
                result = self._execute(spec)
                self.results[spec.number] = result
                if self.fail_fast and result.status == "FAIL":
                    break

            ordered = [self.results[n] for n in sorted(self.results)]
            causes = core.fingerprint_failures(ordered)
            cascade = core.compress_cascade(ordered, causes)
            status = "PASS" if not any(p.status != "PASS" for p in ordered) else "FAIL"
            return core.DiagnosticReport(started_at=started, elapsed_s=round(core.time.time()-started,3), status=status, phases=ordered, root_causes=causes, cascade=cascade, architecture=self.architecture)
        finally:
            core.PHASES = original


def run_all(*, mode: str = "all", timeout_scale: float = 1.0, fail_fast: bool = False, phases: Iterable[int] | None = None) -> core.DiagnosticReport:
    return UnifiedDiagnosticEngine(mode=mode, timeout_scale=timeout_scale, fail_fast=fail_fast).run(phases)
