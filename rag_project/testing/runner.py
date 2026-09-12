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
from .advanced_phases import contract_triangulation, cross_layer_invariants, diagnostic_chain, metamorphic
from .robust_probes import information_loss, retrieval_microscope
from .strict_phases import phase10_production_generation
from .strict_v2 import phase7_real_pdf_lab, phase15_resource_stability
from .production_answer_probes import phase10_canonical_answer_engine
from .production_benchmark_probes import phase14_production_benchmark
from .production_diagnostic_probes import phase12_stable_fingerprinting, phase13_known_causal_graph, phase17_strict_completion
from .production_path_probes import phase16_production_ingestion_benchmark

PHASES = tuple(
    replace(p, markers=("generation", "intelligence")) if p.number == 10 else
    replace(p, markers=("slow",)) if p.number == 14 else
    replace(p, markers=("high_level",)) if p.number == 16 else p
    for p in core.PHASES
)

ROOT = Path(__file__).resolve().parents[2]


def _hardened_mutation_phase(phase: core.PhaseSpec) -> core.PhaseResult:
    """Kill multiple executable source mutants with independent pytest processes."""
    result = core.PhaseResult(phase.number, phase.key, phase.name, status="FAIL", started_at=core.time.time())
    target = ROOT / "rag_project" / "utils" / "text_utils.py"
    mutants = []
    try:
        source = target.read_text(encoding="utf-8")
        original = 'return re.sub(r"\\s+", " ", value or "").strip()'
        if original not in source:
            raise RuntimeError("mutation target changed and no safe mutation can be applied")
        replacements = [
            ("return_raw", 'return value or ""'),
            ("no_collapse", 'return re.sub(r"\\s+", " ", value or "")'),
            ("collapse_to_tab", 'return re.sub(r"\\s+", "\\t", value or "").strip()'),
            ("collapse_only_left", 'return re.sub(r"\\s+", " ", value or "").lstrip()'),
        ]
        with tempfile.TemporaryDirectory(prefix="rag_mutation_suite_v2_") as td:
            root = Path(td)
            for name, replacement in replacements:
                mutant_module = root / f"text_utils_{name}.py"
                mutant_module.write_text(source.replace(original, replacement, 1), encoding="utf-8")
                test_file = root / f"test_{name}.py"
                test_file.write_text(
                    "from importlib.util import spec_from_file_location, module_from_spec\n"
                    f"spec=spec_from_file_location('mutant_{name}', r'{mutant_module}')\n"
                    "m=module_from_spec(spec); spec.loader.exec_module(m)\n"
                    "def test_contract():\n"
                    "    assert m.normalize_whitespace('  diabetes   mellitus  ') == 'diabetes mellitus'\n"
                    "    assert m.normalize_whitespace('\\u00a0HbA1c\\tthreshold\\u00a0') == 'HbA1c threshold'\n",
                    encoding="utf-8",
                )
                proc = subprocess.run([sys.executable, "-m", "pytest", "-q", str(test_file)], cwd=ROOT, text=True, capture_output=True, timeout=60)
                mutants.append({"name": name, "returncode": proc.returncode, "killed": proc.returncode != 0, "stdout": proc.stdout[-700:], "stderr": proc.stderr[-700:]})
        applicable = len(mutants)
        killed = sum(int(item["killed"]) for item in mutants)
        score = killed / max(applicable, 1)
        result.details = {
            "strategy": "four executable source mutants + independent pytest process per mutant",
            "mutants_applicable": applicable,
            "mutants_killed": killed,
            "kill_score": round(score, 3),
            "mutation_results": mutants,
            "target": str(target.relative_to(ROOT)),
            "real_pytest_subprocess": True,
        }
        result.score = round(score, 3)
        result.status = "PASS" if applicable == 4 and killed == applicable else "FAIL"
        if result.status == "FAIL":
            result.failures.append({"location": str(target.relative_to(ROOT)), "exception": "SurvivingMutant", "message": f"kill score={score:.3f}"})
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "phase 11 hardened mutation suite", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(core.time.time() - result.started_at, 3)
    return result


def _hardened_fingerprinting(spec: core.PhaseSpec, results: dict[int, core.PhaseResult]) -> core.PhaseResult:
    result = core.PhaseResult(spec.number, spec.key, spec.name, status="PASS", started_at=core.time.time())
    fingerprints = []
    for number, phase_result in sorted(results.items()):
        for index, failure in enumerate(phase_result.failures):
            location = str(failure.get("location") or "unknown").replace("\\", "/")
            exception = str(failure.get("exception") or "UnknownFailure")
            message = re.sub(r"0x[0-9a-fA-F]+|\b\d+(?:\.\d+)?\b", "#", str(failure.get("message") or failure.get("detail") or ""))
            normalized = "|".join((location.split(":", 1)[0], exception, " ".join(message.casefold().split())))
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
            fingerprints.append({"id": f"p{number}f{index}", "phase": number, "fingerprint": digest, "location": location, "exception": exception, "normalized_message": message})
    multiplicity = {}
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


def _hardened_causal_graph(spec: core.PhaseSpec, results: dict[int, core.PhaseResult]) -> core.PhaseResult:
    result = core.PhaseResult(spec.number, spec.key, spec.name, started_at=core.time.time())
    try:
        spec_map = {item.number: item for item in PHASES}
        nodes, edges = [], []
        for number, phase_result in sorted(results.items()):
            for index, failure in enumerate(phase_result.failures):
                nodes.append({"id": f"p{number}f{index}", "phase": number, "location": failure.get("location"), "exception": failure.get("exception"), "message": failure.get("message")})
        for left in nodes:
            for right in nodes:
                if left["id"] == right["id"] or left["phase"] >= right["phase"]:
                    continue
                same_exception = bool(left["exception"]) and left["exception"] == right["exception"]
                left_module = str(left.get("location") or "").split(":", 1)[0]
                right_module = str(right.get("location") or "").split(":", 1)[0]
                shared_module = bool(left_module) and left_module == right_module
                dependency = left["phase"] in set(spec_map.get(right["phase"], core.PhaseSpec(0,"","","","",())).dependencies)
                shared_terms = set(re.findall(r"[a-z_]{5,}", str(left.get("message") or "").casefold())) & set(re.findall(r"[a-z_]{5,}", str(right.get("message") or "").casefold()))
                if dependency or (same_exception and shared_module) or (shared_terms and shared_module):
                    reasons = []
                    if dependency: reasons.append("declared_phase_dependency")
                    if same_exception: reasons.append("same_exception")
                    if shared_module: reasons.append("shared_project_module")
                    if shared_terms: reasons.append("shared_failure_terms")
                    confidence = 0.95 if dependency and same_exception else 0.85 if dependency or (same_exception and shared_module) else 0.70
                    edges.append({"from": left["id"], "to": right["id"], "reason": reasons, "confidence": confidence})
        roots = [node["id"] for node in nodes if not any(edge["to"] == node["id"] for edge in edges)]
        result.details = {"evidence_level": "graph_causal_hypothesis", "algorithm": "declared dependency + shared module + exception/message overlap", "nodes": nodes, "edges": edges, "candidate_roots": roots, "root_count": len(roots), "independent_failure_count": max(0, len(nodes)-len(edges))}
        result.score = 1.0 if all(0.0 < e["confidence"] <= 1.0 for e in edges) and (not nodes or roots) else 0.0
        result.status = "PASS" if result.score == 1.0 else "FAIL"
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "phase 13 hardened causal graph", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(core.time.time() - result.started_at, 3)
    return result


def _phase17_strict(spec: core.PhaseSpec, results: dict[int, core.PhaseResult]) -> core.PhaseResult:
    return phase17_strict_completion(spec, results)


class UnifiedDiagnosticEngine(core.DiagnosticEngine):
    """Dependency-aware engine with production-path execution for all 17 phases."""

    def _blocked(self, spec: core.PhaseSpec) -> core.PhaseResult | None:
        missing = [dep for dep in spec.dependencies if dep not in self.results]
        if not missing:
            return None
        result = core.PhaseResult(spec.number, spec.key, spec.name, status="BLOCKED", blocked_by=sorted(missing), started_at=core.time.time())
        result.failures.append({"location": f"phase:{missing[0]}", "exception": "DependencyMissing", "message": "prerequisite phase result was not produced"})
        return result

    def _upstream_failure_context(self, spec: core.PhaseSpec, result: core.PhaseResult) -> core.PhaseResult:
        failed = [dep for dep in spec.dependencies if self.results.get(dep) and self.results[dep].status == "FAIL"]
        if failed:
            result.details.setdefault("upstream_failed_phases", failed)
        return result

    def _execute(self, spec: core.PhaseSpec) -> core.PhaseResult:
        blocked = self._blocked(spec)
        if blocked:
            return blocked
        if spec.number == 1: result = self._phase1(spec)
        elif spec.number == 2: result = core._fast_health(spec)
        elif spec.number == 3: result = diagnostic_chain(spec)
        elif spec.number == 4: result = contract_triangulation(spec)
        elif spec.number == 5: result = cross_layer_invariants(spec)
        elif spec.number == 6: result = information_loss(spec)
        elif spec.number == 7: result = phase7_real_pdf_lab(spec)
        elif spec.number == 8: result = metamorphic(spec)
        elif spec.number == 9: result = retrieval_microscope(spec)
        elif spec.number == 10: result = phase10_canonical_answer_engine(spec)
        elif spec.number == 11: result = _hardened_mutation_phase(spec)
        elif spec.number == 12: result = phase12_stable_fingerprinting(spec, self.results)
        elif spec.number == 13: result = phase13_known_causal_graph(spec, self.results)
        elif spec.number == 14: result = phase14_production_benchmark(spec)
        elif spec.number == 15: result = phase15_resource_stability(spec)
        elif spec.number == 16: result = phase16_production_ingestion_benchmark(spec)
        elif spec.number == 17: result = phase17_strict_completion(spec, self.results)
        else: raise RuntimeError(f"unimplemented diagnostic phase: {spec.number}")
        return self._upstream_failure_context(spec, result)

    def run(self, phases: Iterable[int] | None = None) -> core.DiagnosticReport:
        started = core.time.time(); wanted = set(phases or range(1,18))
        if self.mode == "fast": wanted &= {1,2,3,4,5,6,8,11,12,13,17}
        elif self.mode == "deep": wanted &= set(range(1,18))
        original = core.PHASES
        try:
            core.PHASES = PHASES; self.results = {}; self.architecture = core.build_architecture()
            for spec in PHASES:
                if spec.number not in wanted: continue
                result = self._execute(spec); self.results[spec.number] = result
                if self.fail_fast and result.status == "FAIL": break
            ordered = [self.results[n] for n in sorted(self.results)]
            causes = core.fingerprint_failures(ordered); cascade = core.compress_cascade(ordered, causes)
            status = "PASS" if not any(p.status != "PASS" for p in ordered) else "FAIL"
            return core.DiagnosticReport(started_at=started, elapsed_s=round(core.time.time()-started,3), status=status, phases=ordered, root_causes=causes, cascade=cascade, architecture=self.architecture)
        finally:
            core.PHASES = original


def run_all(*, mode: str = "all", timeout_scale: float = 1.0, fail_fast: bool = False, phases: Iterable[int] | None = None) -> core.DiagnosticReport:
    return UnifiedDiagnosticEngine(mode=mode, timeout_scale=timeout_scale, fail_fast=fail_fast).run(phases)
