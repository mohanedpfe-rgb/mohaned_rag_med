"""Authoritative entry point for the unified 17-phase RAG diagnostic system."""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from . import deep_diagnostics as core
from .advanced_phases import (
    contract_triangulation,
    cross_layer_invariants,
    diagnostic_chain,
    metamorphic,
)
from .robust_probes import information_loss, retrieval_microscope
from .strict_phases import (
    phase10_production_generation,
    phase11_mutation_testing,
    phase13_causal_graph,
    phase17_strict_certification,
    wrap_phase7,
    wrap_phase14,
    wrap_phase15,
)
from .final_probes import phase16_independent_gold

PHASES = tuple(
    replace(p, markers=("generation", "intelligence")) if p.number == 10 else
    replace(p, markers=("slow",)) if p.number == 14 else
    replace(p, markers=("high_level",)) if p.number == 16 else p
    for p in core.PHASES
)


class UnifiedDiagnosticEngine(core.DiagnosticEngine):
    """Dependency-aware engine with bounded, project-aware execution for all 17 phases."""

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
        if spec.number == 1:
            result = self._phase1(spec)
        elif spec.number == 2:
            result = core._fast_health(spec)
        elif spec.number == 3:
            result = diagnostic_chain(spec)
        elif spec.number == 4:
            result = contract_triangulation(spec)
        elif spec.number == 5:
            result = cross_layer_invariants(spec)
        elif spec.number == 6:
            result = information_loss(spec)
        elif spec.number == 7:
            result = wrap_phase7(spec)
        elif spec.number == 8:
            result = metamorphic(spec)
        elif spec.number == 9:
            result = retrieval_microscope(spec)
        elif spec.number == 10:
            result = phase10_production_generation(spec)
        elif spec.number == 11:
            result = phase11_mutation_testing(spec)
        elif spec.number == 12:
            result = core.PhaseResult(spec.number, spec.key, spec.name, status="PASS", started_at=core.time.time())
            fingerprints = core.fingerprint_failures(self.results.values())
            result.details = {"algorithm": "structured location + exception + normalized message fingerprint", "unique_fingerprints": len(fingerprints), "fingerprints": fingerprints[:50], "evidence_phases": sorted(self.results)}
            result.score = 1.0
            result.duration_s = round(core.time.time() - result.started_at, 3)
        elif spec.number == 13:
            result = phase13_causal_graph(spec, self.results)
        elif spec.number == 14:
            result = wrap_phase14(spec)
        elif spec.number == 15:
            result = wrap_phase15(spec)
        elif spec.number == 16:
            result = phase16_independent_gold(spec)
        elif spec.number == 17:
            result = phase17_strict_certification(spec, self.results)
        else:
            raise RuntimeError(f"unimplemented diagnostic phase: {spec.number}")
        return self._upstream_failure_context(spec, result)

    def run(self, phases: Iterable[int] | None = None) -> core.DiagnosticReport:
        started = core.time.time()
        wanted = set(phases or range(1, 18))
        if self.mode == "fast":
            wanted &= {1, 2, 3, 4, 5, 6, 8, 11, 12, 13, 17}
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
            status = "PASS" if not any(p.status == "FAIL" for p in ordered) else "FAIL"
            if status == "PASS" and any(p.status == "WARN" for p in ordered):
                status = "WARN"
            return core.DiagnosticReport(started_at=started, elapsed_s=round(core.time.time() - started, 3), status=status, phases=ordered, root_causes=causes, cascade=cascade, architecture=self.architecture)
        finally:
            core.PHASES = original


def run_all(*, mode: str = "all", timeout_scale: float = 1.0, fail_fast: bool = False, phases: Iterable[int] | None = None) -> core.DiagnosticReport:
    return UnifiedDiagnosticEngine(mode=mode, timeout_scale=timeout_scale, fail_fast=fail_fast).run(phases)
