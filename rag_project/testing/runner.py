"""Authoritative entry point for the unified 17-phase RAG diagnostic system."""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from . import deep_diagnostics as core
from .advanced_phases import (
    cross_layer_invariants,
    golden_benchmark,
    information_loss,
    metamorphic,
    mutation_detection,
    performance,
    resources,
)


PHASES = tuple(
    replace(p, markers=("generation", "intelligence")) if p.number == 10 else
    replace(p, markers=("slow",)) if p.number == 14 else
    replace(p, markers=("high_level",)) if p.number == 16 else p
    for p in core.PHASES
)


class UnifiedDiagnosticEngine(core.DiagnosticEngine):
    """Dependency-aware engine with bounded, project-aware execution for every phase."""

    def _blocked(self, spec: core.PhaseSpec) -> core.PhaseResult | None:
        missing = [dep for dep in spec.dependencies if dep not in self.results]
        failed = [dep for dep in spec.dependencies if self.results.get(dep) and self.results[dep].status == "FAIL"]
        if not missing and not failed:
            return None
        blockers = sorted(set(missing + failed))
        result = core.PhaseResult(
            spec.number,
            spec.key,
            spec.name,
            status="BLOCKED",
            blocked_by=blockers,
            started_at=core.time.time(),
        )
        reason = "missing prerequisite phase results" if missing else "upstream prerequisite failed"
        result.failures.append({
            "location": f"phase:{blockers[0]}",
            "exception": "DependencyBlocked",
            "message": f"{reason}; downstream execution would mostly create non-diagnostic symptoms",
        })
        return result

    def _execute(self, spec: core.PhaseSpec) -> core.PhaseResult:
        blocked = self._blocked(spec)
        if blocked:
            return blocked
        if spec.number == 1:
            return self._phase1(spec)
        if spec.number == 2:
            return core._fast_health(spec)
        if spec.number in {3, 4, 7, 9, 10}:
            timeout = {3: 90, 4: 150, 7: 240, 9: 240, 10: 300}[spec.number]
            return core._pytest_phase(spec, timeout=int(timeout * self.timeout_scale), maxfail=8)
        if spec.number == 5:
            return cross_layer_invariants(spec)
        if spec.number == 6:
            return information_loss(spec)
        if spec.number == 8:
            return metamorphic(spec)
        if spec.number == 11:
            return mutation_detection(spec)
        if spec.number == 12:
            return self._phase_analysis(spec, "fingerprinting")
        if spec.number == 13:
            return self._phase_analysis(spec, "cascade")
        if spec.number == 14:
            return performance(spec)
        if spec.number == 15:
            return resources(spec)
        if spec.number == 16:
            return golden_benchmark(spec)
        if spec.number == 17:
            return self._phase_analysis(spec, "certification")
        raise RuntimeError(f"unimplemented diagnostic phase: {spec.number}")

    def run(self, phases: Iterable[int] | None = None) -> core.DiagnosticReport:
        started = core.time.time()
        wanted = set(phases or range(1, 18))
        if self.mode == "fast":
            wanted &= {1, 2, 3, 4, 5, 6, 12, 13, 17}
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
            status = "PASS"
            if any(p.status == "FAIL" for p in ordered):
                status = "FAIL"
            elif any(p.status == "WARN" for p in ordered):
                status = "WARN"
            return core.DiagnosticReport(
                started_at=started,
                elapsed_s=round(core.time.time() - started, 3),
                status=status,
                phases=ordered,
                root_causes=causes,
                cascade=cascade,
                architecture=self.architecture,
            )
        finally:
            core.PHASES = original


def run_all(*, mode: str = "all", timeout_scale: float = 1.0, fail_fast: bool = False, phases: Iterable[int] | None = None) -> core.DiagnosticReport:
    return UnifiedDiagnosticEngine(mode=mode, timeout_scale=timeout_scale, fail_fast=fail_fast).run(phases)
