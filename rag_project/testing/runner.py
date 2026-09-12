"""Authoritative entry point for the unified 17-phase RAG diagnostic system."""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from . import deep_diagnostics as core
from .advanced_phases import (
    cascade_phase,
    certification_phase,
    cross_layer_invariants,
    golden_benchmark,
    information_loss,
    metamorphic,
    mutation_detection,
    performance,
    resources,
    root_cause_phase,
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
        """Block only when a prerequisite result is absent.

        A failed prerequisite is evidence, not a reason to suppress every downstream
        diagnostic. Independent probes must still run so the final report can identify
        the true root cause and distinguish it from downstream symptoms.
        """
        missing = [dep for dep in spec.dependencies if dep not in self.results]
        if not missing:
            return None
        result = core.PhaseResult(
            spec.number,
            spec.key,
            spec.name,
            status="BLOCKED",
            blocked_by=sorted(missing),
            started_at=core.time.time(),
        )
        result.failures.append({
            "location": f"phase:{missing[0]}",
            "exception": "DependencyMissing",
            "message": "prerequisite phase result was not produced; diagnostic execution is incomplete",
        })
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
        elif spec.number in {3, 4, 7, 9, 10}:
            timeout = {3: 90, 4: 150, 7: 240, 9: 240, 10: 300}[spec.number]
            result = core._pytest_phase(spec, timeout=int(timeout * self.timeout_scale), maxfail=8)
        elif spec.number == 5:
            result = cross_layer_invariants(spec)
        elif spec.number == 6:
            result = information_loss(spec)
        elif spec.number == 8:
            result = metamorphic(spec)
        elif spec.number == 11:
            result = mutation_detection(spec)
        elif spec.number == 12:
            result = root_cause_phase(spec, self.results)
        elif spec.number == 13:
            result = cascade_phase(spec, self.results)
        elif spec.number == 14:
            result = performance(spec)
        elif spec.number == 15:
            result = resources(spec)
        elif spec.number == 16:
            result = golden_benchmark(spec)
        elif spec.number == 17:
            result = certification_phase(spec, self.results)
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
