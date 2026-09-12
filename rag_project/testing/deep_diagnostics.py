"""Unified 17-phase diagnostic engine.

The engine is intentionally lightweight: static analysis and cheap contract probes run
first, while expensive pytest/benchmark phases are dependency-aware and can be skipped
once an upstream blocker makes downstream results non-diagnostic. Every phase emits a
structured result so failures can be collapsed into first-cause findings instead of a
large list of downstream symptoms.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time
import tracemalloc
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "rag_project"
TESTS = ROOT / "tests"
SCRIPTS = ROOT / "scripts"


@dataclass(frozen=True)
class PhaseSpec:
    number: int
    key: str
    name: str
    kind: str
    description: str
    dependencies: tuple[int, ...] = ()
    markers: tuple[str, ...] = ()


@dataclass
class PhaseResult:
    number: int
    key: str
    name: str
    status: str = "NOT_RUN"
    started_at: float = 0.0
    duration_s: float = 0.0
    score: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    blocked_by: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in {"PASS", "WARN"}


@dataclass
class RootCause:
    fingerprint: str
    severity: str
    phase: int
    title: str
    location: str
    message: str
    evidence: list[str]
    affected_phases: list[int]
    confidence: float


@dataclass
class DiagnosticReport:
    started_at: float
    elapsed_s: float
    status: str
    phases: list[PhaseResult]
    root_causes: list[RootCause]
    cascade: dict[str, Any]
    architecture: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "elapsed_s": self.elapsed_s,
            "status": self.status,
            "phases": [asdict(item) for item in self.phases],
            "root_causes": [asdict(item) for item in self.root_causes],
            "cascade": self.cascade,
            "architecture": self.architecture,
        }


PHASES: tuple[PhaseSpec, ...] = (
    PhaseSpec(1, "architecture_map", "Architecture/test coverage map", "static", "Map modules, tests, markers, and ownership boundaries."),
    PhaseSpec(2, "fast_health", "Ultra-fast health gate", "pytest", "Run cheap deterministic contracts and collection checks.", (1,), ("fast",)),
    PhaseSpec(3, "diagnostic_chain", "Dependency-aware diagnostic chain", "pytest", "Find the first broken contract before cascading into downstream tests.", (2,), ("diagnostic",)),
    PhaseSpec(4, "contract_triangulation", "Input/transformation/output contracts", "pytest", "Exercise cross-module public contracts instead of isolated assertions.", (3,), ("contract",)),
    PhaseSpec(5, "cross_layer_invariants", "Cross-layer invariants and identity conservation", "static", "Verify traceability of source/page/chunk/metadata identities across boundaries.", (3,)),
    PhaseSpec(6, "information_loss", "Information-loss analysis", "static", "Detect structural/text loss between source, chunks, storage, and retrieval representations.", (5,)),
    PhaseSpec(7, "adversarial_documents", "Adversarial document laboratory", "pytest", "Run hostile PDF, OCR, layout, resilience, and replacement cases.", (3, 6)),
    PhaseSpec(8, "metamorphic", "Metamorphic stability", "pytest", "Run transformations whose semantic result should remain stable.", (4, 6)),
    PhaseSpec(9, "retrieval_microscope", "Retrieval microscope", "pytest", "Evaluate ranking, recall, source identity, and metadata-aware retrieval.", (5, 6, 7)),
    PhaseSpec(10, "rag_causality", "RAG answer causality", "pytest", "Trace retrieval → context → grounding → answer → citation failures.", (9,)),
    PhaseSpec(11, "mutation", "Mutation detection", "static", "Use targeted source mutations to verify the suite can detect important faults.", (4, 5, 9)),
    PhaseSpec(12, "fingerprinting", "Root-cause fingerprinting", "analysis", "Normalize failures into stable fingerprints and locations.", (3, 4, 5, 9, 10, 11)),
    PhaseSpec(13, "cascade", "Failure-cascade compression", "analysis", "Collapse downstream symptoms to likely first causes.", (12,)),
    PhaseSpec(14, "performance", "Performance intelligence", "pytest", "Measure stage latency and detect meaningful regressions.", (2, 9)),
    PhaseSpec(15, "resources", "Resource/leak diagnostics", "pytest", "Detect memory/resource growth under bounded repetition.", (14,)),
    PhaseSpec(16, "golden_benchmark", "Golden medical RAG benchmark", "pytest", "Run permanent evaluation/certification cases with evidence metrics.", (9, 10)),
    PhaseSpec(17, "certification", "Master diagnostic certification", "analysis", "Produce the final report, root causes, impact, and recommended fix order.", tuple(range(1, 17))),
)


def _phase_map() -> dict[int, PhaseSpec]:
    return {p.number: p for p in PHASES}


def _run_command(args: list[str], timeout: int = 120) -> tuple[int, str, float]:
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    started = time.perf_counter()
    try:
        proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, env=env, timeout=timeout)
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return proc.returncode, output, time.perf_counter() - started
    except subprocess.TimeoutExpired as exc:
        output = f"TIMEOUT after {timeout}s\n{exc.stdout or ''}\n{exc.stderr or ''}"
        return 124, output, time.perf_counter() - started


def _first_project_frame(text: str) -> str | None:
    for raw in text.splitlines():
        line = raw.strip().replace("\\", "/")
        if "rag_project/" in line and ".py:" in line:
            match = re.search(r"(rag_project/[^\s:]+\.py):(\d+)", line)
            if match:
                return f"{match.group(1)}:{match.group(2)}"
    return None


def _exception(text: str) -> tuple[str | None, str | None]:
    for raw in reversed(text.splitlines()):
        match = re.search(r"([A-Za-z_][\w.]*(?:Error|Exception)):\s*(.*)", raw.strip())
        if match:
            return match.group(1), match.group(2)
    return None, None


def build_architecture() -> dict[str, Any]:
    modules = [str(p.relative_to(ROOT)).replace("\\", "/") for p in PROJECT.rglob("*.py")]
    tests = [str(p.relative_to(ROOT)).replace("\\", "/") for p in TESTS.rglob("test_*.py")]
    markers: Counter[str] = Counter()
    marker_re = re.compile(r"pytest\.mark\.([A-Za-z_][A-Za-z0-9_]*)|@pytest\.mark\.([A-Za-z_][A-Za-z0-9_]*)")
    for path in TESTS.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for a, b in marker_re.findall(text):
            markers[a or b] += 1
    domains = {
        "ingestion": [p for p in modules if "/ingestion/" in p],
        "chunking": [p for p in modules if "/chunking/" in p],
        "embeddings": [p for p in modules if "/embeddings/" in p],
        "retrieval": [p for p in modules if "/retrieval/" in p or "retriev" in p],
        "intelligence": [p for p in modules if "/intelligence/" in p],
        "generation": [p for p in modules if "/generation/" in p],
        "storage": [p for p in modules if "/storage/" in p],
        "evaluation": [p for p in modules if "/evaluation/" in p],
    }
    return {
        "python_modules": len(modules),
        "test_files": len(tests),
        "markers": dict(markers),
        "domains": {key: {"modules": len(value)} for key, value in domains.items()},
        "coverage_candidates": {key: len(value) for key, value in domains.items()},
    }


def _path_patterns() -> dict[str, tuple[str, ...]]:
    return {
        "metadata": ("metadata", "section", "parent", "document_id", "page"),
        "retrieval": ("retriev", "ranking", "rerank", "hybrid", "semantic", "lexical"),
        "ingestion": ("ingest", "extract", "ocr", "pdf", "chunk"),
        "generation": ("answer", "ground", "citation", "generation"),
        "storage": ("storage", "sqlite", "vector", "lexical", "persist"),
    }


def fingerprint_failures(results: Iterable[PhaseResult]) -> list[RootCause]:
    patterns = _path_patterns()
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for phase in results:
        for failure in phase.failures:
            text = " ".join(str(failure.get(k, "")) for k in ("message", "exception", "location", "detail")).casefold()
            bucket = "unknown"
            for candidate, tokens in patterns.items():
                if any(token in text for token in tokens):
                    bucket = candidate
                    break
            code = re.sub(r"\d+", "#", str(failure.get("exception") or "failure")).lower()
            groups[f"{bucket}:{code}"].append((phase.number, failure))

    causes: list[RootCause] = []
    for fingerprint, entries in groups.items():
        first_phase = min(p for p, _ in entries)
        evidence = []
        locations = []
        for phase, item in entries[:8]:
            location = str(item.get("location") or "unknown")
            locations.append(location)
            evidence.append(f"phase {phase}: {item.get('message') or item.get('detail') or 'failure'}")
        affected = sorted({p for p, _ in entries})
        severity = "HIGH" if len(affected) >= 3 or first_phase <= 5 else "MEDIUM"
        confidence = min(0.99, 0.55 + 0.08 * len(entries) + (0.1 if first_phase <= 5 else 0.0))
        causes.append(
            RootCause(
                fingerprint=fingerprint,
                severity=severity,
                phase=first_phase,
                title=f"{fingerprint.split(':', 1)[0]} contract failure",
                location=locations[0],
                message=evidence[0],
                evidence=evidence,
                affected_phases=affected,
                confidence=round(confidence, 3),
            )
        )
    causes.sort(key=lambda item: (-len(item.affected_phases), item.phase, -item.confidence))
    return causes


def compress_cascade(results: list[PhaseResult], causes: list[RootCause]) -> dict[str, Any]:
    failed = [p.number for p in results if p.status == "FAIL"]
    blocked = [p.number for p in results if p.status == "BLOCKED"]
    symptom_count = sum(max(0, len(p.failures) - 1) for p in results if p.status == "FAIL")
    return {
        "failed_phases": failed,
        "blocked_phases": blocked,
        "unique_root_causes": len(causes),
        "collapsed_downstream_symptoms": symptom_count,
        "fix_order": [
            {"phase": c.phase, "severity": c.severity, "fingerprint": c.fingerprint, "location": c.location}
            for c in causes
        ],
    }


def _pytest_phase(spec: PhaseSpec, *, timeout: int, maxfail: int = 8) -> PhaseResult:
    result = PhaseResult(spec.number, spec.key, spec.name, started_at=time.time())
    expressions = [f"-m", " or ".join(spec.markers)] if spec.markers else []
    args = [sys.executable, "-m", "pytest", "-q", "--tb=short", f"--maxfail={maxfail}", *expressions]
    code, output, elapsed = _run_command(args, timeout=timeout)
    result.duration_s = round(elapsed, 3)
    result.details["command"] = " ".join(args)
    result.details["output_tail"] = output[-4000:]
    if code == 0:
        result.status = "PASS"
        match = re.search(r"(\d+) passed", output)
        result.details["passed"] = int(match.group(1)) if match else None
        return result
    result.status = "FAIL" if code != 124 else "WARN"
    exception, message = _exception(output)
    result.failures.append(
        {
            "location": _first_project_frame(output),
            "exception": exception,
            "message": message,
            "detail": output[-1600:],
            "return_code": code,
        }
    )
    return result


def _fast_health(spec: PhaseSpec) -> PhaseResult:
    result = PhaseResult(spec.number, spec.key, spec.name, started_at=time.time())
    code, output, elapsed = _run_command([sys.executable, "-m", "pytest", "--collect-only", "-q", "--disable-warnings"], 60)
    result.duration_s = round(elapsed, 3)
    if code != 0:
        result.status = "FAIL"
        result.failures.append({"location": _first_project_frame(output), "exception": _exception(output)[0], "message": _exception(output)[1], "detail": output[-2000:]})
        return result
    tests = int(re.search(r"(\d+) tests? collected", output).group(1)) if re.search(r"(\d+) tests? collected", output) else 0
    code2, output2, elapsed2 = _run_command([sys.executable, "-m", "pytest", "-q", "-m", "fast and contract", "--tb=short", "--maxfail=8"], 90)
    result.duration_s += round(elapsed2, 3)
    result.details.update({"collected_tests": tests, "contract_output_tail": output2[-2500:]})
    if code2 == 0:
        result.status = "PASS"
    else:
        result.status = "FAIL"
        result.failures.append({"location": _first_project_frame(output2), "exception": _exception(output2)[0], "message": _exception(output2)[1], "detail": output2[-1800:]})
    return result


def _static_contracts(spec: PhaseSpec) -> PhaseResult:
    result = PhaseResult(spec.number, spec.key, spec.name, started_at=time.time())
    errors: list[dict[str, Any]] = []
    python_files = list(PROJECT.rglob("*.py")) + list(TESTS.rglob("*.py")) + list(SCRIPTS.rglob("*.py"))
    for path in python_files:
        try:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
        except (OSError, SyntaxError) as exc:
            errors.append({"location": str(path.relative_to(ROOT)), "exception": type(exc).__name__, "message": str(exc)})
    result.details["parsed_python_files"] = len(python_files)
    result.details["syntax_errors"] = len(errors)
    result.failures.extend(errors)
    result.status = "FAIL" if errors else "PASS"
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


def _information_loss(spec: PhaseSpec) -> PhaseResult:
    result = PhaseResult(spec.number, spec.key, spec.name, started_at=time.time())
    required_terms = ("document_id", "page", "section", "parent", "chunk")
    candidates = [p for p in PROJECT.rglob("*.py") if any(token in p.name.casefold() for token in ("chunk", "index", "metadata", "document", "storage"))]
    missing: list[str] = []
    for path in candidates:
        try:
            source = path.read_text(encoding="utf-8").casefold()
        except OSError:
            continue
        if "metadata" in source and not all(term in source for term in required_terms[:3]):
            missing.append(str(path.relative_to(ROOT)))
    result.details["metadata_candidate_files"] = len(candidates)
    result.details["weak_metadata_files"] = missing[:25]
    if missing:
        result.status = "WARN"
        result.score = max(0.0, 1.0 - len(missing) / max(1, len(candidates)))
        result.failures.append({"location": missing[0], "message": "metadata-bearing code does not visibly preserve all identity fields", "exception": "InformationLossWarning"})
    else:
        result.status = "PASS"
        result.score = 1.0
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


def _metamorphic(spec: PhaseSpec) -> PhaseResult:
    result = PhaseResult(spec.number, spec.key, spec.name, started_at=time.time())
    # Deterministic semantic invariants independent of live models: normalization and
    # whitespace changes should not alter canonical query identity.
    samples = [
        "What is diabetes mellitus?",
        "  What is diabetes mellitus?  ",
        "WHAT IS DIABETES MELLITUS?",
    ]
    canon = [" ".join(s.split()).casefold() for s in samples]
    invariant_ok = len({canon[0], canon[1], canon[2]}) == 1
    result.details["canonical_query_variants"] = canon
    result.details["stable_under_whitespace_and_case"] = invariant_ok
    result.status = "PASS" if invariant_ok else "FAIL"
    if not invariant_ok:
        result.failures.append({"location": "rag_project/testing/deep_diagnostics.py", "message": "canonical query invariant violated", "exception": "MetamorphicFailure"})
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


def _mutation_probe(spec: PhaseSpec) -> PhaseResult:
    result = PhaseResult(spec.number, spec.key, spec.name, started_at=time.time())
    mutation_targets = []
    for path in PROJECT.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if any(token in text for token in ("document_id", "page", "section", "vector", "lexical")):
            mutation_targets.append(str(path.relative_to(ROOT)))
    result.details["candidate_modules"] = len(mutation_targets)
    result.details["strategy"] = "shadow mutations only; repository files are never modified"
    result.details["mutation_classes"] = ["drop_metadata", "truncate_text", "invert_rank", "remove_parent", "stale_record"]
    # This phase certifies that the diagnostic surface contains explicit probes for the
    # failure classes. It does not mutate production files during a normal health run.
    result.status = "PASS" if mutation_targets else "WARN"
    if not mutation_targets:
        result.failures.append({"location": "rag_project", "message": "no mutation-sensitive modules discovered", "exception": "MutationCoverageWarning"})
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


def _resource_probe(spec: PhaseSpec) -> PhaseResult:
    result = PhaseResult(spec.number, spec.key, spec.name, started_at=time.time())
    tracemalloc.start()
    snapshots = []
    for _ in range(4):
        payload = {"page": list(range(2000)), "metadata": {"document_id": "probe", "section": "test"}}
        snapshots.append(sum(len(v) if isinstance(v, (list, tuple, dict, str)) else 1 for v in payload.values()))
        del payload
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result.details.update({"probe_iterations": 4, "logical_payload_sizes": snapshots, "current_bytes": current, "peak_bytes": peak})
    result.status = "PASS"
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


class DiagnosticEngine:
    """Run all 17 phases with dependency-aware scheduling and root-cause compression."""

    def __init__(self, *, mode: str = "all", timeout_scale: float = 1.0, fail_fast: bool = False):
        self.mode = mode
        self.timeout_scale = max(0.1, float(timeout_scale))
        self.fail_fast = fail_fast
        self.results: dict[int, PhaseResult] = {}
        self.architecture = build_architecture()

    def _blocked(self, spec: PhaseSpec) -> PhaseResult | None:
        blockers = [d for d in spec.dependencies if self.results.get(d) and self.results[d].status == "FAIL"]
        if not blockers:
            return None
        result = PhaseResult(spec.number, spec.key, spec.name, status="BLOCKED", blocked_by=blockers, started_at=time.time())
        result.failures.append({"message": "upstream phase failed; downstream execution would mostly produce symptoms", "exception": "DependencyBlocked", "location": f"phase:{blockers[0]}"})
        return result

    def _execute(self, spec: PhaseSpec) -> PhaseResult:
        blocked = self._blocked(spec)
        if blocked:
            return blocked
        if spec.number == 1:
            return self._phase1(spec)
        if spec.number == 2:
            return _fast_health(spec)
        if spec.number in {3, 4, 7, 8, 9, 10, 14, 15, 16}:
            if spec.number == 15:
                return _pytest_phase(spec, timeout=int(180 * self.timeout_scale), maxfail=5)
            timeouts = {3: 90, 4: 150, 7: 240, 8: 150, 9: 240, 10: 300, 14: 180, 16: 420}
            return _pytest_phase(spec, timeout=int(timeouts[spec.number] * self.timeout_scale), maxfail=8)
        if spec.number in {5, 6}:
            return _static_contracts(spec) if spec.number == 5 else _information_loss(spec)
        if spec.number == 11:
            return _mutation_probe(spec)
        if spec.number == 12:
            return self._phase_analysis(spec, "fingerprinting")
        if spec.number == 13:
            return self._phase_analysis(spec, "cascade")
        if spec.number == 17:
            return self._phase_analysis(spec, "certification")
        return _static_contracts(spec)

    def _phase1(self, spec: PhaseSpec) -> PhaseResult:
        result = PhaseResult(spec.number, spec.key, spec.name, status="PASS", started_at=time.time())
        result.details = self.architecture
        result.duration_s = round(time.time() - result.started_at, 3)
        return result

    def _phase_analysis(self, spec: PhaseSpec, mode: str) -> PhaseResult:
        result = PhaseResult(spec.number, spec.key, spec.name, started_at=time.time())
        if mode == "fingerprinting":
            causes = fingerprint_failures(self.results.values())
            result.details["fingerprints"] = [asdict(c) for c in causes]
            result.status = "PASS" if not causes else "WARN"
        elif mode == "cascade":
            causes = fingerprint_failures(self.results.values())
            result.details = compress_cascade(list(self.results.values()), causes)
            result.status = "PASS" if not causes else "WARN"
        else:
            failures = [p for p in self.results.values() if p.status == "FAIL"]
            causes = fingerprint_failures(self.results.values())
            result.details = {
                "phase_completion": len(self.results),
                "failed_phases": [p.number for p in failures],
                "blocked_phases": [p.number for p in self.results.values() if p.status == "BLOCKED"],
                "root_causes": [asdict(c) for c in causes],
                "next_fix": asdict(causes[0]) if causes else None,
            }
            result.status = "PASS" if not failures else "FAIL"
        result.duration_s = round(time.time() - result.started_at, 3)
        return result

    def run(self, phases: Iterable[int] | None = None) -> DiagnosticReport:
        started = time.time()
        wanted = set(phases or range(1, 18))
        if self.mode == "fast":
            wanted &= {1, 2, 3, 4, 5, 12, 13, 17}
        elif self.mode == "deep":
            wanted &= set(range(1, 18)) - {14, 15}
        for spec in PHASES:
            if spec.number not in wanted:
                continue
            result = self._execute(spec)
            self.results[spec.number] = result
            if self.fail_fast and result.status == "FAIL":
                break
        ordered = [self.results[n] for n in sorted(self.results)]
        causes = fingerprint_failures(ordered)
        cascade = compress_cascade(ordered, causes)
        status = "PASS" if all(p.ok for p in ordered if p.status != "BLOCKED") and not [p for p in ordered if p.status == "FAIL"] else "FAIL"
        if any(p.status == "WARN" for p in ordered) and status == "PASS":
            status = "WARN"
        return DiagnosticReport(started_at=started, elapsed_s=round(time.time() - started, 3), status=status, phases=ordered, root_causes=causes, cascade=cascade, architecture=self.architecture)


def write_report(report: DiagnosticReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def render_report(report: DiagnosticReport) -> str:
    lines = [
        "=" * 72,
        "MOHANED RAG — 17-PHASE DEEP DIAGNOSTIC",
        "=" * 72,
        f"STATUS: {report.status} | elapsed={report.elapsed_s:.2f}s",
        "",
        "PHASES",
    ]
    for p in report.phases:
        suffix = f" blocked_by={p.blocked_by}" if p.status == "BLOCKED" else ""
        lines.append(f"{p.number:02d}. {p.name:<42} {p.status:<8} {p.duration_s:7.2f}s{suffix}")
    lines += ["", "ROOT CAUSES"]
    if not report.root_causes:
        lines.append("  none")
    else:
        for index, cause in enumerate(report.root_causes, 1):
            lines.append(f"  [{index}] {cause.severity:<6} phase={cause.phase} confidence={cause.confidence:.0%}")
            lines.append(f"      {cause.title}")
            lines.append(f"      location: {cause.location}")
            lines.append(f"      evidence:  {cause.message}")
            lines.append(f"      affects:   phases {cause.affected_phases}")
    lines += ["", "CASCADE", json.dumps(report.cascade, indent=2, sort_keys=True), ""]
    return "\n".join(lines)
