"""Fast, deterministic test diagnostics for the RAG system.

This tool is intentionally separate from the full pytest suite. It answers three
questions quickly:

1. Which public runtime symbol changed?
2. Which installer changed it first?
3. Which small probe reproduces the failure?

Usage on Windows:
    python scripts/test_doctor.py
    python scripts/test_doctor.py --audit-runtime
    python scripts/test_doctor.py --probe
    python scripts/test_doctor.py --full
"""

from __future__ import annotations

import argparse
import inspect
import os
import subprocess
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import FunctionType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SymbolSpec:
    label: str
    module: str
    attribute: str


HOT_SYMBOLS = (
    SymbolSpec("public follow-up", "rag_project.intelligence.top_level_pipeline", "rewrite_follow_up"),
    SymbolSpec("integrity follow-up", "rag_project.intelligence.pipeline_integrity", "safe_rewrite_follow_up"),
    SymbolSpec("numeric consistency", "rag_project.intelligence.evidence_guard", "numeric_consistency"),
    SymbolSpec("god-mode enhancer", "rag_project.intelligence.god_mode_100", "enhance_result"),
)

PROBES = (
    ("follow-up public contract", "tests/test_intelligence_adversarial_extra.py::test_top_level_rewrite_follow_up_is_contextual", "follow-up"),
    ("follow-up integrity contract", "tests/test_pipeline_integrity.py::test_real_followup_is_rewritten_without_protocol_metadata", "follow-up"),
    ("numeric structured contract", "tests/test_full_44_intelligence.py::test_numeric_consistency_rejects_unsupported_measurement", "numeric"),
    ("enhancer call contract", "tests/test_answer_system_orchestration_deep.py::test_enhance_result_accepts_grounded_final_answer[Diabetes is chronic.]", "god-mode"),
    ("document replacement", "tests/test_hardening.py::test_modified_document_replaces_old_indexed_content", "storage"),
    ("lexical persistence", "tests/test_index_consistency.py::test_lexical_index_is_persistent_and_returns_matching_chunks", "storage"),
    ("repair consistency", "tests/test_quality_gate.py::test_repaired_lexical_content_matches_authoritative_vector_record", "storage"),
)


@dataclass
class Snapshot:
    identity: int | None
    module: str
    qualname: str
    source: str
    line: int | None
    signature: str
    flags: tuple[str, ...]


def _snapshot(spec: SymbolSpec) -> Snapshot:
    module = import_module(spec.module)
    obj = getattr(module, spec.attribute)
    source = "<unknown>"
    line = None
    try:
        source = inspect.getsourcefile(obj) or inspect.getfile(obj)
        _, line = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        pass
    try:
        signature = str(inspect.signature(obj))
    except (TypeError, ValueError):
        signature = "<unavailable>"
    flags = tuple(sorted(k for k, v in vars(obj).items() if k.startswith("_runtime_") and v)) if hasattr(obj, "__dict__") else ()
    return Snapshot(
        identity=id(obj) if callable(obj) else None,
        module=getattr(obj, "__module__", type(obj).__module__),
        qualname=getattr(obj, "__qualname__", repr(obj)),
        source=str(source),
        line=line,
        signature=signature,
        flags=flags,
    )


def _format_snapshot(s: Snapshot) -> str:
    return (
        f"{s.module}.{s.qualname} | {s.source}:{s.line or '?'} | "
        f"signature={s.signature} | flags={','.join(s.flags) or '-'}"
    )


def audit_runtime() -> int:
    print("\n=== RAG TEST DOCTOR: RUNTIME PROVENANCE ===")
    runtime = import_module("rag_project.runtime")
    installers = runtime._load_installers()
    before = {spec.label: _snapshot(spec) for spec in HOT_SYMBOLS}
    current = dict(before)
    first_change: dict[str, tuple[str, Snapshot, Snapshot]] = {}

    print("Baseline before installers:")
    for spec in HOT_SYMBOLS:
        print(f"  [{spec.label}] {_format_snapshot(before[spec.label])}")

    for installer in installers:
        name = getattr(installer, "__module__", "?") + "." + getattr(installer, "__name__", repr(installer))
        try:
            installer()
        except Exception as exc:  # one broken installer must not hide later provenance
            print(f"  [INSTALLER ERROR] {name}: {type(exc).__name__}: {exc}")
            continue
        for spec in HOT_SYMBOLS:
            after = _snapshot(spec)
            previous = current[spec.label]
            changed = (
                after.identity != previous.identity
                or after.source != previous.source
                or after.line != previous.line
                or after.signature != previous.signature
            )
            if changed and spec.label not in first_change:
                first_change[spec.label] = (name, previous, after)
                print(f"\nFIRST CHANGE: {spec.label}")
                print(f"  installer: {name}")
                print(f"  before:   {_format_snapshot(previous)}")
                print(f"  after:    {_format_snapshot(after)}")
            current[spec.label] = after

    print("\nFinal runtime state:")
    for spec in HOT_SYMBOLS:
        print(f"  [{spec.label}] {_format_snapshot(current[spec.label])}")

    print("\nExact first-mutator map:")
    for spec in HOT_SYMBOLS:
        change = first_change.get(spec.label)
        if change:
            print(f"  {spec.label}: {change[0]}")
        else:
            print(f"  {spec.label}: NOT MUTATED BY INSTALLER STACK")
    return 0


def classify_output(text: str) -> list[tuple[str, str]]:
    rules = (
        ("signature drift", "takes from 2 to 3 positional arguments but 4 were given", "API signature changed; inspect enhancer entrypoint/provenance"),
        ("return-shape drift", "object is not subscriptable", "caller expects structured diagnostics but received a scalar"),
        ("follow-up protocol leak", "Follow-up:' not in", "legacy prefix leaked into a clean public follow-up contract"),
        ("stale index", "OLD_CONTENT_UNIQUE", "document replacement did not remove the old version"),
        ("lexical persistence", "assert [] == ['vec-a']", "lexical index write/query path is disconnected or not persisted"),
        ("repair corruption", "NoneType' object is not subscriptable", "repair routine removed or failed to restore the authoritative lexical record"),
        ("name resolution", "NameError", "runtime patch references a helper that is not in its global namespace"),
    )
    return [(label, hint) for label, needle, hint in rules if needle in text]


def run_probes(maxfail: int) -> int:
    print("\n=== RAG TEST DOCTOR: FAST FAILURE PROBES ===")
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    command = [sys.executable, "-m", "pytest", "-q", "--tb=short", f"--maxfail={maxfail}", *(nodeid for _, nodeid, _ in PROBES)]
    print("$", " ".join(command))
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, env=env)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    findings = classify_output(combined)
    if findings:
        print("\n=== GROUPED ROOT-CAUSE HINTS ===")
        for label, hint in findings:
            print(f"  [{label}] {hint}")
        print("\nRun --audit-runtime next for the first installer that changed the implicated symbol.")
    else:
        print("\nNo known failure signature matched the probe output.")
    return proc.returncode


def run_full(maxfail: int) -> int:
    print("\n=== RAG TEST DOCTOR: FULL PYTEST ===")
    command = [sys.executable, "-m", "pytest", "-q", "--tb=short", f"--maxfail={maxfail}"]
    print("$", " ".join(command))
    return subprocess.call(command, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast RAG runtime/test diagnosis")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--audit-runtime", action="store_true", help="trace the first installer that mutates hot runtime symbols")
    group.add_argument("--probe", action="store_true", help="run the focused failure probes")
    group.add_argument("--full", action="store_true", help="run the entire pytest suite")
    parser.add_argument("--maxfail", type=int, default=7, help="pytest maxfail for probe/full modes")
    args = parser.parse_args()

    if args.audit_runtime:
        return audit_runtime()
    if args.probe:
        return run_probes(args.maxfail)
    if args.full:
        return run_full(args.maxfail)

    audit_runtime()
    return run_probes(args.maxfail)


if __name__ == "__main__":
    raise SystemExit(main())
