"""Fast, deterministic diagnostics for the RAG test/runtime stack.

The doctor is intentionally independent from the full pytest run. It answers:

* Does the diagnostic tool itself bootstrap correctly?
* Which runtime installer first changes a hot public symbol?
* What source file, line, signature and runtime flag owns that symbol?
* Which runtime files statically assign the hot symbol?
* Which focused regression probe reproduces the contract failure?

Windows examples::

    python scripts/test_doctor.py
    python scripts/test_doctor.py --audit-runtime
    python scripts/test_doctor.py --probe
    python scripts/test_doctor.py --contracts
    python scripts/test_doctor.py --static-map
    python scripts/test_doctor.py --full
"""

from __future__ import annotations

import argparse
import ast
import inspect
import os
import subprocess
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class SymbolSpec:
    label: str
    module: str
    attribute: str
    expected: str


HOT_SYMBOLS = (
    SymbolSpec("public follow-up", "rag_project.intelligence.top_level_pipeline", "rewrite_follow_up", "clean-contextual-text"),
    SymbolSpec("integrity follow-up", "rag_project.intelligence.pipeline_integrity", "safe_rewrite_follow_up", "clean-contextual-text"),
    SymbolSpec("numeric consistency", "rag_project.intelligence.evidence_guard", "numeric_consistency", "structured-dict"),
    SymbolSpec("god-mode enhancer", "rag_project.intelligence.god_mode_100", "enhance_result", "4-arg-entrypoint"),
)

PROBES = (
    ("follow-up public contract", "tests/test_intelligence_adversarial_extra.py::test_top_level_rewrite_follow_up_is_contextual"),
    ("follow-up integrity contract", "tests/test_pipeline_integrity.py::test_real_followup_is_rewritten_without_protocol_metadata"),
    ("numeric structured contract", "tests/test_full_44_intelligence.py::test_numeric_consistency_rejects_unsupported_measurement"),
    ("enhancer call contract", "tests/test_answer_system_orchestration_deep.py::test_enhance_result_accepts_grounded_final_answer[Diabetes is chronic.]"),
    ("document replacement", "tests/test_hardening.py::test_modified_document_replaces_old_indexed_content"),
    ("lexical persistence", "tests/test_index_consistency.py::test_lexical_index_is_persistent_and_returns_matching_chunks"),
    ("repair consistency", "tests/test_quality_gate.py::test_repaired_lexical_content_matches_authoritative_vector_record"),
)

TARGET_ASSIGNMENTS = {
    "top_level_pipeline.rewrite_follow_up": HOT_SYMBOLS[0],
    "pipeline_integrity.safe_rewrite_follow_up": HOT_SYMBOLS[1],
    "evidence_guard.numeric_consistency": HOT_SYMBOLS[2],
    "god_mode_100.enhance_result": HOT_SYMBOLS[3],
}


@dataclass(frozen=True)
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
    flags = ()
    if hasattr(obj, "__dict__"):
        flags = tuple(sorted(k for k, v in vars(obj).items() if k.startswith("_runtime_") and v))
    return Snapshot(
        identity=id(obj) if callable(obj) else None,
        module=getattr(obj, "__module__", type(obj).__module__),
        qualname=getattr(obj, "__qualname__", repr(obj)),
        source=str(source),
        line=line,
        signature=signature,
        flags=flags,
    )


def _format_snapshot(snapshot: Snapshot) -> str:
    return (
        f"{snapshot.module}.{snapshot.qualname} | {snapshot.source}:{snapshot.line or '?'} | "
        f"signature={snapshot.signature} | flags={','.join(snapshot.flags) or '-'}"
    )


def _expected_contract(snapshot: Snapshot, spec: SymbolSpec) -> str:
    source = snapshot.source.casefold()
    signature = snapshot.signature
    if spec.expected == "4-arg-entrypoint":
        return "OK: accepts production entrypoint shape" if signature.count(",") >= 2 else "BAD: enhancer signature drift"
    if spec.expected == "structured-dict":
        return "OK: authoritative evidence_guard owner" if "evidence_guard.py" in source else "BAD: structured API replaced by runtime shim"
    if spec.expected == "clean-contextual-text":
        return "BAD: runtime compatibility layer owns clean public API" if "runtime_final_contracts_v" in source else "OK: clean public implementation"
    return "UNKNOWN CONTRACT"


def _load_installers() -> tuple[Callable[[], None], ...]:
    runtime = import_module("rag_project.runtime")
    return runtime._load_installers()


def _attribute_chain(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def static_mutation_map() -> int:
    print("\n=== RAG TEST DOCTOR: STATIC RUNTIME MUTATION MAP ===")
    runtime_dirs = [ROOT / "rag_project"]
    matches: list[tuple[str, int, str, str]] = []
    for root in runtime_dirs:
        for path in root.rglob("*.py"):
            if not (path.name.startswith("runtime_") or "contract" in path.name or path.name == "runtime.py"):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    chain = _attribute_chain(target)
                    if chain in TARGET_ASSIGNMENTS:
                        spec = TARGET_ASSIGNMENTS[chain]
                        matches.append((str(path.relative_to(ROOT)), getattr(node, "lineno", 0), chain, spec.expected))
    if not matches:
        print("No runtime hot-symbol assignments found.")
        return 0
    for path, line, chain, expected in sorted(matches):
        print(f"  {chain:<48} {path}:{line}  expected={expected}")
    print(f"\nFound {len(matches)} direct runtime assignments. These are the first static locations to inspect when provenance changes.")
    return 0


def audit_runtime() -> int:
    print("\n=== RAG TEST DOCTOR: RUNTIME PROVENANCE ===")
    installers = _load_installers()
    before = {spec.label: _snapshot(spec) for spec in HOT_SYMBOLS}
    current = dict(before)
    first_change: dict[str, tuple[str, Snapshot, Snapshot]] = {}
    installer_count = len(installers)

    print(f"Repository root: {ROOT}")
    print(f"Installer count: {installer_count}")
    print("\nBaseline before installers:")
    for spec in HOT_SYMBOLS:
        print(f"  [{spec.label}] expected={spec.expected}")
        print(f"      {_format_snapshot(before[spec.label])}")

    for index, installer in enumerate(installers, 1):
        name = f"{getattr(installer, '__module__', '?')}.{getattr(installer, '__name__', repr(installer))}"
        try:
            installer()
        except Exception as exc:
            print(f"  [INSTALLER ERROR {index}/{installer_count}] {name}: {type(exc).__name__}: {exc}")
            continue
        for spec in HOT_SYMBOLS:
            after = _snapshot(spec)
            previous = current[spec.label]
            changed = after.identity != previous.identity or after.source != previous.source or after.line != previous.line or after.signature != previous.signature
            if changed and spec.label not in first_change:
                first_change[spec.label] = (name, previous, after)
                print(f"\nFIRST MUTATION: {spec.label}")
                print(f"  installer #{index}: {name}")
                print(f"  before: {_format_snapshot(previous)}")
                print(f"  after : {_format_snapshot(after)}")
                print(f"  diagnosis: {_expected_contract(after, spec)}")
            current[spec.label] = after

    print("\nFinal runtime state:")
    for spec in HOT_SYMBOLS:
        snapshot = current[spec.label]
        print(f"  [{spec.label}] expected={spec.expected}")
        print(f"      {_format_snapshot(snapshot)}")
        print(f"      {_expected_contract(snapshot, spec)}")

    print("\nFIRST-MUTATOR MAP:")
    for spec in HOT_SYMBOLS:
        change = first_change.get(spec.label)
        print(f"  {spec.label}: {change[0] if change else 'no installer mutation'}")
    return 0


def classify_output(text: str) -> list[tuple[str, str]]:
    rules = (
        ("signature drift", "takes from 2 to 3 positional arguments but 4 were given", "god-mode enhancer was replaced by a legacy 2/3-argument callable"),
        ("structured-return drift", "object is not subscriptable", "structured diagnostic API returned a scalar; inspect numeric_consistency provenance"),
        ("follow-up protocol leak", "Follow-up:' not in", "clean follow-up API received the legacy protocol marker"),
        ("stale-index failure", "OLD_CONTENT_UNIQUE", "replacement ingestion left old vector records"),
        ("lexical persistence failure", "assert [] == ['vec-a']", "lexical write/query persistence is broken or tokenisation is not updated"),
        ("repair-record failure", "NoneType' object is not subscriptable", "repair path reports valid state but authoritative lexical row is absent"),
        ("module-resolution failure", "No module named 'rag_project'", "diagnostic process was launched outside the repository import root"),
        ("name-resolution failure", "NameError", "runtime patch references a helper outside its defining module namespace"),
    )
    return [(label, hint) for label, needle, hint in rules if needle in text]


def run_probes(maxfail: int) -> int:
    print("\n=== RAG TEST DOCTOR: FAST FAILURE PROBES ===")
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    command = [sys.executable, "-m", "pytest", "-q", "--tb=short", f"--maxfail={maxfail}", *(nodeid for _, nodeid in PROBES)]
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
    else:
        print("\nNo known failure signature matched the probe output.")
    return proc.returncode


def run_contracts() -> int:
    print("\n=== RAG TEST DOCTOR: FAST CONTRACT GATE ===")
    command = [sys.executable, "-m", "pytest", "-q", "-m", "fast and contract", "--tb=short"]
    print("$", " ".join(command))
    return subprocess.call(command, cwd=ROOT)


def run_full(maxfail: int) -> int:
    print("\n=== RAG TEST DOCTOR: FULL PYTEST ===")
    command = [sys.executable, "-m", "pytest", "-q", "--tb=short", f"--maxfail={maxfail}"]
    print("$", " ".join(command))
    return subprocess.call(command, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast RAG runtime/test diagnosis")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--audit-runtime", action="store_true", help="trace the first installer that mutates hot runtime symbols")
    group.add_argument("--probe", action="store_true", help="run focused failure probes")
    group.add_argument("--contracts", action="store_true", help="run the fast contract gate")
    group.add_argument("--static-map", action="store_true", help="list direct runtime assignments to hot symbols")
    group.add_argument("--full", action="store_true", help="run the entire pytest suite")
    parser.add_argument("--maxfail", type=int, default=7, help="pytest maxfail for probe/full modes")
    args = parser.parse_args()

    if args.audit_runtime:
        return audit_runtime()
    if args.probe:
        return run_probes(args.maxfail)
    if args.contracts:
        return run_contracts()
    if args.static_map:
        return static_mutation_map()
    if args.full:
        return run_full(args.maxfail)

    static_mutation_map()
    result = audit_runtime()
    if result:
        return result
    return run_probes(args.maxfail)


if __name__ == "__main__":
    raise SystemExit(main())
