"""Fast, deterministic diagnostics for the RAG test/runtime stack."""
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
    flags = tuple(sorted(k for k, v in vars(obj).items() if k.startswith("_runtime_") and v)) if hasattr(obj, "__dict__") else ()
    return Snapshot(id(obj) if callable(obj) else None, getattr(obj, "__module__", type(obj).__module__), getattr(obj, "__qualname__", repr(obj)), str(source), line, signature, flags)


def _format_snapshot(s: Snapshot) -> str:
    return f"{s.module}.{s.qualname} | {s.source}:{s.line or '?'} | signature={s.signature} | flags={','.join(s.flags) or '-'}"


def _semantic_probe(spec: SymbolSpec) -> tuple[bool, str]:
    try:
        obj = getattr(import_module(spec.module), spec.attribute)
        if spec.expected == "clean-contextual-text":
            result = obj("What about this?", [("What is diabetes?", "Diabetes is a metabolic disease.")])
            ok = "What is diabetes?" in result and "Follow-up:" not in str(result)
            return ok, "clean contextual text" if ok else "legacy protocol prefix leaked"
        if spec.expected == "structured-dict":
            result = obj("Dose is 600 mg.", "The recommended dose is 500 mg.")
            ok = isinstance(result, dict) and result.get("checked") is True and result.get("mismatch") is True
            return ok, "structured numeric diagnostics" if ok else f"returned {type(result).__name__}: {result!r}"
        if spec.expected == "4-arg-entrypoint":
            signature = inspect.signature(obj)
            ok = len(signature.parameters) >= 4
            return ok, "4-argument entrypoint" if ok else f"signature={signature}"
    except Exception as exc:
        return False, f"probe raised {type(exc).__name__}: {exc}"
    return False, "unknown contract"


def _contract_status(snapshot: Snapshot, spec: SymbolSpec) -> tuple[bool, str]:
    semantic_ok, semantic_reason = _semantic_probe(spec)
    if not semantic_ok:
        return False, f"BAD: {semantic_reason}"
    source = snapshot.source.casefold()
    if spec.expected in {"clean-contextual-text", "structured-dict"} and ("runtime_final_contracts_v" in source or "runtime_contract_compat.py" in source):
        return False, "BAD: runtime compatibility layer owns authoritative public contract"
    return True, f"OK: {semantic_reason}"


def _load_installers() -> tuple[Callable[[], None], ...]:
    return import_module("rag_project.runtime")._load_installers()


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
    matches: list[tuple[str, int, str, str]] = []
    for path in (ROOT / "rag_project").rglob("*.py"):
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
                    matches.append((str(path.relative_to(ROOT)), getattr(node, "lineno", 0), chain, TARGET_ASSIGNMENTS[chain].expected))
    for path, line, chain, expected in sorted(matches):
        print(f"  {chain:<48} {path}:{line}  expected={expected}")
    print(f"\nFound {len(matches)} direct runtime assignments. Static assignment != automatic defect.")
    return 0


def audit_runtime() -> int:
    print("\n=== RAG TEST DOCTOR: RUNTIME PROVENANCE ===")
    installers = _load_installers()
    current = {spec.label: _snapshot(spec) for spec in HOT_SYMBOLS}
    first_mutation: dict[str, tuple[int, str, Snapshot, Snapshot]] = {}
    first_bad: dict[str, tuple[int, str, Snapshot, str]] = {}
    final_bad: dict[str, tuple[int, str, Snapshot, str]] = {}
    transitions: dict[str, list[tuple[int, str, Snapshot, bool, str]]] = {spec.label: [] for spec in HOT_SYMBOLS}

    print(f"Repository root: {ROOT}")
    print(f"Installer count: {len(installers)}")
    for spec in HOT_SYMBOLS:
        snap = current[spec.label]
        ok, reason = _contract_status(snap, spec)
        print(f"BASELINE [{spec.label}] {'PASS' if ok else 'FAIL'} expected={spec.expected}")
        print(f"  {_format_snapshot(snap)}")
        print(f"  {reason}")

    for index, installer in enumerate(installers, 1):
        name = f"{getattr(installer, '__module__', '?')}.{getattr(installer, '__name__', repr(installer))}"
        try:
            installer()
        except Exception as exc:
            print(f"  [INSTALLER ERROR {index}] {name}: {type(exc).__name__}: {exc}")
            continue
        for spec in HOT_SYMBOLS:
            after = _snapshot(spec)
            before = current[spec.label]
            changed = after.identity != before.identity or after.source != before.source or after.line != before.line or after.signature != before.signature
            if not changed:
                continue
            ok, reason = _contract_status(after, spec)
            transitions[spec.label].append((index, name, after, ok, reason))
            if spec.label not in first_mutation:
                first_mutation[spec.label] = (index, name, before, after)
            if not ok:
                final_bad[spec.label] = (index, name, after, reason)
                if spec.label not in first_bad:
                    first_bad[spec.label] = (index, name, after, reason)
                    print(f"\nFIRST BAD TRANSITION [{spec.label}]")
                    print(f"  installer #{index}: {name}")
                    print(f"  owner: {_format_snapshot(after)}")
                    print(f"  {reason}")
            current[spec.label] = after

    print("\n=== EXACT TRANSITION CHAINS ===")
    for spec in HOT_SYMBOLS:
        print(f"\n[{spec.label}]")
        for index, name, snap, ok, reason in transitions[spec.label]:
            print(f"  #{index:<2} {'PASS' if ok else 'FAIL':4} {name}")
            print(f"      {_format_snapshot(snap)}")
            if not ok:
                print(f"      {reason}")
        if not transitions[spec.label]:
            print("  no installer mutation")

    print("\n=== ROOT-CAUSE MAP ===")
    for spec in HOT_SYMBOLS:
        mutation = first_mutation.get(spec.label)
        bad = first_bad.get(spec.label)
        final = final_bad.get(spec.label)
        print(f"\n[{spec.label}]")
        print(f"  first mutation: {mutation[1] if mutation else 'none'}")
        print(f"  first bad owner: {bad[1] if bad else 'none'}")
        if bad:
            print(f"      {_format_snapshot(bad[2])}")
            print(f"      {bad[3]}")
        print(f"  final bad owner: {final[1] if final else 'none'}")
        if final and (not bad or final[0] != bad[0] or final[2].identity != bad[2].identity):
            print(f"      {_format_snapshot(final[2])}")
            print(f"      {final[3]}")
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
    group.add_argument("--audit-runtime", action="store_true", help="trace every hot-symbol transition and report first/final bad owners")
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
    audit_runtime()
    return run_probes(args.maxfail)


if __name__ == "__main__":
    raise SystemExit(main())
