"""Intelligent, fast test orchestration and failure localization for the RAG system.

The doctor is deliberately independent from pytest's full suite runtime. It performs
cheap checks first, traces runtime mutation provenance, runs focused contract/storage
checks, and finally parses any pytest failures to the first project-owned traceback
frame. The goal is to answer: *what failed, where, and which layer caused it?*
"""
from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class HotSymbol:
    label: str
    module: str
    attribute: str
    contract: str


HOT_SYMBOLS = (
    HotSymbol("public follow-up", "rag_project.intelligence.top_level_pipeline", "rewrite_follow_up", "clean-follow-up"),
    HotSymbol("integrity follow-up", "rag_project.intelligence.pipeline_integrity", "safe_rewrite_follow_up", "clean-follow-up"),
    HotSymbol("numeric consistency", "rag_project.intelligence.evidence_guard", "numeric_consistency", "structured-numeric"),
    HotSymbol("god-mode enhancer", "rag_project.intelligence.god_mode_100", "enhance_result", "four-argument"),
)

FOCUSED_TESTS = (
    "tests/diagnostics/test_runtime_contract_integrity.py",
    "tests/diagnostics/test_storage_contract_integrity.py",
    "tests/test_pipeline_integrity.py::test_real_followup_is_rewritten_without_protocol_metadata",
    "tests/test_full_44_intelligence.py::test_numeric_consistency_rejects_unsupported_measurement",
    "tests/test_answer_system_orchestration_deep.py::test_enhance_result_accepts_grounded_final_answer[Diabetes is chronic.]",
    "tests/test_hardening.py::test_modified_document_replaces_old_indexed_content",
    "tests/test_index_consistency.py::test_lexical_index_is_persistent_and_returns_matching_chunks",
    "tests/test_quality_gate.py::test_repaired_lexical_content_matches_authoritative_vector_record",
)


@dataclass(frozen=True)
class Snapshot:
    identity: int | None
    module: str
    qualname: str
    source: str
    line: int | None
    signature: str
    flags: tuple[str, ...]


def snapshot(symbol: HotSymbol) -> Snapshot:
    module = importlib.import_module(symbol.module)
    obj = getattr(module, symbol.attribute)
    source = "<unknown>"
    line: int | None = None
    try:
        source = inspect.getsourcefile(obj) or inspect.getfile(obj)
        _, line = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        pass
    try:
        signature = str(inspect.signature(obj))
    except (TypeError, ValueError):
        signature = "<unavailable>"
    flags = tuple(
        sorted(
            key
            for key, value in vars(obj).items()
            if key.startswith("_runtime_") and value
        )
    ) if hasattr(obj, "__dict__") else ()
    return Snapshot(
        identity=id(obj) if callable(obj) else None,
        module=getattr(obj, "__module__", type(obj).__module__),
        qualname=getattr(obj, "__qualname__", repr(obj)),
        source=str(source),
        line=line,
        signature=signature,
        flags=flags,
    )


def format_snapshot(item: Snapshot) -> str:
    return (
        f"{item.module}.{item.qualname} | {item.source}:{item.line or '?'} | "
        f"signature={item.signature} | flags={','.join(item.flags) or '-'}"
    )


def semantic_contract(symbol: HotSymbol, obj) -> tuple[bool, str]:
    try:
        if symbol.contract == "clean-follow-up":
            result = str(obj(
                "What about this?",
                [("What is diabetes?", "Diabetes mellitus is a metabolic disease.")],
            ))
            ok = "What is diabetes?" in result and "Follow-up:" not in result
            return ok, "clean contextual text" if ok else f"legacy protocol leaked: {result!r}"
        if symbol.contract == "structured-numeric":
            result = obj("Dose is 600 mg.", "The recommended dose is 500 mg.")
            ok = isinstance(result, dict) and result.get("checked") is True and result.get("mismatch") is True
            return ok, "structured diagnostics" if ok else f"returned {type(result).__name__}: {result!r}"
        if symbol.contract == "four-argument":
            count = len(inspect.signature(obj).parameters)
            return count >= 4, f"{count}-argument callable" if count >= 4 else f"only {count} parameters"
    except Exception as exc:
        return False, f"probe raised {type(exc).__name__}: {exc}"
    return False, "unknown contract"


def contract_ok(symbol: HotSymbol, snap: Snapshot) -> tuple[bool, str]:
    obj = getattr(importlib.import_module(symbol.module), symbol.attribute)
    semantic, reason = semantic_contract(symbol, obj)
    if not semantic:
        return False, reason
    source = snap.source.casefold()
    if symbol.contract != "four-argument" and (
        "runtime_final_contracts_v" in source or "runtime_contract_compat.py" in source
    ):
        return False, "authoritative contract is owned by a runtime adapter"
    return True, reason


def installer_list() -> tuple:
    runtime = importlib.import_module("rag_project.runtime")
    return tuple(runtime._load_installers())


def attribute_chain(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def static_runtime_map(verbose: bool = True) -> dict[str, list[tuple[str, int]]]:
    wanted = {
        "top_level_pipeline.rewrite_follow_up",
        "pipeline_integrity.safe_rewrite_follow_up",
        "evidence_guard.numeric_consistency",
        "god_mode_100.enhance_result",
    }
    found: dict[str, list[tuple[str, int]]] = {name: [] for name in wanted}
    for path in (ROOT / "rag_project").rglob("*.py"):
        if not (path.name.startswith("runtime") or "contract" in path.name):
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
                chain = attribute_chain(target)
                if chain in wanted:
                    found[chain].append((str(path.relative_to(ROOT)), int(getattr(node, "lineno", 0))))
    if verbose:
        print("\n=== STATIC MUTATION MAP ===")
        total = 0
        for key in sorted(found):
            rows = found[key]
            total += len(rows)
            print(f"\n{key}")
            if not rows:
                print("  none")
            for path, line in rows:
                print(f"  {path}:{line}")
        print(f"\nDirect hot-symbol assignments: {total}")
    return found


def audit_runtime() -> bool:
    print("\n=== RUNTIME PROVENANCE AUDIT ===")
    installers = installer_list()
    current = {item.label: snapshot(item) for item in HOT_SYMBOLS}
    bad_owner: dict[str, tuple[int, str, Snapshot, str]] = {}
    transitions: dict[str, list[tuple[int, str, Snapshot, bool, str]]] = {item.label: [] for item in HOT_SYMBOLS}

    print(f"Installers: {len(installers)}")
    for item in HOT_SYMBOLS:
        ok, reason = contract_ok(item, current[item.label])
        print(f"BASELINE {item.label}: {'PASS' if ok else 'FAIL'}")
        print(f"  {format_snapshot(current[item.label])}")
        print(f"  {reason}")

    for index, installer in enumerate(installers, 1):
        name = f"{getattr(installer, '__module__', '?')}.{getattr(installer, '__name__', repr(installer))}"
        try:
            installer()
        except Exception as exc:
            print(f"INSTALLER ERROR #{index} {name}: {type(exc).__name__}: {exc}")
            continue
        for item in HOT_SYMBOLS:
            after = snapshot(item)
            before = current[item.label]
            changed = (
                after.identity != before.identity
                or after.source != before.source
                or after.line != before.line
                or after.signature != before.signature
            )
            if not changed:
                continue
            ok, reason = contract_ok(item, after)
            transitions[item.label].append((index, name, after, ok, reason))
            if not ok and item.label not in bad_owner:
                bad_owner[item.label] = (index, name, after, reason)
            current[item.label] = after

    print("\n=== FIRST BAD OWNER ===")
    healthy = True
    for item in HOT_SYMBOLS:
        print(f"\n{item.label}")
        for index, name, snap, ok, reason in transitions[item.label]:
            print(f"  #{index:<2} {'PASS' if ok else 'FAIL':4} {name}")
            print(f"      {format_snapshot(snap)}")
            if not ok:
                print(f"      {reason}")
        if item.label in bad_owner:
            healthy = False
            index, name, snap, reason = bad_owner[item.label]
            print(f"  FIRST BAD: installer #{index} -> {name}")
            print(f"             {format_snapshot(snap)}")
            print(f"             {reason}")
        else:
            print("  FIRST BAD: none")
    return healthy


def parse_failure_locations(output: str) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw_line in output.splitlines():
        line = raw_line.strip()
        match = re.search(r"(?P<path>(?:[A-Za-z]:)?[^\s:]+\.py):(?P<line>\d+)(?::\d+)?", line)
        if match:
            normalized = match.group("path").replace("\\", "/")
            if normalized.startswith("rag_project/") or "/rag_project/" in normalized:
                current = {
                    "path": normalized,
                    "line": match.group("line"),
                    "detail": line,
                }
                failures.append(current)
        exc = re.search(r"(?P<etype>[A-Za-z_][\w.]*(?:Error|Exception|Warning)):\s*(?P<message>.*)$", line)
        if exc and current is not None:
            current["exception"] = exc.group("etype")
            current["message"] = exc.group("message")
    return failures


def classify_failure(output: str) -> list[tuple[str, str]]:
    rules = (
        ("runtime-overwrite", r"authoritative contract is owned by a runtime adapter|runtime compatibility layer owns authoritative public contract", "late runtime installer overwrote a canonical public API"),
        ("follow-up-protocol", r"Follow-up:", "legacy follow-up protocol text escaped into a clean public query"),
        ("numeric-return-drift", r"object is not subscriptable|returned bool", "numeric consistency returned the legacy scalar contract"),
        ("signature-drift", r"takes from .* positional arguments but .* given|only \d+ parameters", "public callable signature was narrowed by a compatibility layer"),
        ("stale-version", r"OLD_CONTENT_UNIQUE", "replacement ingestion did not fully retire the previous vector version"),
        ("lexical-persistence", r"\[\] == \['vec-a'\]|received ids=\[\[\]\]", "SQLite lexical mirror was not queryable after reopen"),
        ("lexical-repair", r"NoneType.*subscriptable|row is None", "lexical repair returned healthy state without restoring the authoritative row"),
        ("import-resolution", r"No module named 'rag_project'", "test runner was not launched with repository root on sys.path"),
        ("name-resolution", r"NameError", "runtime patch crossed module namespaces and referenced an undefined helper"),
        ("syntax", r"SyntaxError", "source file could not be parsed"),
    )
    return [(label, hint) for label, pattern, hint in rules if re.search(pattern, output, re.I)]


def run_pytest(label: str, args: list[str], *, timeout: int | None = None) -> tuple[int, str, float]:
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    started = time.perf_counter()
    command = [sys.executable, "-m", "pytest", *args]
    print(f"\n=== {label} ===")
    print("$", " ".join(command))
    try:
        proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, env=env, timeout=timeout)
        elapsed = time.perf_counter() - started
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        print(output)
        print(f"Elapsed: {elapsed:.2f}s")
        return proc.returncode, output, elapsed
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        output = (exc.stdout or "") + "\n" + (exc.stderr or "") + f"\nTIMEOUT after {timeout}s"
        print(output)
        return 124, output, elapsed


def run_static() -> bool:
    print("\n=== STATIC SOURCE HEALTH ===")
    errors: list[str] = []
    python_files = list((ROOT / "rag_project").rglob("*.py")) + list((ROOT / "tests").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py"))
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            errors.append(f"{path.relative_to(ROOT)} -> {type(exc).__name__}: {exc}")
    if errors:
        print("SOURCE ERRORS:")
        for error in errors:
            print(" ", error)
        return False
    print(f"Parsed {len(python_files)} Python files successfully.")
    return True


def run_collection() -> bool:
    code, output, _ = run_pytest("PYTEST COLLECTION HEALTH", ["--collect-only", "-q", "--disable-warnings"], timeout=90)
    if code == 0:
        match = re.search(r"(\d+) tests collected", output)
        print(f"Collected tests: {match.group(1) if match else 'unknown'}")
        return True
    locations = parse_failure_locations(output)
    for item in locations[:10]:
        print(f"  FIRST PROJECT FRAME: {item['path']}:{item['line']}")
    return False


def smart(maxfail: int) -> int:
    started = time.perf_counter()
    print("=== RAG TEST DOCTOR: SMART MODE ===")
    print(f"Repository: {ROOT}")
    print("Strategy: static -> collection -> runtime provenance -> focused contracts -> adaptive analysis")

    ok = run_static()
    static_runtime_map(verbose=True)
    ok = run_collection() and ok
    ok = audit_runtime() and ok

    probe_code, probe_output, _ = run_pytest(
        "FOCUSED CONTRACT + STORAGE PROBES",
        ["-q", "--tb=short", f"--maxfail={maxfail}", *FOCUSED_TESTS],
        timeout=180,
    )
    ok = probe_code == 0 and ok

    findings = classify_failure(probe_output)
    locations = parse_failure_locations(probe_output)
    print("\n=== INTELLIGENT FAILURE MAP ===")
    if findings:
        for label, hint in findings:
            print(f"  [{label}] {hint}")
    else:
        print("  No known failure family matched.")
    if locations:
        print("\n  First project-owned traceback frames:")
        seen: set[tuple[str, str]] = set()
        for location in locations:
            key = (location["path"], location["line"])
            if key in seen:
                continue
            seen.add(key)
            detail = location.get("exception", "failure")
            message = location.get("message", "")
            print(f"    {location['path']}:{location['line']} -> {detail}: {message}")
    else:
        print("  No project-owned traceback frame was recovered from pytest output.")

    contract_code, contract_output, _ = run_pytest(
        "FAST CONTRACT GATE",
        ["-q", "-m", "fast and contract", "--tb=short", "--maxfail=20"],
        timeout=180,
    )
    contract_findings = classify_failure(contract_output)
    if contract_findings:
        print("\n=== CONTRACT FAILURE CLUSTERS ===")
        for label, hint in contract_findings:
            print(f"  [{label}] {hint}")
    ok = contract_code == 0 and ok

    total = time.perf_counter() - started
    print("\n=== TEST DOCTOR RESULT ===")
    print(f"Overall: {'HEALTHY' if ok else 'FAILURES LOCALIZED'}")
    print(f"Diagnostic time: {total:.2f}s")
    print("Next escalation: python scripts/test_matrix.py")
    print("Release gate: python scripts/test_doctor.py --full")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Intelligent RAG test orchestrator and failure localizer")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smart", action="store_true", help="run every fast diagnostic layer and localize failures")
    group.add_argument("--audit-runtime", action="store_true", help="trace runtime installer mutations")
    group.add_argument("--probe", action="store_true", help="run focused contract/storage probes")
    group.add_argument("--contracts", action="store_true", help="run the fast contract gate")
    group.add_argument("--static-map", action="store_true", help="show runtime assignments to hot APIs")
    group.add_argument("--full", action="store_true", help="run the complete pytest suite")
    parser.add_argument("--maxfail", type=int, default=10)
    args = parser.parse_args()

    if args.audit_runtime:
        return 0 if audit_runtime() else 1
    if args.static_map:
        static_runtime_map(verbose=True)
        return 0
    if args.probe:
        return run_pytest(
            "FOCUSED PROBES",
            ["-q", "--tb=short", f"--maxfail={args.maxfail}", *FOCUSED_TESTS],
            timeout=180,
        )[0]
    if args.contracts:
        return run_pytest("FAST CONTRACT GATE", ["-q", "-m", "fast and contract", "--tb=short"], timeout=180)[0]
    if args.full:
        return run_pytest("FULL TEST SUITE", ["-q", "--tb=short", f"--maxfail={args.maxfail}"], timeout=1800)[0]
    return smart(args.maxfail)


if __name__ == "__main__":
    raise SystemExit(main())
