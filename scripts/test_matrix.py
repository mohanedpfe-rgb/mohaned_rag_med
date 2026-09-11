"""Run a fast, lane-aware diagnostic matrix for the RAG test suite.

The matrix distinguishes PASS, FAIL, and EMPTY marker lanes. It also reports
failing nodeids, failure families, and traceback locations so a lane tells you
what broke rather than merely saying that a marker group exited non-zero.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Lane:
    name: str
    expression: str

LANES = (
    Lane("contracts", "fast and contract"),
    Lane("runtime-diagnostics", "fast and diagnostic"),
    Lane("storage", "fast and storage"),
    Lane("intelligence", "fast and intelligence"),
    Lane("ingestion", "fast and ingestion"),
    Lane("regression", "fast and regression"),
)

def _run(command: list[str]) -> tuple[int, str, float]:
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                          env={**os.environ, "PYTHONUNBUFFERED": "1"})
    return proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or ""), time.perf_counter() - started

def _collect_count(expression: str) -> tuple[int, str, float]:
    return _run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", expression])

def _failing_nodeids(output: str) -> list[str]:
    found = []
    for line in output.splitlines():
        match = re.search(r"FAILED\s+([^\s]+)", line.strip())
        if match and match.group(1) not in found:
            found.append(match.group(1))
    return found

def _traceback_locations(output: str) -> list[str]:
    locations = []
    pattern = re.compile(r"(?P<path>(?:[A-Za-z]:)?[^\n]*?(?:tests|rag_project)[/\\][^\n:]+\.py):(?P<line>\d+)")
    for line in output.splitlines():
        match = pattern.search(line.strip())
        if match:
            location = f"{match.group('path')}:{match.group('line')}"
            if location not in locations:
                locations.append(location)
    return locations

def _failure_family(output: str) -> str:
    rules = (
        ("FOLLOW_UP_PROTOCOL_LEAK", "Follow-up:"),
        ("NUMERIC_RETURN_DRIFT", "object is not subscriptable"),
        ("STALE_INDEX", "OLD_CONTENT_UNIQUE"),
        ("LEXICAL_PERSISTENCE", "vec-a"),
        ("LEXICAL_REPAIR", "NoneType' object is not subscriptable"),
        ("IMPORT_FAILURE", "ModuleNotFoundError"),
        ("NAME_RESOLUTION", "NameError"),
        ("COLLECTION_FAILURE", "ERROR collecting"),
    )
    for label, needle in rules:
        if needle in output:
            return label
    return "UNCLASSIFIED"

def run_lane(lane: Lane) -> tuple[Lane, int, int, str, float]:
    collect_code, collect_output, collect_elapsed = _collect_count(lane.expression)
    if collect_code != 0:
        return lane, collect_code, 0, collect_output, collect_elapsed
    match = re.search(r"(\d+)\s+tests?\s+collected", collect_output)
    collected = int(match.group(1)) if match else 0
    if collected == 0:
        return lane, 0, 0, collect_output, collect_elapsed
    code, output, elapsed = _run([
        sys.executable, "-m", "pytest", "-q", "-m", lane.expression,
        "--tb=short", "--maxfail=5",
    ])
    return lane, code, collected, output, collect_elapsed + elapsed

def main() -> int:
    print("=== RAG FAST TEST MATRIX ===")
    print(f"Repository: {ROOT}")
    print(f"Lanes: {len(LANES)}")
    with ThreadPoolExecutor(max_workers=min(3, len(LANES))) as executor:
        futures = [executor.submit(run_lane, lane) for lane in LANES]
        results = [future.result() for future in as_completed(futures)]

    failures = 0
    empty = 0
    for lane, code, collected, output, elapsed in sorted(results, key=lambda item: item[0].name):
        if collected == 0 and code == 0:
            print(f"\n[EMPTY] {lane.name} ({elapsed:.2f}s)")
            print(f"  marker: {lane.expression}")
            print("  No tests carry this marker. This is a test-taxonomy gap, not a product failure.")
            empty += 1
            continue
        status = "PASS" if code == 0 else "FAIL"
        print(f"\n[{status}] {lane.name} ({elapsed:.2f}s, {collected} collected)")
        print(f"  marker: {lane.expression}")
        if code != 0:
            failures += 1
            print(f"  failure family: {_failure_family(output)}")
            for nodeid in _failing_nodeids(output)[:10]:
                print(f"  failing test: {nodeid}")
            for location in _traceback_locations(output)[:10]:
                print(f"  traceback: {location}")

    print("\n=== MATRIX RESULT ===")
    print(f"Failed lanes: {failures}/{len(LANES)}")
    print(f"Empty lanes:  {empty}/{len(LANES)}")
    if empty:
        print("Taxonomy note: add missing markers before treating lane coverage as complete.")
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
