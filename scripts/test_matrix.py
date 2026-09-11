"""Run a fast, lane-aware diagnostic matrix for the RAG test suite.

The matrix distinguishes PASS, FAIL, and EMPTY marker lanes. It uses pytest's
actual deselection totals rather than --collect-only totals, then reports failing
nodeids, root-cause families, traceback locations, and per-lane timing.
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
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    return proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or ""), time.perf_counter() - started


def _selected_count(output: str) -> int:
    """Recover the number of tests actually selected by pytest -m.

    Pytest's --collect-only summary counts the whole collection before marker
    deselection, so it cannot be used to decide whether a lane is empty.
    Runtime summaries expose deselected counts; derive selected from the known
    collection total in the same process when possible.
    """
    deselected = 0
    collected = None
    match = re.search(r"(\d+)\s+deselected", output)
    if match:
        deselected = int(match.group(1))
    match = re.search(r"(\d+)\s+tests?\s+collected", output)
    if match:
        collected = int(match.group(1))
    if collected is not None:
        return max(0, collected - deselected)

    # Passing/failing summaries expose selected test counts directly.
    total = 0
    for pattern in (
        r"(\d+)\s+passed",
        r"(\d+)\s+failed",
        r"(\d+)\s+skipped",
        r"(\d+)\s+xfailed",
        r"(\d+)\s+xpassed",
        r"(\d+)\s+errors?",
    ):
        for raw in re.findall(pattern, output):
            total += int(raw)
    return total


def _failing_nodeids(output: str) -> list[str]:
    found: list[str] = []
    for line in output.splitlines():
        match = re.search(r"FAILED\s+([^\s]+)", line.strip())
        if match and match.group(1) not in found:
            found.append(match.group(1))
    return found


def _traceback_locations(output: str) -> list[str]:
    locations: list[str] = []
    pattern = re.compile(
        r"(?P<path>(?:[A-Za-z]:)?[^\n]*?(?:tests|rag_project)[/\\][^\n:]+\.py):(?P<line>\d+)"
    )
    for line in output.splitlines():
        match = pattern.search(line.strip())
        if match:
            location = f"{match.group('path')}:{match.group('line')}"
            if location not in locations:
                locations.append(location)
    return locations


def _failure_families(output: str) -> list[str]:
    rules = (
        ("FOLLOW_UP_PROTOCOL_LEAK", "Follow-up:"),
        ("NUMERIC_RETURN_DRIFT", "object is not subscriptable"),
        ("STALE_INDEX", "OLD_CONTENT_UNIQUE"),
        ("LEXICAL_REPAIR", "NoneType' object is not subscriptable"),
        ("LEXICAL_PERSISTENCE", "vec-a"),
        ("IMPORT_FAILURE", "ModuleNotFoundError"),
        ("NAME_RESOLUTION", "NameError"),
        ("COLLECTION_FAILURE", "ERROR collecting"),
        ("SIGNATURE_DRIFT", "positional arguments but"),
        ("ASSERTION_FAILURE", "AssertionError"),
    )
    return [label for label, needle in rules if needle in output]


def run_lane(lane: Lane) -> tuple[Lane, int, int, str, float]:
    code, output, elapsed = _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-m",
            lane.expression,
            "--tb=short",
            "--maxfail=5",
        ]
    )
    selected = _selected_count(output)
    return lane, code, selected, output, elapsed


def main() -> int:
    print("=== RAG FAST TEST MATRIX ===")
    print(f"Repository: {ROOT}")
    print(f"Lanes: {len(LANES)}")

    with ThreadPoolExecutor(max_workers=min(3, len(LANES))) as executor:
        futures = [executor.submit(run_lane, lane) for lane in LANES]
        results = [future.result() for future in as_completed(futures)]

    failures = 0
    empty = 0
    for lane, code, selected, output, elapsed in sorted(results, key=lambda item: item[0].name):
        if selected == 0:
            status = "EMPTY" if code == 0 else "ERROR"
            print(f"\n[{status}] {lane.name} ({elapsed:.2f}s, 0 selected)")
            print(f"  marker: {lane.expression}")
            if code == 0:
                print("  No tests were selected by this marker. This is a taxonomy gap, not a product failure.")
                empty += 1
            else:
                print("  Pytest could not execute the lane; inspect the collection/import error below.")
                failures += 1
            for location in _traceback_locations(output)[:5]:
                print(f"  traceback: {location}")
            continue

        status = "PASS" if code == 0 else "FAIL"
        print(f"\n[{status}] {lane.name} ({elapsed:.2f}s, {selected} selected)")
        print(f"  marker: {lane.expression}")
        if code != 0:
            failures += 1
            families = _failure_families(output)
            print(f"  failure families: {', '.join(families) if families else 'UNCLASSIFIED'}")
            for nodeid in _failing_nodeids(output)[:10]:
                print(f"  failing test: {nodeid}")
            for location in _traceback_locations(output)[:10]:
                print(f"  traceback: {location}")

    print("\n=== MATRIX RESULT ===")
    print(f"Failed lanes: {failures}/{len(LANES)}")
    print(f"Empty lanes:  {empty}/{len(LANES)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
