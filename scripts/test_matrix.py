"""Run the RAG test suite as a small, parallel diagnostic matrix.

Each lane is intentionally independent. A lane failure is reported with its exact
pytest expression, elapsed time, and representative project traceback frame. The
matrix is designed to find the failing subsystem in seconds rather than waiting for
the complete suite.
"""
from __future__ import annotations

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


def run_lane(lane: Lane) -> tuple[Lane, int, str, float]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-m",
        lane.expression,
        "--tb=short",
        "--maxfail=5",
    ]
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    return lane, proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or ""), elapsed


def first_project_frame(output: str) -> str:
    for line in output.splitlines():
        normalized = line.replace("\\", "/")
        if "rag_project/" in normalized and ".py:" in normalized:
            return line.strip()
    return "none"


def main() -> int:
    print("=== RAG FAST TEST MATRIX ===")
    print(f"Repository: {ROOT}")
    print(f"Lanes: {len(LANES)}")
    failures = 0

    with ThreadPoolExecutor(max_workers=min(3, len(LANES))) as executor:
        futures = {executor.submit(run_lane, lane): lane for lane in LANES}
        results = []
        for future in as_completed(futures):
            lane, code, output, elapsed = future.result()
            results.append((lane, code, output, elapsed))

    for lane, code, output, elapsed in sorted(results, key=lambda item: item[0].name):
        status = "PASS" if code == 0 else "FAIL"
        print(f"\n[{status}] {lane.name}  ({elapsed:.2f}s)")
        print(f"  marker: {lane.expression}")
        if code != 0:
            failures += 1
            print(f"  first project frame: {first_project_frame(output)}")
            summary = re.findall(r"FAILED .*|ERROR .*|={3,}.*={3,}", output)
            for line in summary[:8]:
                print(f"  {line.strip()}")

    print("\n=== MATRIX RESULT ===")
    print(f"Failed lanes: {failures}/{len(LANES)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
