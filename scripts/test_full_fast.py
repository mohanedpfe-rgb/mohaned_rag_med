"""Bounded parallel full-suite runner for local development.

Collects the test suite once, partitions it by test file instead of individual
node IDs (avoiding Windows command-line limits), and enforces a hard wall-clock
budget so a slow test cannot make the full gate run indefinitely.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def collect_test_files() -> tuple[list[str], int]:
    command = [sys.executable, "-m", "pytest", "--collect-only", "-q", "--disable-warnings"]
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        timeout=30,
    )
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"pytest collection failed:\n{output[-12000:]}")

    total_tests = 0
    import re

    match = re.search(r"(\d+) tests collected", output)
    if match:
        total_tests = int(match.group(1))

    files: list[str] = []
    seen: set[str] = set()
    for raw in output.splitlines():
        line = raw.strip()
        if not line or "::" not in line or line.startswith("="):
            continue
        path = line.split("::", 1)[0].strip()
        if path and path.endswith(".py") and path not in seen:
            seen.add(path)
            files.append(path)

    if not files:
        raise RuntimeError("pytest collection produced no test files")
    return files, total_tests


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full pytest suite in parallel with a hard time budget")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=105.0, help="Per-worker timeout in seconds")
    parser.add_argument("--budget", type=float, default=120.0, help="Overall wall-clock budget in seconds")
    args = parser.parse_args()

    workers = max(1, min(int(args.workers), 8))
    budget = max(30.0, float(args.budget))
    per_worker_timeout = max(10.0, min(float(args.timeout), budget - 5.0))

    started = time.perf_counter()
    print("=== BOUNDED PARALLEL FULL TEST SUITE ===")
    print(f"Workers: {workers}")
    print(f"Budget: {budget:.0f}s")

    try:
        test_files, total_tests = collect_test_files()
    except Exception as exc:
        print(f"COLLECTION FAILURE: {type(exc).__name__}: {exc}")
        return 2

    print(f"Collected: {total_tests or '?'} tests across {len(test_files)} files")

    buckets: list[list[str]] = [[] for _ in range(workers)]
    for index, test_file in enumerate(test_files):
        buckets[index % workers].append(test_file)

    processes: dict[int, subprocess.Popen[str]] = {}
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "--disable-warnings",
            *bucket,
        ]
        processes[index] = subprocess.Popen(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

    deadline = started + budget
    results: dict[int, tuple[int, str, float]] = {}
    while processes and time.perf_counter() < deadline:
        now = time.perf_counter()
        for index, proc in list(processes.items()):
            return_code = proc.poll()
            if return_code is None:
                if now - started >= per_worker_timeout:
                    proc.kill()
                    output = proc.stdout.read() if proc.stdout else ""
                    results[index] = (124, output + f"\nWORKER {index} TIMEOUT", now - started)
                    processes.pop(index, None)
                continue
            output = proc.stdout.read() if proc.stdout else ""
            results[index] = (int(return_code), output, now - started)
            processes.pop(index, None)
            print(f"worker {index}: code={return_code}, {len(buckets[index])} files, {now - started:.1f}s")
            if return_code != 0:
                print(output[-12000:])
        if processes:
            time.sleep(0.15)

    if processes:
        for index, proc in list(processes.items()):
            proc.kill()
            output = proc.stdout.read() if proc.stdout else ""
            results[index] = (124, output + f"\nWORKER {index} HARD TIMEOUT", time.perf_counter() - started)
            processes.pop(index, None)

    elapsed = time.perf_counter() - started
    failed = [index for index, (code, _, _) in results.items() if code != 0]

    print("=== BOUNDED FULL RESULT ===")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Collected: {total_tests or '?'}")
    print(f"Workers completed: {len(results)}/{len(buckets)}")
    print(f"Failed/timeout workers: {failed}")

    if elapsed > budget + 1.0:
        print("FAIL: overall budget exceeded")
        return 124
    if failed:
        print("FAIL: one or more workers failed or timed out")
        return 1
    print("PASS: full suite completed within budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
