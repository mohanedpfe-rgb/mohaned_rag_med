"""Bounded parallel full-suite runner for local development.

Collects pytest node IDs once, partitions them across a small number of worker
processes, and enforces a hard wall-clock budget so a single slow test cannot
make the full gate run indefinitely.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def collect_nodeids() -> list[str]:
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
    nodeids: list[str] = []
    for raw in output.splitlines():
        line = raw.strip()
        if not line or line.endswith("tests collected"):
            continue
        if line.startswith("=") or line.startswith("warning"):
            continue
        if line.startswith("<"):
            continue
        # Pytest -q --collect-only emits one nodeid per line.
        if "::" in line and not line.startswith("collected"):
            nodeids.append(line)
    if not nodeids:
        raise RuntimeError("pytest collection produced no test node IDs")
    return nodeids


def run_worker(index: int, nodeids: list[str], timeout: float) -> tuple[int, str, float]:
    started = time.perf_counter()
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        "--disable-warnings",
        *nodeids,
    ]
    try:
        proc = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            timeout=timeout,
        )
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return proc.returncode, output, time.perf_counter() - started
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + "\n" + (exc.stderr or "")
        return 124, output + f"\nWORKER {index} TIMEOUT", time.perf_counter() - started


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

    nodeids = collect_nodeids()
    print(f"Collected: {len(nodeids)} tests")
    buckets: list[list[str]] = [[] for _ in range(workers)]
    for index, nodeid in enumerate(nodeids):
        buckets[index % workers].append(nodeid)

    results: dict[int, tuple[int, str, float]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_worker, index, bucket, per_worker_timeout): index
            for index, bucket in enumerate(buckets)
            if bucket
        }
        deadline = started + budget
        for future in as_completed(futures, timeout=max(1.0, budget - (time.perf_counter() - started))):
            index = futures[future]
            results[index] = future.result()
            code, output, elapsed = results[index]
            print(f"worker {index}: code={code}, {len(buckets[index])} tests, {elapsed:.1f}s")
            if code != 0:
                print(output[-12000:])

            if time.perf_counter() >= deadline:
                break

    elapsed = time.perf_counter() - started
    missing = [index for index in range(len(buckets)) if buckets[index] and index not in results]
    failed = [index for index, (code, _, _) in results.items() if code != 0]

    print("=== BOUNDED FULL RESULT ===")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Collected: {len(nodeids)}")
    print(f"Workers completed: {len(results)}/{len(buckets)}")
    print(f"Failed/timeout workers: {failed}")
    print(f"Missing workers: {missing}")

    if elapsed > budget:
        print("FAIL: overall budget exceeded")
        return 124
    if missing:
        print("FAIL: overall budget expired before every worker completed")
        return 124
    if failed:
        print("FAIL: one or more workers failed")
        return 1
    print("PASS: full suite completed within budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
