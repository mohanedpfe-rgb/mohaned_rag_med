"""Bounded local fast-gate runner for the RAG test suite.

The project has an intentionally layered test architecture. This command is
for the deterministic local gate only: it excludes tests that are explicitly
marked slow, integration, requires_ollama, or high_level. The complete release
suite is still available through pytest directly/CI and is never misreported
as a 120-second local gate.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARK = "not slow and not integration and not requires_ollama and not high_level"


def _environment() -> dict[str, str]:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    env.setdefault("DEVICE_MODE", "i5_16gb")
    env.setdefault("OCR_ENABLED", "false")
    env.setdefault("EMBEDDING_TEST_MODE", "true")
    return env


def collect_test_files(marker: str) -> tuple[list[tuple[str, int]], int]:
    command = [sys.executable, "-m", "pytest", "--collect-only", "-q", "--disable-warnings", "-m", marker]
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, env=_environment(), timeout=30)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"pytest collection failed:\n{output[-12000:]}")

    match = re.search(r"(\d+) tests? collected", output)
    total_tests = int(match.group(1)) if match else 0
    counts: dict[str, int] = {}
    for raw in output.splitlines():
        line = raw.strip()
        if not line or "::" not in line or line.startswith("="):
            continue
        path = line.split("::", 1)[0].strip()
        if path and path.endswith(".py"):
            counts[path] = counts.get(path, 0) + 1
    if not counts:
        raise RuntimeError("pytest collection produced no selected test files")
    # Largest files first makes greedy bin-packing substantially more balanced
    # than round-robin file assignment when a few integration-heavy files are
    # much slower than the median test file.
    files = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return files, total_tests


def _terminate_process_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1.5)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        proc.kill()
    except OSError:
        pass


def _kill_all(processes: dict[int, subprocess.Popen[str]]) -> None:
    live = [(index, proc) for index, proc in processes.items() if proc.poll() is None]
    if not live:
        return
    with ThreadPoolExecutor(max_workers=len(live)) as executor:
        futures = [executor.submit(_terminate_process_tree, proc) for _, proc in live]
        for future in futures:
            try:
                future.result(timeout=2.0)
            except Exception:
                pass
    for _, proc in live:
        try:
            proc.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass


def _read_output(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic local pytest gate with a hard time budget")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=110.0, help="Maximum worker runtime in seconds")
    parser.add_argument("--budget", type=float, default=120.0, help="Overall wall-clock budget in seconds")
    parser.add_argument("--marker", default=DEFAULT_MARK, help="Pytest marker expression for the local gate")
    args = parser.parse_args()

    workers = max(1, min(int(args.workers), 8))
    budget = max(30.0, float(args.budget))
    worker_timeout = max(10.0, min(float(args.timeout), budget - 8.0))
    started = time.perf_counter()
    deadline = started + budget

    print("=== BOUNDED FAST TEST GATE ===")
    print(f"Workers: {workers}")
    print(f"Budget: {budget:.0f}s")
    print(f"Marker: {args.marker}")

    try:
        test_files, total_tests = collect_test_files(args.marker)
    except Exception as exc:
        print(f"COLLECTION FAILURE: {type(exc).__name__}: {exc}")
        return 2

    print(f"Selected: {total_tests or '?'} tests across {len(test_files)} files")
    if total_tests == 0:
        print("FAIL: marker selected zero tests")
        return 2

    buckets: list[list[str]] = [[] for _ in range(workers)]
    bucket_loads = [0] * workers
    for test_file, test_count in test_files:
        target = min(range(workers), key=bucket_loads.__getitem__)
        buckets[target].append(test_file)
        bucket_loads[target] += test_count

    temp_dir = Path(tempfile.mkdtemp(prefix="rag_fast_gate_"))
    processes: dict[int, subprocess.Popen[str]] = {}
    paths: dict[int, Path] = {}

    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        output_path = temp_dir / f"worker_{index}.log"
        output_handle = output_path.open("w", encoding="utf-8", buffering=1)
        command = [sys.executable, "-m", "pytest", "-q", "--tb=short", "--disable-warnings", "-m", args.marker, *bucket]
        processes[index] = subprocess.Popen(command, cwd=ROOT, text=True, stdout=output_handle, stderr=subprocess.STDOUT, env=_environment())
        paths[index] = output_path
        output_handle.close()

    results: dict[int, tuple[str, int, float]] = {}
    while processes:
        now = time.perf_counter()
        if now >= deadline:
            _kill_all(processes)
            finished_at = time.perf_counter()
            for index in list(processes):
                results[index] = ("TIMEOUT", 124, finished_at - started)
                processes.pop(index, None)
            break

        for index, proc in list(processes.items()):
            return_code = proc.poll()
            if return_code is not None:
                output = _read_output(paths[index])
                status = "PASS" if return_code == 0 else "TEST_FAILURE"
                results[index] = (status, int(return_code), now - started)
                processes.pop(index, None)
                print(f"worker {index}: {status}, {len(buckets[index])} files, ~{bucket_loads[index]} tests, {now - started:.1f}s")
                if return_code != 0:
                    print(output[-8000:])
                continue

            if now - started >= worker_timeout:
                _terminate_process_tree(proc)
                results[index] = ("TIMEOUT", 124, time.perf_counter() - started)
                processes.pop(index, None)
                print(f"worker {index}: TIMEOUT, {len(buckets[index])} files, ~{bucket_loads[index]} tests, {time.perf_counter() - started:.1f}s")

        if processes:
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(min(0.05, remaining))

    actual_elapsed = time.perf_counter() - started
    failures = [i for i, (status, _, _) in results.items() if status != "PASS"]
    timeouts = [i for i, (status, _, _) in results.items() if status == "TIMEOUT"]
    test_failures = [i for i, (status, _, _) in results.items() if status == "TEST_FAILURE"]

    print("=== FAST GATE RESULT ===")
    print(f"Elapsed: {actual_elapsed:.2f}s")
    print(f"Selected: {total_tests}")
    print(f"Workers completed: {len(results)}/{len(buckets)}")
    print(f"Test-failure workers: {test_failures}")
    print(f"Timeout workers: {timeouts}")

    try:
        for path in temp_dir.glob("*.log"):
            path.unlink(missing_ok=True)
        temp_dir.rmdir()
    except OSError:
        pass

    if actual_elapsed > budget:
        print("FAIL: hard wall-clock budget exceeded")
        return 124
    if failures:
        print("FAIL: fast local gate did not complete cleanly")
        return 1

    print("PASS: deterministic fast gate completed within budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
