"""Bounded local fast-gate runner for the RAG test suite.

The project has an intentionally layered test architecture.  This command is
for the deterministic local gate only: it excludes tests that are explicitly
marked slow, integration, or requires_ollama.  The complete release suite is
still available through pytest directly/CI and is never misreported as a
120-second local gate.

The runner collects once, partitions by test file to avoid Windows command-line
limits, streams worker output into temporary files, and enforces a hard
wall-clock budget.  A timeout is reported as TIMEOUT, while an actual pytest
failure is reported as TEST_FAILURE with the worker's last output.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARK = "not slow and not integration and not requires_ollama"


def _environment() -> dict[str, str]:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    # Match the repository's deterministic local/CI mode.  Do not require a
    # live Ollama service merely to run the fast gate.
    env.setdefault("DEVICE_MODE", "i5_16gb")
    env.setdefault("OCR_ENABLED", "false")
    env.setdefault("EMBEDDING_TEST_MODE", "true")
    return env


def collect_test_files(marker: str) -> tuple[list[str], int]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "--disable-warnings",
        "-m",
        marker,
    ]
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=_environment(),
        timeout=30,
    )
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"pytest collection failed:\n{output[-12000:]}")

    total_tests = 0
    match = re.search(r"(\d+) tests? collected", output)
    if match:
        total_tests = int(match.group(1))

    # Collecting with -m still prints node ids for the selected tests; reduce
    # those node ids to unique files so Windows never receives thousands of
    # command-line arguments.
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
        raise RuntimeError("pytest collection produced no selected test files")
    return files, total_tests


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    """Terminate a worker and its descendants on Windows without waiting long."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
    else:
        proc.kill()
    try:
        proc.wait(timeout=2)
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
    for index, test_file in enumerate(test_files):
        buckets[index % workers].append(test_file)

    temp_dir = Path(tempfile.mkdtemp(prefix="rag_fast_gate_"))
    processes: dict[int, subprocess.Popen[str]] = {}
    paths: dict[int, Path] = {}

    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        output_path = temp_dir / f"worker_{index}.log"
        output_handle = output_path.open("w", encoding="utf-8", buffering=1)
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "--disable-warnings",
            "-m",
            args.marker,
            *bucket,
        ]
        processes[index] = subprocess.Popen(
            command,
            cwd=ROOT,
            text=True,
            stdout=output_handle,
            stderr=subprocess.STDOUT,
            env=_environment(),
        )
        paths[index] = output_path
        output_handle.close()

    results: dict[int, tuple[str, int | None, float]] = {}
    while processes:
        now = time.perf_counter()
        if now >= deadline:
            for index, proc in list(processes.items()):
                _terminate_process(proc)
                results[index] = ("TIMEOUT", 124, now - started)
                processes.pop(index, None)
            break

        for index, proc in list(processes.items()):
            return_code = proc.poll()
            if return_code is not None:
                output = _read_output(paths[index])
                status = "PASS" if return_code == 0 else "TEST_FAILURE"
                results[index] = (status, int(return_code), now - started)
                processes.pop(index, None)
                print(f"worker {index}: {status}, {len(buckets[index])} files, {now - started:.1f}s")
                if return_code != 0:
                    print(output[-8000:])
                continue

            if now - started >= worker_timeout:
                _terminate_process(proc)
                results[index] = ("TIMEOUT", 124, now - started)
                processes.pop(index, None)
                print(f"worker {index}: TIMEOUT, {len(buckets[index])} files, {now - started:.1f}s")

        if processes:
            # Keep polling frequent enough that the deadline remains genuinely
            # hard; 50ms is plenty for a local development gate.
            sleep_for = min(0.05, max(0.0, deadline - time.perf_counter()))
            if sleep_for:
                time.sleep(sleep_for)

    elapsed = min(time.perf_counter() - started, budget)
    failures = [i for i, (status, _, _) in results.items() if status != "PASS"]
    timeouts = [i for i, (status, _, _) in results.items() if status == "TIMEOUT"]
    test_failures = [i for i, (status, _, _) in results.items() if status == "TEST_FAILURE"]

    print("=== FAST GATE RESULT ===")
    print(f"Elapsed: {elapsed:.2f}s")
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

    if failures:
        if timeouts:
            print("FAIL: fast local gate exceeded its hard time budget")
        elif test_failures:
            print("FAIL: one or more fast-gate workers reported pytest failures")
        return 1

    if time.perf_counter() > deadline + 0.25:
        print("FAIL: wall-clock budget was exceeded")
        return 124

    print("PASS: deterministic fast gate completed within budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
