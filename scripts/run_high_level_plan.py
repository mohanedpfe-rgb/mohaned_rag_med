"""Run the exact 13-phase high-level behavior plan and emit a strict JSON report."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PHASES = {
    1: ("01_ingestion", "Ingestion"),
    2: ("02_retrieval", "Retrieval"),
    3: ("03_query_intelligence", "Query Intelligence"),
    4: ("04_answer_paths", "Answer Paths"),
    5: ("05_grounding_safety", "Grounding Safety"),
    6: ("06_latency_performance", "Latency Performance"),
    7: ("07_multilingual", "Multilingual"),
    8: ("08_storage_index", "Storage Index"),
    9: ("09_conversation", "Conversation"),
    10: ("10_security_privacy", "Security Privacy"),
    11: ("11_resilience", "Resilience"),
    12: ("12_production_contracts", "Production Contracts"),
    13: ("13_end_to_end", "End to End"),
}
EXPECTED_PHASE_NUMBERS = tuple(PHASES)


def _run_phase(root: Path, number: int, directory: str, *, keyword: str | None, timeout: int) -> dict:
    command = [sys.executable, "-m", "pytest", "-q", f"tests/high_level/{directory}", "-m", "high_level"]
    if keyword:
        command.extend(["-k", keyword])
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=timeout, check=False)
        returncode = completed.returncode
        output = (completed.stdout + "\n" + completed.stderr).strip()
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        output = f"PHASE TIMEOUT after {timeout}s\nstdout={exc.stdout or ''}\nstderr={exc.stderr or ''}".strip()
        timed_out = True
    elapsed = time.perf_counter() - started
    return {
        "number": number,
        "directory": directory,
        "name": PHASES[number][1],
        "status": "PASS" if returncode == 0 else "FAIL",
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": round(elapsed, 3),
        "output_tail": output[-10000:],
    }


def _validate_shape(root: Path) -> dict:
    high_level_root = root / "tests" / "high_level"
    missing = []
    empty = []
    for number in EXPECTED_PHASE_NUMBERS:
        directory = PHASES[number][0]
        phase_root = high_level_root / directory
        if not phase_root.is_dir():
            missing.append(directory)
            continue
        if not any(phase_root.rglob("test_*.py")):
            empty.append(directory)
    return {
        "expected_phase_count": len(EXPECTED_PHASE_NUMBERS),
        "expected_phase_numbers": list(EXPECTED_PHASE_NUMBERS),
        "missing_directories": missing,
        "empty_directories": empty,
        "shape_valid": not missing and not empty,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, action="append", choices=sorted(PHASES), help="Run only selected phase(s).")
    parser.add_argument("--keyword", help="Optional pytest -k expression applied to every selected phase.")
    parser.add_argument("--timeout", type=int, default=900, help="Per-phase subprocess timeout in seconds.")
    parser.add_argument("--json", type=Path, help="Write the machine-readable report to this path.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed phase.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    selected = args.phase or list(PHASES)
    shape = _validate_shape(root)
    report = {
        "plan": "tests/high_level exact 13-phase plan",
        "shape": shape,
        "requested_phase_count": len(selected),
        "phases": [],
        "status": "PASS" if shape["shape_valid"] else "FAIL",
    }
    if not shape["shape_valid"]:
        print(json.dumps(shape, indent=2))
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return 2

    for number in selected:
        directory, _ = PHASES[number]
        result = _run_phase(root, number, directory, keyword=args.keyword, timeout=args.timeout)
        report["phases"].append(result)
        print(f"[{number:02d}] {result['name']}: {result['status']} ({result['elapsed_seconds']}s)")
        if result["status"] != "PASS":
            report["status"] = "FAIL"
            if args.fail_fast:
                break

    phase_numbers = [item["number"] for item in report["phases"]]
    report["all_requested_phases_executed"] = phase_numbers == selected
    report["all_13_phases_executed"] = phase_numbers == list(EXPECTED_PHASE_NUMBERS) if not args.phase else False
    report["failed_phases"] = [item for item in report["phases"] if item["status"] != "PASS"]
    report["certified"] = bool(report["status"] == "PASS" and report["all_requested_phases_executed"] and shape["shape_valid"])

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["certified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
