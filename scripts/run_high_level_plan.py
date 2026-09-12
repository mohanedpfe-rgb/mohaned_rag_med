"""Run the exact 13-phase high-level behavior plan and emit a JSON report."""
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


def _run_phase(root: Path, number: int, directory: str, *, keyword: str | None, timeout: int) -> dict:
    command = [sys.executable, "-m", "pytest", "-q", f"tests/high_level/{directory}", "-m", "high_level"]
    if keyword:
        command.extend(["-k", keyword])
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=timeout, check=False)
    elapsed = time.perf_counter() - started
    output = (completed.stdout + "\n" + completed.stderr).strip()
    return {
        "number": number,
        "directory": directory,
        "name": PHASES[number][1],
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "returncode": completed.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "output_tail": output[-8000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, action="append", choices=sorted(PHASES), help="Run only selected phase(s).")
    parser.add_argument("--keyword", help="Optional pytest -k expression applied to every selected phase.")
    parser.add_argument("--timeout", type=int, default=900, help="Per-phase subprocess timeout in seconds.")
    parser.add_argument("--json", type=Path, help="Write the machine-readable report to this path.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    selected = args.phase or list(PHASES)
    report = {
        "plan": "tests/high_level 13-phase plan",
        "phases": [],
        "status": "PASS",
    }

    for number in selected:
        directory, _ = PHASES[number]
        result = _run_phase(root, number, directory, keyword=args.keyword, timeout=args.timeout)
        report["phases"].append(result)
        print(f"[{number:02d}] {result['name']}: {result['status']} ({result['elapsed_seconds']}s)")
        if result["status"] != "PASS":
            report["status"] = "FAIL"
            break

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" and len(report["phases"]) == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
