"""Command-line entry point for the authoritative 17-phase test doctor.

Examples:
    python scripts/test_doctor.py --fast
    python scripts/test_doctor.py --deep
    python scripts/test_doctor.py --all --json reports/deep_diagnostic.json
    python scripts/test_doctor.py --phase 1 5 9 12 13 17
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_project.testing.deep_diagnostics import render_report, write_report
from rag_project.testing.runner import PHASES, run_all


def _parse_phases(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    phases: list[int] = []
    for value in values:
        for token in value.split(","):
            if not token.strip():
                continue
            number = int(token)
            if number < 1 or number > 17:
                raise argparse.ArgumentTypeError("phase must be between 1 and 17")
            phases.append(number)
    return sorted(set(phases))


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast, deep, dependency-aware RAG test diagnosis")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fast", action="store_true", help="run the fast diagnostic subset")
    group.add_argument("--deep", action="store_true", help="run all 17 phases except explicit performance/resource policy changes")
    group.add_argument("--all", action="store_true", help="run all 17 phases")
    group.add_argument("--phase", nargs="+", metavar="N", help="run selected phase numbers")
    parser.add_argument("--timeout-scale", type=float, default=1.0, help="scale phase timeouts")
    parser.add_argument("--fail-fast", action="store_true", help="stop after the first failed phase")
    parser.add_argument("--json", metavar="PATH", help="write the complete structured report as JSON")
    args = parser.parse_args()

    phases = _parse_phases(args.phase)
    mode = "fast" if args.fast else "deep" if args.deep else "all"
    if phases is not None:
        mode = "all"

    print(f"Authoritative phases: {len(PHASES)}")
    print("Order: " + " → ".join(f"{p.number}:{p.key}" for p in PHASES))

    report = run_all(mode=mode, timeout_scale=args.timeout_scale, fail_fast=args.fail_fast, phases=phases)
    print(render_report(report))

    if args.json:
        output = write_report(report, Path(args.json))
        print(f"JSON report: {output}")

    return 0 if report.status in {"PASS", "WARN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
