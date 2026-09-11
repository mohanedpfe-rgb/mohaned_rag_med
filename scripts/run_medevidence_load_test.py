"""Pre-human-test synthetic load benchmark.

Runs the deterministic callable benchmark only; it never sends real traffic to
an external service and never needs patient data.
"""
from __future__ import annotations
import argparse
import json
import time
from rag_project.intelligence.production_ops import benchmark_callable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    def synthetic_work(item: int) -> int:
        # Deterministic CPU work approximating orchestration overhead without a real LLM call.
        value = item
        for _ in range(2000):
            value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
        return value

    result = benchmark_callable(synthetic_work, list(range(max(1, args.queries))), workers=args.workers)
    payload = {**result.__dict__, "target_concurrency": 100, "target_qpm": 1000,
               "structural_smoke_only": True, "created": time.time()}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
