"""MedEvidence Pro load validation.

Two modes are supported:
- offline smoke: deterministic local callable benchmark;
- HTTP validation: real requests against the running MedEvidence API.
The HTTP mode is the certification path for the plan's 100+ concurrent / 1000+
queries-per-minute requirement and can persist evidence to an artifact file.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rag_project.intelligence.production_ops import benchmark_callable


def run_offline(queries: int, workers: int) -> dict[str, object]:
    def synthetic_work(item: int) -> int:
        value = item
        for _ in range(2000):
            value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
        return value

    result = benchmark_callable(synthetic_work, list(range(max(1, queries))), workers=workers)
    return {
        **result.__dict__,
        "mode": "offline_smoke",
        "target_concurrency": 100,
        "target_qpm": 1000,
        "certification_eligible": False,
    }


def _request(url: str, query: str, timeout: float) -> tuple[float, int, str | None]:
    body = json.dumps({"query": query}).encode("utf-8")
    req = Request(url.rstrip("/") + "/query", data=body, headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as response:
            response.read()
            return (time.perf_counter() - started) * 1000.0, int(response.status), None
    except HTTPError as exc:
        return (time.perf_counter() - started) * 1000.0, int(exc.code), str(exc)
    except URLError as exc:
        return (time.perf_counter() - started) * 1000.0, 0, str(exc)
    except Exception as exc:
        return (time.perf_counter() - started) * 1000.0, 0, str(exc)


def run_http(url: str, queries: int, concurrency: int, timeout: float) -> dict[str, object]:
    prompts = [
        "What is hypertension?",
        "What is metformin used for?",
        "What are common adverse effects of amoxicillin?",
        "What does HbA1c measure?",
        "What is the mechanism of action of ibuprofen?",
    ]
    inputs = [prompts[i % len(prompts)] for i in range(max(1, queries))]
    started = time.perf_counter()
    timings: list[float] = []
    statuses: list[int] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(_request, url, query, timeout) for query in inputs]
        for future in as_completed(futures):
            latency, status, error = future.result()
            timings.append(latency)
            statuses.append(status)
            if error:
                errors.append(error)
    total = max(time.perf_counter() - started, 1e-9)
    successful = sum(200 <= code < 300 for code in statuses)
    ordered = sorted(timings)
    percentile = lambda q: ordered[min(len(ordered) - 1, max(0, int(q * len(ordered)) - 1))] if ordered else 0.0
    qpm = len(inputs) / total * 60.0
    return {
        "mode": "http",
        "url": url,
        "count": len(inputs),
        "concurrency": concurrency,
        "total_seconds": total,
        "achieved_qpm": qpm,
        "success_rate": successful / len(inputs) if inputs else 0.0,
        "errors": len(errors),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "mean_ms": statistics.fmean(timings) if timings else 0.0,
        "target_concurrency": 100,
        "target_qpm": 1000,
        "target_met_in_this_run": concurrency >= 100 and qpm >= 1000 and not errors,
        "certification_eligible": True,
        "sample_errors": errors[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Running MedEvidence API base URL; enables real HTTP validation")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", default=None, help="Optional JSON evidence path")
    args = parser.parse_args()

    if args.url:
        payload = run_http(args.url, args.queries, args.workers, args.timeout)
    else:
        payload = run_offline(args.queries, min(args.workers, 32))
    payload["created"] = time.time()
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload.get("mode") == "offline_smoke":
        return 0
    return 0 if payload.get("target_met_in_this_run") else 2


if __name__ == "__main__":
    raise SystemExit(main())
