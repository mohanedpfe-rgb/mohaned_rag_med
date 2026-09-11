"""Run the gold-set KPI benchmark against a live MedEvidence API.

The runner emits measurements only; it never fabricates clinical accuracy. Gold-set
labels drive deterministic checks for path, required terms, citation presence,
and latency. Clinical correctness still requires reviewed gold labels/sign-off.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _class_for(item: dict) -> str:
    ident = str(item.get("id", ""))
    if "numeric" in ident:
        return "numeric"
    if "complex" in ident or "comparison" in ident or "mechanism" in ident:
        return "complex"
    return "simple"


def _request(base_url: str, question: str, timeout: float) -> tuple[float, int, dict | None, str | None]:
    body = json.dumps({"query": question, "stream": False}).encode("utf-8")
    req = Request(base_url.rstrip("/") + "/query", data=body, headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
        payload = json.loads(raw)
        return (time.perf_counter() - started), int(response.status), payload, None
    except HTTPError as exc:
        return (time.perf_counter() - started), int(exc.code), None, str(exc)
    except (URLError, ValueError, json.JSONDecodeError) as exc:
        return (time.perf_counter() - started), 0, None, str(exc)
    except Exception as exc:
        return (time.perf_counter() - started), 0, None, str(exc)


def _text(payload: dict | None, key: str) -> str:
    value = (payload or {}).get(key, "")
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def run(base_url: str, gold_path: Path, timeout: float) -> dict[str, object]:
    cases = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    results: list[dict] = []
    for item in cases:
        elapsed, status, payload, error = _request(base_url, str(item["question"]), timeout)
        answer = _text(payload, "answer")
        citations = (payload or {}).get("citations") or (payload or {}).get("sources") or []
        route = _text(payload, "generation_path")
        route_obj = (payload or {}).get("route")
        if isinstance(route_obj, dict):
            route = str(route_obj.get("path") or route_obj.get("generation_path") or route)
        lowered = answer.lower()
        missing_terms = [term for term in item.get("must_contain", []) if str(term).lower() not in lowered]
        citation_ok = bool(citations) if item.get("must_cite") else True
        latency_ok = elapsed <= float(item.get("max_latency_s", 9999))
        path_ok = not item.get("expected_path") or str(item["expected_path"]).lower() in route.lower()
        results.append({
            "id": item.get("id"),
            "class": _class_for(item),
            "http_status": status,
            "latency_s": elapsed,
            "expected_path": item.get("expected_path"),
            "observed_path": route,
            "path_ok": path_ok,
            "missing_terms": missing_terms,
            "citation_ok": citation_ok,
            "latency_ok": latency_ok,
            "transport_ok": error is None and 200 <= status < 300,
            "error": error,
        })

    by_class: dict[str, list[dict]] = {}
    for result in results:
        by_class.setdefault(result["class"], []).append(result)

    accuracy = {
        cls: (sum(r["path_ok"] and not r["missing_terms"] and r["transport_ok"] for r in rows) / len(rows) if rows else 0.0)
        for cls, rows in by_class.items()
    }
    latencies = [float(r["latency_s"]) * 1000 for r in results]
    overall = {
        "case_count": len(results),
        "accuracy_by_class": accuracy,
        "weighted_accuracy": statistics.fmean(accuracy.values()) if accuracy else 0.0,
        "citation_rate": (sum(bool(r["citation_ok"]) for r in results) / len(results) if results else 0.0),
        "latency_ms": {
            "p50": sorted(latencies)[max(0, int(0.50 * len(latencies)) - 1)] if latencies else 0.0,
            "p95": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)] if latencies else 0.0,
            "p99": sorted(latencies)[max(0, int(0.99 * len(latencies)) - 1)] if latencies else 0.0,
        },
        "review_required": True,
        "clinical_correctness_claimed": False,
    }
    payload = {"created": time.time(), "base_url": base_url, "gold_set": str(gold_path), "overall": overall, "results": results}
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--gold", default="tests/support/gold_sets/core.jsonl")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", default="artifacts/med_evidence_kpi.json")
    args = parser.parse_args()
    payload = run(args.url, Path(args.gold), args.timeout)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
