"""Production operations for the MedEvidence Pro 20-week plan.

Stdlib-only by design so the existing deterministic CI environment stays stable.
The module covers the plan's production logging, metrics, feedback analysis,
A/B testing, backup/retention, retry/circuit-breaker, and benchmark contracts.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import statistics
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS query_log (
    query_id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    intent TEXT,
    complexity REAL,
    path TEXT,
    timestamp REAL NOT NULL,
    user_id TEXT,
    session_id TEXT,
    metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS answer_log (
    query_id TEXT PRIMARY KEY,
    answer_text TEXT,
    confidence REAL,
    latency_ms REAL,
    created REAL NOT NULL,
    FOREIGN KEY(query_id) REFERENCES query_log(query_id)
);
CREATE TABLE IF NOT EXISTS verification_log (
    query_id TEXT PRIMARY KEY,
    alerts_json TEXT NOT NULL,
    failed_checks_json TEXT NOT NULL,
    final_confidence REAL,
    created REAL NOT NULL,
    FOREIGN KEY(query_id) REFERENCES query_log(query_id)
);
CREATE TABLE IF NOT EXISTS feedback_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id TEXT NOT NULL,
    user_feedback TEXT NOT NULL,
    feedback_text TEXT,
    timestamp REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS metric_sample (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    labels_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ab_assignment (
    experiment TEXT NOT NULL,
    query_id TEXT NOT NULL,
    variant TEXT NOT NULL,
    created REAL NOT NULL,
    PRIMARY KEY(experiment, query_id)
);
CREATE TABLE IF NOT EXISTS ab_result (
    experiment TEXT NOT NULL,
    query_id TEXT NOT NULL,
    variant TEXT NOT NULL,
    success REAL,
    latency_ms REAL,
    satisfaction REAL,
    created REAL NOT NULL,
    PRIMARY KEY(experiment, query_id)
);
CREATE INDEX IF NOT EXISTS idx_query_timestamp ON query_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_feedback_query ON feedback_log(query_id);
CREATE INDEX IF NOT EXISTS idx_metric_name ON metric_sample(metric);
"""


@dataclass(frozen=True)
class RetryPolicy:
    retries: int = 2
    base_delay_seconds: float = 0.15
    max_delay_seconds: float = 2.0
    backoff: float = 2.0

    def delays(self) -> Iterable[float]:
        for attempt in range(max(0, self.retries)):
            yield min(self.max_delay_seconds, self.base_delay_seconds * (self.backoff**attempt))


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_seconds: float = 30.0):
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_seconds = max(0.1, recovery_seconds)
        self._failures = 0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at <= 0:
                return "closed"
            if time.time() - self._opened_at >= self.recovery_seconds:
                return "half_open"
            return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = time.time()


class ResilientCall:
    def __init__(self, retry: RetryPolicy | None = None, breaker: CircuitBreaker | None = None):
        self.retry = retry or RetryPolicy()
        self.breaker = breaker or CircuitBreaker()

    def __call__(self, fn, *args, **kwargs):
        if not self.breaker.allow():
            raise RuntimeError("circuit_open")
        try:
            value = fn(*args, **kwargs)
            self.breaker.record_success()
            return value
        except Exception:
            self.breaker.record_failure()
            delays = list(self.retry.delays())
            last = True
            for delay in delays:
                last = False
                time.sleep(delay)
                try:
                    value = fn(*args, **kwargs)
                    self.breaker.record_success()
                    return value
                except Exception:
                    last = True
            if last:
                raise
            raise RuntimeError("call_failed")


class OperationsStore:
    """SQLite store matching the plan's query/answer/verification/feedback model."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as db:
            db.executescript(SCHEMA)
            db.commit()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=10)
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def record_result(self, query_id: str, query: str, result: Mapping[str, Any], latency_ms: float,
                      *, user_id: str | None = None, session_id: str | None = None) -> None:
        route = result.get("route") if isinstance(result.get("route"), Mapping) else {}
        confidence = result.get("confidence") if isinstance(result.get("confidence"), Mapping) else {}
        verification = result.get("verification") if isinstance(result.get("verification"), Mapping) else {}
        now = time.time()
        metadata = {"status": result.get("status"), "emergency": result.get("emergency_flag", False),
                    "needs_review": result.get("needs_review", False)}
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO query_log VALUES (?,?,?,?,?,?,?,?,?)", (
                query_id, query[:4000], route.get("intent"), float(route.get("complexity", 0.0) or 0.0),
                result.get("generation_path"), now, user_id, session_id, json.dumps(metadata, ensure_ascii=False)))
            db.execute("INSERT OR REPLACE INTO answer_log VALUES (?,?,?,?,?)", (
                query_id, str(result.get("answer", ""))[:20000], float(confidence.get("evidence_confidence", 0.0) or 0.0),
                float(latency_ms), now))
            db.execute("INSERT OR REPLACE INTO verification_log VALUES (?,?,?,?,?)", (
                query_id,
                json.dumps(verification.get("alerts", []), ensure_ascii=False),
                json.dumps(verification.get("failed_checks", []), ensure_ascii=False),
                float(confidence.get("evidence_confidence", verification.get("supported_ratio", 0.0)) or 0.0), now))
            db.commit()

    def add_feedback(self, query_id: str, feedback_type: str, feedback_text: str | None = None) -> None:
        allowed = {"correct", "wrong", "incomplete", "helpful", "unclear"}
        if feedback_type not in allowed:
            raise ValueError(f"feedback_type must be one of {sorted(allowed)}")
        with self._connect() as db:
            db.execute("INSERT INTO feedback_log(query_id,user_feedback,feedback_text,timestamp) VALUES(?,?,?,?)",
                       (query_id, feedback_type, feedback_text, time.time()))
            db.commit()

    def wrong_feedback_rows(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("""
                SELECT q.query, q.intent, q.complexity, q.path, f.user_feedback, f.feedback_text
                FROM feedback_log f JOIN query_log q ON q.query_id=f.query_id
                WHERE f.user_feedback IN ('wrong','incomplete')
                ORDER BY f.timestamp DESC
            """).fetchall()
        return [dict(zip(("query","intent","complexity","path","feedback","feedback_text"), r)) for r in rows]

    def samples(self, metric: str, labels: Mapping[str, str] | None = None) -> list[float]:
        labels = labels or {}
        with self._connect() as db:
            rows = db.execute("SELECT value, labels_json FROM metric_sample WHERE metric=?", (metric,)).fetchall()
        out = []
        for value, raw in rows:
            try:
                data = json.loads(raw)
            except Exception:
                data = {}
            if all(data.get(k) == v for k, v in labels.items()):
                out.append(float(value))
        return out

    def observe(self, metric: str, value: float, labels: Mapping[str, str] | None = None) -> None:
        with self._connect() as db:
            db.execute("INSERT INTO metric_sample(timestamp,metric,value,labels_json) VALUES(?,?,?,?)",
                       (time.time(), metric, float(value), json.dumps(dict(labels or {}), sort_keys=True)))
            db.commit()


class MetricsService:
    def __init__(self, store: OperationsStore):
        self.store = store

    @staticmethod
    def _percentile(values: Sequence[float], q: float) -> float:
        if not values:
            return 0.0
        vals = sorted(values)
        idx = min(len(vals) - 1, max(0, math.ceil(q * len(vals)) - 1))
        return float(vals[idx])

    def snapshot(self) -> dict[str, Any]:
        with self.store._connect() as db:
            rows = db.execute("SELECT q.intent,q.path,a.latency_ms,q.query_id,q.timestamp FROM query_log q JOIN answer_log a ON a.query_id=q.query_id").fetchall()
            feedback = db.execute("SELECT user_feedback, COUNT(*) FROM feedback_log GROUP BY user_feedback").fetchall()
        by_intent: dict[str, dict[str, int]] = {}
        paths: dict[str, int] = {}
        latencies = []
        for intent, path, latency, _, _ in rows:
            by_intent.setdefault(intent or "unknown", {"total": 0, "positive": 0, "negative": 0})["total"] += 1
            paths[path or "unknown"] = paths.get(path or "unknown", 0) + 1
            latencies.append(float(latency or 0.0))
        positive = sum(int(count) for kind, count in feedback if kind in {"correct", "helpful"})
        negative = sum(int(count) for kind, count in feedback if kind in {"wrong", "incomplete"})
        total_feedback = positive + negative
        return {
            "accuracy_by_type": {k: {**v, "observed_accuracy": (v["positive"] / v["total"] if v["total"] else None)} for k, v in by_intent.items()},
            "latency_percentiles": {"p50": self._percentile(latencies, .50), "p95": self._percentile(latencies, .95), "p99": self._percentile(latencies, .99)},
            "path_usage": paths,
            "feedback": dict(feedback),
            "error_rates": {"negative_feedback_rate": (negative / total_feedback if total_feedback else 0.0)},
        }

    def alerts(self, *, latency_p95_ms: float = 20000, failure_rate: float = 0.15, accuracy_drop: float = 0.05) -> list[str]:
        snap = self.snapshot()
        alerts: list[str] = []
        if snap["latency_percentiles"]["p95"] > latency_p95_ms:
            alerts.append("latency_p95_above_threshold")
        if snap["error_rates"]["negative_feedback_rate"] > failure_rate:
            alerts.append("negative_feedback_above_threshold")
        accuracies = [v["observed_accuracy"] for v in snap["accuracy_by_type"].values() if v["observed_accuracy"] is not None]
        if accuracies and min(accuracies) < 1.0 - accuracy_drop:
            alerts.append("observed_accuracy_below_threshold")
        return alerts


class ABTestManager:
    """Deterministic config-based experiment assignment and basic significance helpers."""
    def __init__(self, store: OperationsStore):
        self.store = store

    @staticmethod
    def assign(experiment: str, query_id: str, variants: Sequence[str] = ("A", "B"), weights: Sequence[float] | None = None) -> str:
        if not variants:
            raise ValueError("variants cannot be empty")
        weights = list(weights or [1.0] * len(variants))
        if len(weights) != len(variants) or any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("weights must be non-negative and match variants")
        value = int(hashlib.sha256(f"{experiment}:{query_id}".encode()).hexdigest()[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
        cumulative = 0.0
        total = float(sum(weights))
        for variant, weight in zip(variants, weights):
            cumulative += weight / total
            if value < cumulative:
                return variant
        return variants[-1]

    def assignment(self, experiment: str, query_id: str) -> str:
        with self.store._connect() as db:
            row = db.execute("SELECT variant FROM ab_assignment WHERE experiment=? AND query_id=?", (experiment, query_id)).fetchone()
            if row:
                return str(row[0])
            variant = self.assign(experiment, query_id)
            db.execute("INSERT INTO ab_assignment VALUES(?,?,?,?)", (experiment, query_id, variant, time.time()))
            db.commit()
            return variant

    def record(self, experiment: str, query_id: str, variant: str, *, success: float | None = None,
               latency_ms: float | None = None, satisfaction: float | None = None) -> None:
        with self.store._connect() as db:
            db.execute("INSERT OR REPLACE INTO ab_result VALUES(?,?,?,?,?,?,?)",
                       (experiment, query_id, variant, success, latency_ms, satisfaction, time.time()))
            db.commit()

    def compare(self, experiment: str) -> dict[str, Any]:
        with self.store._connect() as db:
            rows = db.execute("SELECT variant,success,latency_ms,satisfaction FROM ab_result WHERE experiment=?", (experiment,)).fetchall()
        groups: dict[str, list[tuple[float,float,float]]] = {}
        for variant, success, latency, satisfaction in rows:
            groups.setdefault(variant, []).append((float(success or 0), float(latency or 0), float(satisfaction or 0)))
        summary = {}
        for variant, values in groups.items():
            summary[variant] = {
                "n": len(values),
                "success_rate": statistics.fmean(v[0] for v in values) if values else 0.0,
                "latency_ms": statistics.fmean(v[1] for v in values) if values else 0.0,
                "satisfaction": statistics.fmean(v[2] for v in values) if values else 0.0,
            }
        winner = None
        if summary:
            winner = max(summary, key=lambda k: (summary[k]["success_rate"], -summary[k]["latency_ms"], summary[k]["satisfaction"]))
        return {"experiment": experiment, "variants": summary, "winner": winner, "statistical_test": "descriptive; use scipy/stats externally for formal p-values"}


class RetrainingManager:
    """Turns wrong/incomplete feedback into a deterministic review/retraining manifest."""
    def __init__(self, store: OperationsStore, output_dir: str | Path):
        self.store = store
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_manifest(self) -> dict[str, Any]:
        rows = self.store.wrong_feedback_rows()
        by_intent: dict[str, int] = {}
        by_path: dict[str, int] = {}
        for row in rows:
            by_intent[row.get("intent") or "unknown"] = by_intent.get(row.get("intent") or "unknown", 0) + 1
            by_path[row.get("path") or "unknown"] = by_path.get(row.get("path") or "unknown", 0) + 1
        manifest = {
            "created": time.time(),
            "failure_count": len(rows),
            "by_intent": by_intent,
            "by_path": by_path,
            "retrain": {
                "intent_classifier": any(v >= 10 for v in by_intent.values()),
                "complexity_scorer": any(float(r.get("complexity") or 0) in (0.3, 0.5, 0.7) for r in rows),
                "retrieval_reranker": bool(rows),
                "confidence_threshold_review": bool(rows),
            },
            "kb_review_required": bool(rows),
            "samples": rows[:500],
        }
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (self.output_dir / f"retraining-manifest-{stamp}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest


class BackupManager:
    def __init__(self, source_dir: str | Path, backup_dir: str | Path, keep: int = 28):
        self.source_dir = Path(source_dir)
        self.backup_dir = Path(backup_dir)
        self.keep = max(1, keep)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def backup(self) -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        destination = self.backup_dir / stamp
        destination.mkdir(parents=True, exist_ok=True)
        if self.source_dir.exists():
            for item in self.source_dir.rglob("*"):
                if item.is_file():
                    target = destination / item.relative_to(self.source_dir)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target)
        manifests = sorted((p for p in self.backup_dir.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)
        for stale in manifests[self.keep:]:
            shutil.rmtree(stale, ignore_errors=True)
        return destination


@dataclass(frozen=True)
class BenchmarkResult:
    count: int
    total_seconds: float
    throughput_qps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    errors: int


def benchmark_callable(fn, inputs: Sequence[Any], workers: int = 8) -> BenchmarkResult:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    timings: list[float] = []
    errors = 0
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = []
        for item in inputs:
            item_started = time.perf_counter()
            futures.append((item_started, pool.submit(fn, item)))
        for item_started, future in futures:
            try:
                future.result()
            except Exception:
                errors += 1
            timings.append((time.perf_counter() - item_started) * 1000.0)
    total = time.perf_counter() - started
    return BenchmarkResult(
        count=len(inputs), total_seconds=total, throughput_qps=(len(inputs) / total if total else 0.0),
        p50_ms=MetricsService._percentile(timings, .50), p95_ms=MetricsService._percentile(timings, .95),
        p99_ms=MetricsService._percentile(timings, .99), errors=errors)


__all__ = [
    "OperationsStore", "MetricsService", "ABTestManager", "RetrainingManager", "BackupManager",
    "RetryPolicy", "CircuitBreaker", "ResilientCall", "BenchmarkResult", "benchmark_callable",
]