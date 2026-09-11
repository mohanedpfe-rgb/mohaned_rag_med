"""Strict production operations for the MedEvidence Pro plan."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .production_ops import (
    ABTestManager as _ABTestManager,
    BackupManager as _BackupManager,
    CircuitBreaker,
    OperationsStore,
    RetryPolicy,
    ResilientCall,
)


class MetricsService:
    """Feedback-joined operational metrics."""
    def __init__(self, store: OperationsStore): self.store = store

    @staticmethod
    def _percentile(values: Sequence[float], q: float) -> float:
        if not values: return 0.0
        vals = sorted(float(v) for v in values)
        return float(vals[min(len(vals) - 1, max(0, math.ceil(q * len(vals)) - 1))])

    def snapshot(self) -> dict[str, Any]:
        with self.store._connect() as db:
            rows = db.execute("SELECT q.intent,q.path,a.latency_ms,q.query_id FROM query_log q JOIN answer_log a ON a.query_id=q.query_id").fetchall()
            feedback_rows = db.execute("SELECT q.intent,q.path,f.user_feedback,f.query_id FROM feedback_log f JOIN query_log q ON q.query_id=f.query_id").fetchall()
        by_intent: dict[str, dict[str, int]] = {}
        by_path: dict[str, int] = {}
        latencies: list[float] = []
        for intent, path, latency, _ in rows:
            bucket = by_intent.setdefault(intent or "unknown", {"total": 0, "positive": 0, "negative": 0})
            bucket["total"] += 1
            by_path[path or "unknown"] = by_path.get(path or "unknown", 0) + 1
            latencies.append(float(latency or 0.0))
        feedback_counts: dict[str, int] = {}
        for intent, _path, kind, _qid in feedback_rows:
            kind = str(kind); feedback_counts[kind] = feedback_counts.get(kind, 0) + 1
            bucket = by_intent.setdefault(intent or "unknown", {"total": 0, "positive": 0, "negative": 0})
            if kind in {"correct", "helpful"}: bucket["positive"] += 1
            elif kind in {"wrong", "incomplete"}: bucket["negative"] += 1
        positive = sum(v for k, v in feedback_counts.items() if k in {"correct", "helpful"})
        negative = sum(v for k, v in feedback_counts.items() if k in {"wrong", "incomplete"})
        accuracy = {
            k: {**v, "observed_accuracy": v["positive"] / (v["positive"] + v["negative"]) if v["positive"] + v["negative"] else None,
                "feedback_count": v["positive"] + v["negative"]}
            for k, v in by_intent.items()
        }
        return {"accuracy_by_type": accuracy,
                "latency_percentiles": {"p50": self._percentile(latencies,.50), "p95": self._percentile(latencies,.95), "p99": self._percentile(latencies,.99)},
                "path_usage": by_path, "feedback": feedback_counts,
                "error_rates": {"negative_feedback_rate": negative / (positive + negative) if positive + negative else 0.0}}

    def alerts(self, *, latency_p95_ms: float = 20_000.0, failure_rate: float = 0.15, accuracy_floor: float = 0.95) -> list[str]:
        snap = self.snapshot(); alerts: list[str] = []
        if snap["latency_percentiles"]["p95"] > latency_p95_ms: alerts.append("latency_p95_above_threshold")
        if snap["error_rates"]["negative_feedback_rate"] > failure_rate: alerts.append("negative_feedback_above_threshold")
        observed = [x["observed_accuracy"] for x in snap["accuracy_by_type"].values() if x["observed_accuracy"] is not None]
        if observed and min(observed) < accuracy_floor: alerts.append("accuracy_below_threshold")
        return alerts


@dataclass(frozen=True)
class SignificanceResult:
    method: str; statistic: float | None; p_value: float | None; significant_at_05: bool; n_a: int; n_b: int
    def as_dict(self) -> dict[str, Any]: return self.__dict__.copy()


def _normal_cdf(x: float) -> float: return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_proportion_z_test(success_a: Sequence[float], success_b: Sequence[float]) -> SignificanceResult:
    a = [1.0 if float(x) > 0.5 else 0.0 for x in success_a]; b = [1.0 if float(x) > 0.5 else 0.0 for x in success_b]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2: return SignificanceResult("two-proportion-z", None, None, False, n1, n2)
    p1, p2 = statistics.fmean(a), statistics.fmean(b)
    pooled = (sum(a) + sum(b)) / (n1 + n2)
    se = math.sqrt(max(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2), 1e-18))
    z = (p1 - p2) / se; p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return SignificanceResult("two-proportion-z", z, p, p < 0.05, n1, n2)


class ABTestManager(_ABTestManager):
    """Deterministic A/B assignment plus in-process significance and rollout guard."""
    def compare(self, experiment: str, *, min_samples_per_variant: int = 500) -> dict[str, Any]:
        with self.store._connect() as db:
            rows = db.execute("SELECT variant,success,latency_ms,satisfaction FROM ab_result WHERE experiment=?", (experiment,)).fetchall()
        groups: dict[str, list[tuple[float,float,float]]] = {}
        for variant, success, latency, satisfaction in rows:
            groups.setdefault(str(variant), []).append((float(success or 0.0), float(latency or 0.0), float(satisfaction or 0.0)))
        summary = {v: {"n": len(xs), "success_rate": statistics.fmean(x[0] for x in xs) if xs else 0.0,
                       "latency_ms": statistics.fmean(x[1] for x in xs) if xs else 0.0,
                       "satisfaction": statistics.fmean(x[2] for x in xs) if xs else 0.0} for v, xs in groups.items()}
        variants = sorted(groups); significance = None; candidate = None
        if len(variants) >= 2:
            a, b = variants[0], variants[1]
            significance = two_proportion_z_test([x[0] for x in groups[a]], [x[0] for x in groups[b]]).as_dict()
            candidate = max(variants, key=lambda k: (summary[k]["success_rate"], -summary[k]["latency_ms"], summary[k]["satisfaction"]))
        eligible = bool(significance and significance["significant_at_05"] and all(summary[v]["n"] >= min_samples_per_variant for v in variants[:2]))
        return {"experiment": experiment, "variants": summary, "candidate_winner": candidate, "winner": candidate if eligible else None,
                "rollout_eligible": eligible, "min_samples_per_variant": min_samples_per_variant, "statistical_test": significance}


class VerifiedBackupManager(_BackupManager):
    """Crash-consistent SQLite backup with checksum and restore verification."""
    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _copy_file(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)

    def backup_sqlite(self, source_db: str | Path, destination_db: str | Path) -> Path:
        source = Path(source_db); destination = Path(destination_db); destination.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(source) as src, sqlite3.connect(destination) as dst:
            src.execute("PRAGMA wal_checkpoint(FULL)")
            src.backup(dst)
            dst.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        if not self.verify_sqlite(destination): raise RuntimeError(f"backup integrity check failed: {destination}")
        return destination

    def backup(self) -> Path:
        destination = super().backup()
        manifest: dict[str, str] = {}
        for item in sorted(destination.rglob("*")):
            if item.is_file(): manifest[str(item.relative_to(destination))] = self._sha256(item)
        (destination / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return destination

    @classmethod
    def verify_backup(cls, backup_dir: str | Path) -> dict[str, Any]:
        root = Path(backup_dir); raw = root / "SHA256SUMS.json"
        if not raw.exists(): return {"ok": False, "reason": "checksum_manifest_missing"}
        expected = json.loads(raw.read_text(encoding="utf-8')); failures = []
        for rel, digest in expected.items():
            path = root / rel
            if not path.exists() or cls._sha256(path) != digest: failures.append(rel)
        return {"ok": not failures, "checked": len(expected), "failures": failures}

    @staticmethod
    def verify_sqlite(db_path: str | Path) -> bool:
        with sqlite3.connect(db_path) as db: return str(db.execute("PRAGMA integrity_check").fetchone()[0]).lower() == "ok"


__all__ = ["OperationsStore", "MetricsService", "ABTestManager", "VerifiedBackupManager", "CircuitBreaker", "RetryPolicy", "ResilientCall", "two_proportion_z_test", "SignificanceResult"]
