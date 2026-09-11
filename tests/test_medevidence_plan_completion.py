from __future__ import annotations

import tempfile
import time
from pathlib import Path

from rag_project.intelligence.production_ops import (
    ABTestManager, BackupManager, CircuitBreaker, MetricsService,
    OperationsStore, ResilientCall, RetrainingManager, benchmark_callable,
)
from rag_project.knowledge.medical_kb import counts, initialize


def test_medical_kb_schema_is_complete():
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "kb.sqlite3"
        result = initialize(db)
        assert set(result) == {"drugs", "interactions", "guidelines", "contraindications", "disease_graph"}
        assert all(v == 0 for v in result.values())
        assert counts(db) == result


def test_operations_store_records_feedback_and_metrics():
    with tempfile.TemporaryDirectory() as td:
        store = OperationsStore(Path(td) / "ops.sqlite3")
        result = {
            "query_id": "q1", "status": "SUCCESS", "answer": "evidence [S1]",
            "route": {"intent": "factual", "complexity": 0.2},
            "generation_path": "PATH_A_EXTRACTIVE",
            "confidence": {"evidence_confidence": 0.95},
            "verification": {"alerts": [], "failed_checks": [], "supported_ratio": 0.95},
        }
        store.record_result("q1", "What is hypertension?", result, 42.0)
        store.add_feedback("q1", "correct", "clear")
        store.observe("latency_ms", 42.0, {"intent": "factual"})
        snapshot = MetricsService(store).snapshot()
        assert snapshot["path_usage"]["PATH_A_EXTRACTIVE"] == 1
        assert snapshot["feedback"]["correct"] == 1
        assert snapshot["latency_percentiles"]["p50"] == 42.0


def test_ab_assignment_is_deterministic_and_comparable():
    with tempfile.TemporaryDirectory() as td:
        store = OperationsStore(Path(td) / "ops.sqlite3")
        ab = ABTestManager(store)
        first = ab.assignment("prompt-v2", "q-123")
        second = ab.assignment("prompt-v2", "q-123")
        assert first == second
        ab.record("prompt-v2", "q-123", first, success=1, latency_ms=20, satisfaction=5)
        other = "B" if first == "A" else "A"
        ab.record("prompt-v2", "q-456", other, success=0, latency_ms=40, satisfaction=2)
        report = ab.compare("prompt-v2")
        assert report["winner"] == first


def test_retraining_manifest_uses_wrong_feedback():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        store = OperationsStore(root / "ops.sqlite3")
        result = {"query_id": "q1", "status": "SUCCESS", "answer": "x", "route": {"intent": "numeric", "complexity": 0.5}, "generation_path": "PATH_B_TEMPLATE", "confidence": {"evidence_confidence": 0.6}, "verification": {}}
        store.record_result("q1", "dose?", result, 90)
        store.add_feedback("q1", "wrong", "incorrect dose")
        manifest = RetrainingManager(store, root / "manifests").build_manifest()
        assert manifest["failure_count"] == 1
        assert manifest["retrain"]["retrieval_reranker"] is True
        assert list((root / "manifests").glob("retraining-manifest-*.json"))


def test_backup_manager_copies_and_rotates():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); source = root / "source"; backups = root / "backups"
        source.mkdir(); (source / "x.txt").write_text("ok", encoding="utf-8")
        manager = BackupManager(source, backups, keep=1)
        first = manager.backup(); assert (first / "x.txt").read_text() == "ok"
        time.sleep(0.01); manager.backup()
        assert len([p for p in backups.iterdir() if p.is_dir()]) == 1


def test_circuit_breaker_and_retry_recover():
    breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=0.01)
    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        if calls["n"] < 2: raise RuntimeError("temporary")
        return "ok"
    assert ResilientCall(breaker=breaker)(flaky) == "ok"
    assert breaker.state == "closed"


def test_benchmark_contract():
    result = benchmark_callable(lambda value: value + 1, list(range(16)), workers=4)
    assert result.count == 16
    assert result.errors == 0
    assert result.throughput_qps > 0
    assert result.p95_ms >= result.p50_ms
