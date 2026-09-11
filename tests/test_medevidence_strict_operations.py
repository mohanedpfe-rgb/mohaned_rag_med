from pathlib import Path

from rag_project.intelligence.production_ops_strict import (
    ABTestManager,
    MetricsService,
    OperationsStore,
    VerifiedBackupManager,
    two_proportion_z_test,
)


def _record(store: OperationsStore, query_id: str, intent: str) -> None:
    store.record_result(
        query_id,
        f"question {query_id}",
        {
            "route": {"intent": intent, "complexity": 0.2},
            "generation_path": "PATH_A_EXTRACTIVE",
            "answer": "evidence-backed answer",
            "confidence": {"evidence_confidence": 0.9},
            "verification": {"supported_ratio": 0.9},
        },
        100.0,
    )


def test_metrics_accuracy_is_joined_to_intent(tmp_path: Path):
    store = OperationsStore(tmp_path / "ops.sqlite3")
    _record(store, "q1", "factual")
    _record(store, "q2", "factual")
    store.add_feedback("q1", "correct")
    store.add_feedback("q2", "wrong")
    snap = MetricsService(store).snapshot()
    assert snap["accuracy_by_type"]["factual"]["positive"] == 1
    assert snap["accuracy_by_type"]["factual"]["negative"] == 1
    assert snap["accuracy_by_type"]["factual"]["observed_accuracy"] == 0.5


def test_ab_significance_detects_clear_difference(tmp_path: Path):
    store = OperationsStore(tmp_path / "ops.sqlite3")
    manager = ABTestManager(store)
    for i in range(200):
        manager.record("exp", f"a-{i}", "A", success=1.0 if i < 190 else 0.0)
        manager.record("exp", f"b-{i}", "B", success=1.0 if i < 120 else 0.0)
    result = manager.compare("exp")
    assert result["statistical_test"]["p_value"] is not None
    assert result["statistical_test"]["significant_at_05"] is True
    assert result["winner"] == "A"


def test_two_proportion_test_is_conservative_with_small_samples():
    result = two_proportion_z_test([1, 1, 1], [1, 0, 1])
    assert result.significant_at_05 is False
    assert result.p_value is not None


def test_verified_backup_has_checksum_manifest_and_integrity(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "example.txt").write_text("evidence", encoding="utf-8")
    backup_root = tmp_path / "backups"
    manager = VerifiedBackupManager(source, backup_root)
    destination = manager.backup()
    result = VerifiedBackupManager.verify_backup(destination)
    assert result["ok"] is True
    assert (destination / "SHA256SUMS.json").exists()
