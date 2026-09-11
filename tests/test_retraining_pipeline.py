from __future__ import annotations

import json
from pathlib import Path

from rag_project.intelligence.retraining_pipeline import train_task
from rag_project.intelligence.production_ops_strict import ExecutableRetrainingManager


def _write_dataset(path: Path) -> None:
    rows = []
    for i in range(30):
        rows.append({"text": f"drug question medication dose {i}", "label": "medication"})
        rows.append({"text": f"symptom diagnosis patient fever {i}", "label": "diagnosis"})
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_train_task_creates_versioned_artifact(tmp_path: Path):
    dataset = tmp_path / "intent.jsonl"
    _write_dataset(dataset)
    report = train_task("intent_classifier", dataset, tmp_path / "artifacts")
    artifact = Path(report.artifact)
    assert artifact.exists()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["model"]["type"] == "multinomial_text_nb"
    assert payload["training_report"]["examples"] == 60
    assert report.labels == 2


def test_executable_retraining_manager_runs_multiple_tasks(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset)
    reports = ExecutableRetrainingManager(tmp_path / "artifacts").train({"intent": dataset, "complexity": dataset})
    assert len(reports) == 2
    assert all(Path(report.artifact).exists() for report in reports)
