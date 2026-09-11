"""Operational CLI: metrics, alerts, backup, and feedback-manifest generation."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from rag_project.intelligence.production_ops import BackupManager, MetricsService, OperationsStore, RetrainingManager

p=argparse.ArgumentParser(); p.add_argument("--db",default="data/med_evidence_ops.sqlite3"); p.add_argument("--report",action="store_true"); p.add_argument("--retrain",action="store_true"); p.add_argument("--backup-source",default="data"); p.add_argument("--backup-dir",default="backups")
a=p.parse_args(); store=OperationsStore(a.db)
if a.report:
    print(json.dumps({"metrics":MetricsService(store).snapshot(),"alerts":MetricsService(store).alerts()},ensure_ascii=False,indent=2))
if a.retrain:
    print(json.dumps(RetrainingManager(store,"data/retraining").build_manifest(),ensure_ascii=False,indent=2))
if not a.report and not a.retrain:
    print(BackupManager(Path(a.backup_source),Path(a.backup_dir)).backup())
