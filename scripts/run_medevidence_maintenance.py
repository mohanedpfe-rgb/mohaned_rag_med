"""Operational maintenance entrypoint: backup + failure analysis/retraining manifest."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from rag_project.intelligence.production_ops import BackupManager, OperationsStore, RetrainingManager


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--backup", action="store_true")
    parser.add_argument("--retraining", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    store = OperationsStore(root / "data" / "med_evidence_ops.sqlite3")
    out = {"root": str(root)}
    if args.backup:
        out["backup"] = str(BackupManager(root / "data", root / "backups" / "daily").backup())
    if args.retraining:
        out["retraining"] = RetrainingManager(store, root / "data" / "retraining").build_manifest()
    if not args.backup and not args.retraining:
        out["retraining_preview"] = RetrainingManager(store, root / "data" / "retraining").build_manifest()
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
