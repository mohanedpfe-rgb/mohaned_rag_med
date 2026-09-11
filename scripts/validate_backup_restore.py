"""Create a measured backup/restore validation artifact.

The command is intentionally destructive only to a temporary workspace. It proves
SQLite integrity before/after backup and restore, plus checksum-manifest coverage.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
import time
from pathlib import Path

from rag_project.intelligence.production_ops_strict import VerifiedBackupManager


def run(output: Path) -> dict[str, object]:
    # Windows can retain SQLite handles briefly during backup verification. Ignore
    # only temporary-workspace cleanup failures; all integrity assertions remain strict.
    with tempfile.TemporaryDirectory(prefix="medevidence-backup-", ignore_cleanup_errors=True) as raw:
        root = Path(raw)
        source_dir = root / "source"
        backup_dir = root / "backups"
        source_dir.mkdir()
        db_path = source_dir / "operations.sqlite3"
        with sqlite3.connect(db_path) as db:
            db.execute("CREATE TABLE evidence(id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
            db.executemany("INSERT INTO evidence(value) VALUES(?)", [("alpha",), ("beta",), ("gamma",)])
            db.commit()

        manager = VerifiedBackupManager(source_dir, backup_dir)
        started = time.perf_counter()
        backup_dir_created = manager.backup()
        file_backup = backup_dir_created / "operations.sqlite3"
        sqlite_backup = backup_dir_created / "operations-online-backup.sqlite3"
        manager.backup_sqlite(db_path, sqlite_backup)
        source_ok = manager.verify_sqlite(db_path)
        backup_ok = manager.verify_sqlite(file_backup)
        online_ok = manager.verify_sqlite(sqlite_backup)
        checksum = manager.verify_backup(backup_dir_created)

        restored = root / "restored.sqlite3"
        with sqlite3.connect(file_backup) as src, sqlite3.connect(restored) as dst:
            src.backup(dst)
        restore_ok = manager.verify_sqlite(restored)
        with sqlite3.connect(restored) as db:
            row_count = int(db.execute("SELECT COUNT(*) FROM evidence").fetchone()[0])

        payload = {
            "created": time.time(),
            "duration_seconds": time.perf_counter() - started,
            "source_integrity_ok": source_ok,
            "file_backup_integrity_ok": backup_ok,
            "online_backup_integrity_ok": online_ok,
            "checksum_manifest_ok": bool(checksum.get("ok")),
            "restore_integrity_ok": restore_ok,
            "restored_row_count": row_count,
            "passed": all((source_ok, backup_ok, online_ok, checksum.get("ok"), restore_ok, row_count == 3)),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/backup_restore_validation.json")
    args = parser.parse_args()
    payload = run(Path(args.output))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
