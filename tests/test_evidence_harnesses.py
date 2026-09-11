from __future__ import annotations

import json
from pathlib import Path

from scripts.run_medevidence_load_test import run_offline
from scripts.validate_backup_restore import run as run_backup_validation


def test_offline_load_harness_is_explicitly_non_certifying() -> None:
    result = run_offline(queries=8, workers=2)
    assert result["mode"] == "offline_smoke"
    assert result["certification_eligible"] is False
    assert result["errors"] == 0


def test_backup_restore_harness_produces_passed_artifact(tmp_path: Path) -> None:
    output = tmp_path / "backup_restore.json"
    result = run_backup_validation(output)
    assert result["passed"] is True
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["checksum_manifest_ok"] is True
    assert saved["restore_integrity_ok"] is True
    assert saved["restored_row_count"] == 3
