from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()
_THREAD: threading.Thread | None = None
_STOP = threading.Event()
_STATE: dict[str, Any] = {
    "enabled": False,
    "started_at": None,
    "last_scan_at": None,
    "last_action_at": None,
    "last_action": "idle",
    "last_file": None,
    "last_result": None,
    "last_error": None,
    "scans": 0,
    "auto_started": 0,
    "completed": 0,
    "failed": 0,
    "recovered": 0,
}


_ACTIVE = {
    "RUNNING", "DISCOVERED", "VALIDATING", "EXTRACTING", "OCR",
    "CHUNKING", "EMBEDDING", "INDEXING", "VALIDATING_INDEX", "BUILDING",
}
_TERMINAL_NO_RETRY = {"READY", "COMPLETED", "DEGRADED_LEXICAL", "QUARANTINED"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_path(system: Any) -> Path:
    return Path(system.settings.log_dir).resolve() / "auto_supervisor_state.json"


def _persist(system: Any) -> None:
    try:
        target = _state_path(system)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".tmp")
        with _LOCK:
            payload = dict(_STATE)
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        temp.replace(target)
    except Exception:
        # Diagnostics must never stop ingestion.
        pass


def snapshot(system: Any | None = None) -> dict[str, Any]:
    with _LOCK:
        result = dict(_STATE)
    if system is not None:
        try:
            path = _state_path(system)
            if path.is_file():
                disk = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(disk, dict):
                    result.update({k: v for k, v in disk.items() if k not in result or result[k] is None})
        except Exception:
            pass
    return result


def _stored_mtime(document: dict[str, Any]) -> float | None:
    value = document.get("modified_at")
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def _same_file_revision(path: Path, document: dict[str, Any]) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    recorded_size = int(document.get("file_size") or 0)
    recorded_mtime = _stored_mtime(document)
    if recorded_mtime is None:
        return recorded_size == stat.st_size
    # SQLite stores the document mtime as an ISO timestamp with microsecond
    # precision. A sub-second tolerance avoids false "changed" detections caused
    # by filesystem timestamp conversion while still catching real edits.
    return recorded_size == stat.st_size and abs(recorded_mtime - stat.st_mtime) < 1.0


def _candidate(system: Any, path: Path) -> bool:
    try:
        document = system.state_store.get_by_path(str(path.resolve()))
        if document is None:
            return True
        status = str(document.get("status") or "").upper()
        if status in _ACTIVE:
            return False
        same_revision = _same_file_revision(path, document)
        if status in _TERMINAL_NO_RETRY and same_revision:
            return False
        if status.startswith("FAILED") and same_revision:
            return False
        if status in {"", "INTERRUPTED", "RECOVERING"}:
            return True
        return not same_revision
    except (OSError, ValueError):
        return False


def _scan_once(system: Any) -> None:
    incoming = Path(system.settings.incoming_dir).resolve()
    incoming.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        _STATE["last_scan_at"] = _now()
        _STATE["scans"] += 1
    try:
        recovered = int(system.state_store.recover_stale_documents() or 0)
    except Exception:
        recovered = 0
    if recovered:
        with _LOCK:
            _STATE["recovered"] += recovered
            _STATE["last_action"] = f"recovered {recovered} stale job(s)"
            _STATE["last_action_at"] = _now()
        _persist(system)

    try:
        candidates = sorted(
            (p for p in incoming.glob("*.pdf") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
        )
    except OSError:
        candidates = []

    for path in candidates:
        if not _candidate(system, path):
            continue
        with _LOCK:
            _STATE["last_file"] = path.name
            _STATE["last_action"] = f"auto-ingesting {path.name}"
            _STATE["last_action_at"] = _now()
            _STATE["auto_started"] += 1
            _STATE["last_error"] = None
        _persist(system)
        try:
            result = system.ingest_file(path)
            result_status = str((result or {}).get("status") or "unknown").lower()
            with _LOCK:
                _STATE["last_result"] = result
                if result_status in {"success", "skipped"}:
                    _STATE["completed"] += 1
                elif result_status == "failed":
                    _STATE["failed"] += 1
                _STATE["last_action"] = f"completed {path.name} · {result_status}"
                _STATE["last_action_at"] = _now()
            _persist(system)
        except Exception as exc:
            with _LOCK:
                _STATE["failed"] += 1
                _STATE["last_error"] = str(exc)
                _STATE["last_action"] = f"error on {path.name}"
                _STATE["last_action_at"] = _now()
            _persist(system)
        # One canonical ingestion at a time. The production system already uses
        # a serialized local index; the next scan naturally advances to the next
        # pending PDF.
        break
    _persist(system)


def _loop(system: Any, interval_seconds: float) -> None:
    with _LOCK:
        _STATE["enabled"] = True
        _STATE["started_at"] = _STATE["started_at"] or _now()
    _persist(system)
    # Run once immediately so files already sitting in incoming/ are discovered
    # without waiting for the first polling interval.
    while not _STOP.is_set():
        try:
            _scan_once(system)
        except Exception as exc:
            with _LOCK:
                _STATE["last_error"] = str(exc)
                _STATE["last_action"] = "supervisor cycle failed"
                _STATE["last_action_at"] = _now()
            _persist(system)
        if _STOP.wait(max(1.0, float(interval_seconds))):
            break


def start(system: Any, interval_seconds: float = 3.0) -> dict[str, Any]:
    """Start one process-level autonomous PDF supervisor.

    The supervisor is independent from the Streamlit rerun loop: once started,
    it continuously watches the configured incoming directory, recovers expired
    leases, and invokes the production ingestion path without requiring another
    UI click or the browser to remain on the Documents page.
    """
    global _THREAD
    with _LOCK:
        if _THREAD and _THREAD.is_alive():
            return dict(_STATE)
        _STOP.clear()
        _THREAD = threading.Thread(
            target=_loop,
            args=(system, float(interval_seconds)),
            name=f"bookrag-auto-supervisor-{uuid.uuid4().hex[:8]}",
            daemon=True,
        )
        _THREAD.start()
        return dict(_STATE)


def stop() -> None:
    _STOP.set()
    with _LOCK:
        _STATE["enabled"] = False


__all__ = ["start", "stop", "snapshot"]
