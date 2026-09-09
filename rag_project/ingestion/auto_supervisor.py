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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_path(system: Any) -> Path:
    return Path(system.settings.log_dir).resolve() / "auto_supervisor_state.json"


def _persist(system: Any) -> None:
    try:
        target = _state_path(system)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(json.dumps(_STATE, indent=2, sort_keys=True, default=str), encoding="utf-8")
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
                    for key, value in disk.items():
                        result.setdefault(key, value)
        except Exception:
            pass
    return result


def _signature(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _is_retryable(document: dict[str, Any] | None, signature: str) -> bool:
    if not document:
        return True
    status = str(document.get("status") or "").upper()
    stored_sig = f"{int(document.get('file_size') or 0)}:{int((datetime.fromisoformat(str(document.get('modified_at')).replace('Z', '+00:00')).timestamp() * 1_000_000_000) if document.get('modified_at') else 0)}"
    if status in {"READY", "COMPLETED", "DEGRADED_LEXICAL", "QUARANTINED"} and signature == stored_sig:
        return False
    if status.startswith("FAILED") and signature == stored_sig:
        return False
    return status in {"", "INTERRUPTED", "RECOVERING"} or signature != stored_sig


def _candidate(system: Any, path: Path) -> bool:
    try:
        document = system.state_store.get_by_path(str(path.resolve()))
        signature = _signature(path)
        if not _is_retryable(document, signature):
            return False
        status = str((document or {}).get("status") or "").upper()
        if status in {"RUNNING", "DISCOVERED", "VALIDATING", "EXTRACTING", "OCR", "CHUNKING", "EMBEDDING", "INDEXING", "VALIDATING_INDEX"}:
            return False
        return True
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

    candidates = sorted((p for p in incoming.glob("*.pdf") if p.is_file()), key=lambda p: p.stat().st_mtime)
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
        break
    _persist(system)


def _loop(system: Any, interval_seconds: float) -> None:
    with _LOCK:
        _STATE["enabled"] = True
        _STATE["started_at"] = _STATE["started_at"] or _now()
    _persist(system)
    while not _STOP.wait(max(1.0, interval_seconds)):
        try:
            _scan_once(system)
        except Exception as exc:
            with _LOCK:
                _STATE["last_error"] = str(exc)
                _STATE["last_action"] = "supervisor cycle failed"
                _STATE["last_action_at"] = _now()
            _persist(system)


def start(system: Any, interval_seconds: float = 3.0) -> dict[str, Any]:
    """Start one process-level autonomous PDF supervisor.

    The supervisor is independent from the Streamlit rerun loop: once started,
    it keeps watching the configured incoming directory, recovering stale jobs,
    and invoking the production ingestion path without requiring another UI click.
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
