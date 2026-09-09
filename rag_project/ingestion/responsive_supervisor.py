from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

_LOCK = threading.RLock()
_THREAD: threading.Thread | None = None
_OBSERVER: Observer | None = None
_STOP = threading.Event()
_WAKE = threading.Event()
_WORKING: set[str] = set()
_CONTENT_CACHE: dict[str, tuple[str, str | None]] = {}
_STABILITY: dict[str, tuple[str, float]] = {}
_LOCK_HANDLE: int | None = None
_STATE: dict[str, Any] = {
    "enabled": False,
    "watchdog": False,
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
    "in_flight": 0,
}

_ACTIVE = {
    "RUNNING", "DISCOVERED", "VALIDATING", "EXTRACTING", "OCR",
    "CHUNKING", "EMBEDDING", "INDEXING", "VALIDATING_INDEX", "BUILDING",
}
_TERMINAL_NO_RETRY = {"READY", "COMPLETED", "DEGRADED_LEXICAL", "QUARANTINED"}
_STABLE_SECONDS = 0.75
_CACHE_MAX = 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_path(system: Any) -> Path:
    return Path(system.settings.log_dir).resolve() / "auto_supervisor_state.json"


def _lock_path(system: Any) -> Path:
    return Path(system.settings.log_dir).resolve() / "auto_supervisor.lock"


def _persist(system: Any) -> None:
    try:
        target = _state_path(system)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".tmp")
        with _LOCK:
            payload = dict(_STATE)
            payload["in_flight"] = len(_WORKING)
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        temp.replace(target)
    except Exception:
        pass


def _load_persisted_state(system: Any) -> None:
    try:
        path = _state_path(system)
        if not path.is_file():
            return
        disk = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(disk, dict):
            return
        with _LOCK:
            for key in _STATE:
                if key in disk and disk[key] is not None and key != "in_flight":
                    _STATE[key] = disk[key]
    except Exception:
        pass


def snapshot(system: Any | None = None) -> dict[str, Any]:
    with _LOCK:
        payload = dict(_STATE)
        payload["in_flight"] = len(_WORKING)
        return payload


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _acquire_process_lock(system: Any) -> bool:
    global _LOCK_HANDLE
    path = _lock_path(system)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(handle, f"{os.getpid()}\n{_now()}\n".encode("utf-8"))
        _LOCK_HANDLE = handle
        return True
    except FileExistsError:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            pid = int(lines[0]) if lines else -1
        except (OSError, ValueError):
            pid = -1
        if pid > 0 and _process_alive(pid):
            return False
        try:
            path.unlink()
        except OSError:
            return False
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(handle, f"{os.getpid()}\n{_now()}\n".encode("utf-8"))
            _LOCK_HANDLE = handle
            return True
        except OSError:
            return False
    except OSError:
        return False


def _release_process_lock(system: Any) -> None:
    global _LOCK_HANDLE
    handle = _LOCK_HANDLE
    _LOCK_HANDLE = None
    if handle is not None:
        try:
            os.close(handle)
        except OSError:
            pass
    try:
        path = _lock_path(system)
        if path.is_file():
            try:
                pid = int(path.read_text(encoding="utf-8").splitlines()[0])
            except (OSError, ValueError, IndexError):
                pid = -1
            if pid == os.getpid():
                path.unlink()
    except OSError:
        pass


def _file_signature(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _stable_enough(path: Path, signature: str) -> bool:
    now = time.monotonic()
    key = str(path.resolve())
    with _LOCK:
        previous = _STABILITY.get(key)
        if previous is None or previous[0] != signature:
            _STABILITY[key] = (signature, now)
            return False
        return now - previous[1] >= _STABLE_SECONDS


def _same_content(system: Any, path: Path, document: dict[str, Any], signature: str) -> bool:
    cache_key = str(path.resolve())
    cached = _CONTENT_CACHE.get(cache_key)
    if cached and cached[0] == signature:
        return cached[1] == str(document.get("content_hash") or "")
    try:
        digest = str(system._hash_file(path))
    except Exception:
        digest = None
    with _LOCK:
        _CONTENT_CACHE[cache_key] = (signature, digest)
    return digest is not None and digest == str(document.get("content_hash") or "")


def _trim_caches(existing_paths: set[str]) -> None:
    with _LOCK:
        for cache in (_CONTENT_CACHE, _STABILITY):
            for key in list(cache):
                if key not in existing_paths:
                    cache.pop(key, None)
            while len(cache) > _CACHE_MAX:
                cache.pop(next(iter(cache)))


def _candidate(system: Any, path: Path) -> bool:
    key = str(path.resolve())
    with _LOCK:
        if key in _WORKING:
            return False
    try:
        signature = _file_signature(path)
        if not _stable_enough(path, signature):
            return False
        document = system.state_store.get_by_path(key)
        if document is None:
            return True
        status = str(document.get("status") or "").upper()
        if status in _ACTIVE:
            return False
        same_content = _same_content(system, path, document, signature)
        if status in _TERMINAL_NO_RETRY or status.startswith("FAILED"):
            return not same_content
        if status in {"", "INTERRUPTED", "RECOVERING"}:
            return True
        return not same_content
    except (OSError, ValueError):
        return False


def _finish_job(system: Any, path: Path, result: dict[str, Any] | None = None, error: Exception | None = None) -> None:
    key = str(path.resolve())
    with _LOCK:
        _WORKING.discard(key)
        if result is not None:
            _STATE["last_result"] = result
            result_status = str(result.get("status") or "unknown").lower()
            if result_status in {"success", "skipped"}:
                _STATE["completed"] += 1
            elif result_status in {"failed", "error"}:
                _STATE["failed"] += 1
            _STATE["last_action"] = f"completed {path.name} · {result_status}"
            _STATE["last_error"] = None
        if error is not None:
            _STATE["failed"] += 1
            _STATE["last_error"] = f"{type(error).__name__}: {error}"
            _STATE["last_action"] = f"error on {path.name}"
        _STATE["last_action_at"] = _now()
    _persist(system)


def _run_ingestion(system: Any, path: Path) -> None:
    try:
        result = system.ingest_file(path)
        _finish_job(system, path, result=result)
    except Exception as exc:
        _finish_job(system, path, error=exc)


def _dispatch(system: Any, path: Path) -> None:
    key = str(path.resolve())
    with _LOCK:
        if _WORKING:
            return
        if key in _WORKING:
            return
        _WORKING.add(key)
        _STATE["last_file"] = path.name
        _STATE["last_action"] = f"auto-ingest started · {path.name}"
        _STATE["last_action_at"] = _now()
        _STATE["auto_started"] += 1
        _STATE["last_error"] = None
    _persist(system)
    thread = threading.Thread(target=_run_ingestion, args=(system, path), name=f"bookrag-ingest-{uuid.uuid4().hex[:8]}", daemon=True)
    thread.start()


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
        candidates = sorted((p for p in incoming.glob("*.pdf") if p.is_file()), key=lambda p: p.stat().st_mtime)
    except OSError:
        candidates = []
    _trim_caches({str(p.resolve()) for p in candidates})

    if candidates and not _WORKING:
        for path in candidates:
            if _candidate(system, path):
                _dispatch(system, path)
                break
    _persist(system)


class _PDFEventHandler(FileSystemEventHandler):
    def _wake_for(self, path: str) -> None:
        try:
            if Path(path).suffix.lower() == ".pdf":
                _WAKE.set()
        except (OSError, ValueError):
            pass

    def on_created(self, event) -> None:
        if not event.is_directory:
            self._wake_for(event.src_path)

    def on_modified(self, event) -> None:
        if not event.is_directory:
            self._wake_for(event.src_path)

    def on_moved(self, event) -> None:
        if not event.is_directory:
            self._wake_for(event.dest_path)


def _start_watchdog(incoming: Path) -> bool:
    global _OBSERVER
    try:
        observer = Observer()
        observer.daemon = True
        observer.schedule(_PDFEventHandler(), str(incoming), recursive=False)
        observer.start()
        _OBSERVER = observer
        with _LOCK:
            _STATE["watchdog"] = True
        return True
    except Exception:
        with _LOCK:
            _STATE["watchdog"] = False
        _OBSERVER = None
        return False


def _stop_watchdog() -> None:
    global _OBSERVER
    observer = _OBSERVER
    _OBSERVER = None
    if observer is None:
        return
    try:
        observer.stop()
        observer.join(timeout=2.0)
    except Exception:
        pass


def _loop(system: Any, interval_seconds: float) -> None:
    incoming = Path(system.settings.incoming_dir).resolve()
    incoming.mkdir(parents=True, exist_ok=True)
    _load_persisted_state(system)
    if not _acquire_process_lock(system):
        with _LOCK:
            _STATE["enabled"] = False
            _STATE["watchdog"] = False
            _STATE["last_action"] = "another supervisor process owns this project"
        _persist(system)
        return
    with _LOCK:
        _STATE["enabled"] = True
        _STATE["started_at"] = _STATE["started_at"] or _now()
    _start_watchdog(incoming)
    _persist(system)
    try:
        while not _STOP.is_set():
            try:
                _scan_once(system)
            except Exception as exc:
                with _LOCK:
                    _STATE["last_error"] = f"{type(exc).__name__}: {exc}"
                    _STATE["last_action"] = "supervisor cycle failed"
                    _STATE["last_action_at"] = _now()
                _persist(system)
            _WAKE.wait(max(1.0, float(interval_seconds)))
            _WAKE.clear()
    finally:
        _stop_watchdog()
        _release_process_lock(system)
        with _LOCK:
            _STATE["enabled"] = False
            _STATE["watchdog"] = False
        _persist(system)


def start(system: Any, interval_seconds: float = 1.0) -> dict[str, Any]:
    global _THREAD
    with _LOCK:
        if _THREAD and _THREAD.is_alive():
            return dict(_STATE)
        _STOP.clear()
        _WAKE.clear()
        _THREAD = threading.Thread(
            target=_loop,
            args=(system, max(1.0, float(interval_seconds))),
            name=f"bookrag-responsive-supervisor-{uuid.uuid4().hex[:8]}",
            daemon=True,
        )
        _THREAD.start()
        return dict(_STATE)


def stop() -> None:
    _STOP.set()
    _WAKE.set()
    _stop_watchdog()
    with _LOCK:
        _STATE["enabled"] = False
        _STATE["watchdog"] = False


__all__ = ["start", "stop", "snapshot"]
