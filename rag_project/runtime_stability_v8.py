from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()
_WATCHERS: set[str] = set()
_MAX_UI_JOBS = 256


def _as_list(value: Any) -> list[Any]:
    """Normalize sequence-like UI results without implicit array truth evaluation."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except (TypeError, ValueError):
        return []


def _safe_evidence(result: Any) -> list[Any]:
    if not isinstance(result, dict):
        return []
    value = result.get("evidence")
    if value is None:
        value = result.get("hits")
    if value is None:
        value = result.get("citations")
    return _as_list(value)


def _safe_ollama_health(original, base_url: str):
    from rag_project.security import validate_ollama_url
    import requests

    validated = validate_ollama_url(base_url)
    try:
        response = requests.get(
            f"{validated.rstrip('/')}/api/tags",
            timeout=(2.5, 5),
            allow_redirects=False,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Ollama health response was not an object")
        models = [
            str(item.get("name"))
            for item in _as_list(payload.get("models"))
            if isinstance(item, dict) and item.get("name")
        ]
        return True, "Ollama is reachable", models
    except Exception as exc:
        return False, f"{type(exc).__name__}", []


def _prune_jobs(jobs: dict[str, Any]) -> None:
    items = jobs.get("items")
    if not isinstance(items, dict) or len(items) <= _MAX_UI_JOBS:
        return
    removable = [
        (str(job.get("finished") or 0), str(job_id))
        for job_id, job in items.items()
        if isinstance(job, dict) and job.get("status") != "RUNNING"
    ]
    removable.sort()
    for _, job_id in removable[: max(0, len(items) - _MAX_UI_JOBS)]:
        items.pop(job_id, None)


def _watch_upload_queue(module: Any, system: Any, folder: Path) -> None:
    key = str(folder.resolve())
    with _LOCK:
        if key in _WATCHERS:
            return
        _WATCHERS.add(key)

    try:
        while True:
            jobs = module.get_jobs()
            with jobs["lock"]:
                _prune_jobs(jobs)
                running = any(job.get("status") == "RUNNING" for job in jobs["items"].values())
            pending = sorted(path for path in folder.glob("*.pdf") if path.is_file()) if folder.is_dir() else []
            if not running:
                if not pending:
                    return
                try:
                    module.start_ingestion(system, str(folder), trigger="upload-followup")
                except Exception:
                    return
            time.sleep(0.5)
    finally:
        with _LOCK:
            _WATCHERS.discard(key)


def _safe_start_ingestion(original, module: Any, system: Any, source_dir: str, *, trigger: str = "manual") -> str:
    job_id = original(system, source_dir, trigger=trigger)
    folder = Path(source_dir).expanduser().resolve()
    threading.Thread(
        target=_watch_upload_queue,
        args=(module, system, folder),
        name="bookrag-upload-drain",
        daemon=True,
    ).start()
    return job_id


def install() -> None:
    try:
        from rag_project.app import bookrag_ui
    except Exception:
        return

    if not hasattr(bookrag_ui, "_runtime_v8_original_evidence"):
        bookrag_ui._runtime_v8_original_evidence = bookrag_ui._evidence
        bookrag_ui._evidence = _safe_evidence

    if not hasattr(bookrag_ui, "_runtime_v8_original_ollama_health"):
        original_health = bookrag_ui.ollama_health
        bookrag_ui._runtime_v8_original_ollama_health = original_health
        bookrag_ui.ollama_health = lambda base_url: _safe_ollama_health(original_health, base_url)

    if not hasattr(bookrag_ui, "_runtime_v8_original_start_ingestion"):
        original_start = bookrag_ui.start_ingestion
        bookrag_ui._runtime_v8_original_start_ingestion = original_start
        bookrag_ui.start_ingestion = lambda system, source_dir, *, trigger="manual": _safe_start_ingestion(original_start, bookrag_ui, system, source_dir, trigger=trigger)
