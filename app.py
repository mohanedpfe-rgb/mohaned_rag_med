from __future__ import annotations

import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import streamlit as st

from rag_project.auth import require_auth

st.set_page_config(
    page_title="BookRAG Medical",
    page_icon="BR",
    layout="wide",
    initial_sidebar_state="expanded",
)
require_auth()


def _load_local_env() -> None:
    root = Path(__file__).resolve().parent
    env_file = root / ".env"
    if not env_file.is_file():
        return
    try:
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or key in os.environ:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ[key] = value
    except OSError:
        return


_load_local_env()

_BOOT_LOCK = threading.RLock()
_BOOT: dict[str, object] = {
    "status": "idle",
    "system": None,
    "error": None,
    "started_at": 0.0,
}
_LAZY_SETTINGS: Any | None = None
_LAZY_STATE_STORE: Any | None = None
_LAZY_SYSTEM: Any | None = None


def _clamp_local_embedding_profile() -> None:
    try:
        retries = max(1, min(int(os.getenv("EMBEDDING_RETRIES", "2")), 3))
    except ValueError:
        retries = 2
    try:
        timeout = max(30.0, min(float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "180")), 300.0))
    except ValueError:
        timeout = 180.0
    try:
        batch_size = max(16, min(int(os.getenv("EMBEDDING_BATCH_SIZE", "16")), 32))
    except ValueError:
        batch_size = 16
    os.environ["EMBEDDING_RETRIES"] = str(retries)
    os.environ["EMBEDDING_TIMEOUT_SECONDS"] = str(timeout)
    os.environ["EMBEDDING_BATCH_SIZE"] = str(batch_size)


def _write_bootstrap_diagnostic(exc: BaseException) -> None:
    try:
        log_dir = Path(os.getenv("LOG_DIR", "logs")).expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        target = log_dir / "runtime_bootstrap.log"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        target.open("a", encoding="utf-8").write(
            f"\n=== {timestamp} ===\n{traceback.format_exc()}"
        )
    except OSError:
        pass


def _get_lazy_system() -> Any:
    global _LAZY_SETTINGS, _LAZY_STATE_STORE, _LAZY_SYSTEM
    with _BOOT_LOCK:
        if _LAZY_SYSTEM is not None:
            return _LAZY_SYSTEM
        from rag_project.configuration.settings import Settings
        from rag_project.ingestion.state_store import IngestionStateStore
        _LAZY_SETTINGS = Settings.from_env()
        _LAZY_STATE_STORE = IngestionStateStore(_LAZY_SETTINGS.ingestion_db_path)

        class LazySystem:
            def _real(self) -> Any | None:
                with _BOOT_LOCK:
                    runtime = _BOOT.get("system")
                if isinstance(runtime, tuple) and runtime:
                    return runtime[0]
                return None

            @property
            def settings(self) -> Any:
                return _LAZY_SETTINGS

            @property
            def state_store(self) -> Any:
                return _LAZY_STATE_STORE

            @property
            def ready(self) -> bool:
                return self._real() is not None

            def health_report(self) -> dict[str, Any]:
                real = self._real()
                if real is not None:
                    return real.health_report()
                with _BOOT_LOCK:
                    status = str(_BOOT.get("status") or "starting")
                    error = _BOOT.get("error")
                if status == "error":
                    return {"ready": False, "error": error or "Runtime bootstrap failed."}
                return {
                    "ready": False,
                    "status": "BOOTSTRAPPING",
                    "message": "BookRAG services are initializing in the background.",
                    "embedding": {"ok": False, "error": "Runtime initializing"},
                    "index": {"status": "STARTING"},
                    "feature_contract": {"all_resolved": False},
                }

            def __getattr__(self, name: str) -> Any:
                real = self._real()
                if real is not None:
                    return getattr(real, name)

                def not_ready(*args: Any, **kwargs: Any) -> Any:
                    raise RuntimeError("BookRAG services are still initializing. Please try that action again in a moment.")

                return not_ready

        _LAZY_SYSTEM = LazySystem()
        return _LAZY_SYSTEM


def _clamped_settings_for_runtime() -> Any:
    return _get_lazy_system().settings


def _boot_start() -> None:
    with _BOOT_LOCK:
        if _BOOT["status"] in {"starting", "ready"}:
            return
        _BOOT.update(status="starting", error=None, started_at=time.monotonic())

    def worker() -> None:
        try:
            from datetime import datetime, timezone

            _clamp_local_embedding_profile()

            from rag_project.app import bookrag_ui
            from rag_project.app import rag_system as rag_system_module
            from rag_project.app.bookrag_ui import main as ui_main
            from rag_project.app.live_runtime import render_live_runtime
            from rag_project.ingestion.responsive_supervisor import start as start_auto_supervisor
            from rag_project.application import create_rag_system
            from rag_project.security import (
                register_session_upload,
                validate_ollama_url,
                validate_pdf_payload,
                validate_storage_path,
            )

            bookrag_ui.st.set_page_config = lambda *args, **kwargs: None
            if not callable(getattr(rag_system_module, "utc_now", None)):
                rag_system_module.utc_now = lambda: datetime.now(timezone.utc).isoformat()

            system = create_rag_system(_clamped_settings_for_runtime())
            original_save_pdf = bookrag_ui.save_pdf
            original_start_ingestion = bookrag_ui.start_ingestion
            original_ollama_health = bookrag_ui.ollama_health

            def secure_system():
                return _get_lazy_system()

            def secure_save_pdf(incoming, name, content):
                safe_incoming = validate_storage_path(system.settings.project_root, incoming, "incoming folder")
                validate_pdf_payload(name, content)
                register_session_upload(len(content))
                return original_save_pdf(safe_incoming, name, content)

            def secure_start_ingestion(ignored_system, source_dir, *, trigger="manual"):
                safe_source = validate_storage_path(system.settings.project_root, source_dir, "incoming folder")
                return original_start_ingestion(system, str(safe_source), trigger=trigger)

            def secure_ollama_health(base_url):
                return original_ollama_health(validate_ollama_url(base_url))

            bookrag_ui.get_system = secure_system
            bookrag_ui.save_pdf = secure_save_pdf
            bookrag_ui.start_ingestion = secure_start_ingestion
            bookrag_ui.ollama_health = secure_ollama_health

            with _BOOT_LOCK:
                _BOOT.update(
                    status="ready",
                    system=(system, render_live_runtime, ui_main, start_auto_supervisor),
                    error=None,
                )
        except Exception as exc:
            _write_bootstrap_diagnostic(exc)
            with _BOOT_LOCK:
                _BOOT.update(status="error", error=f"{type(exc).__name__}")

    threading.Thread(target=worker, name="bookrag-runtime-bootstrap", daemon=True).start()


def _render_boot_banner() -> None:
    with _BOOT_LOCK:
        status = str(_BOOT.get("status") or "idle")
        error = _BOOT.get("error")
    if status == "starting":
        st.info("BookRAG services are loading in the background. The workspace is already available.")
    elif status == "error":
        st.error(f"BookRAG runtime is unavailable: {error or 'bootstrap failed'}. See logs/runtime_bootstrap.log for the diagnostic traceback.")


def main() -> None:
    _get_lazy_system()
    _boot_start()

    from rag_project.app import bookrag_ui
    from rag_project.app.bookrag_ui import main as ui_main
    from rag_project.app.live_runtime import render_live_runtime

    bookrag_ui.get_system = lambda: _get_lazy_system()

    _render_boot_banner()
    render_live_runtime(_get_lazy_system())
    ui_main()

    with _BOOT_LOCK:
        runtime = _BOOT.get("system")
    if isinstance(runtime, tuple) and len(runtime) == 4:
        system, _, _, start_auto_supervisor = runtime
        start_auto_supervisor(system, interval_seconds=1.0)


if __name__ == "__main__":
    main()
