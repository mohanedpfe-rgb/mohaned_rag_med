from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any


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

import streamlit as st
from rag_project.security import require_auth

st.set_page_config(
    page_title="BookRAG Medical",
    page_icon="BR",
    layout="wide",
    initial_sidebar_state="expanded",
)

_BOOT_LOCK = threading.RLock()
_BOOT: dict[str, object] = {
    "status": "idle",
    "system": None,
    "error": None,
    "started_at": 0.0,
}
_LAZY_SETTINGS: Any | None = None
_LAZY_STATE_STORE: Any | None = None


def _clamp_local_embedding_profile() -> None:
    """Bound local embedding waits so one Ollama stall cannot become a UI freeze."""
    try:
        retries = max(0, min(int(os.getenv("EMBEDDING_RETRIES", "1")), 1))
    except ValueError:
        retries = 1
    try:
        timeout = max(5.0, min(float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "20")), 20.0))
    except ValueError:
        timeout = 20.0
    try:
        batch_size = max(1, min(int(os.getenv("EMBEDDING_BATCH_SIZE", "4")), 4))
    except ValueError:
        batch_size = 4
    os.environ["EMBEDDING_RETRIES"] = str(retries)
    os.environ["EMBEDDING_TIMEOUT_SECONDS"] = str(timeout)
    os.environ["EMBEDDING_BATCH_SIZE"] = str(batch_size)


def _get_lazy_resources() -> tuple[Any, Any]:
    global _LAZY_SETTINGS, _LAZY_STATE_STORE
    with _BOOT_LOCK:
        if _LAZY_SETTINGS is None:
            from rag_project.configuration.settings import Settings
            _LAZY_SETTINGS = Settings.from_env()
        if _LAZY_STATE_STORE is None:
            from rag_project.ingestion.state_store import IngestionStateStore
            _LAZY_STATE_STORE = IngestionStateStore(_LAZY_SETTINGS.ingestion_db_path)
        return _LAZY_SETTINGS, _LAZY_STATE_STORE


class _LazySystem:
    """Lightweight UI facade used while the heavy RAG runtime initializes."""

    def __init__(self) -> None:
        self.settings, self.state_store = _get_lazy_resources()

    def _real(self) -> Any | None:
        with _BOOT_LOCK:
            runtime = _BOOT.get("system")
        if isinstance(runtime, tuple) and runtime:
            return runtime[0]
        return None

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
        if name == "settings":
            return self.settings
        if name == "state_store":
            return self.state_store

        def not_ready(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("BookRAG services are still initializing. Please try that action again in a moment.")

        return not_ready


_LAZY_SYSTEM = _LazySystem()


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
            from rag_project.configuration.settings import Settings
            from rag_project.security import (
                register_session_upload,
                require_clear_confirmation,
                validate_ollama_url,
                validate_pdf_payload,
                validate_query,
                validate_storage_path,
            )

            bookrag_ui.st.set_page_config = lambda *args, **kwargs: None
            if not callable(getattr(rag_system_module, "utc_now", None)):
                rag_system_module.utc_now = lambda: datetime.now(timezone.utc).isoformat()

            original_save_pdf = bookrag_ui.save_pdf
            original_start_ingestion = bookrag_ui.start_ingestion
            original_ollama_health = bookrag_ui.ollama_health

            system = create_rag_system(Settings.from_env())
            if not getattr(system, "_bookrag_security_wrapped", False):
                original_clear = system.clear_pdf_data
                original_apply = system.apply_settings_in_place
                original_answer = system.answer

                def guarded_clear() -> None:
                    require_clear_confirmation()
                    original_clear()

                def guarded_apply(updates):
                    clean = dict(updates or {})
                    if "ollama_base_url" in clean:
                        clean["ollama_base_url"] = validate_ollama_url(clean["ollama_base_url"])
                    for key in (
                        "incoming_dir", "processed_dir", "failed_dir", "archive_dir", "vector_db_dir",
                        "log_dir", "ingestion_db_path",
                    ):
                        if key in clean:
                            clean[key] = validate_storage_path(system.settings.project_root, clean[key], key)
                    return original_apply(clean)

                def guarded_answer(question, metadata_filter=None):
                    return original_answer(validate_query(question), metadata_filter)

                system.clear_pdf_data = guarded_clear
                system.apply_settings_in_place = guarded_apply
                system.answer = guarded_answer
                system._bookrag_security_wrapped = True

            def secure_system():
                return _LAZY_SYSTEM

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
            with _BOOT_LOCK:
                _BOOT.update(status="error", error=f"{type(exc).__name__}: {exc}")

    threading.Thread(target=worker, name="bookrag-runtime-bootstrap", daemon=True).start()


def _render_boot_banner() -> None:
    with _BOOT_LOCK:
        status = str(_BOOT.get("status") or "idle")
        error = _BOOT.get("error")
    if status == "starting":
        st.info("BookRAG services are loading in the background. The workspace is already available.")
    elif status == "error":
        st.error(f"BookRAG runtime is unavailable: {error}")


def main() -> None:
    if not require_auth():
        return

    # Never gate the main UI on heavy runtime initialization. The workspace opens
    # immediately after authentication and the runtime becomes progressively live.
    _boot_start()

    from rag_project.app import bookrag_ui
    from rag_project.app.bookrag_ui import main as ui_main
    from rag_project.app.live_runtime import render_live_runtime

    # The module-level get_system() is intentionally replaced with the lazy facade
    # before the UI renders, preventing a cache miss from constructing the RAG stack
    # on the Streamlit script thread.
    bookrag_ui.get_system = lambda: _LAZY_SYSTEM

    _render_boot_banner()
    render_live_runtime(_LAZY_SYSTEM)
    ui_main()

    with _BOOT_LOCK:
        runtime = _BOOT.get("system")
    if isinstance(runtime, tuple) and len(runtime) == 4:
        system, _, _, start_auto_supervisor = runtime
        start_auto_supervisor(system, interval_seconds=1.0)


if __name__ == "__main__":
    main()
