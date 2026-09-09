from __future__ import annotations

import os
import threading
import time
from pathlib import Path


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


def _boot_start() -> None:
    with _BOOT_LOCK:
        if _BOOT["status"] in {"starting", "ready"}:
            return
        _BOOT.update(status="starting", error=None, started_at=time.monotonic())

    def worker() -> None:
        try:
            from datetime import datetime, timezone

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
                return system

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


def _render_boot_screen() -> None:
    st.markdown(
        """<style>
        :root{--bg:#080d16;--panel:#0e1623;--line:#243246;--text:#edf3fa;--muted:#8493a7;--accent:#63d9d1;--blue:#7e9cff;}
        html,body,[class*='css']{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
        .stApp{background:radial-gradient(900px 500px at 80% -10%,rgba(126,156,255,.12),transparent 60%),radial-gradient(700px 450px at 10% 0,rgba(99,217,209,.07),transparent 62%),var(--bg);color:var(--text)}
        [data-testid='stHeader']{height:0;background:transparent}
        .block-container{max-width:1180px;padding:58px 28px 40px}
        .br-shell{max-width:760px;margin:8vh auto 0;padding:42px;border:1px solid var(--line);border-radius:28px;background:linear-gradient(145deg,rgba(17,28,43,.96),rgba(9,16,26,.96));box-shadow:0 28px 90px rgba(0,0,0,.38)}
        .br-mark{display:flex;gap:14px;align-items:center;margin-bottom:26px}.br-icon{width:52px;height:52px;border-radius:16px;display:grid;place-items:center;background:linear-gradient(145deg,#63d9d1,#7e9cff);color:#071018;font-weight:900;letter-spacing:.04em}.br-title{font-size:30px;font-weight:800;letter-spacing:-.03em}.br-sub{color:var(--muted);font-size:14px;margin-top:2px}.br-badge{display:inline-flex;padding:6px 10px;border:1px solid rgba(99,217,209,.25);border-radius:999px;color:var(--accent);font-size:12px;font-weight:700;margin-bottom:20px}
        </style>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<div class='br-shell'><div class='br-mark'><div class='br-icon'>BR</div><div><div class='br-title'>BookRAG Medical</div><div class='br-sub'>Local evidence-first research workspace</div></div></div><div class='br-badge'>SECURE LOCAL WORKSPACE</div></div>""",
        unsafe_allow_html=True,
    )


def _render_boot_state() -> None:
    _render_boot_screen()
    with _BOOT_LOCK:
        status = str(_BOOT["status"])
        error = _BOOT["error"]
        started = float(_BOOT["started_at"] or 0.0)
    if status == "starting":
        elapsed = max(0.0, time.monotonic() - started)
        st.info("Opening your workspace… services are warming in the background.")
        st.progress(min(elapsed / 8.0, 0.92), text=f"Preparing BookRAG · {elapsed:.1f}s")
    elif status == "error":
        st.error(f"BookRAG could not start: {error}")
        if st.button("Retry startup", type="primary", use_container_width=True):
            with _BOOT_LOCK:
                _BOOT["status"] = "idle"
            st.rerun()


@st.fragment(run_every="0.25s")
def _bootstrap_fragment() -> None:
    _render_boot_state()
    with _BOOT_LOCK:
        status = _BOOT["status"]
    if status == "ready":
        st.rerun()


def main() -> None:
    if not require_auth():
        return

    with _BOOT_LOCK:
        status = str(_BOOT["status"])
        runtime = _BOOT["system"]

    if status != "ready":
        _boot_start()
        _bootstrap_fragment()
        return

    system, render_live_runtime, ui_main, start_auto_supervisor = runtime  # type: ignore[misc]
    start_auto_supervisor(system, interval_seconds=1.0)
    render_live_runtime(system)
    ui_main()


if __name__ == "__main__":
    main()
