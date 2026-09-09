from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from rag_project.app import bookrag_ui
from rag_project.app import rag_system as rag_system_module
from rag_project.app.bookrag_ui import main as ui_main
from rag_project.ingestion.auto_supervisor import start as start_auto_supervisor
from rag_project.security import (
    register_session_upload,
    require_auth,
    require_clear_confirmation,
    validate_ollama_url,
    validate_pdf_payload,
    validate_query,
    validate_storage_path,
)

# Defensive compatibility guard for older/stale deployments where the helper may
# be absent from the imported RAG module even though the canonical helper exists
# in the persistent ingestion state store.
if not callable(getattr(rag_system_module, "utc_now", None)):
    rag_system_module.utc_now = lambda: datetime.now(timezone.utc).isoformat()

_ORIGINAL_GET_SYSTEM = bookrag_ui.get_system
_ORIGINAL_SAVE_PDF = bookrag_ui.save_pdf
_ORIGINAL_START_INGESTION = bookrag_ui.start_ingestion
_ORIGINAL_OLLAMA_HEALTH = bookrag_ui.ollama_health


def _secure_system():
    system = _ORIGINAL_GET_SYSTEM()
    if getattr(system, "_bookrag_security_wrapped", False):
        return system

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
    return system


_secure_system.clear = getattr(_ORIGINAL_GET_SYSTEM, "clear", lambda: None)
bookrag_ui.get_system = _secure_system


def _secure_save_pdf(incoming, name, content):
    system = _secure_system()
    safe_incoming = validate_storage_path(system.settings.project_root, incoming, "incoming folder")
    validate_pdf_payload(name, content)
    register_session_upload(len(content))
    return _ORIGINAL_SAVE_PDF(safe_incoming, name, content)


def _secure_start_ingestion(system, source_dir, *, trigger="manual"):
    safe_source = validate_storage_path(system.settings.project_root, source_dir, "incoming folder")
    return _ORIGINAL_START_INGESTION(system, str(safe_source), trigger=trigger)


def _secure_ollama_health(base_url):
    return _ORIGINAL_OLLAMA_HEALTH(validate_ollama_url(base_url))


bookrag_ui.save_pdf = _secure_save_pdf
bookrag_ui.start_ingestion = _secure_start_ingestion
bookrag_ui.ollama_health = _secure_ollama_health


def main() -> None:
    if not require_auth():
        return
    system = _secure_system()
    # Start one autonomous, process-level watcher after authentication. The UI
    # remains a consumer of durable state; ingestion does not depend on a button
    # click or on the browser session staying on the Documents page.
    start_auto_supervisor(system, interval_seconds=3.0)
    ui_main()


if __name__ == "__main__":
    main()
