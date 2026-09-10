from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st

from rag_project.app import bookrag_ui
from rag_project.ingestion.responsive_supervisor import start as start_supervisor
from rag_project.security import (
    register_session_upload,
    validate_ollama_url,
    validate_pdf_payload,
    validate_storage_path,
)


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


def _clamp_local_embedding_profile() -> None:
    """Repair legacy/unsafe local embedding settings before application startup."""
    try:
        batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
    except (TypeError, ValueError):
        batch_size = 16
    try:
        retries = int(os.getenv("EMBEDDING_RETRIES", "2"))
    except (TypeError, ValueError):
        retries = 2
    try:
        timeout_seconds = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "180"))
    except (TypeError, ValueError):
        timeout_seconds = 180.0

    os.environ["EMBEDDING_BATCH_SIZE"] = str(max(16, min(batch_size, 32)))
    os.environ["EMBEDDING_RETRIES"] = str(max(1, min(retries, 3)))
    os.environ["EMBEDDING_TIMEOUT_SECONDS"] = str(max(30.0, min(timeout_seconds, 300.0)))


def _install_ui_guards() -> None:
    if getattr(bookrag_ui, "_bookrag_ui_guards_installed", False):
        return

    original_save_pdf = bookrag_ui.save_pdf
    original_start_ingestion = bookrag_ui.start_ingestion
    original_ollama_health = bookrag_ui.ollama_health

    def secure_save_pdf(incoming: Any, name: str, content: bytes) -> str:
        system = bookrag_ui.get_system()
        safe_incoming = validate_storage_path(
            system.settings.project_root,
            incoming,
            "incoming folder",
        )
        validate_pdf_payload(name, content)
        register_session_upload(len(content))
        return original_save_pdf(safe_incoming, name, content)

    def secure_start_ingestion(
        system: Any,
        source_dir: str,
        *,
        trigger: str = "manual",
    ) -> str:
        safe_source = validate_storage_path(
            system.settings.project_root,
            source_dir,
            "incoming folder",
        )
        return original_start_ingestion(system, str(safe_source), trigger=trigger)

    def secure_ollama_health(base_url: str):
        return original_ollama_health(validate_ollama_url(base_url))

    bookrag_ui.save_pdf = secure_save_pdf
    bookrag_ui.start_ingestion = secure_start_ingestion
    bookrag_ui.ollama_health = secure_ollama_health
    bookrag_ui._bookrag_ui_guards_installed = True


def main() -> None:
    _load_local_env()
    _clamp_local_embedding_profile()
    _install_ui_guards()
    system = bookrag_ui.get_system()
    start_supervisor(system, interval_seconds=1.0)
    bookrag_ui.main()


if __name__ == "__main__":
    main()
