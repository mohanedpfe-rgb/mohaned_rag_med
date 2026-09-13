from __future__ import annotations

from typing import Any

import streamlit as st

from rag_project.app import bookrag_ui
from rag_project.app.advanced_intelligence_panel import render_advanced_intelligence_panel
from rag_project.app.intelligence_panel import render_intelligence_panel
from rag_project.app.production_contract_panel import render_production_contract_panel
from rag_project.security import (
    register_session_upload,
    validate_ollama_url,
    validate_pdf_payload,
    validate_storage_path,
)


def install() -> None:
    """Install security guards around the already-composed Streamlit UI surface."""
    if getattr(bookrag_ui, "_bookrag_ui_guards_installed", False):
        return

    save = bookrag_ui.save_pdf
    start = bookrag_ui.start_ingestion
    health = bookrag_ui.ollama_health
    ask = bookrag_ui.ask_page
    evidence_row = getattr(bookrag_ui, "_evidence_row", None)

    def secure_save(incoming: Any, name: str, content: bytes) -> str:
        system = bookrag_ui.get_system()
        safe = validate_storage_path(system.settings.project_root, incoming, "incoming folder")
        validate_pdf_payload(name, content)
        register_session_upload(len(content))
        return save(safe, name, content)

    def secure_start(system: Any, source_dir: str, *, trigger: str = "manual") -> str:
        safe = validate_storage_path(system.settings.project_root, source_dir, "incoming folder")
        return start(system, str(safe), trigger=trigger)

    def secure_health(url: str):
        return health(validate_ollama_url(url))

    def enhanced_ask(system: Any) -> None:
        ask(system)
        result = st.session_state.get("answer_result")
        if isinstance(result, dict):
            render_intelligence_panel(result)
            render_advanced_intelligence_panel(result)
            render_production_contract_panel(result)

    def enhanced_evidence_row(item: Any, index: int):
        if isinstance(item, dict):
            return (
                item.get("file_name")
                or item.get("filename")
                or item.get("document_id")
                or f"Source {index}",
                item.get("page_number")
                or item.get("page")
                or item.get("page_numbers")
                or "—",
                item.get("rerank_score")
                or item.get("score")
                or item.get("similarity")
                or item.get("relevance")
                or "—",
                str(item.get("snippet") or item.get("text") or item.get("content") or ""),
            )

        metadata = getattr(item, "metadata", {}) or {}
        title = (
            metadata.get("file_name")
            or metadata.get("filename")
            or metadata.get("document_id")
            or getattr(item, "doc_id", None)
            or f"Source {index}"
        )
        page = metadata.get("page_numbers") or metadata.get("page_number") or metadata.get("page") or "—"
        score = getattr(item, "score", None)
        if score is None:
            score = getattr(item, "rerank_score", None)
        if score is None:
            score = "—"
        return str(title), str(page), score, str(getattr(item, "text", "") or "")

    bookrag_ui.save_pdf = secure_save
    bookrag_ui.start_ingestion = secure_start
    bookrag_ui.ollama_health = secure_health
    bookrag_ui.ask_page = enhanced_ask
    if evidence_row is not None:
        bookrag_ui._evidence_row = enhanced_evidence_row
    bookrag_ui._bookrag_ui_guards_installed = True


__all__ = ["install"]
