from __future__

import hashlib
import html
import io
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from rag_project.application import create_rag_system
from rag_project.configuration.settings import Settings
from rag_project.security import register_session_upload, validate_pdf_payload

PAGES = ["Home", "Documents", "Live Processing", "Ask BookRAG", "Inspector", "System", "Settings"]
ACTIVE = {
    "RUNNING", "DISCOVERED", "VALIDATING", "EXTRACTING", "OCR", "CHUNKING",
    "EMBEDDING", "INDEXING", "VALIDATING_INDEX", "BUILDING", "INTERRUPTED", "RECOVERING",
}
STAGES = {
    "RUNNING": ("Starting", "The document has entered the pipeline."),
    "DISCOVERED": ("Discovered", "The PDF was accepted and queued."),
    "VALIDATING": ("Validating", "Checking the PDF structure and safety."),
    "EXTRACTING": ("Extracting", "Reading the document page by page."),
    "OCR": ("OCR", "Recovering text from scanned pages."),
    "CHUNKING": ("Chunking", "Turning extracted content into retrieval sections."),
    "EMBEDDING": ("Embedding", "Creating semantic search vectors."),
    "INDEXING": ("Indexing", "Writing semantic and lexical search records."),
    "VALIDATING_INDEX": ("Verifying", "Checking the new index before publication."),
    "READY": ("Ready", "Available for grounded research questions."),
    "COMPLETED": ("Ready", "Available for grounded research questions."),
    "FAILED": ("Failed", "Processing stopped and needs attention."),
    "FAILED_EMBEDDING": ("Embedding failed", "Semantic indexing could not complete."),
    "FAILED_INDEXING": ("Indexing failed", "Final index validation could not complete."),
    "INTERRUPTED": ("Interrupted", "Processing stopped before completion."),
    "RECOVERING": ("Recovering", "Preparing a safe recovery attempt."),
}


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"))


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _elapsed(start: Any, end: Any | None = None) -> float:
    started = _utc(start)
    if not started:
        return 0.0
    ended = _utc(end) or datetime.now(timezone.utc)
    return max(0.0, (ended - started).total_seconds())


def _fmt_bytes(size: Any) -> str:
    n = max(0, _int(size))
    units = ["B", "KB", "MB", "GB"]
    i = 0
    value = float(n)
    while value >= 1024 and i < len(units) - 1:
        value /= 1024
        i += 1
    return f"{value:.1f} {units[i]}"


def _fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


@st.cache_resource(show_spinner=False)
def get_system():
    return create_rag_system(Settings.from_env())


@st.cache_resource(show_spinner=False)
def get_jobs() -> dict[str, Any]:
    return {"lock": threading.RLock(), "items": {}}


def docs(system) -> list[dict[str, Any]]:
    try:
        return list(system.state_store.get_all_documents() or [])
    except Exception:
        return []


def pages(system, document_id: str) -> list[dict[str, Any]]:
    try:
        return list(system.state_store.get_pages(document_id) or [])
    except Exception:
        return []


def events(system, document_id: str | None = None, limit: int = 250) -> list[dict[str, Any]]:
    try:
        return list(system.state_store.get_events(document_id, limit=limit) or [])
    except Exception:
        return []


def ready_docs(system) -> list[dict[str, Any]]:
    return [d for d in docs(system) if str(d.get("status", "")).upper() in {"READY", "COMPLETED"}]


def active_docs(system) -> list[dict[str, Any]]:
    return [d for d in docs(system) if str(d.get("status", "")).upper() in ACTIVE]


def total_chunks(system) -> int:
    return sum(_int(d.get("chunk_count", d.get("chunks", 0))) for d in docs(system))


def total_embeddings(system) -> int:
    return sum(_int(d.get("embedding_count", d.get("embeddings", 0))) for d in docs(system))


def _doc_name(document: dict[str, Any]) -> str:
    return str(document.get("file_name") or document.get("filename") or document.get("name") or document.get("document_id") or "Document")


def _doc_date(document: dict[str, Any]) -> datetime | None:
    for key in ("modified_at", "ingestion_completed_at", "ingestion_started_at", "created_at"):
        value = _utc(document.get(key))
        if value:
            return value
    return None


def _stage(document: dict[str, Any]) -> tuple[str, str]:
    key = str(document.get("current_stage") or document.get("status") or "RUNNING").upper()
    return STAGES.get(key, (key.replace("_", " ").title(), "Processing document."))


def _status_class(value: Any) -> str:
    status = str(value or "UNKNOWN").upper()
    if status in {"READY", "COMPLETED", "PASS", "ONLINE", "HEALTHY", "OK", "GROUNDED"}:
        return "good"
    if status in {"FAILED", "FAILED_EMBEDDING", "FAILED_INDEXING", "ERROR", "OFFLINE", "UNAVAILABLE", "ABSTAIN", "INTERRUPTED"}:
        return "bad"
    if status in ACTIVE or status in {"RUNNING", "PROCESSING", "BUILDING", "WARNING", "WARN"}:
        return "warn"
    return "neutral"


def _status(value: Any) -> str:
    text = str(value or "UNKNOWN")
    return f'<span class="pill pill-{_status_class(text)}"><i></i>{_esc(text)}</span>'


def _progress(document: dict[str, Any]) -> float:
    status = str(document.get("status") or document.get("current_stage") or "RUNNING").upper()
    base = {
        "RUNNING": 0.03, "DISCOVERED": 0.08, "VALIDATING": 0.15, "EXTRACTING": 0.30,
        "OCR": 0.46, "CHUNKING": 0.58, "EMBEDDING": 0.73, "INDEXING": 0.86,
        "VALIDATING_INDEX": 0.96, "READY": 1.0, "COMPLETED": 1.0,
    }.get(status, 0.03)
    if status in {"FAILED", "FAILED_EMBEDDING", "FAILED_INDEXING", "INTERRUPTED"}:
        return 0.0
    total, current = _int(document.get("total_pages")), _int(document.get("current_page"))
    if status in {"EXTRACTING", "OCR"} and total:
        base += min(current / total, 1.0) * (0.14 if status == "EXTRACTING" else 0.10)
    return max(0.0, min(base, 1.0))


def _navigate(page: str) -> None:
    if page in PAGES:
        st.session_state["bookrag_page"] = page
        st.rerun()


def _save_pdf(incoming: Path, name: str, content: bytes) -> str:
    validate_pdf_payload(name, content)
    incoming = Path(incoming).expanduser().resolve()
    digest = hashlib.sha256(content).hexdigest()
    stem = "".join(c if c.isalnum() or c in ".-_" else "_" for c in (Path(name).stem or "document")).strip(" ._") or "document"
    incoming.mkdir(parents=True, exist_ok=True)
    target = incoming / f"{stem[:80]}_{digest[:12]}.pdf"
    if not target.exists():
        target.write_bytes(content)
        register_session_upload(len(content))
    return digest


def save_pdf(incoming: Path, name: str, content: bytes) -> str:
    return _save_pdf(incoming, name, content)


def start_ingestion(system, source_dir: str, *, trigger: str = "manual") -> str:
    folder = Path(source_dir).expanduser().resolve()
    root = Path(system.settings.project_root).expanduser().resolve()
