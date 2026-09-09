from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import streamlit as st

from rag_project.ingestion.responsive_supervisor import snapshot as supervisor_snapshot

_ACTIVE = {
    "RUNNING", "DISCOVERED", "VALIDATING", "EXTRACTING", "OCR",
    "CHUNKING", "EMBEDDING", "INDEXING", "VALIDATING_INDEX", "BUILDING",
    "INTERRUPTED", "RECOVERING",
}


def _utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _age(value: Any) -> str:
    parsed = _utc(value)
    if not parsed:
        return "—"
    seconds = max(0, int((datetime.now(timezone.utc) - parsed).total_seconds()))
    if seconds < 2:
        return "now"
    if seconds < 60:
        return f"{seconds}s ago"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s ago"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m ago"


def _progress(document: dict[str, Any]) -> float:
    status = str(document.get("status") or document.get("current_stage") or "RUNNING").upper()
    if status in {"READY", "COMPLETED"}:
        return 1.0
    if status in {"FAILED", "FAILED_EMBEDDING", "FAILED_EXTRACTION", "FAILED_OCR", "FAILED_INDEXING", "DEGRADED_LEXICAL", "QUARANTINED"}:
        return 0.0
    stage_weight = {
        "RUNNING": 0.02,
        "DISCOVERED": 0.06,
        "VALIDATING": 0.12,
        "EXTRACTING": 0.30,
        "OCR": 0.45,
        "CHUNKING": 0.58,
        "EMBEDDING": 0.73,
        "INDEXING": 0.86,
        "VALIDATING_INDEX": 0.96,
        "INTERRUPTED": 0.0,
        "RECOVERING": 0.02,
    }
    value = stage_weight.get(status, 0.02)
    total = int(document.get("total_pages") or 0)
    current = int(document.get("current_page") or 0)
    if status in {"EXTRACTING", "OCR"} and total:
        span = 0.14 if status == "EXTRACTING" else 0.10
        value += min(1.0, max(0.0, current / total)) * span
    return min(0.99, max(0.0, value))


def _safe_documents(system: Any) -> list[dict[str, Any]]:
    try:
        return list(system.state_store.get_all_documents() or [])
    except Exception:
        return []


def _safe_events(system: Any, limit: int = 8) -> list[dict[str, Any]]:
    try:
        return list(system.state_store.get_events(None, limit=limit) or [])
    except Exception:
        return []


@st.fragment(run_every="1s")
def render_live_runtime(system: Any) -> None:
    """Render live runtime state without blocking on document ingestion."""
    documents = _safe_documents(system)
    active = [doc for doc in documents if str(doc.get("status") or "").upper() in _ACTIVE]
    ready = sum(1 for doc in documents if str(doc.get("status") or "").upper() in {"READY", "COMPLETED"})
    supervisor = supervisor_snapshot(system)
    events = _safe_events(system)
    latest = events[-1] if events else {}

    with st.container(border=True):
        left, middle, right = st.columns([1.4, 1.4, 2.2], gap="medium")
        with left:
            enabled = bool(supervisor.get("enabled"))
            st.metric("AUTO INGEST", "ON" if enabled else "OFF", f"scan {_age(supervisor.get('last_scan_at'))}")
        with middle:
            in_flight = int(supervisor.get("in_flight") or 0)
            st.metric("LIVE QUEUE", len(active), f"{ready} ready · {in_flight} worker")
        with right:
            action = str(supervisor.get("last_action") or "idle")
            st.caption("LIVE ACTIVITY")
            st.write(action)
            st.caption(f"watcher: {_age(supervisor.get('last_scan_at'))} · last event: {_age(latest.get('created_at'))}")

        for document in active[:4]:
            name = str(document.get("file_name") or document.get("document_id") or "document")
            stage = str(document.get("current_stage") or document.get("status") or "RUNNING").upper()
            current = int(document.get("current_page") or 0)
            total = int(document.get("total_pages") or 0)
            page_text = f"page {current}/{total}" if total else "page —"
            cols = st.columns([2.5, 1.1, 1.0])
            with cols[0]:
                st.caption(name)
                st.progress(_progress(document), text=f"{stage} · {page_text}")
            with cols[1]:
                st.metric("ELAPSED", _age(document.get("ingestion_started_at")).replace("ago", ""))
            with cols[2]:
                st.metric("HEARTBEAT", _age(document.get("heartbeat_at")))

        if supervisor.get("last_error"):
            st.warning(f"Supervisor: {supervisor['last_error']}")

        st.caption(
            f"Auto-started {int(supervisor.get('auto_started') or 0)} · "
            f"completed {int(supervisor.get('completed') or 0)} · "
            f"failed {int(supervisor.get('failed') or 0)} · "
            f"recovered {int(supervisor.get('recovered') or 0)} · "
            f"scans {int(supervisor.get('scans') or 0)} · "
            f"in-flight {int(supervisor.get('in_flight') or 0)}"
        )


__all__ = ["render_live_runtime"]
