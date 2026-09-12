from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st

from rag_project.app import bookrag_ui
from rag_project.app.advanced_intelligence_panel import render_advanced_intelligence_panel
from rag_project.app.intelligence_panel import render_intelligence_panel
from rag_project.app.production_contract_panel import render_production_contract_panel
from rag_project.app.ingestion_job_contract import summarize_ingestion_results
from rag_project.ingestion.responsive_supervisor import start as start_supervisor
from rag_project.security import register_session_upload, validate_ollama_url, validate_pdf_payload, validate_storage_path
from rag_project.intelligence.pipeline_integrity import install as install_pipeline_integrity
from rag_project.intelligence.production_contract_v2 import install as install_production_contract
from rag_project.ingestion.ingestion_contract import install as install_ingestion_contract
from rag_project.canonical_runtime import install as install_canonical_runtime

_PIPELINE_RUNTIME_VERSION = "2026-09-11-document-aware-v6"


def _load_local_env():
    f = Path(__file__).resolve().parent / ".env"
    if not f.is_file():
        return
    try:
        for raw in f.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if not k or k in os.environ:
                continue
            if len(v) >= 2 and v[0] == v[-1] and v[0] in {'"', "'"}:
                v = v[1:-1]
            os.environ[k] = v
    except OSError:
        pass


def _clamp_local_embedding_profile():
    try:
        b = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
    except (TypeError, ValueError):
        b = 16
    try:
        r = int(os.getenv("EMBEDDING_RETRIES", "2"))
    except (TypeError, ValueError):
        r = 2
    try:
        t = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "180"))
    except (TypeError, ValueError):
        t = 180.0
    os.environ["EMBEDDING_BATCH_SIZE"] = str(max(16, min(b, 32)))
    os.environ["EMBEDDING_RETRIES"] = str(max(1, min(r, 3)))
    os.environ["EMBEDDING_TIMEOUT_SECONDS"] = str(max(30.0, min(t, 300.0)))


def _install_ui_guards():
    if getattr(bookrag_ui, "_bookrag_ui_guards_installed", False):
        return
    save = bookrag_ui.save_pdf
    health = bookrag_ui.ollama_health
    ask = bookrag_ui.ask_page
    evidence_row = getattr(bookrag_ui, "_evidence_row", None)

    def secure_save(incoming: Any, name: str, content: bytes) -> str:
        s = bookrag_ui.get_system()
        safe = validate_storage_path(s.settings.project_root, incoming, "incoming folder")
        validate_pdf_payload(name, content)
        register_session_upload(len(content))
        return save(safe, name, content)

    def secure_start(system: Any, source_dir: str, *, trigger="manual") -> str:
        safe = validate_storage_path(system.settings.project_root, source_dir, "incoming folder")
        return _secure_ingestion_job(system, str(safe), trigger=trigger)

    def enhanced_ask(system: Any):
        ask(system)
        result = st.session_state.get("answer_result")
        if isinstance(result, dict):
            render_intelligence_panel(result)
            render_advanced_intelligence_panel(result)
            render_production_contract_panel(result)

    def enhanced_evidence_row(item: Any, index: int):
        if isinstance(item, dict):
            return (
                item.get("file_name") or item.get("filename") or item.get("document_id") or f"Source {index}",
                item.get("page_number") or item.get("page") or item.get("page_numbers") or "—",
                item.get("rerank_score") or item.get("score") or item.get("similarity") or item.get("relevance") or "—",
                str(item.get("snippet") or item.get("text") or item.get("content") or ""),
            )
        metadata = getattr(item, "metadata", {}) or {}
        title = metadata.get("file_name") or metadata.get("filename") or metadata.get("document_id") or getattr(item, "doc_id", None) or f"Source {index}"
        page = metadata.get("page_numbers") or metadata.get("page_number") or metadata.get("page") or "—"
        score = getattr(item, "score", None)
        if score is None:
            score = getattr(item, "rerank_score", None)
        if score is None:
            score = "—"
        return str(title), str(page), score, str(getattr(item, "text", "") or "")

    def _secure_ingestion_job(system: Any, source_dir: str, *, trigger="manual") -> str:
        return _start_ingestion_with_status_contract(system, source_dir, trigger=trigger)

    bookrag_ui.save_pdf = secure_save
    bookrag_ui.start_ingestion = secure_start
    bookrag_ui.ollama_health = lambda url: health(validate_ollama_url(url))
    bookrag_ui.ask_page = enhanced_ask
    if evidence_row is not None:
        bookrag_ui._evidence_row = enhanced_evidence_row
    bookrag_ui._bookrag_ui_guards_installed = True


def _start_ingestion_with_status_contract(system: Any, source_dir: str, *, trigger: str = "manual") -> str:
    """Run the production ingestion worker with a stable UI result contract."""
    import threading
    import time

    folder = Path(source_dir).expanduser().resolve()
    root = Path(system.settings.project_root).expanduser().resolve()
    try:
        folder.relative_to(root)
    except ValueError as exc:
        raise ValueError("The incoming folder must remain inside the BookRAG project.") from exc
    pdfs = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    if not pdfs:
        raise ValueError("No PDFs are waiting in the incoming folder.")

    jobs = bookrag_ui.get_jobs()
    with jobs["lock"]:
        for job in reversed(list(jobs["items"].values())):
            if job.get("status") == "RUNNING":
                return str(job["id"])
        jid = f"ingest-{time.time_ns()}"
        job = {
            "id": jid,
            "status": "RUNNING",
            "started": time.time(),
            "finished": None,
            "file_count": len(pdfs),
            "completed": 0,
            "failed": 0,
            "result": None,
            "error": None,
            "trigger": trigger,
        }
        jobs["items"][jid] = job

    def worker() -> None:
        try:
            result = system.ingest_directory(str(folder)) or []
            completed, failed, job_status = summarize_ingestion_results(result)
            with jobs["lock"]:
                job.update(result=result, completed=completed, failed=failed, status=job_status)
        except Exception as exc:
            with jobs["lock"]:
                job.update(error=str(exc), status="FAILED")
        finally:
            with jobs["lock"]:
                job["finished"] = time.time()

    threading.Thread(target=worker, name=f"{jid}-worker", daemon=True).start()
    return jid


def _invalidate_stale_runtime_cache() -> None:
    current = st.session_state.get("_bookrag_pipeline_runtime_version")
    if current == _PIPELINE_RUNTIME_VERSION:
        return
    try:
        clear = getattr(bookrag_ui.get_system, "clear", None)
        if callable(clear):
            clear()
    finally:
        st.session_state["_bookrag_pipeline_runtime_version"] = _PIPELINE_RUNTIME_VERSION


def main():
    _load_local_env()
    _clamp_local_embedding_profile()
    install_pipeline_integrity()
    install_production_contract()
    install_ingestion_contract()
    install_canonical_runtime()
    _install_ui_guards()
    _invalidate_stale_runtime_cache()
    system = bookrag_ui.get_system()
    start_supervisor(system, interval_seconds=1.0)
    bookrag_ui.main()


if __name__ == "__main__":
    main()
