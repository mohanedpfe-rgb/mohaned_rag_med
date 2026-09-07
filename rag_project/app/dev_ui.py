from __future__ import annotations

import json
import hashlib
import re
import threading
import time
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from rag_project.app.dev_observer import IngestionObserver
from rag_project.app.resilient_rag import ResilientRAGSystem
from rag_project.configuration.settings import Settings


st.set_page_config(page_title="BookRAG Developer Console", page_icon="🛠️", layout="wide")

_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def _safe_upload_path(directory: Path, original_name: str, content: bytes) -> Path:
    """Create a deterministic ASCII-only filename safe on every Windows filesystem."""
    source_name = Path(str(original_name or "")).name
    source_stem = Path(source_name).stem or "document"
    extension = Path(source_name).suffix.lower()
    if extension != ".pdf":
        extension = ".pdf"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source_stem).strip(" ._") or "document"
    stem = stem[:80]
    if stem.upper().split(".")[0] in _WINDOWS_RESERVED_NAMES:
        stem = f"document_{stem}"
    digest = hashlib.sha256(content).hexdigest()[:12]
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{stem}_{digest}{extension}"


def _save_uploaded_pdf(directory: Path, original_name: str, content: bytes) -> Path:
    if not content:
        raise ValueError(f"Uploaded file '{original_name}' is empty.")
    target = _safe_upload_path(directory, original_name, content)
    target.write_bytes(content)
    return target


@st.cache_resource(show_spinner=False)
def get_system() -> ResilientRAGSystem:
    return ResilientRAGSystem(Settings.from_env())


@st.cache_resource(show_spinner=False)
def get_observer(_state_store: Any) -> IngestionObserver:
    observer = IngestionObserver(_state_store)
    observer.start()
    return observer


@st.cache_resource(show_spinner=False)
def get_job_registry() -> dict[str, dict[str, Any]]:
    return {"lock": threading.Lock(), "jobs": {}}


def _safe_get_model_tags(base_url: str) -> tuple[bool, str, list[str]]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=(3, 5))
        response.raise_for_status()
        payload = response.json()
        models = [str(item.get("name")) for item in payload.get("models", []) if item.get("name")]
        return True, "Ollama reachable", models
    except Exception as exc:
        return False, str(exc), []


def _lexical_count(system: ResilientRAGSystem) -> int:
    return system.vector_store.lexical_count()


def _documents(system: ResilientRAGSystem) -> list[dict[str, Any]]:
    try:
        return system.state_store.get_all_documents()
    except Exception:
        return []


def _start_ingestion(system: ResilientRAGSystem, source_dir: str) -> str:
    registry = get_job_registry()
    job_id = f"ingest-{time.time_ns()}"
    job: dict[str, Any] = {"status": "RUNNING", "started": time.time(), "result": None, "error": None}

    def worker() -> None:
        try:
            job["result"] = system.ingest_directory(source_dir)
            job["status"] = "COMPLETED"
        except Exception as exc:
            job["error"] = repr(exc)
            job["status"] = "FAILED"
        finally:
            job["finished"] = time.time()

    with registry["lock"]:
        registry["jobs"][job_id] = job
    threading.Thread(target=worker, name=f"{job_id}-worker", daemon=True).start()
    return job_id


def _active_job() -> tuple[str | None, dict[str, Any] | None]:
    registry = get_job_registry()
    with registry["lock"]:
        running = [(jid, job) for jid, job in registry["jobs"].items() if job.get("status") == "RUNNING"]
        if not running:
            return None, None
        return running[-1]


def _status_pill(status: str) -> str:
    mapping = {
        "READY": "🟢 READY",
        "RUNNING": "🟡 RUNNING",
        "FAILED": "🔴 FAILED",
        "INTERRUPTED": "🟠 INTERRUPTED",
        "BUILDING": "🔵 BUILDING",
        "PENDING": "⚪ PENDING",
    }
    return mapping.get(status.upper(), f"⚪ {status.upper()}")


def _render_system_health(system: ResilientRAGSystem) -> None:
    observer = get_observer(system.state_store)
    observer.poll_once()
    ok_ollama, ollama_message, ollama_models = _safe_get_model_tags(system.settings.ollama_base_url)
    compatibility = system.vector_store.compatibility_report(system.embedding_service.identity)
    vector_count = system.vector_store.count()
    lexical_count = _lexical_count(system)

    cols = st.columns(5)
    with cols[0]:
        st.metric("Vector chunks", vector_count)
    with cols[1]:
        st.metric("Lexical chunks", lexical_count)
    with cols[2]:
        st.metric("Embedding dim", system.embedding_service.dimension or "—")
    with cols[3]:
        st.metric("Index", compatibility.get("status", "UNKNOWN"))
    with cols[4]:
        st.metric("Ollama", "ONLINE" if ok_ollama else "OFFLINE")

    st.markdown("### Runtime health")
    health_rows = [
        ("Ollama", "PASS" if ok_ollama else "FAIL", ollama_message),
        ("Embedding initialization", "PASS" if system.embedding_startup_error is None else "WARN", system.embedding_startup_error or "Embedding service available"),
        ("Vector index compatibility", "PASS" if compatibility.get("status") == "READY" else "FAIL", compatibility.get("message", "")),
        ("Reranker", "PASS" if getattr(system.reranker, "model", None) is not None else "FALLBACK", "CrossEncoder loaded" if getattr(system.reranker, "model", None) is not None else "CrossEncoder unavailable; score-order fallback is active"),
        (
            "Lexical index",
            "PASS" if lexical_count else "WARN",
            f"{lexical_count} lexical records"
            if lexical_count
            else "No READY lexical records; ingest a document to build the index",
        ),
    ]
    st.dataframe(
        [{"Component": a, "State": b, "Details": c} for a, b, c in health_rows],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Ollama models discovered", expanded=False):
        st.write(ollama_models or "No models reported by Ollama.")


def _render_pipeline() -> None:
    st.markdown("### Actual processing pipeline")
    stages = [
        ("DISCOVERED", "PDF discovered", "input file accepted and registered"),
        ("VALIDATING", "Validated", "file/hash/metadata checks"),
        ("EXTRACTING", "Text extraction", "PDF text parser active"),
        ("OCR", "OCR fallback", "image/scanned pages handled"),
        ("CHUNKING", "Chunking", "parent/child chunks generated"),
        ("EMBEDDING", "Embedding", "vectors generated for chunks"),
        ("INDEXING", "Indexing", "Chroma + lexical index writes"),
        ("VALIDATING_INDEX", "Index validation", "consistency checks"),
        ("READY", "READY", "document is queryable"),
    ]
    for code, title, detail in stages:
        c1, c2, c3 = st.columns([1.2, 2.5, 5])
        c1.code(code, language=None)
        c2.write(f"**{title}**")
        c3.caption(detail)


def _render_ingestion(system: ResilientRAGSystem) -> None:
    observer = get_observer(system.state_store)
    observer.poll_once()
    st.markdown("### Live ingestion monitor")
    left, middle, right = st.columns([4, 1, 1])
    with left:
        source_dir = st.text_input("Ingestion folder", value=str(system.settings.incoming_dir), key="dev_source_dir")
    with middle:
        running_id, _ = _active_job()
        start = st.button("▶ Start", disabled=running_id is not None, use_container_width=True)
    with right:
        live = st.checkbox("Live", value=True, help="Refresh automatically while a job is active.")
    if start:
        _start_ingestion(system, source_dir)
        st.rerun()

    docs = _documents(system)
    rows = []
    for doc in docs[:50]:
        metrics: dict[str, Any] = {}
        try:
            metrics = json.loads(doc.get("ingestion_metrics") or "{}")
        except Exception:
            pass
        rows.append(
            {
                "File": doc.get("file_name"),
                "Status": _status_pill(str(doc.get("status", "UNKNOWN"))),
                "Stage": doc.get("current_stage"),
                "Page": f"{doc.get('current_page', 0)}/{doc.get('total_pages', 0)}",
                "Chunks": metrics.get("chunk_count", "—"),
                "Embeddings": metrics.get("embedding_count", "—"),
                "Dimension": doc.get("embedding_dimension") or "—",
                "Error": str(doc.get("error") or "")[:120],
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    job_id, job = _active_job()
    if job_id and job:
        st.info(f"Worker `{job_id}` is running. The table above is reading durable SQLite state.")
        active = [d for d in docs if str(d.get("status", "")).upper() in {"RUNNING", "DISCOVERED", "VALIDATING", "EXTRACTING", "OCR", "CHUNKING", "EMBEDDING", "INDEXING", "VALIDATING_INDEX"}]
        for doc in active:
            page = int(doc.get("current_page") or 0)
            total = int(doc.get("total_pages") or 0)
            progress = page / total if total else 0.0
            st.write(f"**{doc.get('file_name')}** · {_status_pill(str(doc.get('status', '')))} · `{doc.get('current_stage')}`")
            st.progress(min(max(progress, 0.0), 1.0), text=f"page {page}/{total}" if total else "starting")
    else:
        st.success("No ingestion worker is currently running.")

    st.markdown("### Event timeline")
    events = observer.events(limit=200)
    if events:
        event_rows = []
        for e in events:
            details = e.details or {}
            event_rows.append(
                {
                    "Time (UTC)": e.timestamp,
                    "File": e.file_name,
                    "Stage": e.stage,
                    "Status": e.status,
                    "Page": f"{e.current_page}/{e.total_pages}",
                    "Type": str(details.get("event_type") or "state"),
                    "Message": str(details.get("message") or details.get("error") or "")[:180],
                    "Error": str(details.get("error") or "")[:120],
                }
            )
        st.dataframe(event_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No state transitions have been observed yet.")

    with st.expander("Raw process events", expanded=False):
        raw_events = system.state_store.get_events(limit=200)
        if raw_events:
            st.dataframe(
                [
                    {
                        "Time": event.get("created_at"),
                        "Document": str(event.get("document_id") or "")[:12],
                        "File": event.get("file_name"),
                        "Stage": event.get("stage"),
                        "Status": event.get("status"),
                        "Type": event.get("event_type"),
                        "Message": str(event.get("message") or "")[:180],
                        "Details": json.dumps(event.get("details") or {}, sort_keys=True)[:300],
                    }
                    for event in raw_events
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No process_event records available yet.")

    registry = get_job_registry()
    completed = []
    with registry["lock"]:
        for jid, data in list(registry["jobs"].items())[-10:]:
            completed.append({
                "Job": jid,
                "Status": data.get("status"),
                "Duration": f"{max(0, data.get('finished', time.time()) - data.get('started', time.time())):.1f}s",
                "Error": data.get("error") or "",
            })
    if completed:
        with st.expander("Worker history", expanded=False):
            st.dataframe(completed, use_container_width=True, hide_index=True)

    if live and job_id:
        time.sleep(1.0)
        st.rerun()


def _vector_rows(system: ResilientRAGSystem) -> list[dict[str, Any]]:
    try:
        payload = system.vector_store.get_documents()
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    ids = payload.get("ids", []) or []
    metadatas = payload.get("metadatas", []) or []
    for index, item_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        rows.append(
            {
                "id": str(item_id),
                "document_id": str(metadata.get("document_id") or ""),
                "chunk_id": str(metadata.get("chunk_id") or ""),
                "file_name": str(metadata.get("file_name") or ""),
                "page_numbers": metadata.get("page_numbers", []),
                "index_state": metadata.get("index_state", "UNKNOWN"),
                "version_id": metadata.get("version_id") or "",
                "language": metadata.get("language") or "",
            }
        )
    return rows[:200]


def _document_log_lines(system: ResilientRAGSystem, document_id: str) -> list[str]:
    try:
        events = system.state_store.get_events(document_id=document_id, limit=200)
    except Exception:
        return []
    lines: list[str] = []
    for event in events:
        timestamp = str(event.get("created_at") or "unknown")
        stage = str(event.get("stage") or "UNKNOWN")
        status = str(event.get("status") or "UNKNOWN")
        event_type = str(event.get("event_type") or "state")
        message = str(event.get("message") or "")
        details = event.get("details") or {}
        if details:
            compact = json.dumps(details, sort_keys=True, default=str)
            if len(compact) > 220:
                compact = compact[:220] + "..."
            line = f"[{timestamp}] [{stage}] [{status}] [{event_type}] {message} | {compact}"
        else:
            line = f"[{timestamp}] [{stage}] [{status}] [{event_type}] {message}"
        lines.append(line)
    return lines


def _process_event_rows(system: ResilientRAGSystem) -> list[dict[str, Any]]:
    try:
        events = system.state_store.get_events(limit=500)
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for event in events:
        details = event.get("details") or {}
        rows.append(
            {
                "timestamp": event.get("created_at"),
                "document_id": str(event.get("document_id") or ""),
                "file_name": event.get("file_name") or "",
                "stage": event.get("stage") or "UNKNOWN",
                "status": event.get("status") or "UNKNOWN",
                "event_type": event.get("event_type") or "state",
                "message": event.get("message") or "",
                "details": json.dumps(details, sort_keys=True, default=str),
            }
        )
    return rows


def _render_background_state(system: ResilientRAGSystem) -> None:
    st.markdown("### Live background state monitor")
    refresh = st.button("Refresh live state", use_container_width=True)
    auto_refresh = st.checkbox("Auto-refresh every 2s", value=True, help="Keep the state view in sync while ingestion is running.")
    if refresh:
        st.rerun()

    docs = _documents(system)
    vector_rows = _vector_rows(system)
    event_rows = _process_event_rows(system)
    stage_order = [
        "DISCOVERED",
        "VALIDATING",
        "EXTRACTING",
        "OCR",
        "CHUNKING",
        "EMBEDDING",
        "INDEXING",
        "VALIDATING_INDEX",
        "READY",
        "FAILED",
    ]
    active_stage_counts = {stage: 0 for stage in stage_order}
    for doc in docs:
        current = str(doc.get("current_stage") or "").upper()
        if current in active_stage_counts:
            active_stage_counts[current] += 1

    stage_cols = st.columns(len(stage_order))
    for idx, stage in enumerate(stage_order):
        with stage_cols[idx]:
            st.metric(stage, active_stage_counts.get(stage, 0))

    perf_cols = st.columns(4)
    with perf_cols[0]:
        st.metric("Documents tracked", len(docs))
    with perf_cols[1]:
        st.metric("Ready documents", sum(1 for doc in docs if str(doc.get("status") or "").upper() == "READY"))
    with perf_cols[2]:
        st.metric("Processing now", sum(1 for doc in docs if str(doc.get("status") or "").upper() in {"RUNNING", "DISCOVERED", "VALIDATING", "EXTRACTING", "OCR", "CHUNKING", "EMBEDDING", "INDEXING", "VALIDATING_INDEX"}))
    with perf_cols[3]:
        st.metric("Latest events", len(event_rows))

    doc_table = []
    vector_lookup = {}
    for row in vector_rows:
        key = str(row.get("document_id") or "")
        if key:
            vector_lookup[key] = vector_lookup.get(key, 0) + 1
    for doc in docs[:20]:
        metrics = {}
        try:
            metrics = json.loads(doc.get("ingestion_metrics") or "{}")
        except Exception:
            metrics = {}
        document_id = str(doc.get("document_id") or "")
        doc_table.append(
            {
                "File": doc.get("file_name"),
                "Document ID": document_id[:12],
                "Status": str(doc.get("status") or "UNKNOWN"),
                "Stage": str(doc.get("current_stage") or "UNKNOWN"),
                "Current page": int(doc.get("current_page") or 0),
                "Total pages": int(doc.get("total_pages") or 0),
                "Chunk count": metrics.get("chunk_count", "—"),
                "Embedding count": metrics.get("embedding_count", "—"),
                "Vector rows": vector_lookup.get(document_id, 0),
                "Event count": len([e for e in event_rows if str(e.get("document_id") or "") == document_id]),
                "Version": str(doc.get("version_id") or "")[:12],
                "Error": str(doc.get("error") or "")[:120],
            }
        )
    st.subheader("Document lifecycle")
    if doc_table:
        st.dataframe(doc_table, use_container_width=True, hide_index=True)
    else:
        st.info("No document lifecycle events recorded yet.")

    with st.expander("Process timeline", expanded=True):
        if event_rows:
            stage_filter = st.multiselect(
                "Filter stages",
                sorted({row["stage"] for row in event_rows if row.get("stage")}),
                default=[],
                help="Limit the event stream to selected stages.",
            )
            filtered = event_rows if not stage_filter else [row for row in event_rows if row.get("stage") in stage_filter]
            st.dataframe(
                [
                    {
                        "Time": row.get("timestamp"),
                        "Document": row.get("document_id")[:12],
                        "File": row.get("file_name"),
                        "Stage": row.get("stage"),
                        "Status": row.get("status"),
                        "Type": row.get("event_type"),
                        "Message": row.get("message")[:180],
                    }
                    for row in filtered[-80:]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No background event rows yet.")

    with st.expander("Page checkpoints and extraction state", expanded=True):
        if docs:
            selected_label = st.selectbox("Choose a document", [f"{d.get('file_name')} — {d.get('document_id', '')[:12]}" for d in docs], key="live_state_doc_selector")
            doc = next((d for d in docs if f"{d.get('file_name')} — {d.get('document_id', '')[:12]}" == selected_label), docs[0])
            document_id = str(doc.get("document_id"))
            pages = system.state_store.get_pages(document_id)
            if pages:
                st.dataframe(
                    [
                        {
                            "Page": p.get("page_number"),
                            "Status": p.get("extraction_status"),
                            "Method": p.get("extraction_method"),
                            "OCR": p.get("ocr_status"),
                            "Text chars": len(p.get("text") or ""),
                            "Checksum": str(p.get("checksum") or "")[:16],
                            "Processing error": str(p.get("processing_error") or ""),
                        }
                        for p in pages
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No page checkpoints recorded for the selected document.")
        else:
            st.info("No documents yet.")

    with st.expander("Live terminal log (per document)", expanded=True):
        if docs:
            selected_doc = st.selectbox("Choose a document for the live log", [f"{d.get('file_name')} — {d.get('document_id', '')[:12]}" for d in docs], key="live_terminal_doc_selector")
            document_id = next((str(d.get("document_id")) for d in docs if f"{d.get('file_name')} — {d.get('document_id', '')[:12]}" == selected_doc), str(docs[0].get("document_id")))
            lines = _document_log_lines(system, document_id)
            if lines:
                st.code("\n".join(lines[-120:]), language="text")
            else:
                st.info("No process events recorded for the selected document yet.")
        else:
            st.info("No documents available for live log replay.")

    with st.expander("Vector index live rows", expanded=True):
        if vector_rows:
            st.dataframe(vector_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No vector rows in the Chroma collection yet.")

    with st.expander("Vector details for selected document", expanded=False):
        if docs:
            doc_label = st.selectbox("Select document for vector metadata", [f"{d.get('file_name')} — {d.get('document_id', '')[:12]}" for d in docs], key="vector_detail_doc_selector")
            selected_doc = next((d for d in docs if f"{d.get('file_name')} — {d.get('document_id', '')[:12]}" == doc_label), docs[0])
            document_id = str(selected_doc.get("document_id"))
            try:
                payload = system.vector_store.get_documents(where={"document_id": document_id})
            except Exception:
                payload = {"ids": [], "metadatas": []}
            detail_rows = []
            for idx, item_id in enumerate(payload.get("ids", []) or []):
                metadata = payload.get("metadatas", [])[idx] if idx < len(payload.get("metadatas", [])) else {}
                detail_rows.append({
                    "id": str(item_id),
                    "chunk_id": metadata.get("chunk_id"),
                    "page_numbers": metadata.get("page_numbers"),
                    "index_state": metadata.get("index_state"),
                    "version_id": metadata.get("version_id"),
                    "file_name": metadata.get("file_name"),
                    "language": metadata.get("language"),
                    "metadata": json.dumps(metadata, sort_keys=True, default=str),
                })
            if detail_rows:
                st.dataframe(detail_rows, use_container_width=True, hide_index=True)
                st.json(detail_rows[0] if len(detail_rows) == 1 else detail_rows[:5])
            else:
                st.info("No vector rows for the selected document yet.")
        else:
            st.info("No documents available for vector inspection.")

    with st.expander("Storage / collection metadata", expanded=False):
        collection_meta = getattr(system.vector_store.collection, "metadata", None) or {}
        st.json({
            "collection_name": system.vector_store.collection_name,
            "collection_metadata": collection_meta,
            "vector_count": system.vector_store.count(),
            "lexical_count": _lexical_count(system),
        })

    if auto_refresh:
        time.sleep(2.0)
        st.rerun()


def _render_document_inspector(system: ResilientRAGSystem) -> None:
    st.markdown("### Document / page / vector inspector")
    docs = _documents(system)
    if not docs:
        st.info("No documents recorded.")
        return
    labels = {f"{d.get('file_name')} — {d.get('document_id', '')[:12]}": d for d in docs}
    selected_label = st.selectbox("Document", list(labels))
    doc = labels[selected_label]
    document_id = str(doc.get("document_id"))

    cols = st.columns(5)
    cols[0].metric("Pages", int(doc.get("total_pages") or 0))
    cols[1].metric("Current page", int(doc.get("current_page") or 0))
    cols[2].metric("Status", str(doc.get("status")))
    cols[3].metric("Embedding dim", doc.get("embedding_dimension") or "—")
    cols[4].metric("Index state", str(doc.get("index_state") or "—"))

    metrics: dict[str, Any] = {}
    try:
        metrics = json.loads(doc.get("ingestion_metrics") or "{}")
    except Exception:
        pass
    with st.expander("Ingestion metrics", expanded=True):
        st.json(metrics or {"message": "No ingestion metrics recorded by current backend."})

    pages = system.state_store.get_pages(document_id)
    if pages:
        page_rows = []
        for p in pages:
            page_rows.append(
                {
                    "Page": p.get("page_number"),
                    "Extraction": p.get("extraction_status"),
                    "Method": p.get("extraction_method"),
                    "OCR": p.get("ocr_status"),
                    "Chars": len(p.get("text") or ""),
                    "Checksum": str(p.get("checksum") or "")[:14],
                    "Error": p.get("processing_error") or "",
                }
            )
        st.dataframe(page_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No page checkpoints have been recorded for this document.")

    with st.expander("Index verification", expanded=True):
        try:
            st.json(system.verify_index(document_id))
        except Exception as exc:
            st.error(f"Index verification failed: {exc}")

    with st.expander("Raw document state", expanded=False):
        st.json(doc)


def _render_search_trace(system: ResilientRAGSystem) -> None:
    st.markdown("### Search / RAG trace")
    question = st.text_area("Run a diagnostic question", placeholder="Ask a question and inspect retrieval, reranking, context and grounding.", key="dev_question")
    if st.button("Run diagnostic query", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            started = time.perf_counter()
            try:
                result = system.answer(question)
                st.session_state["dev_last_query"] = result
            except Exception as exc:
                st.session_state["dev_last_query"] = {"error": repr(exc)}
            st.session_state["dev_last_query_elapsed"] = (time.perf_counter() - started) * 1000

    result = st.session_state.get("dev_last_query")
    if not result:
        st.info("No diagnostic query has been run.")
        return
    if result.get("error"):
        st.error(result["error"])
        return
    st.markdown("#### Result")
    st.write(result.get("answer", ""))
    c = result.get("confidence") or {}
    a = result.get("query_analysis") or {}
    e = result.get("evidence_alignment") or {}
    cols = st.columns(6)
    cols[0].metric("Total ms", f"{st.session_state.get('dev_last_query_elapsed', 0):.0f}")
    cols[1].metric("Confidence", str(c.get("level", "unknown")))
    cols[2].metric("Answerability", e.get("answerability", "—"))
    cols[3].metric("Query quality", a.get("query_quality", "—"))
    cols[4].metric("Evidence", e.get("decision", "—"))
    cols[5].metric("Citations", len(result.get("citations", [])))
    with st.expander("Raw RAG trace", expanded=True):
        st.json(
            {
                "query_id": result.get("query_id"),
                "query_analysis": a,
                "evidence_alignment": e,
                "confidence": c,
                "citations": result.get("citations", []),
                "selected_hits": [
                    {
                        "score": h.score,
                        "vector_score": h.vector_score,
                        "lexical_score": h.lexical_score,
                        "metadata": h.metadata,
                        "preview": h.text[:600],
                    }
                    for h in result.get("hits", [])
                ],
            }
        )


def _render_settings(system: ResilientRAGSystem) -> None:
    st.markdown("### Effective configuration")
    config: dict[str, Any] = {}
    for key, value in vars(system.settings).items():
        config[key] = str(value) if isinstance(value, Path) else value
    st.json(config)


def _apply_runtime_settings(system: ResilientRAGSystem) -> None:
    updates: dict[str, object] = {
        "ollama_base_url": st.session_state.console_ollama_host,
        "embedding_model": st.session_state.console_embedding_model,
        "generation_model": st.session_state.console_generation_model,
        "chunk_size": st.session_state.console_chunk_size,
        "chunk_overlap": st.session_state.console_chunk_overlap,
        "top_k": st.session_state.console_top_k,
        "temperature": st.session_state.console_temperature,
        "lexical_mode": st.session_state.console_lexical_mode,
        "vector_weight": st.session_state.console_vector_weight,
        "neighbor_expansion": st.session_state.console_neighbor_expansion,
    }
    success, warnings = system.apply_settings_in_place(updates)
    for warning in warnings:
        st.warning(warning)
    if success:
        st.session_state.console_settings_message = "Settings applied to the live runtime."
    else:
        st.session_state.console_settings_message = "Settings could not be fully applied."


def _render_chat(system: ResilientRAGSystem) -> None:
    st.markdown("### Ask the indexed documents")
    question = st.text_area(
        "Question",
        placeholder="What does the document say about methodology?",
        key="console_question",
    )
    if st.button("Search and answer", type="primary"):
        if not question.strip():
            st.warning("Enter a question before searching.")
        else:
            with st.spinner("Searching indexed documents..."):
                st.session_state.console_answer = system.answer(question)

    result = st.session_state.get("console_answer")
    if not result:
        st.info("Ask a question to see an answer and its citations.")
        return
    st.markdown("### Answer")
    st.write(result.get("answer", ""))
    if result.get("degraded_mode"):
        st.info(f"Degraded mode: retrieval={result.get('retrieval_mode', 'unknown')}")
    confidence = result.get("confidence") or {}
    if confidence:
        st.metric(
            "Evidence confidence",
            str(confidence.get("level", "unknown")).upper(),
            f"top={float(confidence.get('top_score', 0.0)):.3f}, "
            f"margin={float(confidence.get('margin', 0.0)):.3f}",
        )
    st.markdown("### Citations")
    for citation in result.get("citations", []):
        st.write(citation)


def main() -> None:
    system = get_system()
    st.title("BookRAG Studio")
    st.caption("One interface for document upload, chat, ingestion, diagnostics, and configuration.")
    # Overall ingestion status banner
    _render_overall_status(system)

    with st.sidebar:
        st.subheader("Configuration")
        config_defaults = {
            "console_ollama_host": system.settings.ollama_base_url,
            "console_embedding_model": system.settings.embedding_model,
            "console_generation_model": system.settings.generation_model,
            "console_chunk_size": system.settings.chunk_size,
            "console_chunk_overlap": system.settings.chunk_overlap,
            "console_top_k": system.settings.top_k,
            "console_temperature": system.settings.temperature,
            "console_lexical_mode": system.settings.lexical_mode,
            "console_vector_weight": system.settings.vector_weight,
            "console_neighbor_expansion": system.settings.neighbor_expansion,
        }
        for key, value in config_defaults.items():
            st.session_state.setdefault(key, value)
        with st.form("console_configuration_form"):
            st.text_input("Ollama host", key="console_ollama_host")
            st.text_input("Embedding model", key="console_embedding_model")
            st.text_input("Generation model", key="console_generation_model")
            st.number_input("Chunk size", min_value=200, max_value=2000, step=50, key="console_chunk_size")
            st.number_input("Chunk overlap", min_value=20, max_value=300, step=10, key="console_chunk_overlap")
            st.number_input("Top-k retrieval", min_value=1, max_value=20, key="console_top_k")
            st.slider("Generation temperature", 0.0, 1.0, step=0.1, key="console_temperature")
            st.selectbox("Retrieval mode", ["hybrid", "lexical", "vector"], key="console_lexical_mode")
            st.slider("Vector search weight", 0.0, 1.0, step=0.1, key="console_vector_weight")
            st.checkbox("Expand neighboring chunks", key="console_neighbor_expansion")
            st.form_submit_button("Apply settings", on_click=_apply_runtime_settings, args=(system,))
        if st.session_state.get("console_settings_message"):
            st.success(st.session_state.console_settings_message)

        uploads = st.file_uploader("Add PDF files", type=["pdf"], accept_multiple_files=True)
        if uploads:
            saved = 0
            upload_errors: list[str] = []
            for upload in uploads:
                try:
                    _save_uploaded_pdf(
                        system.settings.incoming_dir,
                        upload.name,
                        upload.getvalue(),
                    )
                    saved += 1
                except (OSError, ValueError) as exc:
                    upload_errors.append(f"{upload.name}: {exc}")
            if saved:
                st.success(f"Saved {saved} PDF file(s) to the ingestion queue.")
                # Auto‑start ingestion if no job is running
                running_id, _ = _active_job()
                if not running_id:
                    _start_ingestion(system, system.settings.incoming_dir)
                    st.rerun()
            for upload_error in upload_errors:
                st.error(f"Could not save upload: {upload_error}")

        if st.button("Recreate RAG system"):
            get_system.clear()
            get_observer.clear()
            st.rerun()
        st.write(f"**Project:** `{system.settings.project_root}`")
        st.write(f"**Incoming:** `{system.settings.incoming_dir}`")
        st.write(f"**Vector DB:** `{system.settings.vector_db_dir}`")
        st.divider()
        st.write("All user and developer controls are available in the tabs.")

    tabs = st.tabs(["Chat", "Overview", "Live Ingestion", "Background State", "Document Inspector", "Search Trace", "Configuration"])
    with tabs[0]:
        _render_chat(system)
    with tabs[1]:
        _render_system_health(system)
        _render_pipeline()
    with tabs[2]:
        _render_ingestion(system)
    with tabs[3]:
        _render_background_state(system)
    with tabs[4]:
        _render_document_inspector(system)
    with tabs[5]:
        _render_search_trace(system)
    with tabs[6]:
        _render_settings(system)


if __name__ == "__main__":
    main()
