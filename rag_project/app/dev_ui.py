from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from rag_project.app.dev_observer import IngestionObserver
from rag_project.app.rag_system import RAGSystem
from rag_project.configuration.settings import Settings


st.set_page_config(page_title="BookRAG Developer Console", page_icon="🛠️", layout="wide")


@st.cache_resource(show_spinner=False)
def get_system() -> RAGSystem:
    return RAGSystem(Settings.from_env())


@st.cache_resource(show_spinner=False)
def get_observer(state_store: Any) -> IngestionObserver:
    observer = IngestionObserver(state_store)
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


def _lexical_count(system: RAGSystem) -> int:
    try:
        with system.vector_store._lexical_connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM lexical_chunks").fetchone()[0])
    except Exception:
        return 0


def _documents(system: RAGSystem) -> list[dict[str, Any]]:
    try:
        return system.state_store.get_all_documents()
    except Exception:
        return []


def _start_ingestion(system: RAGSystem, source_dir: str) -> str:
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


def _render_system_health(system: RAGSystem) -> None:
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
        ("Lexical index", "PASS", f"{lexical_count} lexical records"),
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


def _render_ingestion(system: RAGSystem) -> None:
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
        live = st.checkbox("Live", value=False, help="Refresh automatically while a job is active.")
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
        event_rows = [
            {
                "Time (UTC)": e.timestamp,
                "File": e.file_name,
                "Stage": e.stage,
                "Status": e.status,
                "Page": f"{e.current_page}/{e.total_pages}",
                "Error": str(e.details.get("error") or "")[:120],
            }
            for e in events
        ]
        st.dataframe(event_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No state transitions have been observed yet.")

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


def _render_document_inspector(system: RAGSystem) -> None:
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


def _render_search_trace(system: RAGSystem) -> None:
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


def _render_settings(system: RAGSystem) -> None:
    st.markdown("### Effective configuration")
    config: dict[str, Any] = {}
    for key, value in vars(system.settings).items():
        config[key] = str(value) if isinstance(value, Path) else value
    st.json(config)


def main() -> None:
    system = get_system()
    st.title("🛠️ BookRAG Developer Console")
    st.caption("Transparent control room: PDF → extraction/OCR → chunks → embeddings → indexes → retrieval → answer")

    with st.sidebar:
        st.subheader("Controls")
        if st.button("Recreate RAG system"):
            get_system.clear()
            get_observer.clear()
            st.rerun()
        st.write(f"**Project:** `{system.settings.project_root}`")
        st.write(f"**Incoming:** `{system.settings.incoming_dir}`")
        st.write(f"**Vector DB:** `{system.settings.vector_db_dir}`")
        st.divider()
        st.write("Use this console while developing. It exposes internal state that should not be shown to end users.")

    tabs = st.tabs(["Overview", "Live Ingestion", "Document Inspector", "Search Trace", "Configuration"])
    with tabs[0]:
        _render_system_health(system)
        _render_pipeline()
    with tabs[1]:
        _render_ingestion(system)
    with tabs[2]:
        _render_document_inspector(system)
    with tabs[3]:
        _render_search_trace(system)
    with tabs[4]:
        _render_settings(system)


if __name__ == "__main__":
    main()
