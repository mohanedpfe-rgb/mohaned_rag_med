from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from rag_project.application import create_rag_system
from rag_project.configuration.settings import Settings


st.set_page_config(
    page_title="BookRAG Studio",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_system():
    """Build the application through the composition root."""
    return create_rag_system(Settings.from_env())


@st.cache_resource(show_spinner=False)
def get_job_registry() -> dict[str, Any]:
    return {"lock": threading.RLock(), "jobs": {}}


def _job_registry() -> dict[str, Any]:
    return get_job_registry()


def _documents(system) -> list[dict[str, Any]]:
    try:
        return list(system.state_store.get_all_documents() or [])
    except Exception:
        return []


def _running_document_count(system) -> int:
    active = {
        "RUNNING",
        "DISCOVERED",
        "VALIDATING",
        "EXTRACTING",
        "OCR",
        "CHUNKING",
        "EMBEDDING",
        "INDEXING",
        "VALIDATING_INDEX",
    }
    return sum(str(item.get("status") or "").upper() in active for item in _documents(system))


def _active_job() -> tuple[str | None, dict[str, Any] | None]:
    registry = _job_registry()
    with registry["lock"]:
        for job_id, job in reversed(list(registry["jobs"].items())):
            if job.get("status") == "RUNNING":
                return job_id, job
    return None, None


def _start_ingestion(system, source_dir: str) -> str:
    directory = Path(source_dir).expanduser()
    if not directory.is_dir():
        raise ValueError(f"Ingestion folder does not exist: {directory}")

    existing_job, _ = _active_job()
    if existing_job:
        return existing_job

    registry = _job_registry()
    job_id = f"ingest-{time.time_ns()}"
    job = {
        "status": "RUNNING",
        "started": time.time(),
        "finished": None,
        "result": None,
        "error": None,
        "source_dir": str(directory),
    }
    with registry["lock"]:
        registry["jobs"][job_id] = job

    def worker() -> None:
        try:
            job["result"] = system.ingest_directory(str(directory))
            job["status"] = "COMPLETED"
        except Exception as exc:  # pragma: no cover - defensive background boundary
            job["error"] = repr(exc)
            job["status"] = "FAILED"
        finally:
            job["finished"] = time.time()

    threading.Thread(target=worker, name=f"{job_id}-worker", daemon=True).start()
    return job_id


def _safe_save_upload(directory: Path, name: str, content: bytes) -> tuple[Path, str]:
    if not content:
        raise ValueError(f"Uploaded file '{name}' is empty.")
    source_name = Path(str(name or "document.pdf")).name
    stem = Path(source_name).stem or "document"
    safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem).strip(" ._") or "document"
    digest = hashlib.sha256(content).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{safe_stem[:80]}_{digest[:12]}.pdf"
    if not target.exists():
        target.write_bytes(content)
    return target, digest


def _ollama_health(base_url: str) -> tuple[bool, str, list[str]]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=(2.5, 5))
        response.raise_for_status()
        payload = response.json()
        models = [str(item.get("name")) for item in payload.get("models", []) if item.get("name")]
        return True, "Ollama is reachable and returned its model list.", models
    except Exception as exc:
        return False, str(exc), []


def _status_icon(status: str) -> str:
    mapping = {
        "READY": "🟢",
        "RUNNING": "🟡",
        "FAILED": "🔴",
        "INTERRUPTED": "🟠",
        "BUILDING": "🔵",
        "PENDING": "⚪",
    }
    return mapping.get(str(status or "").upper(), "⚪")


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .hero {
            padding: 1.2rem 1.4rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #0f172a 0%, #172554 55%, #0f766e 100%);
            color: white;
            margin-bottom: 1rem;
        }
        .hero h1 { margin-bottom: .3rem; font-size: 2.25rem; }
        .hero p { margin-bottom: .1rem; opacity: .88; }
        .card {
            border: 1px solid rgba(100,116,139,.25);
            border-radius: 16px;
            padding: 1rem;
            background: rgba(255,255,255,.025);
        }
        .good { color: #22c55e; font-weight: 700; }
        .warn { color: #f59e0b; font-weight: 700; }
        .bad { color: #ef4444; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(system) -> None:
    with st.sidebar:
        st.markdown("## 📚 BookRAG Studio")
        st.caption("A beginner-friendly control panel for your local RAG system.")

        st.markdown("### 1. Add PDFs")
        uploads = st.file_uploader(
            "Choose one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="studio_uploads",
            help="Files are copied into the configured incoming folder. The same file is never copied twice during this session.",
        )
        if uploads:
            seen: set[str] = st.session_state.setdefault("saved_upload_hashes", set())
            saved_now = 0
            for upload in uploads:
                try:
                    content = upload.getvalue()
                    digest = hashlib.sha256(content).hexdigest()
                    if digest in seen:
                        continue
                    target, digest = _safe_save_upload(system.settings.incoming_dir, upload.name, content)
                    seen.add(digest)
                    saved_now += int(target.is_file())
                except (OSError, ValueError) as exc:
                    st.error(f"{upload.name}: {exc}")
            if saved_now:
                st.success(f"Saved {saved_now} new PDF file(s).")

        st.markdown("### 2. Process the queue")
        source_dir = st.text_input(
            "Incoming folder",
            value=str(system.settings.incoming_dir),
            key="studio_source_dir",
        )
        active_id, active_job = _active_job()
        disabled = active_id is not None or _running_document_count(system) > 0
        if st.button("▶ Start ingestion", type="primary", use_container_width=True, disabled=disabled):
            try:
                job_id = _start_ingestion(system, source_dir)
                st.session_state["last_job_id"] = job_id
                st.success(f"Ingestion started: {job_id}")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

        if active_id and active_job:
            st.info(f"Running: `{active_id}`")
            elapsed = max(0.0, time.time() - float(active_job.get("started") or time.time()))
            st.caption(f"Background worker time: {elapsed:.1f}s")

        st.markdown("### 3. Maintenance")
        if st.button("🔄 Recreate runtime", use_container_width=True):
            get_system.clear()
            st.rerun()

        confirm = st.checkbox(
            "I understand that cleanup permanently removes managed PDF/index data.",
            key="studio_confirm_cleanup",
        )
        if st.button(
            "🗑️ Clear all PDF data",
            use_container_width=True,
            disabled=not confirm,
        ):
            try:
                removed = system.clear_pdf_data()
                get_system.clear()
                st.session_state.pop("console_answer", None)
                st.success(f"Cleanup complete: {len(removed)} item(s) removed.")
                st.rerun()
            except Exception as exc:
                st.error(f"Cleanup failed: {exc}")

        st.divider()
        st.caption(f"Project: `{system.settings.project_root}`")
        st.caption(f"Vector DB: `{system.settings.vector_db_dir}`")


def _render_overview(system) -> None:
    docs = _documents(system)
    ready = sum(str(doc.get("status") or "").upper() == "READY" for doc in docs)
    failed = sum(str(doc.get("status") or "").upper() in {"FAILED", "INTERRUPTED"} for doc in docs)
    active = _running_document_count(system)
    vector_count = 0
    lexical_count = 0
    try:
        vector_count = int(system.vector_store.count())
        lexical_count = int(system.vector_store.lexical_count())
    except Exception:
        pass

    st.markdown(
        """
        <div class="hero">
          <h1>📚 BookRAG Studio</h1>
          <p>Upload PDFs → index them → ask questions → inspect exactly what the system used.</p>
          <p>Everything runs locally through your configured RAG stack.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Documents", len(docs))
    c2.metric("Ready", ready)
    c3.metric("Processing", active)
    c4.metric("Vector chunks", vector_count)
    c5.metric("Lexical chunks", lexical_count)

    if failed:
        st.warning(f"{failed} document(s) need attention. Open **Ingestion** to see the exact error.")
    elif not docs:
        st.info("Start by uploading a PDF in the left sidebar, then press **Start ingestion**.")
    else:
        st.success("The dashboard is connected to the durable ingestion state and search indexes.")


def _render_health(system) -> None:
    st.subheader("System health")
    ok, message, models = _ollama_health(system.settings.ollama_base_url)
    try:
        compatibility = system.vector_store.compatibility_report(system.embedding_service.identity)
    except Exception as exc:
        compatibility = {"status": "ERROR", "message": str(exc)}

    rows = [
        {"Component": "Ollama", "Status": "PASS" if ok else "FAIL", "Details": message},
        {
            "Component": "Embedding",
            "Status": "PASS" if system.embedding_startup_error is None else "WARN",
            "Details": system.embedding_startup_error or f"model={system.settings.embedding_model}",
        },
        {
            "Component": "Vector index",
            "Status": "PASS" if compatibility.get("status") == "READY" else "WARN",
            "Details": compatibility.get("message", compatibility.get("status", "UNKNOWN")),
        },
        {
            "Component": "Generation model",
            "Status": "INFO",
            "Details": system.settings.generation_model,
        },
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    with st.expander("Models reported by Ollama", expanded=False):
        st.write(models or "No models reported.")


def _render_ingestion(system) -> None:
    st.subheader("Ingestion")
    st.caption("This view reads the real persisted document/page state. Use Refresh after starting a job, or enable live refresh.")
    left, right = st.columns([1, 1])
    with left:
        if st.button("↻ Refresh", use_container_width=True):
            st.rerun()
    with right:
        live = st.checkbox("Live refresh", value=False, help="Refresh every 2 seconds while a worker is active.")

    docs = _documents(system)
    rows = []
    for doc in docs[:50]:
        metrics: dict[str, Any] = {}
        try:
            metrics = json.loads(doc.get("ingestion_metrics") or "{}")
        except Exception:
            pass
        status = str(doc.get("status") or "UNKNOWN").upper()
        rows.append(
            {
                "Status": f"{_status_icon(status)} {status}",
                "File": doc.get("file_name"),
                "Stage": doc.get("current_stage"),
                "Page": f"{doc.get('current_page', 0)}/{doc.get('total_pages', 0)}",
                "Chunks": metrics.get("chunk_count", "—"),
                "Embeddings": metrics.get("embedding_count", "—"),
                "Dimension": doc.get("embedding_dimension") or "—",
                "Error": str(doc.get("error") or "")[:160],
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    job_id, job = _active_job()
    if job_id and job:
        st.info(f"Worker `{job_id}` is running. The backend state above is durable.")
    else:
        st.caption("No local UI worker is currently running.")

    with st.expander("Recent process events", expanded=False):
        try:
            events = system.state_store.get_events(limit=150)
        except Exception:
            events = []
        event_rows = []
        for event in events:
            details = event.get("details") or {}
            event_rows.append(
                {
                    "Time": event.get("created_at"),
                    "File": event.get("file_name"),
                    "Stage": event.get("stage"),
                    "Status": event.get("status"),
                    "Type": event.get("event_type"),
                    "Message": str(event.get("message") or details.get("message") or "")[:180],
                    "Error": str(event.get("error") or details.get("error") or "")[:180],
                }
            )
        st.dataframe(event_rows, use_container_width=True, hide_index=True)

    if live and (job_id or _running_document_count(system) > 0):
        time.sleep(2)
        st.rerun()


def _render_chat(system) -> None:
    st.subheader("Chat with your indexed PDFs")
    st.caption("Answers are generated from retrieved document evidence. Citations and confidence are shown below the answer.")
    question = st.text_area(
        "Your question",
        placeholder="Example: What methodology does the document describe?",
        key="studio_question",
        height=120,
    )
    if st.button("🔎 Search and answer", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            started = time.perf_counter()
            with st.spinner("Retrieving evidence and generating an answer..."):
                try:
                    result = system.answer(question.strip())
                    result["ui_elapsed_ms"] = (time.perf_counter() - started) * 1000
                    st.session_state["console_answer"] = result
                except Exception as exc:
                    st.session_state["console_answer"] = {"error": repr(exc)}

    result = st.session_state.get("console_answer")
    if not result:
        st.info("Ask a question to start.")
        return
    if result.get("error"):
        st.error(result["error"])
        return

    answer = result.get("answer") or "No answer was produced."
    st.markdown("### Answer")
    st.write(answer)

    confidence = result.get("confidence") or {}
    analysis = result.get("query_analysis") or {}
    alignment = result.get("evidence_alignment") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confidence", str(confidence.get("level", "unknown")).upper())
    c2.metric("Answerability", f"{float(alignment.get('answerability', 0.0)):.2f}")
    c3.metric("Query quality", str(analysis.get("query_quality", "unknown")))
    c4.metric("Citations", len(result.get("citations", [])))

    if result.get("degraded_mode"):
        st.warning(f"Degraded retrieval mode: {result.get('retrieval_mode', 'unknown')}")

    st.markdown("### Citations")
    citations = result.get("citations", [])
    if citations:
        for citation in citations:
            st.write(f"📌 {citation}")
    else:
        st.caption("No citations were returned. The answer should be treated as unsupported.")

    with st.expander("Evidence used", expanded=False):
        hits = result.get("hits", [])
        for idx, hit in enumerate(hits, start=1):
            st.markdown(f"**Evidence {idx}**")
            st.caption(json.dumps(hit.metadata or {}, sort_keys=True, default=str))
            st.write(hit.text[:1200])

    with st.expander("Technical trace", expanded=False):
        st.json(
            {
                "query_id": result.get("query_id"),
                "retrieval_mode": result.get("retrieval_mode"),
                "ui_elapsed_ms": result.get("ui_elapsed_ms"),
                "query_analysis": analysis,
                "evidence_alignment": alignment,
                "confidence": confidence,
            }
        )


def _render_inspector(system) -> None:
    st.subheader("Document inspector")
    docs = _documents(system)
    if not docs:
        st.info("No documents yet. Ingest a PDF first.")
        return

    labels = [f"{d.get('file_name')} — {str(d.get('document_id') or '')[:12]}" for d in docs]
    label = st.selectbox("Choose a document", labels, key="inspector_doc")
    doc = docs[labels.index(label)]
    document_id = str(doc.get("document_id"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", str(doc.get("status") or "UNKNOWN"))
    c2.metric("Pages", int(doc.get("total_pages") or 0))
    c3.metric("Embedding dim", doc.get("embedding_dimension") or "—")
    c4.metric("Version", str(doc.get("version_id") or "")[:12] or "—")

    with st.expander("Page checkpoints", expanded=True):
        try:
            pages = system.state_store.get_pages(document_id) or []
        except Exception:
            pages = []
        page_rows = [
            {
                "Page": page.get("page_number"),
                "Extraction": page.get("extraction_status"),
                "Method": page.get("extraction_method"),
                "OCR": page.get("ocr_status"),
                "Characters": len(page.get("text") or ""),
                "Error": page.get("processing_error") or "",
            }
            for page in pages
        ]
        st.dataframe(page_rows, use_container_width=True, hide_index=True)

    with st.expander("Index verification", expanded=True):
        try:
            st.json(system.verify_index(document_id))
        except Exception as exc:
            st.error(f"Index verification failed: {exc}")

    with st.expander("Vector metadata", expanded=False):
        try:
            payload = system.vector_store.get_documents(where={"document_id": document_id})
            ids = payload.get("ids", []) or []
            metadatas = payload.get("metadatas", []) or []
            st.dataframe(
                [
                    {
                        "id": str(item_id),
                        "chunk_id": (metadatas[i] or {}).get("chunk_id"),
                        "pages": (metadatas[i] or {}).get("page_numbers"),
                        "state": (metadatas[i] or {}).get("index_state"),
                        "version": (metadatas[i] or {}).get("version_id"),
                    }
                    for i, item_id in enumerate(ids)
                    if i < len(metadatas)
                ][:200],
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.error(f"Could not read vector metadata: {exc}")

    with st.expander("Raw document state", expanded=False):
        st.json(doc)


def _render_settings(system) -> None:
    st.subheader("Runtime settings")
    st.caption("These controls change the live process. Permanent defaults remain in your environment/configuration.")
    current = system.settings
    with st.form("studio_settings_form"):
        host = st.text_input("Ollama host", value=str(current.ollama_base_url))
        embedding = st.text_input("Embedding model", value=str(current.embedding_model))
        generation = st.text_input("Generation model", value=str(current.generation_model))
        chunk_size = st.number_input("Chunk size", min_value=200, max_value=2000, value=int(current.chunk_size), step=50)
        chunk_overlap = st.number_input("Chunk overlap", min_value=20, max_value=300, value=int(current.chunk_overlap), step=10)
        top_k = st.number_input("Top-k", min_value=1, max_value=20, value=int(current.top_k), step=1)
        temperature = st.slider("Temperature", 0.0, 1.0, value=float(current.temperature), step=0.1)
        vector_weight = st.slider("Vector weight", 0.0, 1.0, value=float(current.vector_weight), step=0.1)
        neighbor_expansion = st.checkbox("Expand neighboring chunks", value=bool(current.neighbor_expansion))
        submitted = st.form_submit_button("Apply live settings", type="primary")\n
    if submitted:
        success, warnings = system.apply_settings_in_place(
            {
                "ollama_base_url": host,
                "embedding_model": embedding,
                "generation_model": generation,
                "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap),
                "top_k": int(top_k),
                "temperature": float(temperature),
                "vector_weight": float(vector_weight),
                "neighbor_expansion": bool(neighbor_expansion),
            }
        )
        if success:
            st.success("Live settings applied successfully.")
        else:
            st.error("Some settings could not be applied.")
        for warning in warnings:
            st.warning(warning)

    with st.expander("Effective configuration", expanded=False):
        rendered: dict[str, Any] = {}
        for key, value in vars(system.settings).items():
            rendered[key] = str(value) if isinstance(value, Path) else value
        st.json(rendered)


def _render_jobs(system) -> None:
    st.subheader("Background state")
    docs = _documents(system)
    job_id, job = _active_job()
    c1, c2, c3 = st.columns(3)
    c1.metric("Active documents", _running_document_count(system))
    c2.metric("UI workers", 1 if job_id else 0)
    c3.metric("Tracked documents", len(docs))

    registry = _job_registry()
    with registry["lock"]:
        history = list(registry["jobs"].items())[-20:]
    if history:
        rows = []
        for jid, item in history:
            rows.append(
                {
                    "Job": jid,
                    "Status": item.get("status"),
                    "Source": item.get("source_dir"),
                    "Duration (s)": round(
                        max(0.0, float(item.get("finished") or time.time()) - float(item.get("started") or time.time())),
                        1,
                    ),
                    "Error": item.get("error") or "",
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No UI worker history yet.")

    if st.button("↻ Refresh background state", use_container_width=True):
        st.rerun()


def main() -> None:
    _inject_css()
    system = get_system()
    _render_sidebar(system)

    tabs = st.tabs([
        "🏠 Overview",
        "💬 Chat",
        "⚙️ System health",
        "⏳ Ingestion",
        "🧭 Background state",
        "🔎 Document inspector",
        "🛠️ Runtime settings",
    ])

    with tabs[0]:
        _render_overview(system)
        _render_health(system)

    with tabs[1]:
        _render_chat(system)

    with tabs[2]:
        _render_health(system)

    with tabs[3]:
        _render_ingestion(system)

    with tabs[4]:
        _render_jobs(system)

    with tabs[5]:
        _render_inspector(system)

    with tabs[6]:
        _render_settings(system)


if __name__ == "__main__":
    main()
