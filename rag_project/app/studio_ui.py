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
    """Build the service through the single application composition root."""
    return create_rag_system(Settings.from_env())


@st.cache_resource(show_spinner=False)
def get_jobs() -> dict[str, Any]:
    return {"lock": threading.RLock(), "items": {}}


def docs(system) -> list[dict[str, Any]]:
    try:
        return list(system.state_store.get_all_documents() or [])
    except Exception:
        return []


def active_document_count(system) -> int:
    active = {
        "RUNNING", "DISCOVERED", "VALIDATING", "EXTRACTING", "OCR",
        "CHUNKING", "EMBEDDING", "INDEXING", "VALIDATING_INDEX",
    }
    return sum(str(d.get("status") or "").upper() in active for d in docs(system))


def active_job() -> tuple[str | None, dict[str, Any] | None]:
    registry = get_jobs()
    with registry["lock"]:
        for job_id, job in reversed(list(registry["items"].items())):
            if job.get("status") == "RUNNING":
                return job_id, job
    return None, None


def start_ingestion(system, source_dir: str) -> str:
    folder = Path(source_dir).expanduser()
    if not folder.is_dir():
        raise ValueError(f"Folder does not exist: {folder}")
    existing, _ = active_job()
    if existing:
        return existing

    registry = get_jobs()
    job_id = f"ingest-{time.time_ns()}"
    job = {
        "status": "RUNNING",
        "started": time.time(),
        "finished": None,
        "result": None,
        "error": None,
        "source_dir": str(folder),
    }
    with registry["lock"]:
        registry["items"][job_id] = job

    def worker() -> None:
        try:
            job["result"] = system.ingest_directory(str(folder))
            job["status"] = "COMPLETED"
        except Exception as exc:
            job["error"] = repr(exc)
            job["status"] = "FAILED"
        finally:
            job["finished"] = time.time()

    threading.Thread(target=worker, name=f"{job_id}-worker", daemon=True).start()
    return job_id


def save_pdf(incoming: Path, name: str, content: bytes) -> str:
    if not content:
        raise ValueError(f"Uploaded file '{name}' is empty.")
    source = Path(name).name
    stem = Path(source).stem or "document"
    safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem).strip(" ._") or "document"
    digest = hashlib.sha256(content).hexdigest()
    target = incoming / f"{safe_stem[:80]}_{digest[:12]}.pdf"
    incoming.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(content)
    return digest


def ollama_health(base_url: str) -> tuple[bool, str, list[str]]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=(2.5, 5))
        response.raise_for_status()
        payload = response.json()
        models = [str(x.get("name")) for x in payload.get("models", []) if x.get("name")]
        return True, "Ollama is reachable.", models
    except Exception as exc:
        return False, str(exc), []


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .br-hero {
            padding: 1.35rem 1.5rem;
            border-radius: 20px;
            background: linear-gradient(135deg, #0f172a, #1e3a8a 55%, #0f766e);
            color: #fff;
            margin-bottom: 1rem;
        }
        .br-hero h1 { margin: 0 0 .35rem 0; font-size: 2.35rem; }
        .br-hero p { margin: .15rem 0; opacity: .88; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(system) -> None:
    with st.sidebar:
        st.markdown("## 📚 BookRAG Studio")
        st.caption("Simple controls for uploading, indexing, searching, and diagnosing your local RAG system.")

        st.markdown("### Add PDFs")
        uploads = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="studio_uploads",
        )
        if uploads:
            seen: set[str] = st.session_state.setdefault("saved_pdf_hashes", set())
            new_count = 0
            for upload in uploads:
                try:
                    content = upload.getvalue()
                    digest = hashlib.sha256(content).hexdigest()
                    if digest in seen:
                        continue
                    save_pdf(Path(system.settings.incoming_dir), upload.name, content)
                    seen.add(digest)
                    new_count += 1
                except (OSError, ValueError) as exc:
                    st.error(f"{upload.name}: {exc}")
            if new_count:
                st.success(f"Saved {new_count} new PDF file(s).")

        st.markdown("### Process queue")
        source_dir = st.text_input(
            "Incoming folder",
            value=str(system.settings.incoming_dir),
            key="studio_source_dir",
        )
        job_id, _ = active_job()
        disabled = job_id is not None or active_document_count(system) > 0
        if st.button("▶ Start ingestion", type="primary", use_container_width=True, disabled=disabled):
            try:
                started_id = start_ingestion(system, source_dir)
                st.session_state["studio_last_job"] = started_id
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

        if job_id:
            st.info(f"Worker `{job_id}` is running")

        st.markdown("### Maintenance")
        if st.button("🔄 Recreate runtime", use_container_width=True):
            get_system.clear()
            st.rerun()

        confirm = st.checkbox("I understand cleanup is permanent.", key="studio_cleanup_confirm")
        if st.button("🗑️ Clear all PDF data", use_container_width=True, disabled=not confirm):
            try:
                removed = system.clear_pdf_data()
                get_system.clear()
                st.session_state.pop("console_answer", None)
                st.session_state.pop("saved_pdf_hashes", None)
                st.success(f"Cleanup complete: {len(removed)} item(s) removed.")
                st.rerun()
            except Exception as exc:
                st.error(f"Cleanup failed: {exc}")


def render_overview(system) -> None:
    items = docs(system)
    ready = sum(str(d.get("status") or "").upper() == "READY" for d in items)
    failed = sum(str(d.get("status") or "").upper() in {"FAILED", "INTERRUPTED"} for d in items)
    try:
        vectors = int(system.vector_store.count())
        lexical = int(system.vector_store.lexical_count())
    except Exception:
        vectors = lexical = 0

    st.markdown(
        """
        <div class="br-hero">
          <h1>📚 BookRAG Studio</h1>
          <p><b>Upload</b> your PDFs → <b>index</b> them → <b>ask</b> questions → <b>inspect</b> the evidence.</p>
          <p>Designed for local, transparent, beginner-friendly RAG work.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Documents", len(items))
    c2.metric("Ready", ready)
    c3.metric("Processing", active_document_count(system))
    c4.metric("Vector chunks", vectors)
    c5.metric("Lexical chunks", lexical)

    if failed:
        st.warning(f"{failed} document(s) need attention. Open **Ingestion** for details.")
    elif not items:
        st.info("No documents yet. Add a PDF from the sidebar to begin.")
    else:
        st.success("Storage, ingestion state, and search indexes are connected.")


def render_health(system) -> None:
    st.subheader("System health")
    ok, message, models = ollama_health(system.settings.ollama_base_url)
    try:
        compatibility = system.vector_store.compatibility_report(system.embedding_service.identity)
    except Exception as exc:
        compatibility = {"status": "ERROR", "message": str(exc)}
    rows = [
        {"Component": "Ollama", "Status": "PASS" if ok else "FAIL", "Details": message},
        {"Component": "Embedding", "Status": "PASS" if system.embedding_startup_error is None else "WARN", "Details": system.embedding_startup_error or system.settings.embedding_model},
        {"Component": "Vector index", "Status": "PASS" if compatibility.get("status") == "READY" else "WARN", "Details": compatibility.get("message", compatibility.get("status", "UNKNOWN"))},
        {"Component": "Generation", "Status": "INFO", "Details": system.settings.generation_model},
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    with st.expander("Ollama models", expanded=False):
        st.write(models or "No models reported by Ollama.")


def render_chat(system) -> None:
    st.subheader("💬 Ask your documents")
    question = st.text_area(
        "Question",
        placeholder="Example: What methodology does the document describe?",
        key="studio_question",
        height=120,
    )
    if st.button("🔎 Search and answer", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Retrieving evidence and generating the answer..."):
                    result = system.answer(question.strip())
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

    st.markdown("### Answer")
    st.write(result.get("answer") or "No answer was produced.")
    confidence = result.get("confidence") or {}
    alignment = result.get("evidence_alignment") or {}
    analysis = result.get("query_analysis") or {}
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
        st.caption("No citations were returned.")

    with st.expander("Evidence used", expanded=False):
        for index, hit in enumerate(result.get("hits", []), start=1):
            st.markdown(f"**Evidence {index}**")
            st.caption(json.dumps(hit.metadata or {}, sort_keys=True, default=str))
            st.write(hit.text[:1600])


def render_ingestion(system) -> None:
    st.subheader("⏳ Ingestion")
    st.caption("The table below reads the real persistent SQLite document state.")
    if st.button("↻ Refresh", use_container_width=True):
        st.rerun()
    live = st.checkbox("Live refresh while processing", value=False)

    rows = []
    for d in docs(system)[:100]:
        metrics: dict[str, Any] = {}
        try:
            metrics = json.loads(d.get("ingestion_metrics") or "{}")
        except Exception:
            pass
        status = str(d.get("status") or "UNKNOWN").upper()
        rows.append(
            {
                "Status": status,
                "File": d.get("file_name"),
                "Stage": d.get("current_stage"),
                "Page": f"{d.get('current_page', 0)}/{d.get('total_pages', 0)}",
                "Chunks": metrics.get("chunk_count", "—"),
                "Embeddings": metrics.get("embedding_count", "—"),
                "Dimension": d.get("embedding_dimension") or "—",
                "Error": str(d.get("error") or "")[:180],
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("Recent process events", expanded=False):
        try:
            events = system.state_store.get_events(limit=200)
        except Exception:
            events = []
        st.dataframe(
            [
                {
                    "Time": e.get("created_at"),
                    "File": e.get("file_name"),
                    "Stage": e.get("stage"),
                    "Status": e.get("status"),
                    "Type": e.get("event_type"),
                    "Message": str(e.get("message") or "")[:180],
                    "Error": str(e.get("error") or "")[:180],
                }
                for e in events
            ],
            use_container_width=True,
            hide_index=True,
        )

    job_id, job = active_job()
    if job_id and job:
        elapsed = max(0.0, time.time() - float(job.get("started") or time.time()))
        st.info(f"Worker `{job_id}` active for {elapsed:.1f}s")
    if live and (job_id or active_document_count(system) > 0):
        time.sleep(2)
        st.rerun()


def render_background(system) -> None:
    st.subheader("🧭 Background state")
    items = docs(system)
    job_id, _ = active_job()
    c1, c2, c3 = st.columns(3)
    c1.metric("Active documents", active_document_count(system))
    c2.metric("UI workers", 1 if job_id else 0)
    c3.metric("Tracked documents", len(items))

    registry = get_jobs()
    with registry["lock"]:
        history = list(registry["items"].items())[-20:]
    if history:
        st.dataframe(
            [
                {
                    "Job": jid,
                    "Status": job.get("status"),
                    "Folder": job.get("source_dir"),
                    "Duration (s)": round(max(0.0, float(job.get("finished") or time.time()) - float(job.get("started") or time.time())), 1),
                    "Error": job.get("error") or "",
                }
                for jid, job in history
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No UI worker history yet.")

    if st.button("↻ Refresh background state", use_container_width=True):
        st.rerun()


def render_inspector(system) -> None:
    st.subheader("🔎 Document inspector")
    items = docs(system)
    if not items:
        st.info("No documents yet.")
        return
    labels = [f"{d.get('file_name')} — {str(d.get('document_id') or '')[:12]}" for d in items]
    selected = st.selectbox("Document", labels, key="studio_inspector_document")
    document = items[labels.index(selected)]
    document_id = str(document.get("document_id"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", str(document.get("status") or "UNKNOWN"))
    c2.metric("Pages", int(document.get("total_pages") or 0))
    c3.metric("Embedding dim", document.get("embedding_dimension") or "—")
    c4.metric("Version", str(document.get("version_id") or "")[:12] or "—")

    with st.expander("Page checkpoints", expanded=True):
        try:
            pages = system.state_store.get_pages(document_id) or []
        except Exception:
            pages = []
        st.dataframe(
            [
                {
                    "Page": p.get("page_number"),
                    "Extraction": p.get("extraction_status"),
                    "Method": p.get("extraction_method"),
                    "OCR": p.get("ocr_status"),
                    "Characters": len(p.get("text") or ""),
                    "Error": p.get("processing_error") or "",
                }
                for p in pages
            ],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Index verification", expanded=True):
        try:
            st.json(system.verify_index(document_id))
        except Exception as exc:
            st.error(f"Verification failed: {exc}")

    with st.expander("Vector metadata", expanded=False):
        try:
            payload = system.vector_store.get_documents(where={"document_id": document_id})
            ids = payload.get("ids", []) or []
            metas = payload.get("metadatas", []) or []
            st.dataframe(
                [
                    {
                        "id": str(item_id),
                        "chunk_id": (metas[i] or {}).get("chunk_id"),
                        "pages": (metas[i] or {}).get("page_numbers"),
                        "state": (metas[i] or {}).get("index_state"),
                        "version": (metas[i] or {}).get("version_id"),
                    }
                    for i, item_id in enumerate(ids)
                    if i < len(metas)
                ][:200],
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.error(f"Vector metadata unavailable: {exc}")

    with st.expander("Raw document state", expanded=False):
        st.json(document)


def render_settings(system) -> None:
    st.subheader("🛠️ Runtime settings")
    st.caption("These values apply to the current process. Permanent defaults remain in configuration/environment files.")
    current = system.settings
    with st.form("studio_settings"):
        host = st.text_input("Ollama host", value=str(current.ollama_base_url))
        embedding = st.text_input("Embedding model", value=str(current.embedding_model))
        generation = st.text_input("Generation model", value=str(current.generation_model))
        chunk_size = st.number_input("Chunk size", min_value=200, max_value=2000, value=int(current.chunk_size), step=50)
        chunk_overlap = st.number_input("Chunk overlap", min_value=20, max_value=300, value=int(current.chunk_overlap), step=10)
        top_k = st.number_input("Top-k", min_value=1, max_value=20, value=int(current.top_k), step=1)
        temperature = st.slider("Temperature", 0.0, 1.0, value=float(current.temperature), step=0.1)
        vector_weight = st.slider("Vector weight", 0.0, 1.0, value=float(current.vector_weight), step=0.1)
        neighbor_expansion = st.checkbox("Expand neighboring chunks", value=bool(current.neighbor_expansion))
        submitted = st.form_submit_button("Apply live settings", type="primary")

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
        st.success("Live settings applied.") if success else st.error("Some settings could not be applied.")
        for warning in warnings:
            st.warning(warning)

    with st.expander("Effective configuration", expanded=False):
        values: dict[str, Any] = {}
        for key, value in vars(system.settings).items():
            values[key] = str(value) if isinstance(value, Path) else value
        st.json(values)


def main() -> None:
    inject_css()
    system = get_system()
    render_sidebar(system)
    tabs = st.tabs([
        "🏠 Overview", "💬 Chat", "⚙️ Health", "⏳ Ingestion",
        "🧭 Background", "🔎 Inspector", "🛠️ Settings",
    ])
    with tabs[0]:
        render_overview(system)
    with tabs[1]:
        render_chat(system)
    with tabs[2]:
        render_health(system)
    with tabs[3]:
        render_ingestion(system)
    with tabs[4]:
        render_background(system)
    with tabs[5]:
        render_inspector(system)
    with tabs[6]:
        render_settings(system)


if __name__ == "__main__":
    main()
