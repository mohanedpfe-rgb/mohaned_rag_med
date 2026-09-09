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
    """Build the production service through the single composition root."""
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


def _status_class(value: str) -> str:
    normalized = value.strip().upper()
    if normalized in {"READY", "PASS", "COMPLETED", "HEALTHY"}:
        return "status-good"
    if normalized in {"FAILED", "FAIL", "ERROR", "INTERRUPTED"}:
        return "status-bad"
    if normalized in {"WARN", "WARNING", "RUNNING", "PROCESSING"}:
        return "status-warn"
    return "status-neutral"


def _metric_card(icon: str, label: str, value: Any, helper: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-top"><span class="metric-icon">{icon}</span><span class="metric-label">{label}</span></div>
          <div class="metric-value">{value}</div>
          <div class="metric-helper">{helper}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section_title(title: str, eyebrow: str = "") -> None:
    label = f'<div class="eyebrow">{eyebrow}</div>' if eyebrow else ""
    st.markdown(f'{label}<div class="section-title">{title}</div>', unsafe_allow_html=True)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f6f8fb;
            --panel: #ffffff;
            --ink: #101828;
            --muted: #667085;
            --line: #e4e7ec;
            --accent: #5b5bd6;
            --accent-soft: #eeefff;
            --success: #12b76a;
            --warning: #f79009;
            --danger: #f04438;
        }
        .stApp { background: var(--bg); color: var(--ink); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"] { visibility: hidden; height: 0; }
        [data-testid="stSidebar"] {
            background: #fbfcfe;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }
        .brand-row { display:flex; align-items:center; gap:.7rem; margin-bottom:1.1rem; }
        .brand-mark { width:38px; height:38px; border-radius:12px; display:flex; align-items:center; justify-content:center; background:#111827; color:white; font-size:20px; box-shadow:0 8px 18px rgba(17,24,39,.18); }
        .brand-name { font-weight:800; font-size:1.05rem; letter-spacing:-.02em; }
        .brand-sub { color:var(--muted); font-size:.76rem; }
        .topbar { display:flex; justify-content:space-between; align-items:center; gap:1rem; margin-bottom:1rem; }
        .topbar-copy h1 { margin:0; font-size:1.85rem; letter-spacing:-.03em; }
        .topbar-copy p { margin:.25rem 0 0; color:var(--muted); }
        .health-pill { display:inline-flex; gap:.45rem; align-items:center; background:#ecfdf3; color:#027a48; border:1px solid #abefc6; padding:.45rem .7rem; border-radius:999px; font-size:.82rem; font-weight:700; }
        .dot { width:8px; height:8px; border-radius:50%; background:currentColor; }
        .hero {
            background: linear-gradient(135deg, #111827 0%, #232a4f 56%, #3f3c89 100%);
            color:#fff; border-radius:20px; padding:1.45rem 1.5rem; margin-bottom:1rem;
            box-shadow:0 18px 40px rgba(17,24,39,.18);
        }
        .hero h2 { margin:0; font-size:1.65rem; letter-spacing:-.03em; }
        .hero p { margin:.5rem 0 0; color:rgba(255,255,255,.76); }
        .metric-grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:.85rem; margin:1rem 0 1.15rem; }
        .metric-card { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:1rem; box-shadow:0 6px 20px rgba(16,24,40,.035); }
        .metric-top { display:flex; align-items:center; gap:.5rem; color:var(--muted); font-size:.78rem; font-weight:700; }
        .metric-icon { width:28px; height:28px; display:inline-flex; align-items:center; justify-content:center; border-radius:9px; background:var(--accent-soft); color:var(--accent); }
        .metric-value { font-size:1.75rem; font-weight:800; margin-top:.35rem; letter-spacing:-.03em; }
        .metric-helper { margin-top:.2rem; color:var(--muted); font-size:.74rem; }
        .section-title { font-size:1.15rem; font-weight:800; letter-spacing:-.02em; margin-bottom:.2rem; }
        .eyebrow { text-transform:uppercase; letter-spacing:.11em; font-size:.68rem; color:var(--muted); font-weight:800; margin-bottom:.2rem; }
        .panel { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:1rem; margin-bottom:.85rem; }
        .status-chip { display:inline-flex; align-items:center; border-radius:999px; padding:.25rem .55rem; font-size:.72rem; font-weight:800; border:1px solid currentColor; }
        .status-good { color:#027a48; background:#ecfdf3; }
        .status-bad { color:#b42318; background:#fef3f2; }
        .status-warn { color:#b54708; background:#fffaeb; }
        .status-neutral { color:#475467; background:#f2f4f7; }
        .answer-box { background:#fcfcff; border:1px solid #dddfff; border-radius:16px; padding:1rem 1.1rem; }
        .answer-label { color:#5b5bd6; text-transform:uppercase; letter-spacing:.09em; font-size:.67rem; font-weight:900; margin-bottom:.4rem; }
        .empty-state { padding:2.4rem 1rem; text-align:center; color:var(--muted); background:#fff; border:1px dashed #d0d5dd; border-radius:16px; }
        .sidebar-divider { margin:.9rem 0; border-top:1px solid var(--line); }
        div[data-testid="stMetric"] { background:transparent; }
        @media (max-width: 900px) { .metric-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(system) -> None:
    with st.sidebar:
        st.markdown(
            '<div class="brand-row"><div class="brand-mark">📚</div><div><div class="brand-name">BookRAG Studio</div><div class="brand-sub">Local document intelligence</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

        st.markdown("### Add PDFs")
        uploads = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="studio_uploads",
            label_visibility="collapsed",
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
            help="Folder scanned by the ingestion service.",
        )
        job_id, _ = active_job()
        disabled = job_id is not None or active_document_count(system) > 0
        if st.button("▶ Start all chunks", type="primary", use_container_width=True, disabled=disabled):
            try:
                started_id = start_ingestion(system, source_dir)
                st.session_state["studio_last_job"] = started_id
                st.session_state["studio_nav"] = "Ingestion"
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

        if job_id:
            st.markdown(f'<span class="status-chip status-warn">● Worker running · {job_id}</span>', unsafe_allow_html=True)

        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
        st.markdown("### Maintenance")
        if st.button("↻ Recreate runtime", use_container_width=True):
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


def render_topbar(system, active_page: str) -> str:
    items = docs(system)
    ready = sum(str(d.get("status") or "").upper() == "READY" for d in items)
    st.markdown(
        f"""
        <div class="topbar">
          <div class="topbar-copy">
            <h1>BookRAG Studio</h1>
            <p>Upload PDFs, index them, ask questions, inspect evidence.</p>
          </div>
          <div class="health-pill"><span class="dot"></span>{ready} ready</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    nav = st.radio(
        "Studio navigation",
        ["Overview", "Ingestion", "Inspector", "Settings", "Chat", "Health", "Background"],
        index=["Overview", "Ingestion", "Inspector", "Settings", "Chat", "Health", "Background"].index(active_page),
        horizontal=True,
        label_visibility="collapsed",
        key="studio_nav_control",
    )
    st.session_state["studio_nav"] = nav
    return nav


def render_overview(system) -> None:
    items = docs(system)
    ready = sum(str(d.get("status") or "").upper() == "READY" for d in items)
    processing = active_document_count(system)
    failed = sum(str(d.get("status") or "").upper() in {"FAILED", "INTERRUPTED"} for d in items)
    try:
        vectors = int(system.vector_store.count())
        lexical = int(system.vector_store.lexical_count())
    except Exception:
        vectors = lexical = 0

    st.markdown(
        """
        <div class="hero">
          <h2>Ask your documents with confidence.</h2>
          <p>Local-first RAG with citations, resilient ingestion, hybrid retrieval, and transparent evidence inspection.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    cols = st.columns(5)
    cards = [
        ("📄", "Documents", len(items), "Tracked in SQLite"),
        ("✓", "Ready", ready, "Available for chat"),
        ("◌", "Processing", processing, "Active pipeline"),
        ("⌘", "Vector chunks", vectors, "Semantic index"),
        ("≡", "Lexical chunks", lexical, "BM25 index"),
    ]
    for col, (icon, label, value, helper) in zip(cols, cards):
        with col:
            _metric_card(icon, label, value, helper)
    st.markdown('</div>', unsafe_allow_html=True)

    left, right = st.columns([1.35, 1])
    with left:
        _section_title("Workspace", "Studio")
        with st.container(border=True):
            st.markdown("**Upload → Index → Ask → Inspect**")
            st.caption("The interface is backed by the real ingestion state, vector store, lexical index, and production RAG answer service.")
            if not items:
                st.info("No documents yet. Add a PDF from the sidebar, then start the ingestion queue.")
            elif failed:
                st.warning(f"{failed} document(s) need attention. Open Ingestion for persistent error details.")
            else:
                st.success("Your workspace is connected and ready for document research.")
    with right:
        _section_title("System snapshot", "Status")
        with st.container(border=True):
            ok, message, models = ollama_health(system.settings.ollama_base_url)
            st.markdown(f"**Ollama** · <span class='status-chip {_status_class('PASS' if ok else 'FAIL')}'>{'PASS' if ok else 'FAIL'}</span>", unsafe_allow_html=True)
            st.caption(message)
            st.markdown(f"**Generation** · `{system.settings.generation_model}`")
            st.markdown(f"**Embedding** · `{system.settings.embedding_model}`")
            st.caption(f"{len(models)} model(s) visible from Ollama")


def render_chat(system) -> None:
    _section_title("Ask your documents", "Chat")
    st.caption("Answers are generated from retrieved evidence and the final result carries grounding and citation metadata.")

    question = st.text_area(
        "Question",
        placeholder="Example: What methodology does the document describe?",
        key="studio_question",
        height=110,
        label_visibility="collapsed",
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
        st.markdown('<div class="empty-state">Ask a question to see a grounded answer, confidence, and citations.</div>', unsafe_allow_html=True)
        return
    if result.get("error"):
        st.error(result["error"])
        return

    st.markdown('<div class="answer-box"><div class="answer-label">Answer</div>', unsafe_allow_html=True)
    st.write(result.get("answer") or "No answer was produced.")
    st.markdown('</div>', unsafe_allow_html=True)

    confidence = result.get("confidence") or {}
    alignment = result.get("evidence_alignment") or {}
    analysis = result.get("query_analysis") or {}
    metrics = st.columns(4)
    metrics[0].metric("Confidence", str(confidence.get("level", "unknown")).upper())
    metrics[1].metric("Answerability", f"{float(alignment.get('answerability', 0.0)):.2f}")
    metrics[2].metric("Query quality", str(analysis.get("query_quality", "unknown")))
    metrics[3].metric("Citations", len(result.get("citations", [])))

    if result.get("degraded_mode"):
        st.warning(f"Degraded retrieval mode: {result.get('retrieval_mode', 'unknown')}")
    if result.get("status") == "MEDICAL_SAFETY_ABSTAIN":
        st.warning("This query crossed the high-risk medical safety boundary. The system withheld a definitive answer because the retrieved support was not strong enough.")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**Citations**")
            citations = result.get("citations", [])
            if citations:
                for citation in citations:
                    st.write(f"📌 {citation}")
            else:
                st.caption("No citations were returned.")
    with col2:
        with st.container(border=True):
            st.markdown("**Answer safety**")
            grounding = result.get("grounding") or {}
            contradictions = result.get("contradictions") or {}
            st.write(f"Grounding: **{grounding.get('status', 'unknown')}**")
            st.write(f"Contradictions: **{contradictions.get('status', 'unknown')}**")
            st.write(f"Degraded mode: **{bool(result.get('degraded_mode'))}**")

    with st.expander("Evidence used", expanded=False):
        hits = result.get("hits", []) or []
        if not hits:
            st.caption("No evidence hits were returned.")
        for index, hit in enumerate(hits, start=1):
            st.markdown(f"**Evidence {index}**")
            st.caption(json.dumps(hit.metadata or {}, sort_keys=True, default=str))
            st.write(hit.text[:1600])

    with st.expander("Query trace", expanded=False):
        trace = result.get("query_trace") or result.get("trace") or {}
        st.json(trace)


def render_health(system) -> None:
    _section_title("System health", "Health")
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
    with st.container(border=True):
        st.dataframe(rows, use_container_width=True, hide_index=True)
    with st.container(border=True):
        st.markdown("**Ollama models**")
        st.write(models or "No models reported by Ollama.")


def render_ingestion(system) -> None:
    _section_title("Ingestion", "Pipeline")
    st.caption("This view reads persisted SQLite state while the worker operates in the background.")
    top = st.columns([1, 1, 4])
    with top[0]:
        if st.button("↻ Refresh", use_container_width=True):
            st.rerun()
    with top[1]:
        live = st.checkbox("Live refresh", value=False)

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
    if rows:
        with st.container(border=True):
            st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.markdown('<div class="empty-state">No ingestion records yet.</div>', unsafe_allow_html=True)

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
    _section_title("Background state", "Operations")
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
        with st.container(border=True):
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
        st.markdown('<div class="empty-state">No UI worker history yet.</div>', unsafe_allow_html=True)

    if st.button("↻ Refresh background state", use_container_width=True):
        st.rerun()


def render_inspector(system) -> None:
    _section_title("Document inspector", "Inspector")
    items = docs(system)
    if not items:
        st.markdown('<div class="empty-state">No documents yet. Upload and index a PDF first.</div>', unsafe_allow_html=True)
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
    _section_title("Runtime settings", "Settings")
    st.caption("Live settings are applied to the current process; persistent defaults remain in configuration/environment files.")
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
        submitted = st.form_submit_button("Apply live settings", type="primary", use_container_width=True)

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
    requested = st.session_state.get("studio_nav", "Overview")
    page = render_topbar(system, requested)

    if page == "Overview":
        render_overview(system)
    elif page == "Ingestion":
        render_ingestion(system)
    elif page == "Inspector":
        render_inspector(system)
    elif page == "Settings":
        render_settings(system)
    elif page == "Chat":
        render_chat(system)
    elif page == "Health":
        render_health(system)
    else:
        render_background(system)


if __name__ == "__main__":
    main()
