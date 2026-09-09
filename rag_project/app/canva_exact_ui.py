from __future__ import annotations

import hashlib
import html
import threading
import time
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from rag_project.application import create_rag_system
from rag_project.configuration.settings import Settings

PAGES = [
    "Overview",
    "Documents",
    "Index them",
    "Ingestion",
    "Chat",
    "Inspector",
    "Health",
    "Settings",
    "Background",
]
ACTIVE_STAGES = {
    "RUNNING",
    "DISCOVERED",
    "VALIDATING",
    "EXTRACTING",
    "OCR",
    "CHUNKING",
    "EMBEDDING",
    "INDEXING",
    "VALIDATING_INDEX",
    "BUILDING",
    "INTERRUPTED",
    "RECOVERING",
}
STAGE_PROGRESS = {
    "RUNNING": 0.03,
    "DISCOVERED": 0.08,
    "VALIDATING": 0.15,
    "EXTRACTING": 0.30,
    "OCR": 0.42,
    "CHUNKING": 0.55,
    "EMBEDDING": 0.70,
    "INDEXING": 0.84,
    "VALIDATING_INDEX": 0.94,
    "READY": 1.0,
    "COMPLETED": 1.0,
    "FAILED": 1.0,
    "FAILED_EMBEDDING": 1.0,
    "INTERRUPTED": 1.0,
    "RECOVERING": 0.12,
}
STAGE_LABELS = {
    "RUNNING": "Starting",
    "DISCOVERED": "Found PDF",
    "VALIDATING": "Checking document",
    "EXTRACTING": "Reading pages",
    "OCR": "Reading scanned pages",
    "CHUNKING": "Preparing searchable sections",
    "EMBEDDING": "Creating search vectors",
    "INDEXING": "Building search index",
    "VALIDATING_INDEX": "Checking index",
    "READY": "Ready for questions",
    "COMPLETED": "Ready for questions",
    "FAILED": "Needs attention",
    "FAILED_EMBEDDING": "Embedding failed",
    "INTERRUPTED": "Interrupted",
    "RECOVERING": "Recovering",
}
STAGE_HELP = {
    "Found PDF": "The PDF was accepted and queued for processing.",
    "Checking document": "We check the PDF structure and classify the document.",
    "Reading pages": "Text is extracted page by page.",
    "Reading scanned pages": "OCR is being used for pages where normal text extraction is insufficient.",
    "Preparing searchable sections": "The extracted text is split into retrieval-friendly chunks.",
    "Creating search vectors": "Each chunk is converted into a semantic vector for similarity search.",
    "Building search index": "Vector and lexical indexes are updated and committed.",
    "Checking index": "The new index is validated before the document becomes available to chat.",
}


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def ready_docs(system) -> list[dict[str, Any]]:
    return [
        item
        for item in docs(system)
        if str(item.get("status", "")).upper() in {"READY", "COMPLETED"}
    ]


def active_document_count(system) -> int:
    return sum(
        str(item.get("status", "")).upper() in ACTIVE_STAGES for item in docs(system)
    )


def total_chunks(system) -> int:
    return sum(
        _safe_int(
            item.get(
                "chunk_count",
                item.get("chunks", item.get("vector_chunks", 0)),
            )
        )
        for item in docs(system)
    )


def total_embeddings(system) -> int:
    return sum(
        _safe_int(
            item.get("embedding_count", item.get("embeddings", 0))
        )
        for item in docs(system)
    )


def _status_class(value: Any) -> str:
    state = str(value or "UNKNOWN").upper()
    if state in {"READY", "PASS", "COMPLETED", "HEALTHY", "OK", "ONLINE"}:
        return "good"
    if state in {
        "FAILED",
        "FAIL",
        "ERROR",
        "UNAVAILABLE",
        "ABSTAIN",
        "INTERRUPTED",
        "OFFLINE",
    }:
        return "bad"
    if state in ACTIVE_STAGES or state in {"RUNNING", "PROCESSING", "BUILDING", "WARN", "WARNING"}:
        return "warn"
    return "neutral"


def _status(value: Any) -> str:
    text = str(value or "UNKNOWN")
    return f'<span class="status { _status_class(text) }">{_esc(text)}</span>'


def _navigate(page: str) -> None:
    st.session_state["studio_nav"] = page if page in PAGES else "Overview"
    st.rerun()


def _refresh() -> None:
    st.rerun()


def _set_job_state(job: dict[str, Any], **updates: Any) -> None:
    registry = get_jobs()
    with registry["lock"]:
        job.update(updates)


def _running_job(registry: dict[str, Any]) -> dict[str, Any] | None:
    with registry["lock"]:
        for job in reversed(list(registry["items"].values())):
            if job.get("status") == "RUNNING":
                return job
    return None


def _job_snapshot() -> list[dict[str, Any]]:
    registry = get_jobs()
    with registry["lock"]:
        return [dict(item) for item in registry["items"].values()]


def _document_progress(document: dict[str, Any]) -> float:
    status = str(document.get("status") or document.get("current_stage") or "RUNNING").upper()
    base = STAGE_PROGRESS.get(status, STAGE_PROGRESS.get(str(document.get("current_stage") or "RUNNING").upper(), 0.03))
    if status in {"EXTRACTING", "OCR"}:
        total = _safe_int(document.get("total_pages"))
        page = _safe_int(document.get("current_page"))
        if total > 0:
            band = 0.17 if status == "EXTRACTING" else 0.10
            base += min(page / total, 1.0) * band
    return min(max(base, 0.0), 1.0)


def _document_stage(document: dict[str, Any]) -> str:
    raw = str(document.get("current_stage") or document.get("status") or "RUNNING").upper()
    return STAGE_LABELS.get(raw, raw.replace("_", " ").title())


def _document_name(document: dict[str, Any]) -> str:
    return str(
        document.get("file_name")
        or document.get("filename")
        or document.get("document_id")
        or "Unnamed document"
    )


def _event_rows(system, limit: int = 120) -> list[dict[str, Any]]:
    try:
        return list(system.state_store.get_events(limit=limit) or [])
    except Exception:
        return []


def _latest_events_by_document(system) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for event in _event_rows(system, limit=500):
        document_id = str(event.get("document_id") or "")
        if document_id:
            latest[document_id] = event
    return latest


def start_ingestion(system, source_dir: str, *, trigger: str = "manual") -> str:
    folder = Path(source_dir).expanduser().resolve()
    root = Path(system.settings.project_root).expanduser().resolve()
    try:
        folder.relative_to(root)
    except ValueError as exc:
        raise ValueError("The incoming folder must stay inside the BookRAG project directory.") from exc
    if not folder.is_dir():
        raise ValueError(f"Folder does not exist: {folder}")
    pdfs = [item for item in folder.iterdir() if item.is_file() and item.suffix.lower() == ".pdf"]
    if not pdfs:
        raise ValueError("There are no PDF files waiting in Incoming.")

    registry = get_jobs()
    with registry["lock"]:
        running = _running_job(registry)
        if running:
            return str(running["id"])
        jid = f"ingest-{time.time_ns()}"
        job = {
            "id": jid,
            "status": "RUNNING",
            "started": time.time(),
            "finished": None,
            "result": None,
            "error": None,
            "source_dir": str(folder),
            "file_count": len(pdfs),
            "file_names": [item.name for item in pdfs],
            "trigger": trigger,
            "completed": 0,
            "failed": 0,
        }
        registry["items"][jid] = job

    def worker() -> None:
        try:
            result = system.ingest_directory(str(folder))
            result = result or []
            completed = sum(1 for row in result if row.get("status") in {"success", "skipped"})
            failed = sum(1 for row in result if row.get("status") == "failed")
            state = "FAILED" if failed and completed == 0 else "COMPLETED"
            _set_job_state(
                job,
                result=result,
                completed=completed,
                failed=failed,
                status=state,
            )
        except Exception as exc:
            _set_job_state(job, error=str(exc), status="FAILED")
        finally:
            _set_job_state(job, finished=time.time())

    threading.Thread(
        target=worker,
        name=f"{jid}-worker",
        daemon=True,
    ).start()
    return jid


def auto_start_after_upload(system, added_count: int) -> str | None:
    if added_count <= 0:
        return None
    try:
        job_id = start_ingestion(
            system,
            str(system.settings.incoming_dir),
            trigger="upload",
        )
        st.session_state["studio_last_job"] = job_id
        st.session_state["studio_auto_ingest_started"] = True
        return job_id
    except Exception as exc:
        st.session_state["studio_auto_ingest_error"] = str(exc)
        return None


def save_pdf(incoming: Path, name: str, content: bytes) -> str:
    if not content:
        raise ValueError(f"Uploaded file '{name}' is empty.")
    if not content.startswith(b"%PDF-"):
        raise ValueError(f"Uploaded file '{name}' is not a valid PDF payload.")
    incoming = Path(incoming).expanduser().resolve()
    stem = Path(name).stem or "document"
    safe_stem = "".join(
        ch if ch.isalnum() or ch in "._-" else "_" for ch in stem
    ).strip(" ._") or "document"
    digest = hashlib.sha256(content).hexdigest()
    incoming.mkdir(parents=True, exist_ok=True)
    target = incoming / f"{safe_stem[:80]}_{digest[:12]}.pdf"
    if not target.exists():
        target.write_bytes(content)
    return digest


def ollama_health(base_url: str) -> tuple[bool, str, list[str]]:
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/api/tags",
            timeout=(2.5, 5),
        )
        response.raise_for_status()
        payload = response.json()
        models = (
            [str(item.get("name")) for item in payload.get("models", []) if item.get("name")]
            if isinstance(payload, dict)
            else []
        )
        if not models:
            return True, "Ollama is online, but no models were reported.", []
        return True, f"Ollama is online and reports {len(models)} model(s).", models
    except Exception as exc:
        return False, str(exc), []


def _ask(
    system,
    question: str,
    *,
    target_key: str,
    metadata_filter: dict[str, Any] | None = None,
) -> None:
    question = str(question or "").strip()
    if not question:
        st.warning("Please enter a question. Example: “What are the main contraindications?”")
        return
    if not ready_docs(system):
        st.warning(
            "No document is ready yet. Upload a PDF and wait until its status says “Ready for questions”."
        )
        return
    with st.spinner("Searching your indexed documents and generating a grounded answer…"):
        try:
            st.session_state[target_key] = system.answer(
                question,
                metadata_filter=metadata_filter,
            )
        except Exception as exc:
            st.error(f"The question could not be answered safely: {exc}")


def _header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="page-header"><div><div class="eyebrow">BOOKRAG STUDIO</div>'
        f'<div class="page-title">{_esc(title)}</div>'
        f'<div class="page-subtitle">{_esc(subtitle)}</div></div></div>',
        unsafe_allow_html=True,
    )


def css() -> None:
    st.markdown(
        """<style>
:root{--bg:#0b1020;--panel:#121a29;--panel2:#0e1624;--line:#29394f;--text:#f2f5fa;--muted:#91a0b5;--accent:#756df1;--good:#47d79a;--warn:#efbd58;--bad:#f27182}
html,body{overflow-x:hidden!important}.stApp{background:var(--bg);color:var(--text)}[data-testid="stHeader"]{display:none}
.block-container{max-width:1440px!important;padding:26px 38px 72px!important}
[data-testid="stSidebar"]>div:first-child{padding:22px 18px 32px!important;overflow-y:auto!important}
[data-testid="stSidebar"] .stButton>button{width:100%!important;min-height:40px!important;background:transparent!important;border:1px solid transparent!important;border-radius:9px!important;color:#a8b5c8!important;text-align:left!important;padding:0 12px!important}
[data-testid="stSidebar"] .stButton>button:hover{background:#172235!important;color:#fff!important;border-color:#2b3c55!important}
.brand{display:flex;align-items:center;gap:11px;margin:4px 7px 26px}.brand-mark{width:42px;height:42px;border-radius:11px;display:grid;place-items:center;background:linear-gradient(135deg,#746cf0,#40c9c1);font-size:20px}.brand-name{font-size:15px;font-weight:850}.brand-sub{font-size:10px;color:#718098;margin-top:2px}.nav-label{margin:17px 8px 7px;color:#62718a;font-size:10px;font-weight:850;text-transform:uppercase;letter-spacing:.12em}.sidebar-divider{height:1px;background:var(--line);margin:16px 0}.sidebar-note{margin:12px 8px;color:#718097;font-size:10px;line-height:1.5}
.page-header{display:flex;justify-content:space-between;gap:20px;margin-bottom:22px}.eyebrow{font-size:10px;font-weight:850;letter-spacing:.15em;color:#6e7d94}.page-title{font-size:31px;line-height:1.1;font-weight:900;letter-spacing:-.045em;margin-top:6px}.page-subtitle{font-size:12px;color:#94a1b4;line-height:1.55;margin-top:8px;max-width:900px}
.section{border:1px solid var(--line);border-radius:14px;background:var(--panel);padding:18px;min-width:0}.section+.section{margin-top:16px}.section-title{font-size:14px;font-weight:850}.section-subtitle{font-size:11px;color:var(--muted);margin-top:4px;line-height:1.5}.metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:16px}.metric{border:1px solid var(--line);border-radius:11px;background:var(--panel2);padding:14px}.metric-label{font-size:10px;color:var(--muted)}.metric-value{font-size:30px;font-weight:900;margin-top:7px}.metric-help{font-size:9px;color:#67778e;margin-top:4px;line-height:1.4}
.grid-2{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(300px,1fr);gap:16px;margin-top:16px}.event{display:grid;grid-template-columns:1fr auto;gap:12px;padding:10px 0;border-bottom:1px solid #202d40}.event:last-child{border-bottom:0}.event-copy{font-size:11px;line-height:1.5}.event-meta{font-size:9px;color:#74829a;margin-top:3px;line-height:1.45}.guide{display:grid;grid-template-columns:34px 1fr;gap:10px;padding:10px 0;border-bottom:1px solid #202d40}.guide:last-child{border-bottom:0}.guide-num{width:26px;height:26px;border-radius:8px;background:#1b2540;display:grid;place-items:center;font-weight:900;color:#bfc8ff}.answer{margin-top:14px;border:1px solid #39367d;background:#17183a;border-radius:11px;padding:13px}.answer-text{font-size:12px;line-height:1.68;overflow-wrap:anywhere}.glass-note{border:1px dashed #34455d;border-radius:10px;padding:12px;color:#8997aa;font-size:11px;line-height:1.55}.hero-upload{border:1px dashed #495c78;border-radius:14px;padding:18px;background:#0e1624}.data-wrap{width:100%;overflow:auto;border:1px solid var(--line);border-radius:10px;margin-top:13px}.data-table{width:100%;min-width:780px;border-collapse:collapse;font-size:10px}.data-table th,.data-table td{padding:9px 10px;border-bottom:1px solid #202d40;white-space:nowrap;text-align:left}.data-table th{background:#101827;color:#687891;font-size:9px;text-transform:uppercase;letter-spacing:.06em}.data-table td{color:#ccd5e1}.status{display:inline-flex;padding:4px 8px;border:1px solid currentColor;border-radius:999px;font-size:9px;font-weight:850}.status.good{color:var(--good);background:rgba(73,217,154,.07)}.status.warn{color:var(--warn);background:rgba(239,189,88,.07)}.status.bad{color:var(--bad);background:rgba(242,113,130,.07)}.status.neutral{color:#a3afc0;background:rgba(139,156,179,.07)}.stage{font-size:10px;font-weight:800}.stage-help{font-size:9px;color:#74829a;margin-top:4px;line-height:1.35}.progress-label{display:flex;justify-content:space-between;gap:10px;font-size:10px;color:#9aa7b9;margin:8px 0 4px}.tiny{font-size:9px;color:#6e7d94;line-height:1.45}.live-dot{display:inline-flex;align-items:center;gap:6px;font-size:10px;color:#93a1b4}.live-dot i{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--good);box-shadow:0 0 0 4px rgba(73,217,154,.08)}
.stTextInput input,.stTextArea textarea,.stSelectbox div[data-baseweb="select"],.stNumberInput input{background:#0d1624!important;color:#f4f7fb!important;border-color:#2b3b52!important}.stFileUploader{background:#0d1624!important;border:0!important}.stButton>button{border-radius:9px!important}.stButton>button[kind="primary"]{background:var(--accent)!important;border-color:var(--accent)!important;color:#fff!important}.stCheckbox label,.stSelectbox label,.stTextInput label,.stTextArea label,.stNumberInput label,.stSlider label{font-size:10px!important;color:#8896aa!important}
@media(max-width:1050px){.block-container{padding:24px 24px 56px!important}.grid-2{grid-template-columns:1fr}.metric-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:700px){.block-container{padding:18px 14px 44px!important}.page-title{font-size:25px}.metric-grid{grid-template-columns:1fr}.data-table{min-width:700px}}
</style>""",
        unsafe_allow_html=True,
    )


@st.fragment(run_every="3s")
def live_workspace_status(system) -> None:
    items = docs(system)
    active = active_document_count(system)
    ready = len(ready_docs(system))
    running = _running_job(get_jobs())
    if running:
        label = "Indexing in progress"
        state = "RUNNING"
    elif active:
        label = f"Processing {active} document(s)"
        state = "RUNNING"
    else:
        label = f"{ready} ready document(s)"
        state = "READY" if ready or not items else "IDLE"
    st.markdown(
        f'<div class="live-dot"><i></i><span>{_esc(label)}</span></div> {_status(state)}',
        unsafe_allow_html=True,
    )


def sidebar(system) -> None:
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-mark">📚</div><div>'
            '<div class="brand-name">BookRAG Studio</div>'
            '<div class="brand-sub">Local document research workspace</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        current = st.session_state.get("studio_nav", "Overview")
        st.markdown('<div class="nav-label">Start here</div>', unsafe_allow_html=True)
        for item in ["Overview", "Index them", "Documents", "Ingestion"]:
            if st.button(
                ("●  " if current == item else "○  ") + item,
                key=f"nav_{item}",
                use_container_width=True,
            ):
                _navigate(item)
        st.markdown('<div class="nav-label">Ask & inspect</div>', unsafe_allow_html=True)
        for item in ["Chat", "Inspector", "Health"]:
            if st.button(
                ("●  " if current == item else "○  ") + item,
                key=f"nav_int_{item}",
                use_container_width=True,
            ):
                _navigate(item)
        st.markdown('<div class="nav-label">Manage</div>', unsafe_allow_html=True)
        for item in ["Settings", "Background"]:
            if st.button(
                ("●  " if current == item else "○  ") + item,
                key=f"nav_admin_{item}",
                use_container_width=True,
            ):
                _navigate(item)
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        live_workspace_status(system)
        st.markdown(
            '<div class="sidebar-note">Upload from <b>Index them</b>. Processing starts automatically; you do not need to press an indexing button.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-label">Safety</div>', unsafe_allow_html=True)
        phrase = st.text_input(
            "Type CLEAR ALL PDF DATA to permanently remove managed PDF data",
            key="bookrag_clear_phrase",
            type="password",
            placeholder="CLEAR ALL PDF DATA",
            help="This removes PDFs, index data, and persisted ingestion state managed by BookRAG.",
            label_visibility="collapsed",
        )
        if st.button(
            "Delete all managed PDF data",
            key="exact_clear",
            use_container_width=True,
            disabled=phrase.strip() != "CLEAR ALL PDF DATA",
        ):
            try:
                system.clear_pdf_data()
                st.session_state["bookrag_clear_phrase"] = ""
                st.session_state["saved_pdf_hashes"] = set()
                st.session_state.pop("exact_answer", None)
                st.session_state.pop("console_answer", None)
                st.success("All managed PDF data was removed.")
                _refresh()
            except Exception as exc:
                st.error(f"Cleanup failed: {exc}")


def topbar() -> None:
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("💬 Ask documents", key="top_chat", use_container_width=True):
            _navigate("Chat")
    with c2:
        if st.button("🩺 System health", key="top_health", use_container_width=True):
            _navigate("Health")
    with c3:
        if st.button("↻ Refresh now", key="top_refresh", use_container_width=True):
            _refresh()


def overview(system) -> None:
    items = docs(system)
    ready = ready_docs(system)
    active = active_document_count(system)
    _header(
        "Overview",
        "A simple control center: upload documents, watch the pipeline, then ask questions when the status becomes Ready.",
    )

    st.markdown(
        '<div class="section"><div class="section-title">How to use BookRAG in 3 steps</div>'
        '<div class="section-subtitle">No indexing knowledge is required.</div>'
        '<div class="guide"><div class="guide-num">1</div><div><b>Upload PDFs</b><div class="tiny">Open “Index them” and choose one or more PDF files.</div></div></div>'
        '<div class="guide"><div class="guide-num">2</div><div><b>Wait for “Ready for questions”</b><div class="tiny">BookRAG extracts pages, prepares chunks, creates vectors, and validates the index automatically.</div></div></div>'
        '<div class="guide"><div class="guide-num">3</div><div><b>Ask a grounded question</b><div class="tiny">Use Chat. Answers are generated from the documents that are ready.</div></div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section"><div class="section-title">Workspace at a glance</div>'
        '<div class="section-subtitle">These numbers come from the shared document state and index.</div>'
        '<div class="metric-grid">',
        unsafe_allow_html=True,
    )
    metrics = [
        ("Documents", len(items), "All known documents"),
        ("Ready", len(ready), "Available to Chat"),
        ("Processing", active, "Currently being indexed"),
        ("Search chunks", total_chunks(system), "Indexed retrieval units"),
    ]
    for label, value, help_text in metrics:
        st.markdown(
            f'<div class="metric"><div class="metric-label">{_esc(label)}</div>'
            f'<div class="metric-value">{_esc(value)}</div>'
            f'<div class="metric-help">{_esc(help_text)}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div></div>', unsafe_allow_html=True)

    left, right = st.columns([1.45, 1], gap="medium")
    with left:
        st.markdown(
            '<div class="section"><div class="section-title">Ask your documents</div>'
            '<div class="section-subtitle">Best for a quick first question. The answer will include the evidence references returned by the RAG system.</div>',
            unsafe_allow_html=True,
        )
        question = st.text_area(
            "Question",
            placeholder="Example: What are the main contraindications described in the documents?",
            height=120,
            key="exact_overview_q",
            help="Ask for a fact, summary, comparison, definition, or explanation that should be supported by your uploaded documents.",
            label_visibility="collapsed",
        )
        if st.button("Search documents and answer", key="exact_overview_ask", use_container_width=True, type="primary"):
            _ask(system, question, target_key="exact_answer")
        result = st.session_state.get("exact_answer")
        if result:
            st.markdown(
                f'<div class="answer"><div class="tiny">Answer</div><div class="answer-text">{_esc(result.get("answer", ""))}</div></div>',
                unsafe_allow_html=True,
            )
            citations = result.get("citations", []) or []
            if citations:
                st.markdown("**Evidence references**")
                for citation in citations:
                    st.write(citation)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown(
            '<div class="section"><div class="section-title">Current runtime</div>'
            '<div class="section-subtitle">This is the same runtime used by indexing, Chat, Inspector, Health, and Settings.</div>',
            unsafe_allow_html=True,
        )
        s = system.settings
        values = [
            ("Generation model", s.generation_model),
            ("Embedding model", s.embedding_model),
            ("Ollama host", s.ollama_base_url),
            ("Top results", s.top_k),
        ]
        for label, value in values:
            st.markdown(
                f'<div class="event"><div class="event-copy">{_esc(label)}</div><div>{_esc(value)}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)


def _upload_block(system, *, compact: bool = False) -> None:
    if not compact:
        st.markdown(
            '<div class="hero-upload"><div class="section-title">Add PDFs</div>'
            '<div class="section-subtitle">Choose one or more PDF files. Saving finishes automatically and starts indexing immediately.</div>',
            unsafe_allow_html=True,
        )
    uploads = st.file_uploader(
        "PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="exact_uploads",
        help="PDF files are validated before they are stored. Duplicate content is ignored safely.",
    )
    if uploads:
        saved = st.session_state.setdefault("saved_pdf_hashes", set())
        incoming = Path(system.settings.incoming_dir)
        added = 0
        failures: list[str] = []
        for upload in uploads:
            payload = upload.getvalue()
            digest = hashlib.sha256(payload).hexdigest()
            if digest in saved:
                continue
            try:
                save_pdf(incoming, upload.name, payload)
                saved.add(digest)
                added += 1
            except Exception as exc:
                failures.append(f"{upload.name}: {exc}")
        if failures:
            for message in failures:
                st.error(message)
        if added:
            job_id = auto_start_after_upload(system, added)
            if job_id:
                st.success(
                    f"Added {added} PDF{'s' if added != 1 else ''}. Indexing started automatically (job {job_id[-8:]})."
                )
                st.session_state["studio_nav"] = "Ingestion"
                time.sleep(0.15)
                st.rerun()
            else:
                st.error(
                    st.session_state.get(
                        "studio_auto_ingest_error",
                        "The PDFs were saved, but automatic indexing could not be started.",
                    )
                )
        st.markdown('</div>' if not compact else '', unsafe_allow_html=True)
    elif not compact:
        st.markdown(
            '<div class="tiny" style="margin-top:8px">Tip: you can select multiple files in one upload.</div></div>',
            unsafe_allow_html=True,
        )


def documents(system) -> None:
    _header(
        "Documents",
        "See every PDF BookRAG knows about, its current processing state, and whether it can be used for questions.",
    )
    @st.fragment(run_every="3s")
    def live_documents() -> None:
        items = docs(system)
        st.markdown(
            '<div class="section"><div class="section-title">Document library <span class="live-dot"><i></i>Live</span></div>'
            '<div class="section-subtitle">This table refreshes automatically while documents are processing.</div>',
            unsafe_allow_html=True,
        )
        if not items:
            st.markdown(
                '<div class="glass-note">No documents yet. Open “Index them” to upload your first PDF.</div>',
                unsafe_allow_html=True,
            )
        else:
            headers = ["Status", "File", "Pages", "Chunks", "Embeddings", "Stage", "Progress", "Message"]
            st.markdown(
                '<div class="data-wrap"><table class="data-table"><tr>'
                + ''.join(f'<th>{header}</th>' for header in headers)
                + '</tr>',
                unsafe_allow_html=True,
            )
            latest_events = _latest_events_by_document(system)
            for item in items:
                status = item.get("status", "N/A")
                stage = _document_stage(item)
                pct = _document_progress(item)
                event = latest_events.get(str(item.get("document_id") or ""), {})
                detail = event.get("message") or item.get("error") or STAGE_HELP.get(stage, "Document state is up to date.")
                row = [
                    _status(status),
                    _esc(_document_name(item)),
                    _esc(item.get("total_pages", item.get("pages", "—"))),
                    _esc(item.get("chunk_count", item.get("chunks", "—"))),
                    _esc(item.get("embedding_count", item.get("embeddings", "—"))),
                    _esc(stage),
                    f"{pct * 100:.0f}%",
                    _esc(detail),
                ]
                st.markdown(
                    '<tr>' + ''.join(
                        f'<td>{value}</td>' for value in row
                    ) + '</tr>',
                    unsafe_allow_html=True,
                )
            st.markdown('</table></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    live_documents()


def _live_pipeline(system) -> None:
    @st.fragment(run_every="2s")
    def render() -> None:
        items = docs(system)
        latest_events = _latest_events_by_document(system)
        running = _running_job(get_jobs())
        st.markdown(
            '<div class="section"><div class="section-title">Live pipeline <span class="live-dot"><i></i>Updating every 2 seconds</span></div>'
            '<div class="section-subtitle">The stage, page progress, and event message below come from the shared ingestion state.</div>',
            unsafe_allow_html=True,
        )
        active_items = [item for item in items if str(item.get("status", "")).upper() in ACTIVE_STAGES]
        if running:
            elapsed = max(0.0, time.time() - _safe_float(running.get("started"), time.time()))
            st.info(
                f"Indexing job is active · {running.get('file_count', 0)} PDF(s) · {elapsed:.1f}s elapsed · trigger: {running.get('trigger', 'manual')}"
            )
        if not active_items:
            if running:
                st.info("The worker has started. Document state will appear here as each PDF enters the pipeline.")
            else:
                st.caption("No documents are being processed right now.")
        for item in active_items:
            name = _document_name(item)
            stage = _document_stage(item)
            pct = _document_progress(item)
            total = _safe_int(item.get("total_pages"))
            page = _safe_int(item.get("current_page"))
            event = latest_events.get(str(item.get("document_id") or ""), {})
            detail = event.get("message") or STAGE_HELP.get(stage, "Working on this document…")
            st.markdown(
                f'<div class="stage"><b>{_esc(name)}</b> · {_esc(stage)}</div>'
                f'<div class="stage-help">{_esc(STAGE_HELP.get(stage, detail))}</div>',
                unsafe_allow_html=True,
            )
            st.progress(pct, text=f"Estimated pipeline progress: {pct * 100:.0f}%")
            page_text = f"Page {page} of {total}" if total else "Page progress not available for this stage"
            st.markdown(
                f'<div class="tiny">{_esc(page_text)} · {_esc(detail)}</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)
    render()


def indexing_controls(system) -> None:
    st.markdown(
        '<div class="section"><div class="section-title">Automatic indexing rules</div>'
        '<div class="section-subtitle">BookRAG owns the complete flow: upload → validate → extract → chunk → embed → index → verify → ready.</div>'
        '<div class="guide"><div class="guide-num">✓</div><div><b>You do not need to press “Start indexing”.</b><div class="tiny">The upload action calls the ingestion worker automatically.</div></div></div>'
        '<div class="guide"><div class="guide-num">↻</div><div><b>The monitor is live.</b><div class="tiny">The Ingestion and Background pages refresh automatically while a job is active.</div></div></div>'
        '<div class="guide"><div class="guide-num">!</div><div><b>Errors stay visible.</b><div class="tiny">A failed document remains in state with an error message so you can inspect the cause before retrying.</div></div></div>'
        '</div>',
        unsafe_allow_html=True,
    )


def index_them(system) -> None:
    _header(
        "Index them",
        "This is the only upload step you need. Select PDFs and the complete indexing process starts automatically.",
    )
    _upload_block(system)
    indexing_controls(system)
    if docs(system):
        _live_pipeline(system)


def ingestion(system, *, index_mode: bool = False) -> None:
    _header(
        "Ingestion" if not index_mode else "Index them",
        "Watch every indexing stage live. Processing begins automatically after upload; this page is primarily a monitor.",
    )
    if index_mode:
        _upload_block(system)
    _live_pipeline(system)

    @st.fragment(run_every="3s")
    def recent_result() -> None:
        registry = get_jobs()
        with registry["lock"]:
            jobs = list(registry["items"].values())
        last = next(
            (job for job in reversed(jobs) if job.get("status") in {"COMPLETED", "FAILED"}),
            None,
        )
        if last:
            if last.get("status") == "COMPLETED":
                st.success(
                    f"Latest job completed: {last.get('completed', 0)} succeeded/skipped, {last.get('failed', 0)} failed."
                )
            else:
                st.error(str(last.get("error") or "The latest indexing job failed. Inspect Documents and Background for details."))
    recent_result()

    with st.expander("What does each stage mean?", expanded=False):
        for stage in ["Found PDF", "Checking document", "Reading pages", "Reading scanned pages", "Preparing searchable sections", "Creating search vectors", "Building search index", "Checking index"]:
            st.markdown(
                f"**{stage}:** {STAGE_HELP[stage]}"
            )

    st.markdown(
        '<div class="section"><div class="section-title">Manual recovery</div>'
        '<div class="section-subtitle">Normally you will never need this. Use it only when PDFs remain in Incoming after an interrupted session.</div>',
        unsafe_allow_html=True,
    )
    incoming = Path(system.settings.incoming_dir)
    waiting = [item for item in incoming.glob("*.pdf") if item.is_file()]
    st.write(f"PDFs currently waiting in Incoming: **{len(waiting)}**")
    if waiting:
        st.caption("The recovery button is intentionally separate from normal upload. Automatic indexing is the default path.")
        if st.button("Retry waiting PDFs", key="ingestion_recover", use_container_width=True, type="secondary"):
            try:
                st.session_state["studio_last_job"] = start_ingestion(
                    system,
                    str(incoming),
                    trigger="recovery",
                )
                _refresh()
            except Exception as exc:
                st.error(str(exc))
    st.markdown('</div>', unsafe_allow_html=True)


def chat(system) -> None:
    _header(
        "Chat",
        "Ask questions about the documents that are currently ready. Use plain language; the system handles retrieval and evidence selection.",
    )
    available = ready_docs(system)
    st.markdown(
        '<div class="section"><div class="section-title">Ask a question</div>'
        '<div class="section-subtitle">Examples: “Summarize the treatment options”, “What is the difference between X and Y?”, or “Which contraindications are listed?”</div>',
        unsafe_allow_html=True,
    )
    if not available:
        st.markdown(
            '<div class="glass-note">Chat is waiting for at least one document to become <b>Ready for questions</b>. Upload a PDF from “Index them”.</div>',
            unsafe_allow_html=True,
        )
    names = ["All ready documents"] + [_document_name(item) for item in available]
    scope = st.selectbox(
        "Document scope",
        names,
        key="exact_scope",
        help="Use “All ready documents” to search the full library, or choose one document for a focused answer.",
    )
    selected_filter = None
    if scope != names[0] and available:
        selected_filter = {
            "document_id": available[names.index(scope) - 1].get("document_id")
        }
    nonce = st.session_state.get("studio_chat_nonce", 0)
    question = st.text_area(
        "Question",
        placeholder="Ask something you expect the PDFs to answer…",
        height=150,
        key=f"studio_question_{nonce}",
        help="Avoid vague questions. Mention the exact concept, drug, finding, section, or comparison you need.",
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Search and answer", key="exact_chat_search", use_container_width=True, type="primary"):
            _ask(system, question, target_key="console_answer", metadata_filter=selected_filter)
    with c2:
        if st.button("Clear this answer", key="exact_chat_clear", use_container_width=True):
            st.session_state.pop("console_answer", None)
            st.session_state["studio_chat_nonce"] = nonce + 1
            _refresh()

    result = st.session_state.get("console_answer")
    if result:
        st.markdown(
            f'<div class="answer"><div class="tiny">Answer</div><div class="answer-text">{_esc(result.get("answer", ""))}</div></div>',
            unsafe_allow_html=True,
        )
        citations = result.get("citations", []) or []
        if citations:
            st.markdown("### Evidence references")
            for citation in citations:
                st.write(citation)
        with st.expander("Show evidence and query trace", expanded=False):
            st.json(
                {
                    "evidence": result.get("evidence", []),
                    "query_trace": result.get("query_trace", {}),
                }
            )
    st.markdown('</div>', unsafe_allow_html=True)


def inspector(system) -> None:
    _header(
        "Inspector",
        "For advanced users and troubleshooting: inspect one document's metadata, page checkpoints, events, and index verification result.",
    )
    items = docs(system)
    st.markdown('<div class="section">', unsafe_allow_html=True)
    if not items:
        st.markdown(
            '<div class="glass-note">Nothing to inspect yet. Upload a PDF first.</div>',
            unsafe_allow_html=True,
        )
    else:
        labels = [_document_name(item) for item in items]
        selected_index = st.selectbox(
            "Choose a document",
            range(len(items)),
            format_func=lambda index: labels[index],
            key="exact_inspector_doc",
            help="Select the document whose processing and index state you want to inspect.",
        )
        selected = items[selected_index]
        for label, value in [
            ("Status", selected.get("status")),
            ("Current stage", _document_stage(selected)),
            ("Pages", selected.get("total_pages", selected.get("pages"))),
            ("Chunks", selected.get("chunk_count", selected.get("chunks"))),
            ("Embeddings", selected.get("embedding_count", selected.get("embeddings"))),
            ("Embedding dimension", selected.get("embedding_dimension", selected.get("dimension"))),
            ("Version", selected.get("version_id", selected.get("version"))),
            ("Document ID", selected.get("document_id")),
        ]:
            st.markdown(
                f'<div class="event"><div class="event-copy">{_esc(label)}</div>'
                f'<div>{_status(value) if label == "Status" else _esc(value)}</div></div>',
                unsafe_allow_html=True,
            )
        st.progress(_document_progress(selected), text=f"Estimated progress: {_document_progress(selected) * 100:.0f}%")
        if st.button(
            "Verify this document's index",
            key="exact_verify",
            use_container_width=True,
            type="primary",
            help="Runs the existing vector-store consistency check for this document.",
        ):
            try:
                result = system.verify_index(selected.get("document_id"))
                st.json(result)
            except Exception as exc:
                st.error(f"Index verification failed: {exc}")
        with st.expander("Page checkpoints", expanded=False):
            st.json(selected.get("page_checkpoints", []))
        with st.expander("Processing events", expanded=False):
            try:
                st.json(system.state_store.get_events(selected.get("document_id"), limit=250))
            except Exception as exc:
                st.error(f"Could not load document events: {exc}")
        with st.expander("Raw metadata", expanded=False):
            st.json(selected)
    st.markdown('</div>', unsafe_allow_html=True)


def health(system) -> None:
    _header(
        "Health",
        "A practical readiness check. Green means the component is usable; red tells you what needs attention.",
    )
    try:
        report = system.health_report()
    except Exception as exc:
        report = {
            "ready": False,
            "embedding": {"ok": False, "error": str(exc)},
            "index": {"status": "UNAVAILABLE", "error": str(exc)},
            "audit": {"ok": False, "error": str(exc)},
            "feature_contract": {"all_resolved": False},
        }
    ollama_ok, ollama_message, models = ollama_health(system.settings.ollama_base_url)
    embedding = report.get("embedding", {})
    index = report.get("index", {})
    audit = report.get("audit", {})
    contract = report.get("feature_contract", {})
    index_status = str(index.get("status", "UNAVAILABLE")).upper()
    embedding_ok = bool(embedding.get("ok"))
    audit_ok = bool(
        audit.get("ok", audit.get("status") in {"PASS", "READY", "OK"})
    )
    contract_ok = bool(contract.get("all_resolved", False))
    overall = bool(report.get("ready")) and ollama_ok

    rows = [
        (
            "Ollama service",
            "ONLINE" if ollama_ok else "OFFLINE",
            ollama_message,
            "Start Ollama or fix the configured host if this is offline.",
        ),
        (
            "Embedding engine",
            "PASS" if embedding_ok else "FAIL",
            embedding.get("identity") or embedding.get("error") or system.settings.embedding_model,
            "The embedding model must be available before new documents can be indexed.",
        ),
        (
            "Vector index",
            "READY" if index_status in {"READY", "OK"} else index_status,
            index.get("error") or index_status,
            "A non-ready index can prevent grounded retrieval.",
        ),
        (
            "Production contract",
            "PASS" if contract_ok else "FAIL",
            "All production features resolved" if contract_ok else "Some production features are unresolved.",
            "This is an application self-check, not a document-quality check.",
        ),
        (
            "Index audit",
            "PASS" if audit_ok else "FAIL",
            audit.get("error") or ("Consistency checks complete" if audit_ok else "Audit failed."),
            "Use Inspector for document-specific verification.",
        ),
    ]

    st.markdown(
        f'<div class="section"><div class="section-title">Overall readiness: {_esc("READY" if overall else "ATTENTION REQUIRED")}</div>'
        f'<div class="section-subtitle">BookRAG can answer questions reliably only when the required runtime and index components are healthy.</div>'
        f'<div class="data-wrap"><table class="data-table"><tr><th>Component</th><th>Status</th><th>Current result</th><th>What it means</th></tr>',
        unsafe_allow_html=True,
    )
    for label, status, detail, meaning in rows:
        st.markdown(
            f'<tr><td>{_esc(label)}</td><td>{_status(status)}</td><td>{_esc(detail)}</td><td>{_esc(meaning)}</td></tr>',
            unsafe_allow_html=True,
        )
    st.markdown('</table></div>', unsafe_allow_html=True)
    if models:
        st.markdown("**Models reported by Ollama:**")
        st.write(models)
    else:
        st.caption("No Ollama model list was returned.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="section"><div class="section-title">Before you ask a question</div>'
        '<div class="section-subtitle">Beginner checklist</div>'
        '<div class="guide"><div class="guide-num">1</div><div><b>Ollama is online</b><div class="tiny">The configured local host responds to /api/tags.</div></div></div>'
        '<div class="guide"><div class="guide-num">2</div><div><b>At least one document is Ready</b><div class="tiny">Check Documents and wait for “Ready for questions”.</div></div></div>'
        '<div class="guide"><div class="guide-num">3</div><div><b>Then use Chat</b><div class="tiny">Select a document scope and ask a precise question.</div></div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    if st.button("Recheck health now", key="exact_health_recheck", use_container_width=True):
        _refresh()


def settings(system) -> None:
    _header(
        "Settings",
        "Change runtime behavior with explanations. Fields that require a new runtime are clearly reported after saving.",
    )
    s = system.settings
    st.markdown(
        '<div class="section"><div class="section-title">Basic settings</div>'
        '<div class="section-subtitle">These are the settings most users may need to change.</div>',
        unsafe_allow_html=True,
    )
    with st.form("exact_settings_form"):
        host = st.text_input(
            "Ollama host",
            value=str(s.ollama_base_url),
            help="The local HTTP endpoint where your Ollama service is running.",
        )
        embedding = st.text_input(
            "Embedding model",
            value=str(s.embedding_model),
            help="Model used to convert document chunks into semantic search vectors.",
        )
        generation = st.text_input(
            "Generation model",
            value=str(s.generation_model),
            help="Model used to write the final answer from retrieved evidence.",
        )
        c1, c2 = st.columns(2)
        with c1:
            chunk = st.number_input(
                "Chunk size",
                200,
                4000,
                int(s.chunk_size),
                50,
                help="How much extracted text is placed into one retrieval unit.",
            )
            top_k = st.number_input(
                "Search results to retrieve",
                1,
                50,
                int(s.top_k),
                1,
                help="How many top retrieval candidates are considered before answer generation.",
            )
            vector_weight = st.slider(
                "Semantic search weight",
                0.0,
                1.0,
                float(s.vector_weight),
                0.05,
                help="Higher values give more weight to semantic/vector similarity versus lexical matching.",
            )
        with c2:
            overlap = st.number_input(
                "Chunk overlap",
                0,
                3999,
                int(s.chunk_overlap),
                10,
                help="How much neighboring text is repeated between chunks. It must stay below chunk size.",
            )
            temperature = st.slider(
                "Answer creativity",
                0.0,
                1.0,
                float(s.temperature),
                0.05,
                help="Lower values favor more deterministic answers.",
            )
            neighbor = st.checkbox(
                "Expand neighboring context",
                value=bool(s.neighbor_expansion),
                help="When enabled, nearby chunks from the same document can be added to the answer context.",
            )
        apply = st.form_submit_button("Save and apply settings", use_container_width=True)
    if apply:
        if overlap >= chunk:
            st.error("Chunk overlap must be smaller than chunk size.")
        else:
            updates = {
                "ollama_base_url": host,
                "embedding_model": embedding,
                "generation_model": generation,
                "chunk_size": int(chunk),
                "chunk_overlap": int(overlap),
                "top_k": int(top_k),
                "temperature": float(temperature),
                "vector_weight": float(vector_weight),
                "neighbor_expansion": bool(neighbor),
            }
            try:
                ok, warnings = system.apply_settings_in_place(updates)
                if ok:
                    st.success("Settings were applied to the shared runtime.")
                for warning in warnings or []:
                    st.warning(warning)
            except Exception as exc:
                st.error(f"Settings could not be applied: {exc}")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Advanced configuration", expanded=False):
        st.write("Advanced environment-backed settings remain available through your existing .env/configuration workflow.")
        st.caption("This UI intentionally keeps beginner controls separate from deployment-specific tuning.")


def background(system) -> None:
    _header(
        "Background",
        "Live worker activity plus durable document events. This is where you look when something is processing, waiting, or failed.",
    )
    @st.fragment(run_every="2s")
    def live_background() -> None:
        jobs = _job_snapshot()
        items = docs(system)
        st.markdown(
            '<div class="section"><div class="section-title">Worker activity <span class="live-dot"><i></i>Live</span></div>'
            '<div class="section-subtitle">The worker registry is paired with durable ingestion events, so status remains understandable even when the worker is quiet.</div>',
            unsafe_allow_html=True,
        )
        if jobs:
            for job in reversed(jobs):
                started = _safe_float(job.get("started"), time.time())
                finished = _safe_float(job.get("finished"), time.time()) if job.get("finished") else time.time()
                elapsed = max(0.0, finished - started)
                outcome = job.get("status", "UNKNOWN")
                summary = (
                    f"{job.get('completed', 0)} completed/skipped · {job.get('failed', 0)} failed · "
                    f"{job.get('file_count', 0)} PDF(s) · trigger: {job.get('trigger', 'manual')}"
                )
                if job.get("error"):
                    summary = str(job["error"])
                st.markdown(
                    f'<div class="event"><div class="event-copy"><b>{_esc(Path(job.get("source_dir", "")).name or "Incoming")}</b>'
                    f'<div class="event-meta">{_esc(summary)} · {elapsed:.1f}s</div></div>{_status(outcome)}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="glass-note">No app-session worker has been started yet. Upload a PDF to create the first job.</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="section"><div class="section-title">Durable document states</div>'
            '<div class="section-subtitle">These rows come from the SQLite ingestion state, not only from the Streamlit session.</div>',
            unsafe_allow_html=True,
        )
        if items:
            for item in items:
                stage = _document_stage(item)
                detail = item.get("error") or STAGE_HELP.get(stage, "State recorded by the ingestion pipeline.")
                st.markdown(
                    f'<div class="event"><div class="event-copy"><b>{_esc(_document_name(item))}</b>'
                    f'<div class="event-meta">{_esc(stage)} · {_esc(detail)}</div></div>{_status(item.get("status"))}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No persisted document states yet.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="section"><div class="section-title">Recent ingestion events</div>'
            '<div class="section-subtitle">Useful when you want to understand exactly where a document is spending time.</div>',
            unsafe_allow_html=True,
        )
        events = _event_rows(system, limit=60)
        if events:
            for event in reversed(events[-25:]):
                timestamp = event.get("created_at", "")
                message = event.get("message") or event.get("stage") or event.get("event_type")
                name = event.get("file_name") or event.get("document_id") or "document"
                st.markdown(
                    f'<div class="event"><div class="event-copy"><b>{_esc(name)}</b>'
                    f'<div class="event-meta">{_esc(message)} · {_esc(timestamp)}</div></div>{_status(event.get("status") or event.get("stage"))}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No ingestion events have been recorded yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    live_background()
    if st.button("Refresh background view now", key="exact_background_refresh", use_container_width=True):
        _refresh()


def main() -> None:
    st.set_page_config(
        page_title="BookRAG Studio",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.session_state.setdefault("studio_nav", "Overview")
    st.session_state.setdefault("studio_chat_nonce", 0)
    st.session_state.setdefault("saved_pdf_hashes", set())
    system = get_system()
    css()
    sidebar(system)
    topbar()
    page = st.session_state.get("studio_nav", "Overview")
    if page == "Overview":
        overview(system)
    elif page == "Documents":
        documents(system)
    elif page == "Index them":
        index_them(system)
    elif page == "Ingestion":
        ingestion(system)
    elif page == "Chat":
        chat(system)
    elif page == "Inspector":
        inspector(system)
    elif page == "Health":
        health(system)
    elif page == "Settings":
        settings(system)
    elif page == "Background":
        background(system)
    else:
        st.session_state["studio_nav"] = "Overview"
        st.rerun()


if __name__ == "__main__":
    main()
