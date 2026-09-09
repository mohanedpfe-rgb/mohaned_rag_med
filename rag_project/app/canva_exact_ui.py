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

PAGES = ["Overview", "Documents", "Index them", "Ingestion", "Chat", "Inspector", "Health", "Settings", "Background"]
ACTIVE_STAGES = {"RUNNING", "DISCOVERED", "VALIDATING", "EXTRACTING", "OCR", "CHUNKING", "EMBEDDING", "INDEXING", "VALIDATING_INDEX", "BUILDING"}


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
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
    return [d for d in docs(system) if str(d.get("status", "")).upper() == "READY"]


def active_document_count(system) -> int:
    return sum(str(d.get("status", "")).upper() in ACTIVE_STAGES for d in docs(system))


def total_chunks(system) -> int:
    return sum(_safe_int(d.get("chunk_count", d.get("chunks", d.get("vector_chunks", 0)))) for d in docs(system))


def _status_class(value: Any) -> str:
    state = str(value or "UNKNOWN").upper()
    if state in {"READY", "PASS", "COMPLETED", "HEALTHY", "OK"}:
        return "good"
    if state in {"FAILED", "FAIL", "ERROR", "UNAVAILABLE", "ABSTAIN"}:
        return "bad"
    if state in {"RUNNING", "PROCESSING", "BUILDING", "WARN", "WARNING"}:
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


def start_ingestion(system, source_dir: str) -> str:
    folder = Path(source_dir).expanduser().resolve()
    root = Path(system.settings.project_root).expanduser().resolve()
    try:
        folder.relative_to(root)
    except ValueError as exc:
        raise ValueError("The incoming folder must stay inside the BookRAG project directory.") from exc
    if not folder.is_dir():
        raise ValueError(f"Folder does not exist: {folder}")
    pdfs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    if not pdfs:
        raise ValueError("No PDF files were found in the incoming folder.")

    registry = get_jobs()
    with registry["lock"]:
        for jid, job in reversed(list(registry["items"].items())):
            if job.get("status") == "RUNNING":
                return jid
        jid = f"ingest-{time.time_ns()}"
        registry["items"][jid] = {"id": jid, "status": "RUNNING", "started": time.time(), "finished": None, "result": None, "error": None, "source_dir": str(folder), "file_count": len(pdfs)}

    def worker() -> None:
        job = registry["items"][jid]
        try:
            job["result"] = system.ingest_directory(str(folder))
            job["status"] = "COMPLETED"
        except Exception as exc:
            job["error"] = str(exc)
            job["status"] = "FAILED"
        finally:
            job["finished"] = time.time()

    threading.Thread(target=worker, name=f"{jid}-worker", daemon=True).start()
    return jid


def save_pdf(incoming: Path, name: str, content: bytes) -> str:
    if not content:
        raise ValueError(f"Uploaded file '{name}' is empty.")
    if not content.startswith(b"%PDF-"):
        raise ValueError(f"Uploaded file '{name}' is not a valid PDF payload.")
    incoming = Path(incoming).expanduser().resolve()
    stem = Path(name).stem or "document"
    safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem).strip(" ._") or "document"
    digest = hashlib.sha256(content).hexdigest()
    incoming.mkdir(parents=True, exist_ok=True)
    target = incoming / f"{safe_stem[:80]}_{digest[:12]}.pdf"
    if not target.exists():
        target.write_bytes(content)
    return digest


def ollama_health(base_url: str) -> tuple[bool, str, list[str]]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=(2.5, 5))
        response.raise_for_status()
        payload = response.json()
        models = [str(x.get("name")) for x in payload.get("models", []) if x.get("name")] if isinstance(payload, dict) else []
        return True, "Ollama is reachable.", models
    except Exception as exc:
        return False, str(exc), []


def _ask(system, question: str, *, target_key: str, metadata_filter: dict[str, Any] | None = None) -> None:
    question = str(question or "").strip()
    if not question:
        st.warning("Enter a question first.")
        return
    if not ready_docs(system):
        st.warning("No READY documents are available yet. Upload and index a PDF first.")
        return
    with st.spinner("Searching indexed evidence…"):
        try:
            st.session_state[target_key] = system.answer(question, metadata_filter=metadata_filter)
        except Exception as exc:
            st.error(f"Answer failed safely: {exc}")


def _header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="page-header"><div><div class="eyebrow">BOOKRAG STUDIO</div><div class="page-title">{_esc(title)}</div><div class="page-subtitle">{_esc(subtitle)}</div></div></div>', unsafe_allow_html=True)


def css() -> None:
    st.markdown("""<style>
:root{--bg:#0b1020;--panel:#121a29;--panel2:#0e1624;--line:#27384f;--text:#f2f5fa;--muted:#8e9cb0;--accent:#726af0;--good:#49d99a;--warn:#efbd58;--bad:#f27182}
html,body{overflow-x:hidden!important}.stApp{background:var(--bg);color:var(--text)}[data-testid="stHeader"]{display:none}
.block-container{max-width:1440px!important;padding:30px 38px 72px!important}
[data-testid="stSidebar"]>div:first-child{padding:24px 18px 32px!important;overflow-y:auto!important}
[data-testid="stSidebar"] .stButton>button{width:100%!important;min-height:40px!important;background:transparent!important;border:1px solid transparent!important;border-radius:9px!important;color:#a8b5c8!important;text-align:left!important;padding:0 12px!important}
[data-testid="stSidebar"] .stButton>button:hover{background:#172235!important;color:#fff!important;border-color:#2b3c55!important}
.brand{display:flex;align-items:center;gap:11px;margin:4px 7px 26px}.brand-mark{width:42px;height:42px;border-radius:11px;display:grid;place-items:center;background:linear-gradient(135deg,#746cf0,#40c9c1);font-size:20px}.brand-name{font-size:15px;font-weight:850}.brand-sub{font-size:10px;color:#718098;margin-top:2px}.nav-label{margin:17px 8px 7px;color:#62718a;font-size:10px;font-weight:850;text-transform:uppercase;letter-spacing:.12em}.sidebar-divider{height:1px;background:var(--line);margin:16px 0}.sidebar-note{margin:12px 8px;color:#718097;font-size:10px;line-height:1.5}
.page-header{display:flex;justify-content:space-between;gap:20px;margin-bottom:22px}.eyebrow{font-size:10px;font-weight:850;letter-spacing:.15em;color:#6e7d94}.page-title{font-size:31px;line-height:1.1;font-weight:900;letter-spacing:-.045em;margin-top:6px}.page-subtitle{font-size:12px;color:#94a1b4;line-height:1.5;margin-top:8px;max-width:860px}
.section{border:1px solid var(--line);border-radius:14px;background:var(--panel);padding:18px;min-width:0}.section+.section{margin-top:16px}.section-title{font-size:14px;font-weight:850}.section-subtitle{font-size:11px;color:var(--muted);margin-top:4px}.metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:16px}.metric{border:1px solid var(--line);border-radius:11px;background:var(--panel2);padding:14px}.metric-label{font-size:10px;color:var(--muted)}.metric-value{font-size:30px;font-weight:900;margin-top:7px}.grid-2{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(320px,1fr);gap:16px;margin-top:16px}.event{display:grid;grid-template-columns:1fr auto;gap:12px;padding:10px 0;border-bottom:1px solid #202d40}.event:last-child{border-bottom:0}.event-copy{font-size:11px}.event-meta{font-size:9px;color:#74829a;margin-top:3px}.answer{margin-top:14px;border:1px solid #39367d;background:#17183a;border-radius:11px;padding:13px}.answer-text{font-size:12px;line-height:1.68;overflow-wrap:anywhere}.glass-note{border:1px dashed #34455d;border-radius:10px;padding:12px;color:#8997aa;font-size:11px;line-height:1.5}.data-wrap{width:100%;overflow:auto;border:1px solid var(--line);border-radius:10px;margin-top:13px}.data-table{width:100%;min-width:720px;border-collapse:collapse;font-size:10px}.data-table th,.data-table td{padding:9px 10px;border-bottom:1px solid #202d40;white-space:nowrap;text-align:left}.data-table th{background:#101827;color:#687891;font-size:9px;text-transform:uppercase;letter-spacing:.06em}.data-table td{color:#ccd5e1}.status{display:inline-flex;padding:4px 8px;border:1px solid currentColor;border-radius:999px;font-size:9px;font-weight:850}.status.good{color:var(--good);background:rgba(73,217,154,.07)}.status.warn{color:var(--warn);background:rgba(239,189,88,.07)}.status.bad{color:var(--bad);background:rgba(242,113,130,.07)}.status.neutral{color:#a3afc0;background:rgba(139,156,179,.07)}
.stTextInput input,.stTextArea textarea,.stSelectbox div[data-baseweb="select"],.stNumberInput input{background:#0d1624!important;color:#f4f7fb!important;border-color:#2b3b52!important}.stFileUploader{background:#0d1624!important;border:1px dashed #3b4d65!important;border-radius:10px!important}.stButton>button{border-radius:9px!important}.stButton>button[kind="primary"]{background:var(--accent)!important;border-color:var(--accent)!important;color:#fff!important}.stCheckbox label,.stSelectbox label,.stTextInput label,.stTextArea label,.stNumberInput label{font-size:10px!important;color:#8896aa!important}
@media(max-width:1050px){.block-container{padding:24px 24px 56px!important}.grid-2{grid-template-columns:1fr}.metric-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:700px){.block-container{padding:20px 14px 44px!important}.page-title{font-size:25px}.metric-grid{grid-template-columns:1fr}.data-table{min-width:640px}}
</style>""", unsafe_allow_html=True)


def sidebar(system) -> None:
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-mark">📚</div><div><div class="brand-name">BookRAG Studio</div><div class="brand-sub">Local medical knowledge base</div></div></div>', unsafe_allow_html=True)
        current = st.session_state.get("studio_nav", "Overview")
        st.markdown('<div class="nav-label">Workspace</div>', unsafe_allow_html=True)
        for item in ["Overview", "Documents", "Index them", "Ingestion"]:
            if st.button(("●  " if current == item else "○  ") + item, key=f"nav_{item}", use_container_width=True): _navigate(item)
        st.markdown('<div class="nav-label">Intelligence</div>', unsafe_allow_html=True)
        for item in ["Chat", "Inspector", "Health"]:
            if st.button(("●  " if current == item else "○  ") + item, key=f"nav_int_{item}", use_container_width=True): _navigate(item)
        st.markdown('<div class="nav-label">Administration</div>', unsafe_allow_html=True)
        for item in ["Settings", "Background"]:
            if st.button(("●  " if current == item else "○  ") + item, key=f"nav_admin_{item}", use_container_width=True): _navigate(item)
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-label">Add documents</div>', unsafe_allow_html=True)
        uploads = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True, key="exact_uploads")
        if uploads:
            saved = st.session_state.setdefault("saved_pdf_hashes", set())
            incoming = Path(system.settings.incoming_dir)
            added = 0
            for upload in uploads:
                payload = upload.getvalue(); digest = hashlib.sha256(payload).hexdigest()
                if digest in saved: continue
                try:
                    save_pdf(incoming, upload.name, payload); saved.add(digest); added += 1
                except Exception as exc: st.error(str(exc))
            if added: st.success(f"Added {added} PDF{'s' if added != 1 else ''} to Incoming.")
        if st.button("Start indexing", key="exact_start", use_container_width=True, type="primary"):
            try:
                st.session_state["studio_last_job"] = start_ingestion(system, str(system.settings.incoming_dir))
                _navigate("Ingestion")
            except Exception as exc: st.error(str(exc))
        if st.button("Recreate runtime", key="exact_recreate", use_container_width=True):
            get_system.clear(); _navigate("Overview")
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-label">Danger zone</div>', unsafe_allow_html=True)
        phrase = st.text_input("Type CLEAR ALL PDF DATA", key="bookrag_clear_phrase", type="password", label_visibility="collapsed", placeholder="Type CLEAR ALL PDF DATA")
        if st.button("Clear all PDF data", key="exact_clear", use_container_width=True, disabled=phrase.strip() != "CLEAR ALL PDF DATA"):
            try:
                system.clear_pdf_data()
                st.session_state["bookrag_clear_phrase"] = ""
                st.session_state["saved_pdf_hashes"] = set()
                st.session_state.pop("exact_answer", None); st.session_state.pop("console_answer", None)
                _refresh()
            except Exception as exc: st.error(f"Cleanup failed: {exc}")
        st.markdown('<div class="sidebar-note">One shared runtime for documents, indexing, chat, inspector, health, and settings.</div>', unsafe_allow_html=True)


def topbar() -> None:
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("💬 Chat", key="top_chat", use_container_width=True): _navigate("Chat")
    with c2:
        if st.button("🩺 Health", key="top_health", use_container_width=True): _navigate("Health")
    with c3:
        if st.button("↻ Refresh", key="top_refresh", use_container_width=True): _refresh()


def overview(system) -> None:
    items, ready, active = docs(system), ready_docs(system), active_document_count(system)
    _header("Overview", "Control the complete local RAG workflow from one stable workspace.")
    st.markdown('<div class="section"><div class="section-title">Workspace status</div><div class="section-subtitle">Persistent document state and the same runtime are shared by every page.</div><div class="metric-grid">', unsafe_allow_html=True)
    for label, value in [("Documents", len(items)), ("Ready", len(ready)), ("Processing", active), ("Vector chunks", total_chunks(system))]:
        st.markdown(f'<div class="metric"><div class="metric-label">{_esc(label)}</div><div class="metric-value">{_esc(value)}</div></div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)
    left, right = st.columns([1.45, 1], gap="medium")
    with left:
        st.markdown('<div class="section"><div class="section-title">Ask your documents</div><div class="section-subtitle">Grounded answer generation with citations from the indexed library.</div>', unsafe_allow_html=True)
        question = st.text_area("Question", placeholder="Ask something about your indexed PDFs…", height=120, key="exact_overview_q", label_visibility="collapsed")
        if st.button("Search and answer", key="exact_overview_ask", use_container_width=True, type="primary"):
            _ask(system, question, target_key="exact_answer")
        result = st.session_state.get("exact_answer")
        if result:
            st.markdown(f'<div class="answer"><div class="muted">Answer</div><div class="answer-text">{_esc(result.get("answer", ""))}</div></div>', unsafe_allow_html=True)
            for citation in result.get("citations", []) or []: st.write(citation)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section"><div class="section-title">Runtime summary</div><div class="section-subtitle">Current shared configuration.</div>', unsafe_allow_html=True)
        s = system.settings
        for label, value in [("Generation", s.generation_model), ("Embedding", s.embedding_model), ("Ollama", s.ollama_base_url), ("Top-k", s.top_k)]:
            st.markdown(f'<div class="event"><div class="event-copy">{_esc(label)}</div><div>{_esc(value)}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def documents(system) -> None:
    _header("Documents", "The central document library shared by upload, indexing, chat, and inspection.")
    items = docs(system)
    st.markdown('<div class="section"><div class="section-title">Document library</div><div class="section-subtitle">Status, extraction, chunks, and embedding metadata.</div>', unsafe_allow_html=True)
    if not items:
        st.markdown('<div class="glass-note">No documents yet. Upload PDF files from the sidebar.</div>', unsafe_allow_html=True)
    else:
        headers = ["Status", "File", "Pages", "Chunks", "Embeddings", "Dimension", "Stage", "Error"]
        st.markdown('<div class="data-wrap"><table class="data-table"><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr>', unsafe_allow_html=True)
        for d in items:
            row = [d.get("status", "N/A"), d.get("file_name", d.get("filename", "N/A")), d.get("total_pages", d.get("pages", "N/A")), d.get("chunk_count", d.get("chunks", d.get("vector_chunks", "N/A"))), d.get("embedding_count", d.get("embeddings", "N/A")), d.get("embedding_dimension", d.get("dimension", "N/A")), d.get("current_stage", d.get("stage", "N/A")), d.get("error", "")]
            st.markdown('<tr>' + ''.join(f'<td>{_status(v) if i == 0 else _esc(v)}</td>' for i, v in enumerate(row)) + '</tr>', unsafe_allow_html=True)
        st.markdown('</table></div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        if st.button("Open ingestion", key="documents_ingestion", use_container_width=True): _navigate("Ingestion")
    with b:
        if st.button("Refresh", key="documents_refresh", use_container_width=True): _refresh()
    st.markdown('</div>', unsafe_allow_html=True)


def ingestion(system, *, index_mode: bool = False) -> None:
    _header("Index them" if index_mode else "Ingestion", "Run and monitor the same indexing worker used by the rest of the application.")
    registry = get_jobs()
    with registry["lock"]: jobs = list(registry["items"].values())
    running = next((job for job in reversed(jobs) if job.get("status") == "RUNNING"), None)
    st.markdown('<div class="section"><div class="section-title">Indexing pipeline</div><div class="section-subtitle">A single worker record is shared with Background and reflected in document state.</div>', unsafe_allow_html=True)
    if running:
        elapsed = max(0.0, time.time() - float(running.get("started", time.time())))
        st.info(f"Indexing in progress · {Path(running.get('source_dir', '')).name} · {elapsed:.1f}s · {running.get('file_count', '?')} PDF(s)")
        st.progress(0.35, text="PDF indexing pipeline active…")
    else:
        st.caption("No indexing worker is currently running.")
    if st.button("Start indexing incoming folder", key="ingestion_start_main", use_container_width=True, type="primary"):
        try:
            st.session_state["studio_last_job"] = start_ingestion(system, str(system.settings.incoming_dir))
            _refresh()
        except Exception as exc: st.error(str(exc))
    items = docs(system)
    if items:
        st.markdown('<div class="data-wrap"><table class="data-table"><tr><th>Status</th><th>File</th><th>Stage</th><th>Page</th><th>Chunks</th><th>Embeddings</th><th>Dimension</th><th>Error</th></tr>', unsafe_allow_html=True)
        for d in items:
            values = [d.get("status", "N/A"), d.get("file_name", d.get("filename", "N/A")), d.get("current_stage", d.get("stage", "N/A")), d.get("current_page", d.get("page", "N/A")), d.get("chunk_count", d.get("chunks", d.get("vector_chunks", "N/A"))), d.get("embedding_count", d.get("embeddings", "N/A")), d.get("embedding_dimension", d.get("dimension", "N/A")), d.get("error", "")]
            st.markdown('<tr>' + ''.join(f'<td>{_status(v) if i == 0 else _esc(v)}</td>' for i, v in enumerate(values)) + '</tr>', unsafe_allow_html=True)
        st.markdown('</table></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="glass-note">No document records are available.</div>', unsafe_allow_html=True)
    last = next((job for job in reversed(jobs) if job.get("status") in {"COMPLETED", "FAILED"}), None)
    if last and last.get("status") == "COMPLETED": st.success("Latest indexing job completed successfully.")
    elif last and last.get("status") == "FAILED": st.error(str(last.get("error") or "Indexing failed."))
    a, b = st.columns(2)
    with a:
        if st.button("Refresh progress", key="ingestion_refresh", use_container_width=True): _refresh()
    with b:
        if st.button("Background workers", key="ingestion_background", use_container_width=True): _navigate("Background")
    st.markdown('</div>', unsafe_allow_html=True)


def chat(system) -> None:
    _header("Chat", "Ask questions over the shared indexed library and inspect exactly what supports the answer.")
    available = ready_docs(system)
    st.markdown('<div class="section"><div class="section-title">Grounded search</div><div class="section-subtitle">Choose the whole library or one ready document.</div>', unsafe_allow_html=True)
    names = ["All ready documents"] + [str(d.get("file_name", d.get("filename", "Document"))) for d in available]
    scope = st.selectbox("Search scope", names, key="exact_scope")
    selected_filter = None
    if scope != names[0] and available: selected_filter = {"document_id": available[names.index(scope) - 1].get("document_id")}
    nonce = st.session_state.get("studio_chat_nonce", 0)
    question = st.text_area("Question", placeholder="Ask about your indexed evidence…", height=130, key=f"studio_question_{nonce}")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Search and answer", key="exact_chat_search", use_container_width=True, type="primary"):
            _ask(system, question, target_key="console_answer", metadata_filter=selected_filter)
    with c2:
        if st.button("Clear conversation", key="exact_chat_clear", use_container_width=True):
            st.session_state.pop("console_answer", None); st.session_state["studio_chat_nonce"] = nonce + 1; _refresh()
    result = st.session_state.get("console_answer")
    if result:
        st.markdown(f'<div class="answer"><div class="muted">Answer</div><div class="answer-text">{_esc(result.get("answer", ""))}</div></div>', unsafe_allow_html=True)
        for citation in result.get("citations", []) or []: st.write(citation)
        with st.expander("Evidence and query trace"): st.json({"evidence": result.get("evidence", []), "query_trace": result.get("query_trace", {})})
    st.markdown('</div>', unsafe_allow_html=True)


def inspector(system) -> None:
    _header("Inspector", "Verify document metadata and index state without leaving the shared workspace.")
    items = docs(system)
    st.markdown('<div class="section">', unsafe_allow_html=True)
    if not items:
        st.markdown('<div class="glass-note">No documents are available to inspect.</div>', unsafe_allow_html=True)
    else:
        labels = [str(d.get("file_name", d.get("filename", d.get("document_id", "Document")))) for d in items]
        selected = items[st.selectbox("Document", range(len(items)), format_func=lambda i: labels[i], key="exact_inspector_doc")]
        for label, value in [("Status", selected.get("status")), ("Pages", selected.get("total_pages", selected.get("pages"))), ("Chunks", selected.get("chunk_count", selected.get("chunks"))), ("Embeddings", selected.get("embedding_count", selected.get("embeddings"))), ("Dimension", selected.get("embedding_dimension", selected.get("dimension"))), ("Version", selected.get("version_id", selected.get("version"))), ("Document ID", selected.get("document_id"))]:
            st.markdown(f'<div class="event"><div class="event-copy">{_esc(label)}</div><div>{_status(value) if label == "Status" else _esc(value)}</div></div>', unsafe_allow_html=True)
        if st.button("Verify index", key="exact_verify", use_container_width=True, type="primary"):
            try: st.json(system.verify_index(selected.get("document_id")))
            except Exception as exc: st.error(f"Index verification failed: {exc}")
        with st.expander("Page checkpoints"): st.json(selected.get("page_checkpoints", []))
        with st.expander("Raw metadata"): st.json(selected)
    st.markdown('</div>', unsafe_allow_html=True)


def health(system) -> None:
    _header("Health", "Real readiness diagnostics across Ollama, embeddings, the vector index, contracts, and audit state.")
    try: report = system.health_report()
    except Exception as exc: report = {"ready": False, "embedding": {"ok": False, "error": str(exc)}, "index": {"status": "UNAVAILABLE", "error": str(exc)}, "audit": {"ok": False, "error": str(exc)}, "feature_contract": {"all_resolved": False}}
    ollama_ok, ollama_message, models = ollama_health(system.settings.ollama_base_url)
    embedding = report.get("embedding", {}); index = report.get("index", {}); audit = report.get("audit", {}); contract = report.get("feature_contract", {})
    index_status = str(index.get("status", "UNAVAILABLE")).upper(); embedding_ok = bool(embedding.get("ok")); audit_ok = bool(audit.get("ok", audit.get("status") in {"PASS", "READY", "OK"})); contract_ok = bool(contract.get("all_resolved", False))
    rows = [("Ollama", "PASS" if ollama_ok else "FAIL", ollama_message), ("Embedding", "PASS" if embedding_ok else "FAIL", embedding.get("identity") or embedding.get("error") or system.settings.embedding_model), ("Vector index", "READY" if index_status in {"READY", "OK"} else index_status, index.get("error") or index_status), ("Production contract", "PASS" if contract_ok else "FAIL", "All production features resolved" if contract_ok else "Production feature contract incomplete"), ("Index audit", "PASS" if audit_ok else "FAIL", audit.get("error") or ("Consistency checks complete" if audit_ok else "Audit failed"))]
    overall = bool(report.get("ready")) and ollama_ok
    st.markdown(f'<div class="section"><div class="section-title">Readiness: {_esc("READY" if overall else "ATTENTION REQUIRED")}</div><div class="data-wrap"><table class="data-table"><tr><th>Component</th><th>Status</th><th>Detail</th></tr>', unsafe_allow_html=True)
    for label, status, detail in rows: st.markdown(f'<tr><td>{_esc(label)}</td><td>{_status(status)}</td><td>{_esc(detail)}</td></tr>', unsafe_allow_html=True)
    st.markdown('</table></div>', unsafe_allow_html=True)
    if models: st.write(models)
    if st.button("Recheck health", key="exact_health_recheck", use_container_width=True): _refresh()
    st.markdown('</div>', unsafe_allow_html=True)


def settings(system) -> None:
    _header("Settings", "Apply validated configuration to the same runtime used by every workspace component.")
    s = system.settings
    st.markdown('<div class="section"><div class="section-title">Runtime configuration</div><div class="section-subtitle">Changes are validated before they reach the application.</div>', unsafe_allow_html=True)
    with st.form("exact_settings_form"):
        host = st.text_input("Ollama host", value=str(s.ollama_base_url))
        embedding = st.text_input("Embedding model", value=str(s.embedding_model))
        generation = st.text_input("Generation model", value=str(s.generation_model))
        c1, c2 = st.columns(2)
        with c1:
            chunk = st.number_input("Chunk size", 200, 4000, int(s.chunk_size), 50)
            top_k = st.number_input("Top-k", 1, 50, int(s.top_k), 1)
            vector_weight = st.slider("Vector weight", 0.0, 1.0, float(s.vector_weight), 0.05)
        with c2:
            overlap = st.number_input("Chunk overlap", 0, 3999, int(s.chunk_overlap), 10)
            temperature = st.slider("Temperature", 0.0, 1.0, float(s.temperature), 0.05)
            neighbor = st.checkbox("Neighbor expansion", value=bool(s.neighbor_expansion))
        apply = st.form_submit_button("Apply settings", use_container_width=True)
    if apply:
        if overlap >= chunk:
            st.error("Chunk overlap must be smaller than chunk size.")
        else:
            updates = {"ollama_base_url": host, "embedding_model": embedding, "generation_model": generation, "chunk_size": int(chunk), "chunk_overlap": int(overlap), "top_k": int(top_k), "temperature": float(temperature), "vector_weight": float(vector_weight), "neighbor_expansion": bool(neighbor)}
            try: st.success(f"Settings applied: {system.apply_settings_in_place(updates)}")
            except Exception as exc: st.error(f"Settings could not be applied: {exc}")
    st.markdown('</div>', unsafe_allow_html=True)


def background(system) -> None:
    _header("Background", "Observe the shared indexing worker and completed jobs without losing context.")
    registry = get_jobs()
    with registry["lock"]: jobs = list(registry["items"].values())
    st.markdown('<div class="section"><div class="section-title">Worker activity</div>', unsafe_allow_html=True)
    if not jobs: st.markdown('<div class="glass-note">No background jobs have been started in this application session.</div>', unsafe_allow_html=True)
    for job in reversed(jobs):
        started = float(job.get("started") or time.time()); finished = float(job.get("finished") or time.time()); elapsed = max(0.0, finished - started)
        st.markdown(f'<div class="event"><div class="event-copy">{_esc(Path(job.get("source_dir", "")).name)}<div class="event-meta">{_esc(job.get("error") or f"{job.get('file_count', 0)} PDF(s)")} · {elapsed:.1f}s</div></div>{_status(job.get("status"))}</div>', unsafe_allow_html=True)
    if st.button("Refresh workers", key="exact_background_refresh", use_container_width=True): _refresh()
    st.markdown('</div>', unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="BookRAG Studio", page_icon="📚", layout="wide", initial_sidebar_state="expanded")
    st.session_state.setdefault("studio_nav", "Overview")
    st.session_state.setdefault("studio_chat_nonce", 0)
    system = get_system()
    css()
    sidebar(system)
    topbar()
    page = st.session_state.get("studio_nav", "Overview")
    if page == "Overview": overview(system)
    elif page == "Documents": documents(system)
    elif page == "Index them": ingestion(system, index_mode=True)
    elif page == "Ingestion": ingestion(system)
    elif page == "Chat": chat(system)
    elif page == "Inspector": inspector(system)
    elif page == "Health": health(system)
    elif page == "Settings": settings(system)
    elif page == "Background": background(system)
    else:
        st.session_state["studio_nav"] = "Overview"
        st.rerun()


if __name__ == "__main__": main()
