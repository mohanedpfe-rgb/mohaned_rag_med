from __future__ import annotations

import hashlib
import html
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from rag_project.application import create_rag_system
from rag_project.configuration.settings import Settings

PAGES = ["Home", "Documents", "Live Processing", "Ask BookRAG", "Inspector", "System", "Settings"]
ACTIVE_STAGES = {
    "RUNNING", "DISCOVERED", "VALIDATING", "EXTRACTING", "OCR", "CHUNKING",
    "EMBEDDING", "INDEXING", "VALIDATING_INDEX", "BUILDING", "INTERRUPTED", "RECOVERING",
}
STAGE_INFO = {
    "RUNNING": ("Starting", "The document has entered the processing pipeline."),
    "DISCOVERED": ("Found PDF", "The PDF was accepted and is being prepared."),
    "VALIDATING": ("Checking PDF", "The PDF structure and document type are being checked."),
    "EXTRACTING": ("Reading pages", "BookRAG is reading the PDF page by page."),
    "OCR": ("Reading scanned pages", "OCR is reading pages that do not contain usable text."),
    "CHUNKING": ("Preparing sections", "Extracted text is being split into searchable sections."),
    "EMBEDDING": ("Creating semantic search", "Search vectors are being generated for the sections."),
    "INDEXING": ("Building index", "Vectors and lexical records are being written to the search index."),
    "VALIDATING_INDEX": ("Checking index", "The new index is being checked before it becomes searchable."),
    "READY": ("Ready", "This PDF can now be used for grounded questions."),
    "COMPLETED": ("Ready", "This PDF can now be used for grounded questions."),
    "FAILED": ("Needs attention", "Processing stopped because an error occurred."),
    "FAILED_EMBEDDING": ("Embedding failed", "The embedding service could not create search vectors."),
    "INTERRUPTED": ("Interrupted", "Processing stopped before the PDF was completed."),
    "RECOVERING": ("Recovering", "BookRAG is preparing the document for another attempt."),
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
        stamp = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(stamp)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _elapsed(start: Any, end: Any | None = None) -> float:
    started = _utc(start)
    if not started:
        return 0.0
    finish = _utc(end) if end else datetime.now(timezone.utc)
    if not finish:
        finish = datetime.now(timezone.utc)
    return max(0.0, (finish - started).total_seconds())


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
    return [d for d in docs(system) if str(d.get("status", "")).upper() in ACTIVE_STAGES]


def total_chunks(system) -> int:
    return sum(_int(d.get("chunk_count", d.get("chunks", d.get("vector_chunks", 0)))) for d in docs(system))


def total_embeddings(system) -> int:
    return sum(_int(d.get("embedding_count", d.get("embeddings", 0))) for d in docs(system))


def _stage(document: dict[str, Any]) -> tuple[str, str]:
    raw = str(document.get("current_stage") or document.get("status") or "RUNNING").upper()
    return STAGE_INFO.get(raw, (raw.replace("_", " ").title(), "BookRAG is processing this document."))


def _status_class(value: Any) -> str:
    state = str(value or "UNKNOWN").upper()
    if state in {"READY", "COMPLETED", "PASS", "ONLINE", "HEALTHY", "OK"}:
        return "good"
    if state in {"FAILED", "ERROR", "FAIL", "OFFLINE", "UNAVAILABLE", "INTERRUPTED"}:
        return "bad"
    if state in ACTIVE_STAGES or state in {"RUNNING", "PROCESSING", "BUILDING", "WARNING", "WARN"}:
        return "warn"
    return "neutral"


def _status(value: Any) -> str:
    text = str(value or "UNKNOWN")
    return f'<span class="status status-{_status_class(text)}">{_esc(text)}</span>'


def _navigate(page: str) -> None:
    st.session_state["bookrag_page"] = page if page in PAGES else "Home"
    st.rerun()


def _refresh() -> None:
    st.rerun()


def _snapshot_jobs() -> list[dict[str, Any]]:
    registry = get_jobs()
    with registry["lock"]:
        return [dict(item) for item in registry["items"].values()]


def _running_job() -> dict[str, Any] | None:
    registry = get_jobs()
    with registry["lock"]:
        for job in reversed(list(registry["items"].values())):
            if job.get("status") == "RUNNING":
                return job
    return None


def _set_job(job: dict[str, Any], **updates: Any) -> None:
    registry = get_jobs()
    with registry["lock"]:
        job.update(updates)


def save_pdf(incoming: Path, name: str, content: bytes) -> str:
    if not content:
        raise ValueError(f"{name} is empty.")
    if not content.startswith(b"%PDF-"):
        raise ValueError(f"{name} is not a valid PDF file.")
    incoming = Path(incoming).expanduser().resolve()
    safe_stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in (Path(name).stem or "document"))
    safe_stem = safe_stem.strip(" ._") or "document"
    digest = hashlib.sha256(content).hexdigest()
    incoming.mkdir(parents=True, exist_ok=True)
    target = incoming / f"{safe_stem[:80]}_{digest[:12]}.pdf"
    if not target.exists():
        target.write_bytes(content)
    return digest


def start_ingestion(system, source_dir: str, *, trigger: str = "manual") -> str:
    folder = Path(source_dir).expanduser().resolve()
    root = Path(system.settings.project_root).expanduser().resolve()
    try:
        folder.relative_to(root)
    except ValueError as exc:
        raise ValueError("The Incoming folder must stay inside the BookRAG project directory.") from exc
    if not folder.is_dir():
        raise ValueError(f"Folder does not exist: {folder}")
    pdfs = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    if not pdfs:
        raise ValueError("There are no PDF files waiting in Incoming.")
    registry = get_jobs()
    with registry["lock"]:
        for job in reversed(list(registry["items"].values())):
            if job.get("status") == "RUNNING":
                return str(job["id"])
        jid = f"ingest-{time.time_ns()}"
        job = {
            "id": jid,
            "status": "RUNNING",
            "started": time.time(),
            "finished": None,
            "source_dir": str(folder),
            "file_count": len(pdfs),
            "file_names": [p.name for p in pdfs],
            "trigger": trigger,
            "completed": 0,
            "failed": 0,
            "result": None,
            "error": None,
        }
        registry["items"][jid] = job

    def worker() -> None:
        try:
            result = system.ingest_directory(str(folder)) or []
            completed = sum(1 for row in result if row.get("status") in {"success", "skipped"})
            failed = sum(1 for row in result if row.get("status") == "failed")
            _set_job(job, result=result, completed=completed, failed=failed, status="FAILED" if failed and not completed else "COMPLETED")
        except Exception as exc:
            _set_job(job, error=str(exc), status="FAILED")
        finally:
            _set_job(job, finished=time.time())

    threading.Thread(target=worker, name=f"{jid}-worker", daemon=True).start()
    return jid


def auto_ingest(system, added_count: int) -> str | None:
    if added_count <= 0:
        return None
    try:
        job_id = start_ingestion(system, str(system.settings.incoming_dir), trigger="upload")
        st.session_state["last_ingest_job"] = job_id
        st.session_state["auto_ingest_message"] = f"{added_count} PDF file(s) accepted. Processing started automatically."
        return job_id
    except Exception as exc:
        st.session_state["auto_ingest_error"] = str(exc)
        return None


def _progress(document: dict[str, Any]) -> float:
    status = str(document.get("status") or document.get("current_stage") or "RUNNING").upper()
    weights = {
        "RUNNING": 0.03, "DISCOVERED": 0.08, "VALIDATING": 0.16, "EXTRACTING": 0.32,
        "OCR": 0.48, "CHUNKING": 0.58, "EMBEDDING": 0.72, "INDEXING": 0.85,
        "VALIDATING_INDEX": 0.95, "READY": 1.0, "COMPLETED": 1.0,
        "FAILED": 1.0, "FAILED_EMBEDDING": 1.0, "INTERRUPTED": 1.0,
    }
    value = weights.get(status, 0.03)
    total = _int(document.get("total_pages"))
    current = _int(document.get("current_page"))
    if status in {"EXTRACTING", "OCR"} and total:
        value += min(current / total, 1.0) * (0.13 if status == "EXTRACTING" else 0.10)
    return max(0.0, min(value, 1.0))


def _live_now_text(document: dict[str, Any], latest_event: dict[str, Any] | None) -> str:
    current = _int(document.get("current_page"))
    total = _int(document.get("total_pages"))
    stage, description = _stage(document)
    message = (latest_event or {}).get("message") or description
    page_text = f"Page {current} of {total}" if total else "Page count not available yet"
    return f"{page_text} · {message}"


def _upload_panel(system) -> None:
    st.markdown('<div class="upload-card"><div class="upload-title">Add your PDFs</div><div class="upload-subtitle">Choose one or more PDF files. BookRAG saves them and starts processing automatically.</div>', unsafe_allow_html=True)
    uploads = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True, key="bookrag_uploader", help="You can select multiple PDFs at once. Duplicate files are ignored automatically.")
    if uploads:
        saved = st.session_state.setdefault("uploaded_hashes", set())
        incoming = Path(system.settings.incoming_dir)
        added = 0
        errors: list[str] = []
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
                errors.append(f"{upload.name}: {exc}")
        for message in errors:
            st.error(message)
        if added:
            job_id = auto_ingest(system, added)
            if job_id:
                st.success(f"{added} PDF file(s) added. Automatic processing is now running.")
                st.session_state["bookrag_page"] = "Live Processing"
                st.rerun()
            else:
                st.error(st.session_state.get("auto_ingest_error", "The PDFs were saved, but automatic processing could not be started."))
    st.markdown('</div>', unsafe_allow_html=True)


@st.fragment(run_every="2s")
def live_status_bar(system) -> None:
    items = docs(system)
    active = active_docs(system)
    ready = ready_docs(system)
    running = _running_job()
    if active or running:
        state, label = "RUNNING", f"{len(active)} document(s) processing"
    elif ready:
        state, label = "READY", f"{len(ready)} document(s) ready"
    elif items:
        state, label = "IDLE", "No document is processing"
    else:
        state, label = "IDLE", "Waiting for your first PDF"
    st.markdown(f'<div class="status-bar"><span class="live-pulse"></span><b>Live</b><span>{_esc(label)}</span><span class="status-spacer"></span>{_status(state)}</div>', unsafe_allow_html=True)


@st.fragment(run_every="2s")
def live_processing_panel(system, *, compact: bool = False) -> None:
    items = active_docs(system)
    latest = {str(e.get("document_id")): e for e in events(system, limit=500)}
    st.markdown('<div class="panel"><div class="panel-head"><div><div class="panel-title">Live processing</div><div class="panel-subtitle">Exact document stage, current page, elapsed time and latest activity.</div></div><div class="live-label"><span class="live-pulse"></span>updates every 2s</div></div>', unsafe_allow_html=True)
    if not items:
        if _running_job():
            st.info("The worker is active. Waiting for the document state to appear…")
        else:
            st.markdown('<div class="empty">Nothing is processing right now.</div>', unsafe_allow_html=True)
    for document in items:
        document_id = str(document.get("document_id") or "")
        stage, description = _stage(document)
        event = latest.get(document_id)
        elapsed = _elapsed(document.get("ingestion_started_at"))
        current = _int(document.get("current_page"))
        total = _int(document.get("total_pages"))
        pct = _progress(document)
        name = str(document.get("file_name") or document.get("filename") or document_id)
        page_text = f"Page {current} / {total}" if total else "Page information pending"
        st.markdown(f'<div class="doc-live"><div class="doc-live-top"><div><div class="doc-name">{_esc(name)}</div><div class="doc-stage">{_esc(stage)}</div></div><div class="doc-time">{elapsed:.1f}s</div></div><div class="stage-desc">{_esc(description)}</div>', unsafe_allow_html=True)
        st.progress(pct, text=f"Pipeline progress · {pct * 100:.0f}%")
        st.markdown(f'<div class="live-grid"><div><span>Current page</span><b>{_esc(page_text)}</b></div><div><span>Chunks</span><b>{_esc(document.get("chunk_count", "—"))}</b></div><div><span>Embeddings</span><b>{_esc(document.get("embedding_count", "—"))}</b></div><div><span>Activity</span><b>{_esc(_live_now_text(document, event))}</b></div></div></div>', unsafe_allow_html=True)
        if not compact:
            page_rows = pages(system, document_id)
            if page_rows:
                with st.expander(f"Page-by-page progress · {name}", expanded=False):
                    completed = [p for p in page_rows if str(p.get("extraction_status", "")).upper() == "COMPLETED"]
                    st.caption(f"{len(completed)} / {len(page_rows)} page record(s) completed")
                    for page in page_rows:
                        pno = _int(page.get("page_number"))
                        pstatus = str(page.get("extraction_status") or "PENDING")
                        ocr = str(page.get("ocr_status") or "not_required")
                        chars = len(str(page.get("text") or ""))
                        updated = page.get("updated_at") or ""
                        state = "Completed" if pstatus.upper() == "COMPLETED" else "Pending"
                        st.markdown(f'<div class="page-row"><div><b>Page {pno}</b><small>{_esc(updated)}</small></div><div>{_status(state)}</div><div>OCR: {_esc(ocr)}</div><div>{chars:,} chars</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def _header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><div class="eyebrow">BOOKRAG · LOCAL DOCUMENT RESEARCH</div><h1>{_esc(title)}</h1><p>{_esc(subtitle)}</p></div>', unsafe_allow_html=True)


def css() -> None:
    st.markdown("""<style>
:root{--bg:#f4f7fb;--surface:#fff;--surface2:#f8fafc;--ink:#142033;--muted:#66758a;--line:#dfe6ef;--blue:#2563eb;--teal:#0f9f8c;--green:#13966b;--amber:#c78208;--red:#d9485f;--shadow:0 8px 28px rgba(24,42,70,.07)}
.stApp{background:var(--bg);color:var(--ink)}[data-testid="stHeader"]{background:transparent}.block-container{max-width:1400px;padding:24px 34px 64px}.hero{margin:6px 0 22px}.eyebrow{font-size:10px;letter-spacing:.14em;font-weight:800;color:#7890ad}.hero h1{font-size:34px;line-height:1.08;margin:7px 0 6px;color:#102039;letter-spacing:-.035em}.hero p{margin:0;max-width:840px;color:var(--muted);font-size:13px;line-height:1.6}
.status-bar{display:flex;align-items:center;gap:8px;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:10px 13px;box-shadow:var(--shadow);font-size:12px;margin-bottom:16px}.status-spacer{flex:1}.live-pulse{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 0 5px rgba(19,150,107,.10);display:inline-block}.live-label{font-size:10px;color:#6e7e93;display:flex;gap:7px;align-items:center}.panel{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:18px;box-shadow:var(--shadow);margin-top:16px}.panel-head{display:flex;justify-content:space-between;gap:18px;align-items:flex-start;margin-bottom:14px}.panel-title{font-size:16px;font-weight:850}.panel-subtitle{font-size:11px;color:var(--muted);margin-top:4px;line-height:1.5}.doc-live{border:1px solid #e3e9f1;background:#fbfcfe;border-radius:13px;padding:14px;margin-top:12px}.doc-live-top{display:flex;justify-content:space-between;gap:16px}.doc-name{font-size:13px;font-weight:850;overflow-wrap:anywhere}.doc-stage{font-size:11px;color:var(--blue);font-weight:750;margin-top:3px}.doc-time{font-size:12px;color:#53657b;font-variant-numeric:tabular-nums}.stage-desc{font-size:10px;color:var(--muted);margin:7px 0 7px;line-height:1.45}.live-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;margin-top:10px}.live-grid>div{background:#f5f8fc;border:1px solid #e5ebf2;border-radius:9px;padding:8px}.live-grid span{display:block;font-size:9px;color:#75869c;text-transform:uppercase;letter-spacing:.05em}.live-grid b{display:block;font-size:11px;margin-top:3px;overflow-wrap:anywhere}.page-row{display:grid;grid-template-columns:1.3fr .7fr 1fr .7fr;gap:10px;padding:8px 0;border-bottom:1px solid #e7edf4;font-size:10px;align-items:center}.page-row:last-child{border-bottom:0}.page-row small{display:block;color:#8a98aa;font-size:9px;margin-top:2px}.status{display:inline-flex;align-items:center;border-radius:999px;border:1px solid currentColor;padding:3px 8px;font-size:9px;font-weight:800}.status-good{color:var(--green);background:#edf9f4}.status-warn{color:var(--amber);background:#fff7e6}.status-bad{color:var(--red);background:#fff0f2}.status-neutral{color:#6d7d92;background:#f1f4f8}.upload-card{background:linear-gradient(135deg,#eef5ff,#f8fbff);border:1px solid #d8e4f3;border-radius:16px;padding:18px;box-shadow:var(--shadow)}.upload-title{font-size:17px;font-weight:900}.upload-subtitle{font-size:11px;color:var(--muted);margin:5px 0 12px;line-height:1.5}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.metric{background:var(--surface);border:1px solid var(--line);border-radius:13px;padding:14px;box-shadow:var(--shadow)}.metric-label{font-size:10px;color:var(--muted)}.metric-value{font-size:28px;font-weight:900;margin-top:5px}.metric-help{font-size:9px;color:#8593a5;margin-top:4px}.two{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(300px,.9fr);gap:16px;margin-top:16px}.guide{display:grid;grid-template-columns:32px 1fr;gap:10px;padding:10px 0;border-bottom:1px solid #e6ecf3}.guide:last-child{border-bottom:0}.guide-num{width:28px;height:28px;border-radius:8px;background:#eaf1ff;color:var(--blue);display:grid;place-items:center;font-weight:900}.muted{color:var(--muted);font-size:11px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:11px;margin-top:10px}.table-wrap table{width:100%;border-collapse:collapse;min-width:800px;font-size:10px}.table-wrap th,.table-wrap td{padding:9px 10px;border-bottom:1px solid #e7edf4;text-align:left;white-space:nowrap}.table-wrap th{background:#f7f9fc;color:#74859a;font-size:9px;text-transform:uppercase;letter-spacing:.05em}.empty{border:1px dashed #cfd8e4;border-radius:11px;padding:20px;text-align:center;color:#7b8a9e;font-size:11px}
[data-testid="stSidebar"]>div:first-child{padding:20px 16px 28px!important;background:#fff;border-right:1px solid #e3e8ef}[data-testid="stSidebar"] .stButton>button{border-radius:9px!important;border:1px solid transparent!important;text-align:left!important;background:transparent!important;color:#51627a!important;min-height:38px!important;font-size:12px!important}[data-testid="stSidebar"] .stButton>button:hover{background:#eef4ff!important;color:#1f55b9!important;border-color:#d7e4fb!important}.brand{display:flex;gap:10px;align-items:center;padding:4px 5px 19px}.brand-icon{width:38px;height:38px;border-radius:10px;background:linear-gradient(135deg,#2563eb,#0f9f8c);display:grid;place-items:center;color:#fff;font-weight:900}.brand-name{font-size:14px;font-weight:900}.brand-sub{font-size:9px;color:#8593a5;margin-top:2px}.nav-title{font-size:9px;font-weight:850;letter-spacing:.1em;color:#94a2b4;text-transform:uppercase;margin:16px 7px 6px}.side-note{font-size:9px;color:#8593a5;line-height:1.5;padding:7px}.danger{border-top:1px solid #e6ebf2;margin-top:16px;padding-top:12px}
.stTextInput input,.stTextArea textarea,.stSelectbox div[data-baseweb="select"],.stNumberInput input{background:#fff!important;color:#172234!important;border:1px solid #d5deea!important;border-radius:9px!important}.stFileUploader{background:transparent!important}.stButton>button{border-radius:9px!important}.stButton>button[kind="primary"]{background:var(--blue)!important;border-color:var(--blue)!important;color:#fff!important}.stButton>button[kind="secondary"]{border-color:#cfd9e6!important}.stProgress>div>div>div>div{background:var(--blue)!important}.stAlert{border-radius:10px}.stExpander{border:1px solid #e1e8f0;border-radius:10px}.stMarkdown,.stCaption{font-size:12px}
@media(max-width:1000px){.block-container{padding:20px}.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.two{grid-template-columns:1fr}.live-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:650px){.block-container{padding:16px}.hero h1{font-size:27px}.cards{grid-template-columns:1fr}.live-grid{grid-template-columns:1fr}.page-row{grid-template-columns:1fr 1fr}.panel-head{flex-direction:column}}
</style>""", unsafe_allow_html=True)


def sidebar(system) -> None:
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-icon">BR</div><div><div class="brand-name">BookRAG</div><div class="brand-sub">Local document assistant</div></div></div>', unsafe_allow_html=True)
        current = st.session_state.get("bookrag_page", "Home")
        st.markdown('<div class="nav-title">Workspace</div>', unsafe_allow_html=True)
        for item in ["Home", "Documents", "Live Processing", "Ask BookRAG"]:
            if st.button(("●  " if current == item else "○  ") + item, key=f"side_{item}", use_container_width=True):
                _navigate(item)
        st.markdown('<div class="nav-title">Tools</div>', unsafe_allow_html=True)
        for item in ["Inspector", "System", "Settings"]:
            if st.button(("●  " if current == item else "○  ") + item, key=f"side_tool_{item}", use_container_width=True):
                _navigate(item)
        st.markdown('<div class="nav-title">Live status</div>', unsafe_allow_html=True)
        live_status_bar(system)
        st.markdown('<div class="side-note">The sidebar can be collapsed with Streamlit’s menu button. Everything continues running while it is closed.</div>', unsafe_allow_html=True)
        st.markdown('<div class="danger"><div class="nav-title" style="margin-top:0">Danger zone</div>', unsafe_allow_html=True)
        phrase = st.text_input("Confirmation phrase", key="clear_phrase", type="password", placeholder="CLEAR ALL PDF DATA", help="Use only when you intentionally want to remove PDFs, index data and persisted ingestion state.")
        if st.button("Delete all managed PDF data", key="side_clear", use_container_width=True, disabled=phrase.strip() != "CLEAR ALL PDF DATA"):
            try:
                system.clear_pdf_data()
                st.session_state["uploaded_hashes"] = set()
                st.session_state.pop("exact_answer", None)
                st.session_state.pop("chat_answer", None)
                _refresh()
            except Exception as exc:
                st.error(f"Cleanup failed: {exc}")
        st.markdown('</div>', unsafe_allow_html=True)


def top_actions() -> None:
    a, b, c = st.columns([1, 1, 1])
    with a:
        if st.button("＋ Add PDFs", key="top_add", use_container_width=True):
            _navigate("Documents")
    with b:
        if st.button("⌕ Ask BookRAG", key="top_ask", use_container_width=True):
            _navigate("Ask BookRAG")
    with c:
        if st.button("↻ Refresh", key="top_refresh", use_container_width=True):
            _refresh()


def home(system) -> None:
    _header("Welcome to BookRAG", "A beginner-friendly local workspace for turning PDFs into a searchable, grounded knowledge base.")
    top_actions()
    items, ready, active = docs(system), ready_docs(system), active_docs(system)
    st.markdown('<div class="cards">', unsafe_allow_html=True)
    for label, value, help_text in [
        ("Documents", len(items), "PDFs known to BookRAG"),
        ("Ready", len(ready), "Available for questions"),
        ("Processing", len(active), "Currently being handled"),
        ("Search sections", total_chunks(system), "Retrieval units in the index"),
    ]:
        st.markdown(f'<div class="metric"><div class="metric-label">{_esc(label)}</div><div class="metric-value">{_esc(value)}</div><div class="metric-help">{_esc(help_text)}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    if active:
        live_processing_panel(system, compact=True)
    else:
        _upload_panel(system)
        st.markdown('<div class="two"><div class="panel"><div class="panel-title">How it works</div><div class="muted">You do not need to understand RAG internals.</div>', unsafe_allow_html=True)
        for number, title, body in [
            ("1", "Upload", "Choose one or more PDF files. They are validated and saved automatically."),
            ("2", "Watch", "BookRAG reads pages, prepares sections, creates search vectors and verifies the index."),
            ("3", "Ask", "When a document says Ready, open Ask BookRAG and ask a question in normal language."),
        ]:
            st.markdown(f'<div class="guide"><div class="guide-num">{number}</div><div><b>{title}</b><div class="muted">{body}</div></div></div>', unsafe_allow_html=True)
        st.markdown('</div><div class="panel"><div class="panel-title">Good questions</div><div class="muted">Examples users can copy.</div><div class="guide"><div class="guide-num">?</div><div>“Summarize the treatment options for condition X.”</div></div><div class="guide"><div class="guide-num">?</div><div>“What contraindications are listed?”</div></div><div class="guide"><div class="guide-num">?</div><div>“Compare X and Y using only the PDFs.”</div></div></div></div>', unsafe_allow_html=True)


def documents_page(system) -> None:
    _header("Documents", "Your PDF library. Upload here or inspect what has already been processed.")
    _upload_panel(system)
    @st.fragment(run_every="3s")
    def live_library() -> None:
        items = docs(system)
        st.markdown('<div class="panel"><div class="panel-head"><div><div class="panel-title">Document library</div><div class="panel-subtitle">Live view from the persistent ingestion database.</div></div><div class="live-label"><span class="live-pulse"></span>live</div></div>', unsafe_allow_html=True)
        if not items:
            st.markdown('<div class="empty">No PDFs yet. Use the upload box above.</div>', unsafe_allow_html=True)
        else:
            headers = ["Status", "File", "Stage", "Page", "Progress", "Time", "Chunks", "Embeddings"]
            st.markdown('<div class="table-wrap"><table><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr>', unsafe_allow_html=True)
            for item in items:
                stage, _ = _stage(item)
                pct = _progress(item)
                page = _int(item.get("current_page")); total = _int(item.get("total_pages"))
                page_text = f"{page}/{total}" if total else "—"
                elapsed = _elapsed(item.get("ingestion_started_at"), item.get("ingestion_completed_at"))
                st.markdown('<tr>' + ''.join([
                    f'<td>{_status(item.get("status"))}</td>',
                    f'<td>{_esc(item.get("file_name", item.get("filename", "—")))}</td>',
                    f'<td>{_esc(stage)}</td>',
                    f'<td>{_esc(page_text)}</td>',
                    f'<td>{pct * 100:.0f}%</td>',
                    f'<td>{elapsed:.1f}s</td>',
                    f'<td>{_esc(item.get("chunk_count", item.get("chunks", "—")))}</td>',
                    f'<td>{_esc(item.get("embedding_count", item.get("embeddings", "—")))}</td>',
                ]) + '</tr>', unsafe_allow_html=True)
            st.markdown('</table></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    live_library()


def processing_page(system) -> None:
    _header("Live Processing", "Watch exactly what BookRAG is doing. The display refreshes automatically while the worker runs.")
    live_processing_panel(system)
    @st.fragment(run_every="2s")
    def timeline() -> None:
        running = _running_job()
        st.markdown('<div class="panel"><div class="panel-title">Job control</div><div class="panel-subtitle">Normal uploads start automatically. Recovery is only for PDFs left in Incoming after an interrupted run.</div>', unsafe_allow_html=True)
        if running:
            elapsed = max(0.0, time.time() - _float(running.get("started"), time.time()))
            st.info(f"Worker running · {running.get('file_count', 0)} PDF(s) · {elapsed:.1f}s elapsed · started by {running.get('trigger', 'manual')}.")
        waiting = sorted(Path(system.settings.incoming_dir).glob("*.pdf")) if Path(system.settings.incoming_dir).exists() else []
        st.write(f"PDFs waiting in Incoming: **{len(waiting)}**")
        if waiting and not running:
            if st.button("Retry waiting PDFs", key="recover_waiting", use_container_width=True, type="secondary"):
                try:
                    start_ingestion(system, str(system.settings.incoming_dir), trigger="recovery")
                    _refresh()
                except Exception as exc:
                    st.error(str(exc))
        st.markdown('</div>', unsafe_allow_html=True)
        jobs = _snapshot_jobs()
        if jobs:
            st.markdown('<div class="panel"><div class="panel-title">Recent jobs</div>', unsafe_allow_html=True)
            for job in reversed(jobs[-10:]):
                elapsed = max(0.0, _float(job.get("finished"), time.time()) - _float(job.get("started"), time.time())) if job.get("finished") else max(0.0, time.time() - _float(job.get("started"), time.time()))
                summary = f"{job.get('completed', 0)} completed · {job.get('failed', 0)} failed · {job.get('file_count', 0)} PDF(s) · {elapsed:.1f}s"
                st.markdown(f'<div class="page-row"><div><b>{_esc(Path(job.get("source_dir", "Incoming")).name)}</b><small>{_esc(job.get("trigger", "manual"))}</small></div><div>{_status(job.get("status"))}</div><div>{_esc(summary)}</div><div>{_esc(job.get("error") or "")}</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        all_events = events(system, limit=100)
        if all_events:
            st.markdown('<div class="panel"><div class="panel-title">Live event timeline</div><div class="panel-subtitle">Every recorded stage/event with its UTC timestamp.</div>', unsafe_allow_html=True)
            for event in reversed(all_events[-30:]):
                stamp = event.get("created_at") or ""
                name = event.get("file_name") or event.get("document_id") or "document"
                message = event.get("message") or event.get("stage") or event.get("event_type") or "event"
                page = f"page {event.get('current_page')} / {event.get('total_pages')}" if event.get("total_pages") else ""
                st.markdown(f'<div class="page-row"><div><b>{_esc(name)}</b><small>{_esc(stamp)}</small></div><div>{_status(event.get("status") or event.get("stage"))}</div><div>{_esc(page)}</div><div>{_esc(message)}</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    timeline()


def ask_page(system) -> None:
    _header("Ask BookRAG", "Ask in plain language. Answers are generated from the PDF evidence that is currently ready.")
    available = ready_docs(system)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    if not available:
        st.info("No document is ready yet. Upload a PDF and wait until its status becomes Ready.")
    names = ["All ready documents"] + [str(d.get("file_name", d.get("filename", "Document"))) for d in available]
    scope = st.selectbox("Search scope", names, help="Choose the whole ready library or one document.", key="ask_scope")
    selected_filter = None
    if scope != names[0] and available:
        selected_filter = {"document_id": available[names.index(scope) - 1].get("document_id")}
    q = st.text_area("Your question", key=f"ask_question_{st.session_state.get('chat_nonce', 0)}", height=150, placeholder="Example: What are the main contraindications described in the PDFs?", help="Ask for a fact, definition, summary or comparison that your PDFs can support.")
    c1, c2 = st.columns([2, 1])
    with c1:
        if st.button("Get grounded answer", key="ask_run", use_container_width=True, type="primary"):
            if not q.strip():
                st.warning("Please enter a question first.")
            elif not available:
                st.warning("Wait for at least one document to become Ready.")
            else:
                with st.spinner("Searching documents and generating an evidence-grounded answer…"):
                    try:
                        st.session_state["chat_answer"] = system.answer(q.strip(), metadata_filter=selected_filter)
                    except Exception as exc:
                        st.error(f"The question could not be answered safely: {exc}")
    with c2:
        if st.button("Clear answer", key="ask_clear", use_container_width=True):
            st.session_state.pop("chat_answer", None)
            st.session_state["chat_nonce"] = st.session_state.get("chat_nonce", 0) + 1
            _refresh()
    answer = st.session_state.get("chat_answer")
    if answer:
        st.markdown(f'<div class="panel"><div class="panel-title">Answer</div><div class="panel-subtitle">Generated from the retrieved document evidence.</div><div style="margin-top:10px;line-height:1.7;font-size:13px;white-space:pre-wrap">{_esc(answer.get("answer", ""))}</div></div>', unsafe_allow_html=True)
        citations = answer.get("citations", []) or []
        if citations:
            st.markdown("### Evidence references")
            for citation in citations:
                st.write(citation)
        with st.expander("Show retrieved evidence and query trace"):
            st.json({"evidence": answer.get("evidence", []), "query_trace": answer.get("query_trace", {})})
    st.markdown('</div>', unsafe_allow_html=True)


def inspector_page(system) -> None:
    _header("Inspector", "A clear troubleshooting view: document state, exact page records, events and index verification.")
    items = docs(system)
    if not items:
        st.markdown('<div class="empty">No document is available yet.</div>', unsafe_allow_html=True)
        return
    labels = [str(d.get("file_name") or d.get("filename") or d.get("document_id")) for d in items]
    index = st.selectbox("Choose a document", range(len(items)), format_func=lambda i: labels[i], key="inspect_doc")
    selected = items[index]
    stage, description = _stage(selected)
    st.markdown('<div class="panel"><div class="panel-title">Document state</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="table-wrap"><table><tr><th>Field</th><th>Current value</th></tr><tr><td>Status</td><td>{_status(selected.get("status"))}</td></tr><tr><td>Stage</td><td>{_esc(stage)}</td></tr><tr><td>Current page</td><td>{_esc(selected.get("current_page"))} / {_esc(selected.get("total_pages"))}</td></tr><tr><td>Elapsed time</td><td>{_elapsed(selected.get("ingestion_started_at"), selected.get("ingestion_completed_at")):.2f}s</td></tr><tr><td>Chunks</td><td>{_esc(selected.get("chunk_count", "—"))}</td></tr><tr><td>Embeddings</td><td>{_esc(selected.get("embedding_count", "—"))}</td></tr><tr><td>Embedding dimension</td><td>{_esc(selected.get("embedding_dimension", "—"))}</td></tr><tr><td>Error</td><td>{_esc(selected.get("error"))}</td></tr></table></div>', unsafe_allow_html=True)
    st.progress(_progress(selected), text=f"Pipeline progress · {_progress(selected) * 100:.0f}%")
    st.caption(description)
    if st.button("Verify this document's index", key="verify_index_button", type="primary", use_container_width=True):
        try:
            st.json(system.verify_index(selected.get("document_id")))
        except Exception as exc:
            st.error(f"Index verification failed: {exc}")
    st.markdown('</div>', unsafe_allow_html=True)
    page_rows = pages(system, str(selected.get("document_id")))
    st.markdown('<div class="panel"><div class="panel-title">Page records</div><div class="panel-subtitle">Persisted page-by-page extraction/OCR state.</div>', unsafe_allow_html=True)
    if page_rows:
        for page in page_rows:
            st.markdown(f'<div class="page-row"><div><b>Page {_int(page.get("page_number"))}</b><small>{_esc(page.get("updated_at"))}</small></div><div>{_status(page.get("extraction_status"))}</div><div>OCR: {_esc(page.get("ocr_status"))}</div><div>{len(str(page.get("text") or "")):,} chars</div></div>', unsafe_allow_html=True)
    else:
        st.caption("No page records have been stored yet.")
    st.markdown('</div>', unsafe_allow_html=True)
    with st.expander("Processing event history"):
        st.json(events(system, str(selected.get("document_id")), limit=250))
    with st.expander("Raw metadata"):
        st.json(selected)


@st.fragment(run_every="8s")
def system_snapshot(system) -> None:
    try:
        report = system.health_report()
    except Exception as exc:
        report = {"ready": False, "error": str(exc), "embedding": {}, "index": {}, "audit": {}, "feature_contract": {}}
    try:
        ok, message, models = ollama_health(system.settings.ollama_base_url)
    except Exception as exc:
        ok, message, models = False, str(exc), []
    embedding = report.get("embedding", {})
    index = report.get("index", {})
    audit = report.get("audit", {})
    contract = report.get("feature_contract", {})
    rows = [
        ("Ollama", "ONLINE" if ok else "OFFLINE", message),
        ("Embedding", "PASS" if embedding.get("ok") else "FAIL", embedding.get("identity") or embedding.get("error") or system.settings.embedding_model),
        ("Vector index", str(index.get("status", "UNAVAILABLE")).upper(), index.get("error") or "Index status reported by the runtime."),
        ("Production contract", "PASS" if contract.get("all_resolved") else "FAIL", "Application feature resolution check."),
        ("Index audit", "PASS" if audit.get("ok", False) else "FAIL", audit.get("error") or "Consistency check."),
    ]
    st.markdown('<div class="panel"><div class="panel-head"><div><div class="panel-title">System health</div><div class="panel-subtitle">Live runtime diagnostics. No manual refresh is required.</div></div><div class="live-label"><span class="live-pulse"></span>8s</div></div><div class="table-wrap"><table><tr><th>Component</th><th>Status</th><th>Detail</th></tr>', unsafe_allow_html=True)
    for label, status, detail in rows:
        st.markdown(f'<tr><td>{_esc(label)}</td><td>{_status(status)}</td><td>{_esc(detail)}</td></tr>', unsafe_allow_html=True)
    st.markdown('</table></div>', unsafe_allow_html=True)
    if models:
        st.caption("Models reported by Ollama: " + ", ".join(models))
    st.markdown('</div>', unsafe_allow_html=True)


def system_page(system) -> None:
    _header("System", "See the real services BookRAG depends on and what each result means.")
    system_snapshot(system)
    live_status_bar(system)
    st.markdown('<div class="two"><div class="panel"><div class="panel-title">What healthy means</div><div class="guide"><div class="guide-num">✓</div><div><b>Ollama online</b><div class="muted">The local model service responds to health checks.</div></div></div><div class="guide"><div class="guide-num">✓</div><div><b>Embedding engine ready</b><div class="muted">New PDF sections can be converted into search vectors.</div></div></div><div class="guide"><div class="guide-num">✓</div><div><b>Index ready</b><div class="muted">Stored vectors and lexical records can be used for retrieval.</div></div></div></div><div class="panel"><div class="panel-title">Current runtime</div>', unsafe_allow_html=True)
    s = system.settings
    for label, value in [("Ollama host", s.ollama_base_url), ("Embedding model", s.embedding_model), ("Generation model", s.generation_model), ("Search results", s.top_k)]:
        st.markdown(f'<div class="page-row"><div><b>{_esc(label)}</b></div><div></div><div></div><div>{_esc(value)}</div></div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)


def settings_page(system) -> None:
    _header("Settings", "Beginner-safe settings first. Advanced deployment tuning stays behind the existing configuration system.")
    s = system.settings
    st.markdown('<div class="panel"><div class="panel-title">Basic runtime settings</div><div class="panel-subtitle">These controls affect how documents are searched and how answers are generated.</div>', unsafe_allow_html=True)
    with st.form("basic_settings"):
        host = st.text_input("Ollama host", value=str(s.ollama_base_url), help="Local HTTP address used by Ollama.")
        embedding = st.text_input("Embedding model", value=str(s.embedding_model), help="Turns document sections into semantic search vectors.")
        generation = st.text_input("Generation model", value=str(s.generation_model), help="Writes the grounded final answer from retrieved evidence.")
        a, b = st.columns(2)
        with a:
            top_k = st.number_input("Search results", min_value=1, max_value=50, value=int(s.top_k), step=1, help="More results can improve recall, but can add noise.")
            vector_weight = st.slider("Semantic search weight", 0.0, 1.0, float(s.vector_weight), 0.05, help="Higher values rely more on semantic similarity.")
        with b:
            temperature = st.slider("Answer creativity", 0.0, 1.0, float(s.temperature), 0.05, help="Lower is more deterministic; higher is more varied.")
            neighbor = st.checkbox("Use nearby sections", value=bool(s.neighbor_expansion), help="Adds nearby chunks from the same document when useful.")
        save = st.form_submit_button("Save settings", use_container_width=True, type="primary")
    if save:
        try:
            ok, warnings = system.apply_settings_in_place({"ollama_base_url": host, "embedding_model": embedding, "generation_model": generation, "top_k": int(top_k), "vector_weight": float(vector_weight), "temperature": float(temperature), "neighbor_expansion": bool(neighbor)})
            if ok:
                st.success("Settings saved to the shared runtime.")
            for warning in warnings or []:
                st.warning(warning)
        except Exception as exc:
            st.error(f"Settings could not be saved: {exc}")
    st.markdown('</div>', unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="BookRAG", page_icon="📚", layout="wide", initial_sidebar_state="expanded")
    st.session_state.setdefault("bookrag_page", "Home")
    st.session_state.setdefault("chat_nonce", 0)
    st.session_state.setdefault("uploaded_hashes", set())
    system = get_system()
    css()
    sidebar(system)
    page = st.session_state.get("bookrag_page", "Home")
    if page == "Home":
        home(system)
    elif page == "Documents":
        documents_page(system)
    elif page == "Live Processing":
        processing_page(system)
    elif page == "Ask BookRAG":
        ask_page(system)
    elif page == "Inspector":
        inspector_page(system)
    elif page == "System":
        system_page(system)
    elif page == "Settings":
        settings_page(system)
    else:
        st.session_state["bookrag_page"] = "Home"
        st.rerun()


if __name__ == "__main__":
    main()
