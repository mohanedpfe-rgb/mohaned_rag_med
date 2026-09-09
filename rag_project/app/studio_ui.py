from __future__ import annotations

import hashlib
import html
import json
import threading
import time
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from rag_project.application import create_rag_system
from rag_project.configuration.settings import Settings


st.set_page_config(page_title="BookRAG Studio", page_icon="📚", layout="wide", initial_sidebar_state="expanded")

NAV_ITEMS = ["Overview", "Chat", "Health", "Ingestion", "Background", "Inspector", "Settings"]
ACTIVE_STAGES = {"RUNNING", "DISCOVERED", "VALIDATING", "EXTRACTING", "OCR", "CHUNKING", "EMBEDDING", "INDEXING", "VALIDATING_INDEX"}


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
    if not any(p.is_file() and p.suffix.lower() == ".pdf" for p in folder.iterdir()):
        raise ValueError(f"No PDF files were found in {folder}.")

    registry = get_jobs()
    job_id = f"ingest-{time.time_ns()}"
    job = {"status": "RUNNING", "started": time.time(), "finished": None, "result": None, "error": None, "source_dir": str(folder)}
    with registry["lock"]:
        registry["items"][job_id] = job
        while len(registry["items"]) > 50:
            registry["items"].pop(next(iter(registry["items"])), None)

    def worker() -> None:
        try:
            result = system.ingest_directory(str(folder))
            with registry["lock"]:
                job["result"] = result
                job["status"] = "COMPLETED"
        except Exception as exc:
            with registry["lock"]:
                job["error"] = repr(exc)
                job["status"] = "FAILED"
        finally:
            with registry["lock"]:
                job["finished"] = time.time()

    threading.Thread(target=worker, name=f"{job_id}-worker", daemon=True).start()
    return job_id


def save_pdf(incoming: Path, name: str, content: bytes) -> str:
    if not content:
        raise ValueError(f"Uploaded file '{name}' is empty.")
    if not content.startswith(b"%PDF-"):
        raise ValueError(f"Uploaded file '{name}' is not a valid PDF payload.")
    source = Path(name).name
    stem = Path(source).stem or "document"
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
        models = [str(x.get("name")) for x in response.json().get("models", []) if x.get("name")]
        return True, "Ollama is reachable.", models
    except Exception as exc:
        return False, str(exc), []


def _status_class(value: str) -> str:
    value = value.strip().upper()
    if value in {"READY", "PASS", "COMPLETED", "HEALTHY", "OK"}:
        return "good"
    if value in {"FAILED", "FAIL", "ERROR", "INTERRUPTED", "UNAVAILABLE"}:
        return "bad"
    if value in {"WARN", "WARNING", "RUNNING", "PROCESSING", "BUILDING"}:
        return "warn"
    return "neutral"


def _status(value: Any) -> str:
    text = str(value or "UNKNOWN")
    return f'<span class="status {_status_class(text)}">{html.escape(text)}</span>'


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"))


def _navigate(page: str) -> None:
    if page in NAV_ITEMS:
        st.session_state["studio_nav"] = page
        st.rerun()


def _format_duration(started: Any, finished: Any = None) -> float:
    try:
        end = float(finished or time.time())
        return round(max(0.0, end - float(started or end)), 1)
    except (TypeError, ValueError):
        return 0.0


def _metric_card(icon: str, label: str, value: Any, helper: str = "") -> None:
    st.markdown(f'<div class="metric-card"><div class="metric-top"><span class="metric-icon">{icon}</span><span>{_esc(label)}</span></div><div class="metric-value">{_esc(value)}</div><div class="metric-helper">{_esc(helper)}</div></div>', unsafe_allow_html=True)


def _section_title(title: str, eyebrow: str = "") -> None:
    prefix = f'<div class="eyebrow">{_esc(eyebrow)}</div>' if eyebrow else ""
    st.markdown(f'{prefix}<div class="section-title">{_esc(title)}</div>', unsafe_allow_html=True)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _settings_snapshot(system) -> dict[str, Any]:
    s = system.settings
    names = [
        "ollama_base_url", "embedding_model", "generation_model", "chunk_size", "chunk_overlap",
        "top_k", "temperature", "vector_weight", "neighbor_expansion", "rerank_top_k",
    ]
    return {name: getattr(s, name, None) for name in names}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root{--bg:#07111f;--panel:rgba(17,29,48,.76);--panel2:#0d1a2b;--line:rgba(148,163,184,.16);--ink:#f8fafc;--muted:#94a3b8;--accent:#7c83ff;--accent2:#35d5c5;--good:#35d399;--warn:#f5b94c;--bad:#ff6b7a}
        .stApp{background:radial-gradient(circle at 85% -10%,rgba(124,131,255,.18),transparent 34%),radial-gradient(circle at 10% 0%,rgba(53,213,197,.10),transparent 28%),var(--bg);color:var(--ink)}
        [data-testid="stHeader"]{background:transparent}.block-container{max-width:1500px;padding:1.1rem 2rem 3rem}.stMarkdown,.stText,.stCaption{color:var(--ink)}
        [data-testid="stSidebar"]{background:rgba(5,12,22,.96);border-right:1px solid var(--line)}
        [data-testid="stSidebar"]>div:first-child{padding:1rem .85rem}.brand{display:flex;gap:.75rem;align-items:center;padding:.45rem .35rem 1rem}.brand-mark{width:42px;height:42px;border-radius:13px;background:linear-gradient(135deg,#7c83ff,#35d5c5);display:flex;align-items:center;justify-content:center;font-size:21px;box-shadow:0 12px 28px rgba(124,131,255,.22)}.brand-name{font-weight:850;font-size:1.02rem}.brand-sub{font-size:.71rem;color:var(--muted);margin-top:.12rem}
        .nav-label{font-size:.67rem;text-transform:uppercase;letter-spacing:.12em;color:#64748b;font-weight:800;margin:.7rem .5rem}.nav-active button{border-color:rgba(124,131,255,.45)!important;background:rgba(124,131,255,.13)!important;color:white!important}
        div.stButton>button{border:1px solid var(--line);border-radius:11px;background:rgba(17,29,48,.72);color:var(--ink);font-weight:700;min-height:2.35rem}div.stButton>button:hover{border-color:rgba(124,131,255,.55);color:white}
        .topbar{display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;margin:.2rem 0 1rem}.topbar h1{font-size:2rem;letter-spacing:-.04em;margin:0}.topbar p{color:var(--muted);margin:.3rem 0 0}.live-pill{padding:.42rem .7rem;border:1px solid rgba(53,211,153,.28);border-radius:999px;color:#8af0c7;background:rgba(53,211,153,.08);font-size:.75rem;font-weight:800}
        .hero{border:1px solid rgba(124,131,255,.24);border-radius:22px;padding:1.5rem;background:linear-gradient(135deg,rgba(17,29,48,.95),rgba(45,48,96,.88));box-shadow:0 25px 60px rgba(0,0,0,.24);margin-bottom:1rem}.hero h2{margin:0;font-size:1.6rem;letter-spacing:-.035em}.hero p{color:#cbd5e1;margin:.45rem 0 0}.hero-actions{margin-top:1rem}
        .metric-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:.8rem;margin:1rem 0}.metric-card{border:1px solid var(--line);background:var(--panel);backdrop-filter:blur(16px);border-radius:16px;padding:.95rem;box-shadow:0 12px 32px rgba(0,0,0,.14)}.metric-top{display:flex;align-items:center;gap:.5rem;color:var(--muted);font-size:.74rem;font-weight:750}.metric-icon{display:inline-flex;width:28px;height:28px;align-items:center;justify-content:center;border-radius:9px;background:rgba(124,131,255,.12);color:#aeb3ff}.metric-value{font-size:1.7rem;font-weight:850;letter-spacing:-.04em;margin-top:.35rem}.metric-helper{font-size:.7rem;color:#64748b;margin-top:.15rem}
        .section-title{font-size:1.12rem;font-weight:820;letter-spacing:-.02em;margin-bottom:.2rem}.eyebrow{font-size:.66rem;text-transform:uppercase;letter-spacing:.13em;color:#64748b;font-weight:850;margin-bottom:.2rem}.panel{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:1rem;margin-bottom:.9rem;box-shadow:0 12px 32px rgba(0,0,0,.10)}.panel-head{display:flex;justify-content:space-between;align-items:center;gap:.8rem;margin-bottom:.8rem}.muted{color:var(--muted)}
        .status{display:inline-flex;align-items:center;border-radius:999px;padding:.22rem .52rem;border:1px solid currentColor;font-size:.68rem;font-weight:850}.status.good{color:#6ee7b7;background:rgba(53,211,153,.08)}.status.warn{color:#f8cc72;background:rgba(245,185,76,.08)}.status.bad{color:#ff8b98;background:rgba(255,107,122,.08)}.status.neutral{color:#a8b3c2;background:rgba(148,163,184,.07)}
        .answer{background:rgba(124,131,255,.07);border:1px solid rgba(124,131,255,.2);border-radius:16px;padding:1rem}.answer-label{color:#aeb3ff;font-size:.66rem;text-transform:uppercase;letter-spacing:.12em;font-weight:900}.answer-text{font-size:1rem;line-height:1.7;margin-top:.5rem}.evidence{border:1px solid var(--line);border-radius:12px;padding:.75rem;margin:.5rem 0;background:rgba(2,8,18,.24)}
        .table-wrap{overflow:auto;border:1px solid var(--line);border-radius:14px}.data-table{width:100%;border-collapse:collapse;font-size:.76rem}.data-table th,.data-table td{padding:.62rem .65rem;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}.data-table th{color:#94a3b8;background:rgba(148,163,184,.04);font-weight:800}.data-table td{color:#dbe4ef}.data-table tr:last-child td{border-bottom:0}
        .kv{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.55rem}.kv-item{padding:.72rem;border:1px solid var(--line);border-radius:12px;background:rgba(148,163,184,.035)}.kv-key{font-size:.67rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em}.kv-value{font-weight:750;margin-top:.18rem;overflow-wrap:anywhere}
        .event{display:grid;grid-template-columns:110px 1fr auto;gap:.8rem;align-items:center;padding:.7rem 0;border-bottom:1px solid var(--line)}.event:last-child{border-bottom:0}.event-time{font-size:.7rem;color:#64748b}.event-copy{font-size:.78rem}.event-meta{font-size:.67rem;color:var(--muted)}
        .glass-note{border:1px dashed rgba(124,131,255,.28);border-radius:14px;padding:.8rem;color:#aeb9c8;background:rgba(124,131,255,.035)}
        @media(max-width:1050px){.metric-grid{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:700px){.block-container{padding:.8rem}.metric-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.kv{grid-template-columns:1fr}.topbar{display:block}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(system) -> None:
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-mark">📚</div><div><div class="brand-name">BookRAG Studio</div><div class="brand-sub">Local document intelligence</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-label">Workspace</div>', unsafe_allow_html=True)
        current = st.session_state.get("studio_nav", "Overview")
        for item in NAV_ITEMS:
            if st.button(("●  " if item == current else "○  ") + item, key=f"nav_{item}", use_container_width=True):
                _navigate(item)

        st.markdown('<div class="nav-label">PDF controls</div>', unsafe_allow_html=True)
        uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True, key="studio_uploads")
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

        source_dir = st.text_input("Incoming folder", value=str(system.settings.incoming_dir), key="studio_source_dir")
        running_id, _ = active_job()
        disabled = running_id is not None or active_document_count(system) > 0
        if st.button("▶ Start all chunks", type="primary", use_container_width=True, disabled=disabled):
            try:
                st.session_state["studio_last_job"] = start_ingestion(system, source_dir)
                st.session_state["studio_nav"] = "Ingestion"
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if running_id:
            st.markdown(f"{_status('RUNNING')} <span class='muted'>{_esc(running_id)}</span>", unsafe_allow_html=True)

        st.markdown('<div class="nav-label">Maintenance</div>', unsafe_allow_html=True)
        if st.button("↻ Recreate runtime", use_container_width=True):
            get_system.clear()
            st.rerun()
        confirm = st.checkbox("Confirm permanent cleanup", key="studio_cleanup_confirm")
        if st.button("🗑️ Clear all PDF data", use_container_width=True, disabled=not confirm):
            try:
                removed = system.clear_pdf_data()
                get_system.clear()
                for key in ("console_answer", "saved_pdf_hashes", "selected_document_id"):
                    st.session_state.pop(key, None)
                st.success(f"Cleanup complete: {len(removed)} item(s) removed.")
                st.rerun()
            except Exception as exc:
                st.error(f"Cleanup failed: {exc}")


def render_topbar(system, active_page: str) -> None:
    all_docs = docs(system)
    ready = len(ready_docs(system))
    running = active_document_count(system)
    state = "Processing" if running else "Ready" if ready else "Waiting"
    st.markdown(f'<div class="topbar"><div><h1>BookRAG Studio</h1><p>{_esc(active_page)} workspace · production RAG control plane</p></div><div class="live-pill">● {_esc(state)}</div></div>', unsafe_allow_html=True)


def render_metrics(system) -> None:
    all_docs = docs(system)
    ready = ready_docs(system)
    running = active_document_count(system)
    vector_chunks = sum(_safe_int(d.get("vector_chunks", d.get("chunks", 0))) for d in all_docs)
    lexical_chunks = sum(_safe_int(d.get("lexical_chunks", d.get("chunks", 0))) for d in all_docs)
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    _metric_card("▣", "Documents", len(all_docs), "Indexed or processing")
    _metric_card("✓", "Ready", len(ready), "Ready for retrieval")
    _metric_card("◌", "Processing", running, "Active ingestion stages")
    _metric_card("⌁", "Vector chunks", vector_chunks, "Embedding index")
    _metric_card("≋", "Lexical chunks", lexical_chunks, "Keyword retrieval")
    st.markdown('</div>', unsafe_allow_html=True)


def _document_rows(system) -> list[dict[str, Any]]:
    return docs(system)


def render_overview(system) -> None:
    _section_title("Document intelligence at a glance", "Overview")
    st.markdown('<div class="hero"><h2>From PDF to grounded answers.</h2><p>Ingest, inspect, retrieve, verify, and operate the complete BookRAG pipeline from one local studio.</p></div>', unsafe_allow_html=True)
    render_metrics(system)
    left, right = st.columns([1.35, .65])
    with left:
        _section_title("Recent documents", "Corpus")
        rows = _document_rows(system)
        if not rows:
            st.markdown('<div class="glass-note">No documents yet. Upload PDFs from the sidebar to begin.</div>', unsafe_allow_html=True)
        else:
            table = '<div class="table-wrap"><table class="data-table"><thead><tr><th>Status</th><th>File</th><th>Pages</th><th>Chunks</th><th>Dimension</th></tr></thead><tbody>'
            for d in rows[-12:][::-1]:
                table += f'<tr><td>{_status(d.get("status"))}</td><td>{_esc(d.get("filename", d.get("file", d.get("document_id", "—"))))}</td><td>{_esc(d.get("pages", d.get("page_count")))}</td><td>{_esc(d.get("chunks", d.get("vector_chunks")))}</td><td>{_esc(d.get("embedding_dimension", d.get("dimension")))}</td></tr>'
            st.markdown(table + '</tbody></table></div>', unsafe_allow_html=True)
    with right:
        _section_title("Quick actions", "Operate")
        if st.button("Open Chat", use_container_width=True): _navigate("Chat")
        if st.button("Inspect index", use_container_width=True): _navigate("Inspector")
        if st.button("Check Health", use_container_width=True): _navigate("Health")
        if st.button("Open Ingestion", use_container_width=True): _navigate("Ingestion")
        st.markdown('<div class="glass-note">The UI is intentionally fail-closed: unavailable models or an incompatible index are surfaced instead of silently producing an answer.</div>', unsafe_allow_html=True)


def _answer_text(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("answer") or result.get("text") or result.get("response") or result)
    return str(result)


def _result_field(result: Any, *keys: str, default: Any = None) -> Any:
    if not isinstance(result, dict): return default
    for key in keys:
        if key in result: return result[key]
    return default


def render_chat(system) -> None:
    _section_title("Grounded document chat", "Chat")
    ready = ready_docs(system)
    if not ready:
        st.markdown('<div class="glass-note">No READY documents are available. Ingest and verify a PDF before asking evidence-backed questions.</div>', unsafe_allow_html=True)
    ids = [str(d.get("document_id")) for d in ready if d.get("document_id")]
    labels = [str(d.get("filename", d.get("document_id", "document"))) for d in ready]
    scope_label = st.selectbox("Search scope", ["All ready documents"] + labels, key="chat_scope")
    selected_filter = None if scope_label == "All ready documents" else ids[labels.index(scope_label)] if scope_label in labels else None
    question = st.text_area("Question", key=f"studio_question_{st.session_state.get('studio_chat_nonce', 0)}", height=120, placeholder="Ask a question grounded in your PDFs…")
    c1, c2 = st.columns([1, 1])
    with c1:
        ask = st.button("Search and answer", type="primary", use_container_width=True, disabled=not bool(question.strip()))
    with c2:
        if st.button("Clear chat", use_container_width=True):
            st.session_state["console_answer"] = None
            st.session_state["studio_chat_nonce"] = st.session_state.get("studio_chat_nonce", 0) + 1
            st.rerun()
    if ask:
        metadata_filter = {"document_id": selected_filter} if selected_filter else None
        with st.spinner("Retrieving evidence and generating a grounded answer…"):
            try:
                # The production system performs retrieval, reranking, grounding, citations, and safety gates.
                result = system.answer(question.strip(), metadata_filter=metadata_filter)
                st.session_state["console_answer"] = result
            except Exception as exc:
                st.session_state["console_answer"] = {"status": "ERROR", "answer": f"The answer pipeline failed safely: {exc!s}"}
    result = st.session_state.get("console_answer")
    if result is None:
        st.markdown('<div class="empty-state">Ask a question to see the answer, confidence, answerability, query quality, citations, and supporting evidence.</div>', unsafe_allow_html=True)
        return
    answer = _answer_text(result)
    confidence = _result_field(result, "confidence", "evidence_confidence", default="—")
    answerability = _result_field(result, "answerability", "grounding", "status", default="—")
    query_quality = _result_field(result, "query_quality", "quality", default="—")
    citations = _result_field(result, "citations", default=[]) or []
    st.markdown(f'<div class="answer"><div class="answer-label">Answer</div><div class="answer-text">{html.escape(answer).replace(chr(10), "<br>")}</div></div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, label, value in zip(cols, ["Confidence", "Answerability", "Query quality", "Citations"], [confidence, answerability, query_quality, len(citations) if isinstance(citations, list) else citations]):
        with col:
            st.metric(label, str(value))
    with st.expander("Evidence and trace", expanded=False):
        evidence = _result_field(result, "evidence", "sources", "retrieved_chunks", default=[])
        if isinstance(evidence, list):
            for item in evidence[:12]:
                if isinstance(item, dict):
                    st.markdown(f'<div class="evidence"><b>{_esc(item.get("filename", item.get("document_id", "Evidence")))}</b> · page {_esc(item.get("page", item.get("page_number", "—")))}<br><span class="muted">{_esc(item.get("text", item.get("content", item.get("snippet", ""))))}</span></div>', unsafe_allow_html=True)
                else:
                    st.write(item)
        else:
            st.json(evidence)
        if isinstance(citations, list) and citations:
            st.write("Citations")
            st.json(citations)
        trace = _result_field(result, "query_trace", "trace", default=None)
        if trace is not None:
            st.json(trace)


def render_health(system) -> None:
    _section_title("Runtime health", "Health")
    s = system.settings
    ok, message, models = ollama_health(str(getattr(s, "ollama_base_url", "http://localhost:11434")))
    configured = [str(getattr(s, "embedding_model", "")), str(getattr(s, "generation_model", ""))]
    missing = [m for m in configured if m and m not in models]
    checks = [
        ("Ollama", ok, message),
        ("Embedding", not bool(missing and configured[0] in missing), configured[0] or "Not configured"),
        ("Vector index", _index_ready(system), "Compatibility and visibility check"),
        ("Generation", not (len(configured) > 1 and configured[1] in missing), configured[1] or "Not configured"),
    ]
    if missing:
        st.warning(f"Configured models missing from Ollama: {', '.join(missing)}")
    table = '<div class="table-wrap"><table class="data-table"><thead><tr><th>Component</th><th>Status</th><th>Detail</th></tr></thead><tbody>'
    for name, good, detail in checks:
        table += f'<tr><td>{_esc(name)}</td><td>{_status("HEALTHY" if good else "UNAVAILABLE")}</td><td>{_esc(detail)}</td></tr>'
    st.markdown(table + '</tbody></table></div>', unsafe_allow_html=True)
    if st.button("Recheck index", use_container_width=True):
        st.rerun()
    with st.expander("Runtime report"):
        try: st.json(system.health_report())
        except Exception as exc: st.error(str(exc))


def _index_ready(system) -> bool:
    try:
        report = system.verify_index()
        if isinstance(report, dict):
            return bool(report.get("ready", report.get("compatible", report.get("ok", False))))
        return bool(report)
    except Exception:
        return False


def render_ingestion(system) -> None:
    _section_title("Ingestion control center", "Ingestion")
    running_id, running = active_job()
    if running:
        elapsed = _format_duration(running.get("started"))
        st.progress(0.25, text=f"Active progress · {elapsed}s · {running.get('source_dir')}")
    else:
        st.info("No active ingestion worker.")
    if st.button("Start ingestion", type="primary"):
        try:
            st.session_state["studio_last_job"] = start_ingestion(system, str(system.settings.incoming_dir))
            st.rerun()
        except ValueError as exc: st.error(str(exc))
    rows = docs(system)
    table = '<div class="table-wrap"><table class="data-table"><thead><tr><th>Status</th><th>File</th><th>Stage</th><th>Page</th><th>Chunks</th><th>Embeddings</th><th>Dimension</th><th>Error</th></tr></thead><tbody>'
    for d in rows:
        table += f'<tr><td>{_status(d.get("status"))}</td><td>{_esc(d.get("filename", d.get("file", d.get("document_id"))))}</td><td>{_esc(d.get("stage", d.get("current_stage")))}</td><td>{_esc(d.get("page", d.get("current_page")))}</td><td>{_esc(d.get("chunks", d.get("vector_chunks")))}</td><td>{_esc(d.get("embeddings", d.get("embedded_chunks")))}</td><td>{_esc(d.get("embedding_dimension", d.get("dimension")))}</td><td>{_esc(d.get("error"))}</td></tr>'
    st.markdown(table + '</tbody></table></div>', unsafe_allow_html=True)
    _section_title("Latest ingestion status", "Operations")
    registry = get_jobs()
    with registry["lock"]:
        jobs = list(registry["items"].items())[-10:][::-1]
    if not jobs:
        st.caption("No ingestion jobs recorded in this runtime.")
    for job_id, job in jobs:
        st.markdown(f'<div class="event"><span class="event-time">{time.strftime("%H:%M:%S", time.localtime(job.get("started", time.time())))}</span><span class="event-copy"><b>{_esc(job_id)}</b><br><span class="event-meta">{_esc(job.get("source_dir"))}</span></span>{_status(job.get("status"))}</div>', unsafe_allow_html=True)


def render_background(system) -> None:
    _section_title("Background workers", "Background")
    registry = get_jobs()
    with registry["lock"]:
        jobs = list(registry["items"].items())[::-1]
    if not jobs:
        st.markdown('<div class="empty-state">No background jobs yet.</div>', unsafe_allow_html=True)
        return
    for job_id, job in jobs:
        duration = _format_duration(job.get("started"), job.get("finished"))
        st.markdown(f'<div class="panel"><div class="panel-head"><b>{_esc(job_id)}</b>{_status(job.get("status"))}</div><div class="kv"><div class="kv-item"><div class="kv-key">Source</div><div class="kv-value">{_esc(job.get("source_dir"))}</div></div><div class="kv-item"><div class="kv-key">Duration</div><div class="kv-value">{duration}s</div></div></div>{("<pre>" + html.escape(str(job.get("error"))) + "</pre>") if job.get("error") else ""}</div>', unsafe_allow_html=True)
    if st.button("Refresh background status", use_container_width=True): st.rerun()


def _selected_doc(system) -> dict[str, Any] | None:
    rows = docs(system)
    if not rows: return None
    labels = [str(d.get("filename", d.get("document_id", f"Document {i+1}"))) for i, d in enumerate(rows)]
    selected = st.selectbox("Document", labels, key="inspector_document")
    return rows[labels.index(selected)]


def render_inspector(system) -> None:
    _section_title("Document and index inspector", "Inspector")
    doc = _selected_doc(system)
    if not doc:
        st.markdown('<div class="empty-state">No document metadata is available.</div>', unsafe_allow_html=True)
        return
    status = str(doc.get("status", "UNKNOWN"))
    cards = [
        ("Status", status), ("Pages", doc.get("pages", doc.get("page_count"))),
        ("Dimension", doc.get("embedding_dimension", doc.get("dimension"))), ("Version", doc.get("version", doc.get("ingestion_version"))),
        ("Chunks", doc.get("chunks", doc.get("vector_chunks"))), ("Document ID", doc.get("document_id")),
    ]
    st.markdown('<div class="kv">' + ''.join(f'<div class="kv-item"><div class="kv-key">{_esc(k)}</div><div class="kv-value">{_esc(v)}</div></div>' for k,v in cards) + '</div>', unsafe_allow_html=True)
    st.write("")
    if st.button("Verify index", type="primary"):
        try: st.json(system.verify_index())
        except Exception as exc: st.error(str(exc))
    with st.expander("Page checkpoints", expanded=True):
        checkpoints = doc.get("page_checkpoints", doc.get("checkpoints", []))
        if isinstance(checkpoints, list):
            for checkpoint in checkpoints: st.json(checkpoint)
        else: st.json(checkpoints)
    with st.expander("Raw metadata"):
        st.json(doc)


def render_settings(system) -> None:
    _section_title("Runtime configuration", "Settings")
    current = _settings_snapshot(system)
    st.caption("Live settings are applied to the Streamlit runtime and persisted through the existing Settings configuration model when supported.")
    with st.form("studio_settings_form"):
        host = st.text_input("Ollama host", value=str(current.get("ollama_base_url") or "http://localhost:11434"))
        emb = st.text_input("Embedding model", value=str(current.get("embedding_model") or "nomic-embed-text"))
        gen = st.text_input("Generation model", value=str(current.get("generation_model") or "llama3.2:3b"))
        a,b,c = st.columns(3)
        with a: chunk_size = st.number_input("Chunk size", min_value=64, max_value=4000, value=_safe_int(current.get("chunk_size"),600), step=32)
        with b: overlap = st.number_input("Chunk overlap", min_value=0, max_value=2000, value=_safe_int(current.get("chunk_overlap"),100), step=16)
        with c: top_k = st.number_input("Top-k", min_value=1, max_value=100, value=_safe_int(current.get("top_k"),8), step=1)
        a,b,c = st.columns(3)
        with a: temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=float(current.get("temperature") or .2), step=.05)
        with b: vector_weight = st.number_input("Vector weight", min_value=0.0, max_value=1.0, value=float(current.get("vector_weight") or .65), step=.05)
        with c: neighbor = st.number_input("Neighbor expansion", min_value=0, max_value=10, value=_safe_int(current.get("neighbor_expansion"),1), step=1)
        submitted = st.form_submit_button("Apply live settings", type="primary")
    if submitted:
        if overlap >= chunk_size:
            st.error("Chunk overlap must be smaller than chunk size.")
            return
        values = {"ollama_base_url": host.strip(), "embedding_model": emb.strip(), "generation_model": gen.strip(), "chunk_size": int(chunk_size), "chunk_overlap": int(overlap), "top_k": int(top_k), "temperature": float(temperature), "vector_weight": float(vector_weight), "neighbor_expansion": int(neighbor)}
        applied = 0
        for key, value in values.items():
            if hasattr(system.settings, key):
                try: setattr(system.settings, key, value); applied += 1
                except Exception: pass
        st.success(f"Applied {applied}/{len(values)} settings. Recreate runtime if a component caches the old configuration.")
        st.session_state["studio_settings_applied"] = values
    st.markdown('<div class="glass-note"><b>apply settings:</b> changes are intentionally constrained to known Settings fields; unknown fields are never injected into the production service.</div>', unsafe_allow_html=True)


def render_app() -> None:
    inject_css()
    system = get_system()
    render_sidebar(system)
    page = st.session_state.get("studio_nav", "Overview")
    if page not in NAV_ITEMS: page = "Overview"
    render_topbar(system, page)
    if page == "Overview": render_overview(system)
    elif page == "Chat": render_chat(system)
    elif page == "Health": render_health(system)
    elif page == "Ingestion": render_ingestion(system)
    elif page == "Background": render_background(system)
    elif page == "Inspector": render_inspector(system)
    elif page == "Settings": render_settings(system)


if __name__ == "__main__":
    render_app()
