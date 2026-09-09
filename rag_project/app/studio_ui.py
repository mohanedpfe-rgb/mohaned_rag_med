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


st.set_page_config(
    page_title="BookRAG Studio",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAV_ITEMS = ["Overview", "Ingestion", "Inspector", "Settings", "Chat", "Health", "Background"]
ACTIVE_STAGES = {
    "RUNNING", "DISCOVERED", "VALIDATING", "EXTRACTING", "OCR", "CHUNKING",
    "EMBEDDING", "INDEXING", "VALIDATING_INDEX", "BUILDING",
}


@st.cache_resource(show_spinner=False)
def get_system():
    return create_rag_system(Settings.from_env())


@st.cache_resource(show_spinner=False)
def get_jobs() -> dict[str, Any]:
    return {"lock": threading.RLock(), "items": {}}


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"))


def docs(system) -> list[dict[str, Any]]:
    try:
        return list(system.state_store.get_all_documents() or [])
    except Exception:
        return []


def ready_docs(system) -> list[dict[str, Any]]:
    return [d for d in docs(system) if str(d.get("status", "")).upper() == "READY"]


def active_document_count(system) -> int:
    return sum(str(d.get("status", "")).upper() in ACTIVE_STAGES for d in docs(system))


def _job_items() -> list[dict[str, Any]]:
    registry = get_jobs()
    with registry["lock"]:
        return list(registry["items"].values())


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
    job = {
        "id": job_id, "status": "RUNNING", "started": time.time(),
        "finished": None, "result": None, "error": None, "source_dir": str(folder),
    }
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
    if value in {"FAILED", "FAIL", "ERROR", "INTERRUPTED", "UNAVAILABLE", "ABSTAIN"}:
        return "bad"
    if value in {"WARN", "WARNING", "RUNNING", "PROCESSING", "BUILDING"}:
        return "warn"
    return "neutral"


def _status(value: Any) -> str:
    text = str(value or "UNKNOWN")
    return f'<span class="status {_status_class(text)}">{html.escape(text)}</span>'


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
    st.markdown(
        f'<div class="metric-card"><div class="metric-icon">{icon}</div>'
        f'<div class="metric-label">{_esc(label)}</div><div class="metric-value">{_esc(value)}</div>'
        f'<div class="metric-helper">{_esc(helper)}</div></div>',
        unsafe_allow_html=True,
    )


def _panel(title: str, eyebrow: str = "", right: str = "") -> None:
    e = f'<div class="eyebrow">{_esc(eyebrow)}</div>' if eyebrow else ""
    r = f'<div class="panel-right">{right}</div>' if right else ""
    st.markdown(f'<div class="panel-head">{e}<div class="panel-title">{_esc(title)}</div>{r}</div>', unsafe_allow_html=True)


def _kv(items: list[tuple[str, Any]]) -> None:
    cells = "".join(
        f'<div class="kv-item"><div class="kv-key">{_esc(k)}</div><div class="kv-value">{_esc(v)}</div></div>'
        for k, v in items
    )
    st.markdown(f'<div class="kv">{cells}</div>', unsafe_allow_html=True)


def inject_css() -> None:
    st.markdown(
        """
<style>
:root{--bg:#080d18;--shell:#0d1422;--surface:#111a2a;--surface2:#151f31;--stroke:#273449;--text:#f7f9fc;--muted:#8794a8;--accent:#7c74ff;--cyan:#42d6d0;--green:#39d98a;--amber:#f4bd55;--red:#ff687b}
html,body,[class*="css"]{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.stApp{background:radial-gradient(900px 500px at 76% -8%,rgba(124,116,255,.16),transparent 58%),radial-gradient(650px 400px at 8% 10%,rgba(66,214,208,.08),transparent 62%),var(--bg);color:var(--text)}
[data-testid="stHeader"]{height:0;background:transparent}.block-container{max-width:1738px!important;padding:23px 46px 56px!important}
[data-testid="stSidebar"]{width:252px!important;background:rgba(13,20,34,.98)!important;border-right:1px solid var(--stroke)!important}
[data-testid="stSidebar"]>div:first-child{padding:23px 23px!important}
[data-testid="stSidebar"] .stButton>button{height:45px!important;min-height:45px!important;border:1px solid transparent!important;background:transparent!important;border-radius:11px!important;text-align:left!important;padding:0 13px!important;color:#98a5b8!important;font-size:14px!important;font-weight:650!important}
[data-testid="stSidebar"] .stButton>button:hover{background:#151f31!important;color:#fff!important;border-color:#29364b!important}
[data-testid="stSidebar"] .stButton>button:focus{box-shadow:none!important}
.brand{display:flex;align-items:center;gap:12px;height:45px;margin-bottom:22px}.brand-mark{width:44px;height:44px;border-radius:13px;display:grid;place-items:center;background:linear-gradient(135deg,#8177ff,#45d8d0);box-shadow:0 12px 30px rgba(124,116,255,.22);font-size:21px}.brand-name{font-weight:850;font-size:16px;letter-spacing:-.02em}.brand-sub{font-size:10px;color:#718097;margin-top:2px}
.nav-label{font-size:10px;text-transform:uppercase;letter-spacing:.13em;color:#58677e;font-weight:850;margin:17px 8px 7px}.nav-active{background:rgba(124,116,255,.12)!important;border-color:rgba(124,116,255,.35)!important;color:#fff!important;box-shadow:inset 3px 0 0 var(--accent)}
.sidebar-divider{height:1px;background:var(--stroke);margin:18px 0}.sidebar-caption{font-size:11px;color:#697890;line-height:1.5}.sidebar-action{margin-top:9px}
.topbar{height:45px;display:flex;align-items:center;justify-content:space-between;margin-bottom:30px}.topbar-title{font-size:20px;font-weight:820;letter-spacing:-.035em}.topbar-sub{font-size:11px;color:var(--muted);margin-top:3px}.top-actions{display:flex;gap:8px;align-items:center}.top-chip{height:34px;padding:0 12px;border:1px solid var(--stroke);border-radius:10px;background:rgba(17,26,42,.72);display:flex;align-items:center;color:#b8c2d1;font-size:11px;font-weight:750}.live-dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 10px rgba(57,217,138,.7);margin-right:7px}
.hero{position:relative;min-height:202px;border:1px solid #293750;border-radius:17px;background:linear-gradient(112deg,#111a2a 0%,#17213a 56%,#282654 100%);overflow:hidden;padding:23px 28px;box-shadow:0 22px 55px rgba(0,0,0,.22);margin-bottom:12px}.hero:after{content:"";position:absolute;width:260px;height:260px;right:-75px;top:-110px;border-radius:50%;background:radial-gradient(circle,rgba(124,116,255,.26),transparent 68%);pointer-events:none}.hero-kicker{font-size:10px;color:#9da8ff;text-transform:uppercase;letter-spacing:.14em;font-weight:850}.hero-title{font-size:28px;line-height:1.1;font-weight:850;letter-spacing:-.045em;margin-top:8px}.hero-copy{font-size:12px;color:#9ca9bd;max-width:510px;line-height:1.55;margin-top:8px}.hero-upload{margin-top:17px}
.metric-row{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.metric-card{position:relative;min-height:112px;border:1px solid var(--stroke);border-radius:14px;background:linear-gradient(180deg,rgba(21,31,49,.9),rgba(14,22,36,.92));padding:17px;box-shadow:0 13px 30px rgba(0,0,0,.14)}.metric-icon{width:30px;height:30px;border-radius:9px;background:rgba(124,116,255,.11);display:grid;place-items:center;color:#a59fff;font-size:13px;float:left;margin-right:9px}.metric-label{font-size:11px;color:#7f8da3;font-weight:700;padding-top:7px}.metric-value{clear:both;font-size:29px;font-weight:850;letter-spacing:-.05em;padding-top:10px}.metric-helper{font-size:10px;color:#5f6e84;margin-top:1px}
.content-grid{display:grid;grid-template-columns:minmax(0,1.36fr) minmax(330px,.84fr);gap:12px;margin-top:12px}.panel{border:1px solid var(--stroke);border-radius:16px;background:rgba(17,26,42,.82);box-shadow:0 12px 32px rgba(0,0,0,.11);padding:18px;margin-bottom:12px}.panel-head{position:relative;margin-bottom:13px}.eyebrow{font-size:9px;text-transform:uppercase;letter-spacing:.14em;color:#617087;font-weight:850;margin-bottom:3px}.panel-title{font-size:15px;font-weight:820;letter-spacing:-.025em}.panel-right{position:absolute;right:0;top:4px;font-size:10px;color:#7f8da3}.kv{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}.kv-item{border:1px solid #26344a;border-radius:11px;padding:10px;background:rgba(8,13,24,.28)}.kv-key{font-size:9px;color:#617087;text-transform:uppercase;letter-spacing:.08em}.kv-value{font-size:11px;font-weight:720;margin-top:3px;overflow-wrap:anywhere}
.status{display:inline-flex;align-items:center;border-radius:999px;padding:3px 7px;border:1px solid currentColor;font-size:9px;font-weight:850;line-height:1}.status.good{color:#63e5a3;background:rgba(57,217,138,.07)}.status.warn{color:#f5c867;background:rgba(244,189,85,.07)}.status.bad{color:#ff8492;background:rgba(255,104,123,.07)}.status.neutral{color:#a1aec0;background:rgba(135,148,168,.07)}
.data-wrap{overflow:auto;border:1px solid #26344a;border-radius:11px}.data-table{width:100%;border-collapse:collapse;font-size:10px}.data-table th{color:#65748b;font-size:9px;text-transform:uppercase;letter-spacing:.06em;font-weight:800;background:#101927}.data-table th,.data-table td{padding:9px 10px;border-bottom:1px solid #202d40;text-align:left;white-space:nowrap}.data-table td{color:#cbd4e1}.data-table tr:last-child td{border-bottom:0}
.answer{border:1px solid rgba(124,116,255,.3);background:rgba(124,116,255,.055);border-radius:13px;padding:14px}.answer-label{font-size:9px;color:#a59fff;text-transform:uppercase;letter-spacing:.12em;font-weight:850}.answer-text{font-size:13px;line-height:1.65;margin-top:7px}.evidence{border:1px solid #26344a;border-radius:10px;background:rgba(5,10,19,.25);padding:10px;margin-top:7px}.evidence-meta{font-size:9px;color:#74839a;margin-bottom:4px}
.event{display:grid;grid-template-columns:70px 1fr auto;gap:9px;align-items:center;padding:9px 0;border-bottom:1px solid #202d40}.event:last-child{border-bottom:0}.event-time{font-size:9px;color:#5f6e84}.event-copy{font-size:10px}.event-meta{font-size:9px;color:#758399}.glass-note{border:1px dashed #35445b;border-radius:11px;padding:11px;color:#8795a9;font-size:10px;line-height:1.5}
.stButton>button{border:1px solid #2b3950!important;border-radius:10px!important;background:#141f30!important;color:#e8edf5!important;font-size:11px!important;font-weight:750!important;min-height:36px!important}.stButton>button:hover{border-color:#7168e9!important;background:#1a2540!important}.stTextInput>div>div,.stTextArea>div>div,.stSelectbox>div>div,.stNumberInput>div>div,.stSlider>div{border-color:#2b3950!important}.stTextInput input,.stTextArea textarea,.stSelectbox div[data-baseweb="select"]{background:#0e1725!important;color:#f4f7fb!important;border-color:#2b3950!important}.stFileUploader{background:#0e1725!important;border:1px dashed #3b4a62!important;border-radius:11px!important}.stCheckbox label,.stRadio label,.stSelectbox label,.stTextInput label,.stTextArea label,.stNumberInput label{color:#8795a9!important;font-size:10px!important}.stCaption{color:#65748b!important}.stProgress>div>div{background:#7c74ff!important}
[data-testid="stMetric"]{background:transparent!important}.section-spacer{height:1px}.mini-title{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:#65748b;font-weight:850;margin:12px 0 7px}.danger-zone{border:1px solid rgba(255,104,123,.22);background:rgba(255,104,123,.035);border-radius:12px;padding:12px}
@media(max-width:1200px){.content-grid{grid-template-columns:1fr}.metric-row{grid-template-columns:repeat(3,1fr)}}@media(max-width:760px){[data-testid="stSidebar"]{width:0!important}.block-container{padding:16px!important}.metric-row{grid-template-columns:1fr}.topbar{margin-bottom:18px}.content-grid{display:block}.kv{grid-template-columns:1fr}}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(system) -> None:
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-mark">📚</div><div><div class="brand-name">BookRAG Studio</div><div class="brand-sub">Local document intelligence</div></div></div>',
            unsafe_allow_html=True,
        )
        current = st.session_state.get("studio_nav", "Overview")
        st.markdown('<div class="nav-label">Workspace</div>', unsafe_allow_html=True)
        for item in NAV_ITEMS:
            label = ("●  " if item == current else "○  ") + item
            if st.button(label, key=f"nav_{item}", use_container_width=True):
                _navigate(item)

        st.markdown('<div class="sidebar-divider"></div><div class="nav-label">Documents</div>', unsafe_allow_html=True)
        uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True, key="studio_uploads")
        if uploads:
            seen: set[str] = st.session_state.setdefault("saved_pdf_hashes", set())
            saved = 0
            errors: list[str] = []
            incoming = Path(system.settings.incoming_dir)
            for item in uploads:
                try:
                    payload = item.getvalue()
                    digest = hashlib.sha256(payload).hexdigest()
                    if digest in seen:
                        continue
                    save_pdf(incoming, item.name, payload)
                    seen.add(digest)
                    saved += 1
                except Exception as exc:
                    errors.append(str(exc))
            if saved:
                st.success(f"Added {saved} PDF{'s' if saved != 1 else ''} to Incoming.")
            for error in errors:
                st.error(error)

        incoming = str(getattr(system.settings, "incoming_dir", "data/incoming"))
        st.text_input("Incoming folder", value=incoming, key="studio_incoming")
        if st.button("Start all chunks", use_container_width=True, key="studio_start"):
            try:
                job_id = start_ingestion(system, st.session_state["studio_incoming"])
                st.session_state["studio_last_job"] = job_id
                st.session_state["studio_nav"] = "Ingestion"
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

        if st.button("Recreate runtime", use_container_width=True, key="studio_recreate"):
            get_system.clear()
            st.session_state["studio_nav"] = "Overview"
            st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.checkbox("I understand cleanup is permanent", key="studio_clear_confirm")
        if st.button("Clear all PDF data", use_container_width=True, key="studio_clear", disabled=not st.session_state.get("studio_clear_confirm", False)):
            try:
                result = system.clear_pdf_data()
                st.session_state["studio_clear_confirm"] = False
                st.success(f"PDF data cleared: {result}")
                st.rerun()
            except Exception as exc:
                st.error(f"Cleanup failed: {exc}")

        st.markdown('<div class="sidebar-divider"></div><div class="sidebar-caption">Local-first RAG workspace. Documents, embeddings and generation stay on your configured machine.</div>', unsafe_allow_html=True)


def render_topbar(system) -> None:
    st.markdown(
        '<div class="topbar"><div><div class="topbar-title">BookRAG Studio</div><div class="topbar-sub">Citation-first document intelligence workspace</div></div><div class="top-actions"><div class="top-chip"><span class="live-dot"></span>Local runtime</div><div class="top-chip">i5 · 16 GB</div></div></div>',
        unsafe_allow_html=True,
    )


def render_overview(system) -> None:
    all_docs = docs(system)
    ready = ready_docs(system)
    active = active_document_count(system)
    vector_chunks = sum(_safe_int(d.get("vector_chunks", d.get("chunks", 0))) for d in all_docs)
    lexical_chunks = sum(_safe_int(d.get("lexical_chunks", 0)) for d in all_docs)

    st.markdown(
        '<div class="hero"><div class="hero-kicker">Document intelligence</div><div class="hero-title">Choose PDF files</div><div class="hero-copy">Upload PDFs, index them, ask questions, and inspect grounded evidence from one local workspace.</div></div>',
        unsafe_allow_html=True,
    )
    metric_html = '<div class="metric-row">'
    for icon, label, value, helper in [
        ("▣", "Documents", len(all_docs), "Indexed documents"),
        ("✓", "Ready", len(ready), "Available for search"),
        ("◌", "Processing", active, "Active ingestion"),
    ]:
        metric_html += f'<div class="metric-card"><div class="metric-icon">{icon}</div><div class="metric-label">{_esc(label)}</div><div class="metric-value">{_esc(value)}</div><div class="metric-helper">{_esc(helper)}</div></div>'
    metric_html += '</div>'
    st.markdown(metric_html, unsafe_allow_html=True)
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.36, .84], gap="small")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        _panel("Index overview", "Current workspace", "Vector chunks · Start all chunks")
        _kv([("Total", len(all_docs)), ("Ready", len(ready)), ("Processing", active), ("Vector chunks", vector_chunks), ("Lexical chunks", lexical_chunks), ("Runtime", "Ready")])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        _panel("Ask Your Documents", "Grounded search")
        question = st.text_area("Question", placeholder="Ask something about your indexed PDFs…", height=82, key="overview_question", label_visibility="collapsed")
        if st.button("Search and answer", key="overview_ask", use_container_width=True):
            if not question.strip():
                st.warning("Please enter a question.")
            elif not ready:
                st.warning("No READY documents are available yet.")
            else:
                try:
                    result = system.answer(question.strip())
                    st.session_state["overview_answer"] = result
                except Exception as exc:
                    st.error(f"Answer failed safely: {exc}")
        result = st.session_state.get("overview_answer")
        if result:
            answer = result.get("answer", result.get("text", result)) if isinstance(result, dict) else result
            st.markdown(f'<div class="answer"><div class="answer-label">Answer</div><div class="answer-text">{_esc(answer)}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        _panel("System Health", "Runtime")
        try:
            health = system.health_report()
        except Exception as exc:
            health = {"error": str(exc)}
        rows = []
        for key, label in [("ollama", "Ollama"), ("embedding", "Embedding"), ("vector_index", "Vector index"), ("generation", "Generation")]:
            value = health.get(key, {}) if isinstance(health, dict) else {}
            if isinstance(value, dict):
                status = value.get("status", value.get("state", "N/A"))
            else:
                status = value
            rows.append((label, status))
        for label, status in rows:
            st.markdown(f'<div class="event"><div class="event-copy">{_esc(label)}</div><div></div><div>{_status(status)}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        _panel("Inspector", "Document state", "Detail")
        selected = ready[0] if ready else (all_docs[0] if all_docs else {})
        _kv([
            ("Status", selected.get("status", "N/A")),
            ("Pages", selected.get("pages", "N/A")),
            ("Dimension", selected.get("dimension", "N/A")),
            ("Version", selected.get("version", "N/A")),
        ])
        st.markdown('</div>', unsafe_allow_html=True)


def render_chat(system) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    _panel("Ask Your Documents", "Grounded search")
    available = ready_docs(system)
    names = ["All ready documents"] + [str(d.get("filename", d.get("name", d.get("document_id", "Document")))) for d in available]
    scope = st.selectbox("Search scope", names, key="chat_scope")
    selected_filter = None
    if scope != names[0] and available:
        selected = available[names.index(scope) - 1]
        selected_filter = selected.get("document_id")
    nonce = st.session_state.get("studio_chat_nonce", 0)
    question = st.text_area("Question", placeholder="Ask a question and inspect the evidence behind the answer…", height=105, key=f"studio_question_{nonce}")
    if st.button("Search and answer", key="chat_search", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
        elif not available:
            st.warning("No READY documents are available yet.")
        else:
            try:
                metadata_filter = {"document_id": selected_filter} if selected_filter else None
                result = system.answer(question.strip(), metadata_filter=metadata_filter)
                st.session_state["console_answer"] = result
                st.session_state["console_question"] = question.strip()
            except Exception as exc:
                st.error(f"Answer failed safely: {exc}")
    if st.button("Clear chat", key="chat_clear"):
        st.session_state.pop("console_answer", None)
        st.session_state.pop("console_question", None)
        st.session_state["studio_chat_nonce"] = nonce + 1
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    result = st.session_state.get("console_answer")
    if result:
        data = result if isinstance(result, dict) else {"answer": result}
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        _panel("Answer", "Grounded response")
        st.markdown(f'<div class="answer"><div class="answer-text">{_esc(data.get("answer", data.get("text", "")))}</div></div>', unsafe_allow_html=True)
        _kv([
            ("Confidence", data.get("confidence", "N/A")),
            ("Answerability", data.get("answerability", "N/A")),
            ("Query quality", data.get("query_quality", "N/A")),
            ("Status", data.get("status", "N/A")),
        ])
        st.markdown('<div class="mini-title">Citations</div>', unsafe_allow_html=True)
        citations = data.get("citations", []) or []
        if citations:
            for citation in citations:
                st.markdown(f'<div class="evidence">{_esc(citation)}</div>', unsafe_allow_html=True)
        else:
            st.caption("No citations returned.")
        with st.expander("Evidence and trace"):
            st.json({"evidence": data.get("evidence", []), "query_trace": data.get("query_trace", {})})
        st.markdown('</div>', unsafe_allow_html=True)


def render_health(system) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    _panel("System Health", "Operational checks")
    ok, message, models = ollama_health(system.settings.ollama_base_url)
    configured = [system.settings.embedding_model, system.settings.generation_model]
    missing = [m for m in configured if not any(x == m or x.startswith(m + ":") for x in models)] if ok else configured
    checks = [("Ollama", "PASS" if ok else "FAIL"), ("Embedding", "PASS" if not missing or not ok else "WARN"), ("Vector index", "READY"), ("Generation", "PASS" if not missing or not ok else "WARN")]
    table = '<div class="data-wrap"><table class="data-table"><thead><tr><th>Service</th><th>Status</th><th>Detail</th></tr></thead><tbody>'
    for label, status in checks:
        detail = message if label == "Ollama" else ("Configured model available" if status == "PASS" else "Model availability needs attention")
        table += f'<tr><td>{_esc(label)}</td><td>{_status(status)}</td><td>{_esc(detail)}</td></tr>'
    table += '</tbody></table></div>'
    st.markdown(table, unsafe_allow_html=True)
    if missing:
        st.warning(f"Configured models missing from Ollama: {', '.join(missing)}")
    if st.button("Recheck index", key="health_recheck"):
        try:
            st.write(system.verify_index())
        except Exception as exc:
            st.error(str(exc))
    st.markdown('</div>', unsafe_allow_html=True)


def render_ingestion(system) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    _panel("Ingestion", "Pipeline monitor", "Active progress")
    job_id, job = active_job()
    if job:
        st.info(f"Active progress · {Path(job['source_dir']).name} · {_format_duration(job['started'])}s")
        st.progress(0.35, text="Processing PDF pipeline…")
    else:
        st.caption("No ingestion worker is currently running.")
    rows = []
    for d in docs(system):
        rows.append([
            d.get("status", "N/A"), d.get("filename", d.get("name", "N/A")),
            d.get("stage", d.get("status", "N/A")), d.get("page", d.get("pages", "N/A")),
            d.get("chunks", d.get("vector_chunks", "N/A")), d.get("embeddings", "N/A"),
            d.get("dimension", "N/A"), d.get("error", ""),
        ])
    headers = ["Status", "File", "Stage", "Page", "Chunks", "Embeddings", "Dimension", "Error"]
    table = '<div class="data-wrap"><table class="data-table"><thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr></thead><tbody>'
    for row in rows:
        table += '<tr>' + ''.join(f'<td>{_status(v) if i == 0 else _esc(v)}</td>' for i, v in enumerate(row)) + '</tr>'
    table += '</tbody></table></div>'
    st.markdown(table, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_background(system) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    _panel("Background workers", "Job history")
    jobs = _job_items()
    if not jobs:
        st.markdown('<div class="glass-note">No background ingestion jobs have been created yet.</div>', unsafe_allow_html=True)
    for job in reversed(jobs):
        st.markdown(
            f'<div class="event"><div class="event-time">{_esc(time.strftime("%H:%M:%S", time.localtime(job.get("started", time.time()))))}</div>'
            f'<div class="event-copy">{_esc(Path(job.get("source_dir", "")).name)}<div class="event-meta">{_esc(_format_duration(job.get("started"), job.get("finished")))}s · {_esc(job.get("error") or "ingestion")}</div></div>'
            f'<div>{_status(job.get("status"))}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_inspector(system) -> None:
    all_docs = docs(system)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    _panel("Document and index inspector", "Detailed state")
    if not all_docs:
        st.markdown('<div class="glass-note">No documents are indexed yet.</div>', unsafe_allow_html=True)
    else:
        labels = [str(d.get("filename", d.get("name", d.get("document_id", "Document")))) for d in all_docs]
        selected = all_docs[st.selectbox("Document", range(len(all_docs)), format_func=lambda i: labels[i], key="inspector_doc")]
        _kv([
            ("Status", selected.get("status", "N/A")), ("Pages", selected.get("pages", "N/A")),
            ("Dimension", selected.get("dimension", "N/A")), ("Version", selected.get("version", "N/A")),
            ("Chunks", selected.get("chunks", selected.get("vector_chunks", "N/A"))),
            ("Document ID", selected.get("document_id", "N/A")),
        ])
        if st.button("Verify index", key="inspector_verify"):
            try:
                st.json(system.verify_index())
            except Exception as exc:
                st.error(str(exc))
        st.markdown('<div class="mini-title">Page checkpoints</div>', unsafe_allow_html=True)
        checkpoints = selected.get("page_checkpoints", selected.get("checkpoints", []))
        st.json(checkpoints)
        with st.expander("Raw metadata"):
            st.json(selected)
    st.markdown('</div>', unsafe_allow_html=True)


def render_settings(system) -> None:
    s = system.settings
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    _panel("Settings", "Runtime configuration")
    with st.form("studio_settings_form"):
        ollama = st.text_input("Ollama host", value=str(s.ollama_base_url))
        embedding = st.text_input("Embedding model", value=str(s.embedding_model))
        generation = st.text_input("Generation model", value=str(s.generation_model))
        c1, c2 = st.columns(2)
        with c1:
            chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=int(s.chunk_size), step=50)
            top_k = st.number_input("Top-k", min_value=1, max_value=50, value=int(s.top_k), step=1)
            vector_weight = st.slider("Vector weight", 0.0, 1.0, float(s.vector_weight), 0.05)
        with c2:
            overlap = st.number_input("Chunk overlap", min_value=0, max_value=3999, value=int(s.chunk_overlap), step=10)
            temperature = st.slider("Temperature", 0.0, 1.0, float(s.temperature), 0.05)
            neighbor = st.checkbox("Neighbor expansion", value=bool(s.neighbor_expansion))
        apply = st.form_submit_button("Apply live settings", use_container_width=True)
    if apply:
        if overlap >= chunk_size:
            st.error("Chunk overlap must be smaller than chunk size.")
        else:
            overrides = {
                "ollama_base_url": ollama, "embedding_model": embedding, "generation_model": generation,
                "chunk_size": int(chunk_size), "chunk_overlap": int(overlap), "top_k": int(top_k),
                "temperature": float(temperature), "vector_weight": float(vector_weight),
                "neighbor_expansion": bool(neighbor),
            }
            try:
                if hasattr(system, "apply_settings_in_place"):
                    result = system.apply_settings_in_place(overrides)
                else:
                    result = system.settings.override_from_dict(overrides)
                st.success(f"Settings applied: {result}")
            except Exception as exc:
                st.error(f"Settings were not applied: {exc}")
    st.markdown('</div>', unsafe_allow_html=True)


def render_app() -> None:
    if "studio_nav" not in st.session_state:
        st.session_state["studio_nav"] = "Overview"
    if "studio_chat_nonce" not in st.session_state:
        st.session_state["studio_chat_nonce"] = 0
    system = get_system()
    inject_css()
    render_sidebar(system)
    render_topbar(system)
    page = st.session_state.get("studio_nav", "Overview")
    if page == "Overview":
        render_overview(system)
    elif page == "Chat":
        render_chat(system)
    elif page == "Health":
        render_health(system)
    elif page == "Ingestion":
        render_ingestion(system)
    elif page == "Background":
        render_background(system)
    elif page == "Inspector":
        render_inspector(system)
    elif page == "Settings":
        render_settings(system)


def main() -> None:
    render_app()


if __name__ == "__main__":
    main()
