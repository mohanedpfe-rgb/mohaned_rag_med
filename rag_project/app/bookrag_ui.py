from __future__ import annotations

import hashlib
import html
import io
import json
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
ACTIVE = {
    "RUNNING", "DISCOVERED", "VALIDATING", "EXTRACTING", "OCR", "CHUNKING",
    "EMBEDDING", "INDEXING", "VALIDATING_INDEX", "BUILDING", "INTERRUPTED", "RECOVERING",
}
STAGES = {
    "RUNNING": ("Starting", "The document has entered the pipeline."),
    "DISCOVERED": ("Discovered", "The PDF was accepted and queued."),
    "VALIDATING": ("Validating", "Checking the PDF structure and safety."),
    "EXTRACTING": ("Extracting", "Reading the document page by page."),
    "OCR": ("OCR", "Recovering text from scanned pages."),
    "CHUNKING": ("Chunking", "Turning extracted content into retrieval sections."),
    "EMBEDDING": ("Embedding", "Creating semantic search vectors."),
    "INDEXING": ("Indexing", "Writing semantic and lexical search records."),
    "VALIDATING_INDEX": ("Verifying", "Checking the new index before publication."),
    "READY": ("Ready", "Available for grounded research questions."),
    "COMPLETED": ("Ready", "Available for grounded research questions."),
    "FAILED": ("Failed", "Processing stopped and needs attention."),
    "FAILED_EMBEDDING": ("Embedding failed", "Semantic indexing could not complete."),
    "INTERRUPTED": ("Interrupted", "Processing stopped before completion."),
    "RECOVERING": ("Recovering", "Preparing a safe recovery attempt."),
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
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _elapsed(start: Any, end: Any | None = None) -> float:
    started = _utc(start)
    if not started:
        return 0.0
    ended = _utc(end) or datetime.now(timezone.utc)
    return max(0.0, (ended - started).total_seconds())


def _fmt_bytes(size: Any) -> str:
    n = max(0, _int(size))
    units = ["B", "KB", "MB", "GB"]
    i = 0
    value = float(n)
    while value >= 1024 and i < len(units) - 1:
        value /= 1024
        i += 1
    return f"{value:.1f} {units[i]}"


def _fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


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
    return [d for d in docs(system) if str(d.get("status", "")).upper() in ACTIVE]


def total_chunks(system) -> int:
    return sum(_int(d.get("chunk_count", d.get("chunks", 0))) for d in docs(system))


def total_embeddings(system) -> int:
    return sum(_int(d.get("embedding_count", d.get("embeddings", 0))) for d in docs(system))


def _doc_name(document: dict[str, Any]) -> str:
    return str(document.get("file_name") or document.get("filename") or document.get("name") or document.get("document_id") or "Document")


def _doc_date(document: dict[str, Any]) -> datetime | None:
    for key in ("modified_at", "ingestion_completed_at", "ingestion_started_at", "created_at"):
        value = _utc(document.get(key))
        if value:
            return value
    return None


def _stage(document: dict[str, Any]) -> tuple[str, str]:
    key = str(document.get("current_stage") or document.get("status") or "RUNNING").upper()
    return STAGES.get(key, (key.replace("_", " ").title(), "Processing document."))


def _status_class(value: Any) -> str:
    status = str(value or "UNKNOWN").upper()
    if status in {"READY", "COMPLETED", "PASS", "ONLINE", "HEALTHY", "OK", "GROUNDED"}:
        return "good"
    if status in {"FAILED", "FAILED_EMBEDDING", "ERROR", "OFFLINE", "UNAVAILABLE", "ABSTAIN", "INTERRUPTED"}:
        return "bad"
    if status in ACTIVE or status in {"RUNNING", "PROCESSING", "BUILDING", "WARNING", "WARN"}:
        return "warn"
    return "neutral"


def _status(value: Any) -> str:
    text = str(value or "UNKNOWN")
    return f'<span class="pill pill-{_status_class(text)}"><i></i>{_esc(text)}</span>'


def _progress(document: dict[str, Any]) -> float:
    status = str(document.get("status") or document.get("current_stage") or "RUNNING").upper()
    base = {
        "RUNNING": 0.03, "DISCOVERED": 0.08, "VALIDATING": 0.15, "EXTRACTING": 0.30,
        "OCR": 0.46, "CHUNKING": 0.58, "EMBEDDING": 0.73, "INDEXING": 0.86,
        "VALIDATING_INDEX": 0.96, "READY": 1.0, "COMPLETED": 1.0,
    }.get(status, 1.0 if status in {"FAILED", "FAILED_EMBEDDING", "INTERRUPTED"} else 0.03)
    total, current = _int(document.get("total_pages")), _int(document.get("current_page"))
    if status in {"EXTRACTING", "OCR"} and total:
        base += min(current / total, 1.0) * (0.14 if status == "EXTRACTING" else 0.10)
    return max(0.0, min(base, 1.0))


def _navigate(page: str) -> None:
    if page in PAGES:
        st.session_state["bookrag_page"] = page
        st.rerun()


def _save_pdf(incoming: Path, name: str, content: bytes) -> str:
    if not content or not content.startswith(b"%PDF-"):
        raise ValueError(f"{name} is not a valid PDF payload.")
    incoming = Path(incoming).expanduser().resolve()
    digest = hashlib.sha256(content).hexdigest()
    stem = "".join(c if c.isalnum() or c in ".-_" else "_" for c in (Path(name).stem or "document")).strip(" ._") or "document"
    incoming.mkdir(parents=True, exist_ok=True)
    target = incoming / f"{stem[:80]}_{digest[:12]}.pdf"
    if not target.exists():
        target.write_bytes(content)
    return digest


def save_pdf(incoming: Path, name: str, content: bytes) -> str:
    return _save_pdf(incoming, name, content)


def start_ingestion(system, source_dir: str, *, trigger: str = "manual") -> str:
    folder = Path(source_dir).expanduser().resolve()
    root = Path(system.settings.project_root).expanduser().resolve()
    try:
        folder.relative_to(root)
    except ValueError as exc:
        raise ValueError("The incoming folder must remain inside the BookRAG project.") from exc
    pdfs = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    if not pdfs:
        raise ValueError("No PDFs are waiting in the incoming folder.")
    jobs = get_jobs()
    with jobs["lock"]:
        for job in reversed(list(jobs["items"].values())):
            if job.get("status") == "RUNNING":
                return str(job["id"])
        jid = f"ingest-{time.time_ns()}"
        job = {
            "id": jid, "status": "RUNNING", "started": time.time(), "finished": None,
            "file_count": len(pdfs), "completed": 0, "failed": 0, "result": None,
            "error": None, "trigger": trigger,
        }
        jobs["items"][jid] = job

    def worker() -> None:
        try:
            result = system.ingest_directory(str(folder)) or []
            ok = sum(1 for item in result if item.get("status") in {"success", "skipped"})
            bad = sum(1 for item in result if item.get("status") == "failed")
            with jobs["lock"]:
                job.update(result=result, completed=ok, failed=bad, status="FAILED" if bad and not ok else "COMPLETED")
        except Exception as exc:
            with jobs["lock"]:
                job.update(error=str(exc), status="FAILED")
        finally:
            with jobs["lock"]:
                job["finished"] = time.time()

    threading.Thread(target=worker, name=f"{jid}-worker", daemon=True).start()
    return jid


def auto_ingest(system, count: int) -> str | None:
    if not count:
        return None
    try:
        job_id = start_ingestion(system, str(system.settings.incoming_dir), trigger="upload")
        st.session_state["last_ingest_job"] = job_id
        return job_id
    except Exception as exc:
        st.session_state["auto_ingest_error"] = str(exc)
        return None


def ollama_health(base_url: str):
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=(2.5, 5))
        response.raise_for_status()
        models = [str(item.get("name")) for item in response.json().get("models", []) if item.get("name")]
        return True, "Ollama is reachable", models
    except Exception as exc:
        return False, str(exc), []


def _job_running() -> bool:
    jobs = get_jobs()
    with jobs["lock"]:
        return any(job.get("status") == "RUNNING" for job in jobs["items"].values())


def _metric(result: dict[str, Any], *keys: str) -> Any:
    containers: list[Any] = [result, result.get("query_trace", {})]
    for key in keys:
        for container in containers:
            if isinstance(container, dict) and key in container and container[key] is not None:
                return container[key]
    return None


def _evidence(result: dict[str, Any]) -> list[Any]:
    value = result.get("evidence") or result.get("hits") or result.get("citations") or []
    return list(value) if isinstance(value, (list, tuple)) else []


def _evidence_row(item: Any, index: int) -> tuple[str, str, Any, str]:
    if isinstance(item, dict):
        title = item.get("file_name") or item.get("filename") or item.get("document_id") or f"Source {index}"
        page = item.get("page_number") or item.get("page") or item.get("page_numbers") or "—"
        score = item.get("rerank_score") or item.get("score") or item.get("similarity") or item.get("relevance") or "—"
        snippet = item.get("snippet") or item.get("text") or item.get("content") or ""
        return str(title), str(page), score, str(snippet)
    return f"Source {index}", "—", "—", str(item)


def _command_palette(system) -> None:
    with st.popover("⌘K  Commands", use_container_width=False):
        st.caption("Fast navigation · Enter a destination or action")
        query = st.text_input("Command", placeholder="Search pages…", key="command_palette")
        normalized = (query or "").strip().casefold()
        commands = {
            "home": "Home", "documents": "Documents", "docs": "Documents",
            "processing": "Live Processing", "live": "Live Processing",
            "ask": "Ask BookRAG", "chat": "Ask BookRAG", "inspector": "Inspector",
            "system": "System", "settings": "Settings",
        }
        matches = [(name.title(), page) for name, page in commands.items() if not normalized or normalized in name]
        if not matches:
            st.info("No command matches that search.")
        for label, page in matches[:8]:
            if st.button(f"{label}  →  {page}", key=f"cmd_{label}_{page}", use_container_width=True):
                _navigate(page)
        st.caption("Shortcut hint: Ctrl/Cmd + K · Use the visible command launcher on environments where browser shortcuts are restricted.")


def css() -> None:
    st.markdown(r'''<style>
:root{--bg:#080d16;--panel:#0e1623;--panel2:#111c2b;--line:#243246;--line2:#1b293b;--text:#edf3fa;--muted:#8493a7;--faint:#56667c;--accent:#63d9d1;--blue:#7e9cff;--green:#5fe0a0;--amber:#f2c76b;--red:#ff7285;--shadow:0 24px 70px rgba(0,0,0,.25)}
html,body,[class*="css"]{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.stApp{background:radial-gradient(900px 500px at 80% -10%,rgba(126,156,255,.12),transparent 60%),radial-gradient(700px 450px at 10% 0,rgba(99,217,209,.07),transparent 62%),var(--bg);color:var(--text)}[data-testid=stHeader]{height:0;background:transparent}.block-container{max-width:1520px;padding:26px 42px 80px}.stApp *{letter-spacing:-.005em}
[data-testid=stSidebar]{background:#09111d!important;border-right:1px solid var(--line)!important}[data-testid=stSidebar]>div:first-child{padding:22px 16px 30px!important}.brand{display:flex;align-items:center;gap:11px;padding:4px 7px 24px}.brandmark{width:38px;height:38px;border-radius:11px;background:linear-gradient(135deg,#76e0d9,#718dff);display:grid;place-items:center;color:#07121b;font-size:13px;font-weight:950;box-shadow:0 10px 30px rgba(99,217,209,.16)}.brandname{font-weight:900;font-size:15px}.brandsub{font-size:9px;color:var(--faint);margin-top:2px}.navgroup{margin:17px 7px 7px;font-size:9px;text-transform:uppercase;letter-spacing:.14em;color:#58697e;font-weight:850}
[data-testid=stSidebar] .stButton>button{min-height:40px!important;border:1px solid transparent!important;background:transparent!important;color:#8e9db0!important;border-radius:9px!important;text-align:left!important;font-size:12px!important;font-weight:700!important;padding:0 12px!important}[data-testid=stSidebar] .stButton>button:hover{background:#111d2c!important;border-color:#26364b!important;color:#edf3fa!important}.sidefoot{border-top:1px solid var(--line2);margin-top:18px;padding:14px 7px;color:#68788d;font-size:9px;line-height:1.6}
.top{display:flex;justify-content:space-between;align-items:center;gap:18px;margin-bottom:16px}.crumb{font-size:10px;color:#62738a;text-transform:uppercase;letter-spacing:.13em;font-weight:850}.title{font-size:22px;font-weight:900;letter-spacing:-.04em;margin-top:4px}.topright{display:flex;align-items:center;gap:8px}.chip{height:32px;border:1px solid var(--line);background:rgba(14,22,35,.8);border-radius:9px;padding:0 11px;display:flex;align-items:center;font-size:10px;color:#9aabbe}.dot{width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 11px rgba(95,224,160,.7);margin-right:7px}
.hero{border:1px solid #2a3a51;border-radius:18px;padding:28px;background:linear-gradient(115deg,#0e1725,#121f32 58%,#182a38);box-shadow:var(--shadow);position:relative;overflow:hidden}.hero:after{content:"";position:absolute;right:-110px;top:-150px;width:360px;height:360px;border-radius:50%;background:radial-gradient(circle,rgba(99,217,209,.14),transparent 65%)}.kicker{font-size:9px;text-transform:uppercase;letter-spacing:.16em;color:#76d7d1;font-weight:900}.hero h1{font-size:32px;line-height:1.08;margin:8px 0 7px;letter-spacing:-.045em}.hero p{margin:0;max-width:760px;color:#93a2b5;font-size:12px;line-height:1.65}.hero-actions{margin-top:18px}
.livebar{display:flex;align-items:center;gap:8px;margin-top:12px;padding:9px 12px;border:1px solid var(--line);border-radius:10px;background:rgba(14,22,35,.8);font-size:10px;color:#8fa0b4}.livebar .grow{flex:1}.livepulse{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 0 5px rgba(95,224,160,.07),0 0 12px rgba(95,224,160,.6)}
.grid4{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:11px;margin-top:12px}.stat{border:1px solid var(--line);border-radius:13px;background:linear-gradient(180deg,rgba(17,28,43,.94),rgba(12,20,32,.94));padding:15px;min-height:98px}.statlabel{font-size:9px;color:#718198;text-transform:uppercase;letter-spacing:.08em;font-weight:800}.statvalue{font-size:27px;font-weight:900;margin-top:6px;letter-spacing:-.05em}.stathint{font-size:9px;color:#53657a;margin-top:2px}
.section{border:1px solid var(--line);border-radius:15px;background:rgba(14,22,35,.86);padding:18px;margin-top:12px;box-shadow:0 12px 36px rgba(0,0,0,.10)}.sectionhead{display:flex;justify-content:space-between;align-items:flex-start;gap:15px;margin-bottom:13px}.sectiontitle{font-size:14px;font-weight:850}.sectionsub{font-size:10px;color:#687990;line-height:1.5;margin-top:4px}.twocol{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(300px,.75fr);gap:12px}.threecol{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.docrow{border:1px solid var(--line2);border-radius:11px;padding:12px;background:#0b1421;margin-top:8px}.dochead{display:flex;justify-content:space-between;gap:12px}.docname{font-size:12px;font-weight:800;overflow-wrap:anywhere}.meta{font-size:9px;color:#6f8198;margin-top:3px}.meter{height:5px;background:#1a293a;border-radius:99px;overflow:hidden;margin:11px 0 8px}.meter>i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--blue));border-radius:99px}.microgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}.micro{padding:8px;border:1px solid #1c2b3e;border-radius:8px;background:#0d1725}.micro span{display:block;font-size:8px;color:#62748b;text-transform:uppercase}.micro b{display:block;font-size:10px;margin-top:3px;overflow-wrap:anywhere}.pill{display:inline-flex;align-items:center;gap:5px;border-radius:99px;border:1px solid currentColor;padding:3px 7px;font-size:8px;font-weight:850;white-space:nowrap}.pill i{width:5px;height:5px;border-radius:50%;background:currentColor}.pill-good{color:var(--green);background:rgba(95,224,160,.06)}.pill-warn{color:var(--amber);background:rgba(242,199,107,.06)}.pill-bad{color:var(--red);background:rgba(255,114,133,.06)}.pill-neutral{color:#8494a8;background:rgba(132,148,168,.05)}
.empty{border:1px dashed #304157;border-radius:11px;padding:30px;text-align:center;color:#708197;font-size:10px}.evidence{border:1px solid #23354b;border-radius:11px;padding:12px;background:#0b1421;margin-top:8px}.evidencehead{display:flex;justify-content:space-between;gap:10px}.evidencetitle{font-size:10px;font-weight:800}.evidencemeta{font-size:8px;color:#6e829a}.snippet{font-size:10px;color:#9aabba;line-height:1.6;margin-top:7px}.confidence{display:flex;align-items:center;gap:8px;font-size:9px;color:#708299}.bar{height:4px;flex:1;background:#1a293b;border-radius:99px;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent)}
.metricgrid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-top:10px}.metric{padding:11px;border:1px solid #1d2d41;border-radius:9px;background:#0b1421}.metric label{display:block;font-size:8px;color:#62748b;text-transform:uppercase}.metric b{display:block;font-size:14px;margin-top:5px}.metric small{display:block;font-size:8px;color:#56677c;margin-top:3px}.page{display:grid;grid-template-columns:1.05fr .75fr 1fr .65fr;gap:10px;padding:9px 0;border-bottom:1px solid #1a2839;align-items:center;font-size:9px}.page:last-child{border-bottom:0}.page small{display:block;color:#586a80;font-size:8px;margin-top:2px}.kv{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}.kvitem{border:1px solid #1e2e42;border-radius:9px;padding:10px;background:#0b1421}.kvkey{font-size:8px;color:#60738b;text-transform:uppercase;letter-spacing:.07em}.kvvalue{font-size:10px;font-weight:750;margin-top:4px;overflow-wrap:anywhere}.timeline{border-left:1px solid #26384d;margin:4px 0 0 5px;padding-left:15px}.event{position:relative;padding:0 0 14px;font-size:9px;color:#8fa0b4}.event:before{content:"";position:absolute;left:-19px;top:3px;width:6px;height:6px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 4px #0e1623}.event b{color:#d9e2ed}.event small{display:block;color:#52647b;margin-top:3px}.pipeline{display:flex;gap:5px;margin:13px 0}.step{height:6px;flex:1;border-radius:99px;background:#1b2b3e}.step.on{background:linear-gradient(90deg,var(--accent),var(--blue))}.step.current{box-shadow:0 0 0 2px rgba(99,217,209,.13),0 0 16px rgba(99,217,209,.20)}
.stTextInput input,.stTextArea textarea,.stNumberInput input,.stSelectbox div[data-baseweb=select],[data-baseweb=select] *{background:#0b1421!important;color:#eaf1f8!important;border-color:#26374c!important;border-radius:9px!important}.stTextInput input:focus,.stTextArea textarea:focus{border-color:#4c7884!important;box-shadow:0 0 0 1px #4c7884!important}.stButton>button{border-radius:9px!important;min-height:38px!important;border:1px solid #293a50!important;background:#111d2c!important;color:#dce6f0!important;font-weight:750!important;font-size:10px!important}.stButton>button:hover{border-color:#4e687e!important;background:#162438!important}.stButton>button:focus-visible{outline:2px solid #63d9d1!important;outline-offset:2px}.stButton>button[kind=primary]{background:#4b83a0!important;border-color:#5a9bb7!important;color:#fff!important}.stProgress>div>div>div>div{background:linear-gradient(90deg,var(--accent),var(--blue))!important}.stFileUploader{border:1px dashed #385069!important;border-radius:11px!important;background:#0b1421!important;padding:4px}.stFileUploader label{color:#91a2b6!important}.stExpander{border:1px solid #24364b!important;border-radius:10px!important;background:#0c1522!important}.stAlert{border-radius:9px!important}.caption{font-size:9px;color:#64758b}.chatrow{border:1px solid #213146;border-radius:13px;padding:13px;margin-top:10px;background:#0b1421}.chatrow.user{background:#0e1725}.chatlabel{font-size:8px;color:#60738b;text-transform:uppercase;letter-spacing:.10em;font-weight:850}.answertext{font-size:13px;line-height:1.75;color:#dce6ef;white-space:pre-wrap}.sourcepanel{height:100%;border-left:1px solid #203147;padding-left:15px}.quicktag{display:inline-block;border:1px solid #284058;border-radius:99px;padding:5px 8px;margin:4px 4px 0 0;font-size:8px;color:#7890a9;background:#0b1421}
@media(max-width:1150px){.grid4{grid-template-columns:repeat(2,1fr)}.twocol,.threecol{grid-template-columns:1fr}.metricgrid{grid-template-columns:repeat(2,1fr)}.sourcepanel{border-left:0;border-top:1px solid #203147;padding-left:0;padding-top:15px;margin-top:15px}}@media(max-width:700px){.block-container{padding:18px 14px 50px}.hero h1{font-size:26px}.grid4,.metricgrid{grid-template-columns:1fr}.microgrid,.kv{grid-template-columns:1fr}.page{grid-template-columns:1fr 1fr}.topright{display:none}}
</style>''', unsafe_allow_html=True)


def sidebar(system) -> None:
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brandmark">BR</div><div><div class="brandname">BookRAG Medical</div><div class="brandsub">Evidence-first document research</div></div></div>', unsafe_allow_html=True)
        current = st.session_state.get("bookrag_page", "Home")
        for group, items in [("Workspace", ["Home", "Documents", "Live Processing", "Ask BookRAG"]), ("Research tools", ["Inspector", "System", "Settings"])]:
            st.markdown(f'<div class="navgroup">{group}</div>', unsafe_allow_html=True)
            for item in items:
                prefix = "●  " if item == current else "○  "
                if st.button(prefix + item, key=f"nav_{item}", use_container_width=True):
                    _navigate(item)
        st.markdown('<div class="sidefoot"><b>Local & private</b><br>Documents, retrieval and generation stay inside the configured local runtime.</div>', unsafe_allow_html=True)


def topbar(system, title: str) -> None:
    active = len(active_docs(system))
    st.markdown('<div class="top"><div><div class="crumb">BookRAG / Research workspace</div><div class="title">' + _esc(title) + '</div></div><div class="topright"><div class="chip"><span class="dot"></span>' + str(active) + ' processing</div></div></div>', unsafe_allow_html=True)
    _command_palette(system)


def upload_block(system) -> None:
    st.markdown('<div class="section"><div class="sectionhead"><div><div class="sectiontitle">Add research material</div><div class="sectionsub">Upload one or more PDFs. Files are persisted, hashed and sent to the ingestion pipeline automatically.</div></div></div>', unsafe_allow_html=True)
    files = st.file_uploader("PDF documents", type=["pdf"], accept_multiple_files=True, key="premium_uploader", label_visibility="collapsed")
    if files:
        seen = st.session_state.setdefault("uploaded_hashes", set())
        added, errors = 0, []
        for file in files:
            payload = file.getvalue()
            digest = hashlib.sha256(payload).hexdigest()
            if digest in seen:
                continue
            try:
                save_pdf(Path(system.settings.incoming_dir), file.name, payload)
                seen.add(digest)
                added += 1
            except Exception as exc:
                errors.append(f"{file.name}: {exc}")
        for error in errors:
            st.error(error)
        if added:
            job_id = auto_ingest(system, added)
            if job_id:
                st.session_state["bookrag_page"] = "Live Processing"
                st.success(f"{added} document(s) accepted · live processing started")
                st.rerun()
            elif st.session_state.get("auto_ingest_error"):
                st.error(st.session_state["auto_ingest_error"])
    st.markdown('</div>', unsafe_allow_html=True)


@st.fragment(run_every="2s")
def global_live(system) -> None:
    active = active_docs(system)
    ready = ready_docs(system)
    label = f"{len(active)} document(s) actively processing" if active else (f"{len(ready)} document(s) ready" if ready else "Workspace idle")
    state = "PROCESSING" if active or _job_running() else ("READY" if ready else "IDLE")
    st.markdown(f'<div class="livebar"><span class="livepulse"></span><b>LIVE</b><span>{_esc(label)}</span><span class="grow"></span>{_status(state)}</div>', unsafe_allow_html=True)


def home(system) -> None:
    topbar(system, "Research workspace")
    st.markdown('<div class="hero"><div class="kicker">Medical document intelligence</div><h1>Research with evidence, not guesses.</h1><p>Turn medical PDFs into a private, searchable knowledge base. Watch extraction page by page, inspect the durable state, then ask grounded questions with transparent evidence and abstention.</p><div class="hero-actions">', unsafe_allow_html=True)
    a, b = st.columns([1, 1])
    with a:
        if st.button("＋ Add PDFs", key="hero_add", type="primary", use_container_width=True):
            _navigate("Documents")
    with b:
        if st.button("Ask the library →", key="hero_ask", use_container_width=True):
            _navigate("Ask BookRAG")
    st.markdown('</div></div>', unsafe_allow_html=True)
    global_live(system)
    documents = docs(system)
    ready = ready_docs(system)
    active = active_docs(system)
    stats = [
        ("Documents", len(documents), "managed PDFs"),
        ("Ready", len(ready), "grounded research sources"),
        ("Processing", len(active), "live pipeline jobs"),
        ("Search sections", total_chunks(system), "indexed retrieval units"),
    ]
    st.markdown('<div class="grid4">' + ''.join(f'<div class="stat"><div class="statlabel">{_esc(label)}</div><div class="statvalue">{value:,}</div><div class="stathint">{_esc(hint)}</div></div>' for label, value, hint in stats) + '</div>', unsafe_allow_html=True)
    left, right = st.columns([1.35, .75])
    with left:
        st.markdown('<div class="section"><div class="sectiontitle">Your research workflow</div><div class="sectionsub">A clear path from raw PDF to defensible answer.</div>', unsafe_allow_html=True)
        workflow = [("01", "Upload documents", "Add PDFs and start automatic ingestion.", "Documents"), ("02", "Watch processing", "See exact page, stage, OCR and latest event.", "Live Processing"), ("03", "Inspect the document", "Audit durable page records, chunks and index state.", "Inspector"), ("04", "Ask grounded questions", "Retrieve evidence before generation; abstain when unsupported.", "Ask BookRAG")]
        for num, title, desc, target in workflow:
            st.markdown(f'<div class="evidence"><div class="evidencehead"><div class="evidencetitle">{num} · {_esc(title)}</div><span class="evidencemeta">{_esc(target)}</span></div><div class="snippet">{_esc(desc)}</div></div>', unsafe_allow_html=True)
            if st.button(f"Open {title}", key=f"home_{num}"):
                _navigate(target)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section"><div class="sectiontitle">Trust model</div><div class="sectionsub">What must happen before BookRAG answers.</div>', unsafe_allow_html=True)
        for title, desc in [("Retrieve", "Hybrid search finds relevant document evidence."), ("Rerank", "Evidence is ordered before generation."), ("Ground", "The answer is constrained by retrieved sources."), ("Measure", "Query quality and answerability stay visible."), ("Abstain", "Insufficient evidence can produce a refusal instead of a guess.")]:
            st.markdown(f'<div class="guide"><b>{_esc(title)}</b><div class="muted">{_esc(desc)}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def _filter_documents(ds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    left, mid, right = st.columns([2, 1, 1])
    with left:
        query = st.text_input("Search documents", placeholder="Filename, hash, document ID…", label_visibility="collapsed", key="doc_search")
    with mid:
        status = st.selectbox("Status", ["All", "Ready", "Processing", "Failed"], label_visibility="collapsed", key="doc_status")
    with right:
        sort = st.selectbox("Sort", ["Newest", "Oldest", "Name", "Status", "Size"], label_visibility="collapsed", key="doc_sort")
    f1, f2, f3 = st.columns(3)
    with f1:
        languages = sorted({str(d.get("language")).upper() for d in ds if d.get("language")})
        selected_languages = st.multiselect("Language", languages, placeholder="Any language", key="doc_language")
    with f2:
        min_size = st.number_input("Minimum size (MB)", min_value=0.0, value=0.0, step=1.0, key="doc_min_size")
    with f3:
        max_size = st.number_input("Maximum size (MB)", min_value=0.0, value=0.0, step=1.0, key="doc_max_size")
    rows: list[dict[str, Any]] = []
    needle = query.casefold().strip()
    for document in ds:
        name = _doc_name(document)
        status_value = str(document.get("status") or "UNKNOWN").upper()
        hay = " ".join([name, str(document.get("document_id", "")), str(document.get("content_hash", ""))]).casefold()
        size_mb = _int(document.get("file_size")) / (1024 * 1024)
        language = str(document.get("language") or "").upper()
        if needle and needle not in hay:
            continue
        if status == "Ready" and status_value not in {"READY", "COMPLETED"}:
            continue
        if status == "Processing" and status_value not in ACTIVE:
            continue
        if status == "Failed" and "FAILED" not in status_value:
            continue
        if selected_languages and language not in selected_languages:
            continue
        if min_size and size_mb < min_size:
            continue
        if max_size and size_mb > max_size:
            continue
        rows.append(document)
    if sort == "Name":
        rows.sort(key=lambda d: _doc_name(d).casefold())
    elif sort == "Status":
        rows.sort(key=lambda d: str(d.get("status") or ""))
    elif sort == "Size":
        rows.sort(key=lambda d: _int(d.get("file_size")), reverse=True)
    elif sort == "Oldest":
        rows.sort(key=lambda d: _doc_date(d) or datetime.min.replace(tzinfo=timezone.utc))
    else:
        rows.sort(key=lambda d: _doc_date(d) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return rows


def _document_preview(system, document: dict[str, Any]) -> None:
    document_id = str(document.get("document_id") or "")
    if not document_id:
        return
    with st.expander("Quick preview", expanded=False):
        preview_pages = pages(system, document_id)
        first = preview_pages[0] if preview_pages else {}
        content = str(first.get("text") or first.get("content") or "")
        if content:
            st.markdown(f'**Page {_int(first.get("page_number"))}** · {len(content):,} characters')
            st.text_area("Preview", content[:6000], height=220, label_visibility="collapsed", disabled=True)
        else:
            st.caption("No extracted page text is available yet.")
        st.json({
            "document_id": document_id,
            "source_path": document.get("source_path") or document.get("pdf_path") or document.get("path"),
            "language": document.get("language"),
            "content_hash": document.get("content_hash"),
        })


def documents_page(system) -> None:
    topbar(system, "Documents")
    upload_block(system)
    all_docs = docs(system)
    st.markdown('<div class="section"><div class="sectionhead"><div><div class="sectiontitle">Document library</div><div class="sectionsub">Search, filter, preview and bulk-verify your persistent ingestion state.</div></div></div>', unsafe_allow_html=True)
    rows = _filter_documents(all_docs)
    ids = [str(d.get("document_id")) for d in rows if d.get("document_id")]
    st.session_state.setdefault("selected_docs", set())
    visible_selected = {doc_id for doc_id in st.session_state["selected_docs"] if doc_id in ids}
    select_all = st.checkbox(f"Select all visible · {len(rows)} documents", value=bool(rows) and len(visible_selected) == len(ids), key="doc_select_all")
    if select_all:
        visible_selected = set(ids)
    else:
        visible_selected = {doc_id for doc_id in visible_selected if doc_id in ids}
    st.session_state["selected_docs"] = visible_selected
    if visible_selected:
        act1, act2, act3 = st.columns([1, 1, 2])
        with act1:
            if st.button("Verify selected", type="primary", use_container_width=True):
                results = {}
                for doc_id in sorted(visible_selected):
                    try:
                        results[doc_id] = system.verify_index(doc_id)
                    except Exception as exc:
                        results[doc_id] = {"error": str(exc)}
                st.session_state["bulk_verify_result"] = results
        with act2:
            if st.button("Clear selection", use_container_width=True):
                st.session_state["selected_docs"] = set()
                st.rerun()
        with act3:
            manifest = [{"document_id": d.get("document_id"), "file_name": _doc_name(d), "status": d.get("status"), "pages": d.get("total_pages"), "chunks": d.get("chunk_count"), "embeddings": d.get("embedding_count")} for d in rows if str(d.get("document_id")) in visible_selected]
            st.download_button("Export selection manifest", data=json.dumps(manifest, indent=2).encode(), file_name="bookrag_selection.json", mime="application/json", use_container_width=True)
    if st.session_state.get("bulk_verify_result"):
        with st.expander("Bulk verification results", expanded=False):
            st.json(st.session_state.pop("bulk_verify_result"))
    if not rows:
        st.markdown('<div class="empty">No documents match this view.</div>', unsafe_allow_html=True)
    for document in rows:
        document_id = str(document.get("document_id") or "")
        name = _doc_name(document)
        progress = _progress(document)
        status_value = str(document.get("status") or "UNKNOWN")
        checked = document_id in visible_selected
        st.markdown(f'<div class="docrow"><div class="dochead"><div><div class="docname">{_esc(name)}</div><div class="meta">{_esc(document_id)} · {_fmt_bytes(document.get("file_size"))} · {_esc((document.get("language") or "language unknown").upper())}</div></div>{_status(status_value)}</div><div class="meter"><i style="width:{progress * 100:.1f}%"></i></div><div class="microgrid"><div class="micro"><span>Stage</span><b>{_esc(_stage(document)[0])}</b></div><div class="micro"><span>Pages</span><b>{_int(document.get("current_page"))} / {_int(document.get("total_pages"))}</b></div><div class="micro"><span>Chunks</span><b>{_int(document.get("chunk_count")):,}</b></div><div class="micro"><span>Embeddings</span><b>{_int(document.get("embedding_count")):,}</b></div></div></div>', unsafe_allow_html=True)
        a, b, c = st.columns([1.2, 1.2, 5])
        with a:
            picked = st.checkbox("Select", value=checked, key=f"select_{document_id}", label_visibility="collapsed") if document_id else False
            if document_id and picked != checked:
                if picked:
                    visible_selected.add(document_id)
                else:
                    visible_selected.discard(document_id)
                st.session_state["selected_docs"] = visible_selected
                st.rerun()
        with b:
            if st.button("Inspect", key=f"inspect_{document_id}", use_container_width=True):
                st.session_state["inspect_doc_id"] = document_id
                _navigate("Inspector")
        with c:
            _document_preview(system, document)
    st.markdown('</div>', unsafe_allow_html=True)


@st.fragment(run_every="2s")
def processing_live(system) -> None:
    topbar(system, "Live Processing")
    active = active_docs(system)
    if not active:
        st.markdown('<div class="section"><div class="empty">No document is processing right now. Upload a PDF to start the live pipeline.</div></div>', unsafe_allow_html=True)
        return
    recent_events = events(system, limit=800)
    latest_by_doc: dict[str, dict[str, Any]] = {}
    for event in recent_events:
        latest_by_doc[str(event.get("document_id"))] = event
    stage_keys = list(STAGES.keys())
    for document in active:
        document_id = str(document.get("document_id"))
        name = _doc_name(document)
        stage, description = _stage(document)
        progress = _progress(document)
        current, total = _int(document.get("current_page")), _int(document.get("total_pages"))
        latest = latest_by_doc.get(document_id, {})
        stage_key = str(document.get("current_stage") or document.get("status") or "RUNNING").upper()
        current_idx = stage_keys.index(stage_key) if stage_key in stage_keys else 0
        st.markdown(f'<div class="section"><div class="dochead"><div><div class="docname">{_esc(name)}</div><div class="meta">{_esc(stage)} · {_fmt_time(_elapsed(document.get("ingestion_started_at")))} elapsed</div></div>{_status(document.get("status"))}</div><div class="sectionsub">{_esc(description)}</div>', unsafe_allow_html=True)
        st.progress(progress, text=f"Pipeline progress · {progress * 100:.0f}%")
        st.markdown('<div class="pipeline">' + ''.join(f'<div class="step {"on" if i < current_idx else ""} {"current" if i == current_idx else ""}"></div>' for i, _ in enumerate(stage_keys[:10])) + '</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metricgrid"><div class="metric"><label>Current page</label><b>{current or "—"} / {total or "—"}</b><small>durable checkpoint</small></div><div class="metric"><label>Stage</label><b>{_esc(stage)}</b><small>persisted state</small></div><div class="metric"><label>Latest event</label><b>{_esc(latest.get("message") or latest.get("event_type") or "Working…")}</b><small>most recent state signal</small></div><div class="metric"><label>Elapsed</label><b>{_fmt_time(_elapsed(document.get("ingestion_started_at")))}</b><small>live wall-clock time</small></div></div>', unsafe_allow_html=True)
        page_rows = pages(system, document_id)
        if page_rows:
            done = sum(1 for row in page_rows if str(row.get("extraction_status", "")).upper() == "COMPLETED")
            st.caption(f"{done} / {len(page_rows)} page records completed")
            with st.expander(f"Page-level extraction · {len(page_rows)} records", expanded=True):
                for row in page_rows[-50:]:
                    page_no = _int(row.get("page_number"))
                    text_len = len(str(row.get("text") or ""))
                    st.markdown(f'<div class="page"><div><b>Page {page_no}</b><small>{_esc(row.get("updated_at"))}</small></div><div>{_status(row.get("extraction_status"))}</div><div>OCR · {_esc(row.get("ocr_status") or "not required")}</div><div>{text_len:,} chars</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def processing_page(system) -> None:
    processing_live(system)


def _render_answer_metrics(result: dict[str, Any]) -> None:
    quality = _metric(result, "query_quality") or "—"
    answerability = _metric(result, "answerability")
    relevance = _metric(result, "query_relevance", "relevance")
    confidence = _metric(result, "evidence_confidence", "confidence")
    consistency = _metric(result, "cross_chunk_consistency", "consistency")
    decision = _metric(result, "decision") or ("ABSTAIN" if result.get("abstained") else "GROUNDED")
    metrics = [
        ("Answerability", f"{_float(answerability) * 100:.0f}%" if answerability is not None else "—", "evidence alignment"),
        ("Evidence confidence", f"{_float(confidence) * 100:.0f}%" if confidence is not None else "—", "retrieval support"),
        ("Query quality", str(quality), "question signal"),
        ("Decision", str(decision), "grounding gate"),
    ]
    st.markdown('<div class="metricgrid">' + ''.join(f'<div class="metric"><label>{_esc(label)}</label><b>{_esc(value)}</b><small>{_esc(hint)}</small></div>' for label, value, hint in metrics) + '</div>', unsafe_allow_html=True)
    if relevance is not None or consistency is not None:
        st.caption(f"Diagnostic alignment · relevance {_float(relevance):.2f} · consistency {_float(consistency):.2f}")


def _progressive_answer(answer: str, key: str) -> None:
    if not answer:
        return
    chunks = [answer[i:i + 220] for i in range(0, len(answer), 220)]
    placeholder = st.empty()
    rendered = ""
    for chunk in chunks:
        rendered += chunk
        placeholder.markdown(f'<div class="answertext">{_esc(rendered)}</div>', unsafe_allow_html=True)
        time.sleep(0.015)
    st.session_state[f"revealed_{key}"] = True


def ask_page(system) -> None:
    topbar(system, "Ask BookRAG")
    available = ready_docs(system)
    st.markdown('<div class="section"><div class="sectiontitle">Grounded research console</div><div class="sectionsub">Ask the library, inspect answerability, and see the exact evidence that justified the response.</div>', unsafe_allow_html=True)
    names = ["All ready documents"] + [_doc_name(d) for d in available]
    scope = st.selectbox("Source scope", names, key="ask_scope")
    selected = None if scope == names[0] else {"document_id": available[names.index(scope) - 1].get("document_id")}
    presets = ["What are the main findings?", "What limitations are reported?", "Compare the most relevant results."]
    st.caption("Quick questions")
    for preset in presets:
        if st.button(preset, key="preset_" + hashlib.sha1(preset.encode()).hexdigest()[:8]):
            st.session_state["research_question"] = preset
    question = st.text_area("Question", height=120, placeholder="What does the literature say about…?", key="research_question")
    a, b, c = st.columns([2, 1, 1])
    with a:
        run = st.button("Run grounded search", type="primary", use_container_width=True, key="ask_run")
    with b:
        if st.button("Regenerate", use_container_width=True, disabled=not bool(st.session_state.get("answer_result"))):
            run = True
    with c:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.pop("answer_result", None)
            st.session_state["chat_history"] = []
            st.rerun()
    if run:
        if not question.strip():
            st.warning("Enter a research question first.")
        elif not available:
            st.warning("No ready document is available yet.")
        else:
            with st.spinner("Retrieving evidence and composing a grounded answer…"):
                try:
                    result = system.answer(question.strip(), metadata_filter=selected)
                    st.session_state["answer_result"] = result
                    history = st.session_state.setdefault("chat_history", [])
                    history.append({"role": "user", "content": question.strip()})
                    history.append({"role": "assistant", "content": str(result.get("answer") or "")})
                except Exception as exc:
                    st.error(f"The answer could not be produced safely: {exc}")
    answer_result = st.session_state.get("answer_result")
    if not isinstance(answer_result, dict):
        st.markdown('</div>', unsafe_allow_html=True)
        with st.expander("How to get stronger answers", expanded=True):
            st.write("Ask about one measurable concept, population, intervention, outcome, or comparison. The system will abstain when the indexed evidence is insufficient.")
        return
    st.markdown('</div>', unsafe_allow_html=True)
    answer = str(answer_result.get("answer") or "")
    evidence = _evidence(answer_result)
    grounded = not bool(answer_result.get("abstained")) and str(answer_result.get("status", "")).upper() not in {"ABSTAIN", "SYSTEM_NOT_READY"}
    left, right = st.columns([1.28, .72])
    with left:
        st.markdown(f'<div class="section"><div class="sectionhead"><div><div class="sectiontitle">Answer</div><div class="sectionsub">{("Grounded response" if grounded else "Abstention / caution")} · {len(evidence)} evidence item(s)</div></div>{_status("GROUNDED" if grounded else "ABSTAIN")}</div>', unsafe_allow_html=True)
        reveal_key = hashlib.sha1(answer.encode()).hexdigest()[:12]
        if st.session_state.get(f"revealed_{reveal_key}"):
            st.markdown(f'<div class="answertext">{_esc(answer)}</div>', unsafe_allow_html=True)
        else:
            _progressive_answer(answer, reveal_key)
        st.markdown('</div>', unsafe_allow_html=True)
        _render_answer_metrics(answer_result)
        st.markdown('<div class="section"><div class="sectionhead"><div><div class="sectiontitle">Evidence</div><div class="sectionsub">Open each source to inspect the retrieved passage and its score.</div></div></div>', unsafe_allow_html=True)
        if evidence:
            for i, item in enumerate(evidence, 1):
                title, page, score, snippet = _evidence_row(item, i)
                score_value = _float(score, default=-1)
                score_text = f"{score_value:.3f}" if score_value >= 0 else str(score)
                st.markdown(f'<div class="evidence"><div class="evidencehead"><div class="evidencetitle">{_esc(title)}</div><div class="evidencemeta">Page {_esc(page)} · score {_esc(score_text)}</div></div><div class="snippet">{_esc(snippet[:1200])}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty">No evidence records were returned with this answer.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section sourcepanel"><div class="sectiontitle">Answer controls</div><div class="sectionsub">Portable output and transparent diagnostics.</div>', unsafe_allow_html=True)
        st.download_button("Export answer · Markdown", data=answer.encode(), file_name="bookrag_answer.md", mime="text/markdown", use_container_width=True)
        st.download_button("Export answer · JSON", data=json.dumps(answer_result, default=str, indent=2).encode(), file_name="bookrag_answer.json", mime="application/json", use_container_width=True)
        st.caption("Copy")
        st.code(answer, language="text")
        with st.expander("Query trace", expanded=False):
            st.json(answer_result.get("query_trace", {}))
        with st.expander("Raw response", expanded=False):
            st.json(answer_result)
        st.markdown('</div>', unsafe_allow_html=True)


def inspector_page(system) -> None:
    topbar(system, "Inspector")
    documents = docs(system)
    if not documents:
        st.markdown('<div class="section"><div class="empty">No document is available to inspect.</div></div>', unsafe_allow_html=True)
        return
    document_ids = [str(d.get("document_id")) for d in documents]
    selected_id = str(st.session_state.get("inspect_doc_id") or "")
    default_idx = document_ids.index(selected_id) if selected_id in document_ids else 0
    chosen_idx = st.selectbox("Document", range(len(documents)), index=default_idx, format_func=lambda i: _doc_name(documents[i]), key="inspect_selector")
    document = documents[chosen_idx]
    document_id = str(document.get("document_id") or "")
    st.session_state["inspect_doc_id"] = document_id
    stage, description = _stage(document)
    progress = _progress(document)
    st.markdown(f'<div class="section"><div class="sectionhead"><div><div class="sectiontitle">{_esc(_doc_name(document))}</div><div class="sectionsub">{_esc(description)}</div></div>{_status(document.get("status"))}</div><div class="meter"><i style="width:{progress * 100:.1f}%"></i></div>', unsafe_allow_html=True)
    values = [
        ("Stage", stage), ("Pages", f'{_int(document.get("current_page"))} / {_int(document.get("total_pages"))}'),
        ("Elapsed", _fmt_time(_elapsed(document.get("ingestion_started_at"), document.get("ingestion_completed_at")))),
        ("Chunks", f'{_int(document.get("chunk_count")):,}'), ("Embeddings", f'{_int(document.get("embedding_count")):,}'),
        ("Embedding dimension", document.get("embedding_dimension") or "—"), ("Parser", document.get("parser_version") or "—"),
        ("Version", document.get("version_id") or document.get("content_hash") or "—"),
    ]
    st.markdown('<div class="kv">' + ''.join(f'<div class="kvitem"><div class="kvkey">{_esc(key)}</div><div class="kvvalue">{_esc(value)}</div></div>' for key, value in values) + '</div>', unsafe_allow_html=True)
    if document.get("error"):
        st.error(str(document.get("error")))
    if st.button("Verify index", type="primary", use_container_width=True, key="verify_index"):
        try:
            st.session_state["verify_result"] = system.verify_index(document_id)
        except Exception as exc:
            st.error(str(exc))
    if st.session_state.get("verify_result") is not None:
        with st.expander("Verification result", expanded=True):
            st.json(st.session_state.pop("verify_result"))
    st.markdown('</div>', unsafe_allow_html=True)
    page_rows = pages(system, document_id)
    st.markdown('<div class="section"><div class="sectiontitle">Page records</div><div class="sectionsub">Durable extraction and OCR checkpoints for every physical page.</div>', unsafe_allow_html=True)
    if page_rows:
        for row in page_rows:
            st.markdown(f'<div class="page"><div><b>Page {_int(row.get("page_number"))}</b><small>{_esc(row.get("updated_at"))}</small></div><div>{_status(row.get("extraction_status"))}</div><div>OCR · {_esc(row.get("ocr_status") or "not required")}</div><div>{len(str(row.get("text") or "")):,} chars</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty">No page records stored yet.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    with st.expander("Event timeline", expanded=True):
        event_rows = events(system, document_id, 250)
        if not event_rows:
            st.caption("No events recorded yet.")
        else:
            st.markdown('<div class="timeline">', unsafe_allow_html=True)
            for event in reversed(event_rows[-50:]):
                message = event.get("message") or event.get("event_type") or event.get("stage") or "Event"
                st.markdown(f'<div class="event"><b>{_esc(message)}</b> · {_status(event.get("status") or event.get("stage"))}<small>{_esc(event.get("created_at"))} · page {_esc(event.get("current_page") or "—")} / {_esc(event.get("total_pages") or "—")}</small></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    with st.expander("Raw state", expanded=False):
        st.json(document)


@st.fragment(run_every="8s")
def system_snapshot(system) -> None:
    try:
        report = system.health_report()
    except Exception as exc:
        report = {"ready": False, "error": str(exc)}
    try:
        ollama_ok, ollama_message, models = ollama_health(system.settings.ollama_base_url)
    except Exception as exc:
        ollama_ok, ollama_message, models = False, str(exc), []
    embedding = report.get("embedding", {}) if isinstance(report, dict) else {}
    index = report.get("index", {}) if isinstance(report, dict) else {}
    contract = report.get("feature_contract", {}) if isinstance(report, dict) else {}
    rows = [
        ("Ollama", "ONLINE" if ollama_ok else "OFFLINE", ollama_message),
        ("Embedding", "PASS" if embedding.get("ok") else "UNKNOWN", embedding.get("identity") or embedding.get("error") or system.settings.embedding_model),
        ("Vector index", str(index.get("status", "UNKNOWN")).upper(), index.get("error") or "Runtime-reported index health"),
        ("Production contract", "PASS" if contract.get("all_resolved", False) else "CHECK", "Feature resolution"),
    ]
    st.markdown('<div class="section"><div class="sectionhead"><div><div class="sectiontitle">System health</div><div class="sectionsub">Operational signals refresh automatically.</div></div><span class="chip"><span class="dot"></span>8s</span></div>', unsafe_allow_html=True)
    for label, status, detail in rows:
        st.markdown(f'<div class="page"><div><b>{_esc(label)}</b></div><div>{_status(status)}</div><div>{_esc(detail)}</div><div></div></div>', unsafe_allow_html=True)
    if models:
        st.caption("Ollama models · " + ", ".join(models))
    if report.get("error"):
        st.error(str(report["error"]))
    st.markdown('</div>', unsafe_allow_html=True)


def system_page(system) -> None:
    topbar(system, "System")
    system_snapshot(system)
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section"><div class="sectiontitle">Runtime profile</div><div class="sectionsub">Configured local services.</div>', unsafe_allow_html=True)
        for key, value in [("Ollama", system.settings.ollama_base_url), ("Embedding", system.settings.embedding_model), ("Generation", system.settings.generation_model), ("Top K", system.settings.top_k), ("Chunks", total_chunks(system)), ("Embeddings", total_embeddings(system))]:
            st.markdown(f'<div class="kvitem" style="margin-top:7px"><div class="kvkey">{_esc(key)}</div><div class="kvvalue">{_esc(value)}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section"><div class="sectiontitle">Operational guidance</div><div class="sectionsub">How to interpret the workspace signals.</div>', unsafe_allow_html=True)
        for key, value in [("Ready", "Document is published to retrieval."), ("Processing", "Persistent page state is still changing."), ("Failed", "Inspect the document error and event timeline."), ("Abstain", "The evidence did not justify a confident answer."), ("Offline", "The local model endpoint is unavailable; generation cannot proceed safely.")]:
            st.markdown(f'<div class="evidence"><div class="evidencetitle">{_esc(key)}</div><div class="snippet">{_esc(value)}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def settings_page(system) -> None:
    topbar(system, "Settings")
    settings = system.settings
    st.markdown('<div class="section"><div class="sectiontitle">Research behavior</div><div class="sectionsub">Safe controls are exposed here; validation remains in the application layer.</div>', unsafe_allow_html=True)
    with st.form("settings_form"):
        host = st.text_input("Ollama host", str(settings.ollama_base_url))
        embedding_model = st.text_input("Embedding model", str(settings.embedding_model))
        generation_model = st.text_input("Generation model", str(settings.generation_model))
        a, b = st.columns(2)
        with a:
            top_k = st.number_input("Retrieval results", 1, 50, int(settings.top_k))
            vector_weight = st.slider("Semantic weight", 0.0, 1.0, float(settings.vector_weight), 0.05)
        with b:
            temperature = st.slider("Generation temperature", 0.0, 1.0, float(settings.temperature), 0.05)
            neighbor = st.checkbox("Neighbor expansion", bool(settings.neighbor_expansion))
        save = st.form_submit_button("Save configuration", type="primary", use_container_width=True)
    if save:
        try:
            ok, warnings = system.apply_settings_in_place({
                "ollama_base_url": host,
                "embedding_model": embedding_model,
                "generation_model": generation_model,
                "top_k": int(top_k),
                "vector_weight": float(vector_weight),
                "temperature": float(temperature),
                "neighbor_expansion": bool(neighbor),
            })
            if ok:
                st.success("Configuration saved.")
            for warning in warnings or []:
                st.warning(warning)
        except Exception as exc:
            st.error(f"Settings were not saved: {exc}")
    st.markdown('</div>', unsafe_allow_html=True)
    with st.expander("Advanced diagnostics", expanded=False):
        st.json({
            "project_root": str(settings.project_root),
            "incoming_dir": str(settings.incoming_dir),
            "vector_db_dir": str(settings.vector_db_dir),
            "ingestion_db_path": str(settings.ingestion_db_path),
        })


def main() -> None:
    st.set_page_config(page_title="BookRAG Medical", page_icon="BR", layout="wide", initial_sidebar_state="expanded")
    st.session_state.setdefault("bookrag_page", "Home")
    st.session_state.setdefault("uploaded_hashes", set())
    st.session_state.setdefault("chat_history", [])
    system = get_system()
    css()
    sidebar(system)
    page = st.session_state.get("bookrag_page", "Home")
    renderer = {
        "Home": home,
        "Documents": documents_page,
        "Live Processing": processing_page,
        "Ask BookRAG": ask_page,
        "Inspector": inspector_page,
        "System": system_page,
        "Settings": settings_page,
    }.get(page, home)
    renderer(system)


if __name__ == "__main__":
    main()
