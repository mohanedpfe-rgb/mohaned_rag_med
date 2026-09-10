from __future__ import annotations

import ipaddress
import os
import re
import secrets
import socket
import threading
import time
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

OLLAMA_ALLOWLIST_ENV = "BOOKRAG_OLLAMA_ALLOWLIST"
MAX_PDF_PAGES_ENV = "BOOKRAG_MAX_PDF_PAGES"
CLEAR_PHRASE = "CLEAR ALL PDF DATA"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_SESSION_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_PDF_PAGES = 500
MAX_QUERY_CHARS = 4000
GLOBAL_CONCURRENT_INGESTS = 2
GLOBAL_CONCURRENT_ANSWERS = 4
MAX_FILENAME_CHARS = 180
RATE_STATE_MAX = 4096

_INGEST_LIMITER = threading.BoundedSemaphore(GLOBAL_CONCURRENT_INGESTS)
_ANSWER_LIMITER = threading.BoundedSemaphore(GLOBAL_CONCURRENT_ANSWERS)
_RATE_LOCK = threading.Lock()
_RATE_STATE: dict[str, tuple[float, int]] = {}


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def max_pdf_pages() -> int:
    raw = os.getenv(MAX_PDF_PAGES_ENV, str(MAX_PDF_PAGES)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = MAX_PDF_PAGES
    return max(1, min(value, 5000))


def _session_actor() -> str:
    actor = st.session_state.get("bookrag_actor_id")
    if not actor:
        actor = secrets.token_hex(16)
        st.session_state["bookrag_actor_id"] = actor
    return actor


def audit_event(action: str, *, detail: str = "") -> None:
    from datetime import datetime, timezone
    log_dir = Path(os.getenv("LOG_DIR", "logs")).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_action = "".join(ch for ch in str(action) if ch.isalnum() or ch in ".-_")[:80]
    safe_detail = sanitize_log_text(detail)[:500]
    with (log_dir / "security_audit.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now(timezone.utc).isoformat()} actor={_session_actor()} action={safe_action} detail={safe_detail}\n")


def sanitize_log_text(value: object) -> str:
    text = str(value or "")
    text = "".join(ch for ch in text if ch in "\n\r\t" or unicodedata.category(ch)[0] != "C")
    return text.replace("\r", "\\r").replace("\n", "\\n")


def clear_confirmation_ui() -> None:
    with st.sidebar:
        phrase = st.text_input("Type CLEAR ALL PDF DATA to enable deletion", key="bookrag_clear_phrase", placeholder=CLEAR_PHRASE, label_visibility="collapsed")
    st.session_state["exact_confirm"] = secrets.compare_digest(phrase, CLEAR_PHRASE)


def require_clear_confirmation() -> None:
    if not secrets.compare_digest(str(st.session_state.get("bookrag_clear_phrase", "")), CLEAR_PHRASE):
        raise PermissionError("Typed confirmation required before clearing PDF data.")
    audit_event("clear_authorized")


def validate_storage_path(project_root: Path, candidate: str | Path, label: str = "path") -> Path:
    root = Path(project_root).expanduser().resolve()
    target = Path(candidate).expanduser().resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must remain inside the BookRAG project directory.") from exc
    return target


def _resolved_ips(host: str) -> set[str]:
    try:
        return {item[4][0] for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)}
    except OSError as exc:
        raise ValueError("Ollama hostname could not be resolved safely.") from exc


def _is_disallowed_ip(value: str) -> bool:
    ip = ipaddress.ip_address(value)
    return bool(ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified or ip.is_private)


def validate_ollama_url(value: str) -> str:
    raw = str(value or "").strip().rstrip("/")
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Ollama URL must be an HTTP(S) URL without embedded credentials.")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("Ollama URL must contain only scheme, host and optional port.")
    host = parsed.hostname.lower()
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if host in {"localhost", "ip6-localhost"} or (ip and ip.is_loopback):
        return raw
    allowlist = {h.strip().lower() for h in os.getenv(OLLAMA_ALLOWLIST_ENV, "").split(",") if h.strip()}
    if not allowlist or host not in allowlist:
        raise ValueError("Remote Ollama endpoints are disabled unless the exact hostname is in BOOKRAG_OLLAMA_ALLOWLIST.")
    addresses = {str(ipaddress.ip_address(x)) for x in _resolved_ips(host)}
    if not addresses or any(_is_disallowed_ip(addr) for addr in addresses):
        raise ValueError("Ollama hostname resolves to a private, local, reserved, or otherwise unsafe network address.")
    return raw


def validate_query(value: str) -> str:
    question = str(value or "").strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    if len(question) > MAX_QUERY_CHARS:
        raise ValueError(f"Question is too long; maximum is {MAX_QUERY_CHARS} characters.")
    return question


def validate_pdf_payload(name: str, content: bytes) -> None:
    safe_name = Path(name).name
    if len(safe_name) > MAX_FILENAME_CHARS:
        raise ValueError("PDF filename is too long.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Uploaded file '{safe_name}' exceeds the 50 MB security limit.")
    if not content.startswith(b"%PDF-"):
        raise ValueError(f"Uploaded file '{safe_name}' is not a valid PDF payload.")
    try:
        import fitz
        pdf = fitz.open(stream=content, filetype="pdf")
        try:
            validate_pdf_page_count(pdf.page_count)
        finally:
            pdf.close()
    except Exception as exc:
        raise ValueError(f"Uploaded file '{safe_name}' failed PDF structural validation.") from exc


def validate_pdf_page_count(page_count: int) -> int:
    count = int(page_count)
    limit = max_pdf_pages()
    if count < 0:
        raise ValueError("PDF page count cannot be negative.")
    if count > limit:
        raise ValueError(f"PDF has {count} pages; the security limit is {limit} pages.")
    return count


def register_session_upload(size_bytes: int) -> None:
    size = int(size_bytes)
    if size < 0:
        raise ValueError("Upload size cannot be negative.")
    total = int(st.session_state.get("bookrag_upload_bytes", 0))
    if total + size > MAX_SESSION_UPLOAD_BYTES:
        raise ValueError("This session exceeded the 500 MB cumulative upload security limit.")
    st.session_state["bookrag_upload_bytes"] = total + size


def sanitize_model_text(text: str, *, limit: int) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = "".join(ch for ch in value if ch in "\n\r\t" or unicodedata.category(ch)[0] != "C")
    if len(value) > limit:
        value = value[:limit] + "\n[TRUNCATED_UNTRUSTED_TEXT]"
    return value


def _safe_sequence(value: object) -> list[object]:
    if value is None: return []
    if isinstance(value, list): return value
    if isinstance(value, tuple): return list(value)
    try: return list(value)
    except (TypeError, ValueError): return []


def sanitize_evidence_for_prompt(text: str) -> str:
    value = sanitize_model_text(text, limit=12000)
    patterns = (
        r"(?i)^\s*(ignore|forget|disregard|override|bypass|follow these instructions|new instructions)\b",
        r"(?i)^\s*(system|developer|assistant|user|human|instruction|prompt)\s*[:：\-–—]",
        r"(?i)(reveal|show|print|dump).{0,40}(system prompt|developer prompt|hidden instructions)",
        r"(?i)(jailbreak|admin override|developer mode|simulation mode|roleplay as)",
    )
    lines = []
    for line in value.splitlines():
        lines.append("[REDACTED_UNTRUSTED_INSTRUCTION]" if any(re.search(pattern, line) for pattern in patterns) else line)
    return "\n".join(lines).strip()


def postprocess_medical_output(result: dict) -> dict:
    if not isinstance(result, dict):
        return result
    answer = str(result.get("answer") or "")
    citations = _safe_sequence(result.get("citations"))
    high_risk = re.search(
        r"(?i)(?:\b(?:dose|dosage|mg|mcg|ml|prescri\w*|inject\w*|opioid\w*|chemotherapy\w*|suicid\w*|overdose\w*|emergency\w*)\b|\btake\s+\d+\b|\banticoagul\w*\b|\bpregnan\w*\b|\binsulin\b)",
        answer,
    )
    if high_risk and not citations:
        result = dict(result)
        result["answer"] = "I can't provide a clinically actionable recommendation without cited evidence from the indexed documents. Please verify the relevant source before acting."
        result["safety_backstop"] = "medical_action_without_citation"
    return result


def acquire_ingest_slot(timeout: float = 0.1) -> bool: return _INGEST_LIMITER.acquire(timeout=max(0.0, timeout))
def release_ingest_slot() -> None: _INGEST_LIMITER.release()
def acquire_answer_slot(timeout: float = 0.1) -> bool: return _ANSWER_LIMITER.acquire(timeout=max(0.0, timeout))
def release_answer_slot() -> None: _ANSWER_LIMITER.release()


def consume_rate_limit(bucket: str, *, limit: int, window_seconds: float) -> bool:
    now = time.monotonic(); key = f"{_session_actor()}::{bucket}"
    with _RATE_LOCK:
        start, count = _RATE_STATE.get(key, (now, 0))
        if now - start >= window_seconds: start, count = now, 0
        if count >= limit: return False
        _RATE_STATE[key] = (start, count + 1)
        if len(_RATE_STATE) > RATE_STATE_MAX:
            expired = [k for k, (s, _) in _RATE_STATE.items() if now - s >= window_seconds]
            for k in expired[: RATE_STATE_MAX // 2]: _RATE_STATE.pop(k, None)
            while len(_RATE_STATE) > RATE_STATE_MAX: _RATE_STATE.pop(next(iter(_RATE_STATE)))
        return True


def enforce_private_permissions(root: Path) -> None:
    if os.name == "nt": return
    root = Path(root).resolve()
    for target in (root / "data", root / "logs"):
        if not target.exists(): continue
        try:
            target.chmod(0o700 if target.is_dir() else 0o600)
            if target.is_dir():
                for path in target.rglob("*"): path.chmod(0o700 if path.is_dir() else 0o600)
        except OSError: continue


def harden_system(system):
    if getattr(system, "_bookrag_security_hardened", False): return system
    root = Path(system.settings.project_root).expanduser().resolve(); enforce_private_permissions(root)
    for attr in ("incoming_dir", "processed_dir", "failed_dir", "archive_dir", "vector_db_dir", "log_dir", "ingestion_db_path"):
        value = getattr(system.settings, attr, None)
        if value is not None: validate_storage_path(root, value, attr)
    try:
        import rag_project.app.rag_system as rag_module
        rag_module.sanitize_evidence = sanitize_evidence_for_prompt
    except Exception: pass
    if getattr(system, "conversation_memory", None) is not None:
        memory = system.conversation_memory; original_prompt_context = memory.prompt_context
        memory.prompt_context = lambda: sanitize_model_text(original_prompt_context(), limit=8000)
    original_clear = system.clear_pdf_data; original_apply = system.apply_settings_in_place; original_ingest_directory = system.ingest_directory; original_ingest_file = system.ingest_file; original_answer = system.answer
    def guarded_clear():
        require_clear_confirmation(); audit_event("clear_start")
        try: return original_clear()
        finally: audit_event("clear_finish")
    def guarded_apply(updates):
        clean = dict(updates or {})
        if "ollama_base_url" in clean: clean["ollama_base_url"] = validate_ollama_url(clean["ollama_base_url"])
        for key in ("incoming_dir", "processed_dir", "failed_dir", "archive_dir", "vector_db_dir", "log_dir", "ingestion_db_path"):
            if key in clean: clean[key] = validate_storage_path(root, clean[key], key)
        return original_apply(clean)
    def guarded_ingest_directory(directory=None):
        if not acquire_ingest_slot(0.1): raise RuntimeError("Too many concurrent ingestion jobs. Please retry shortly.")
        try: return original_ingest_directory(directory)
        finally: release_ingest_slot()
    def guarded_ingest_file(pdf_path, *args, **kwargs):
        if not acquire_ingest_slot(0.1): raise RuntimeError("Too many concurrent ingestion jobs. Please retry shortly.")
        try: return original_ingest_file(pdf_path, *args, **kwargs)
        finally: release_ingest_slot()
    def guarded_answer(question, *args, **kwargs):
        question = validate_query(question)
        if not consume_rate_limit("answer", limit=30, window_seconds=60): raise RuntimeError("Too many questions in a short period. Please wait a moment and retry.")
        if not acquire_answer_slot(0.1): raise RuntimeError("Too many concurrent answer jobs. Please retry shortly.")
        try: return postprocess_medical_output(original_answer(question, *args, **kwargs))
        finally: release_answer_slot()
    system.clear_pdf_data = guarded_clear; system.apply_settings_in_place = guarded_apply; system.ingest_directory = guarded_ingest_directory; system.ingest_file = guarded_ingest_file; system.answer = guarded_answer; system._bookrag_security_hardened = True
    return system
