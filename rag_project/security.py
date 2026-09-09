from __future__ import annotations

import ipaddress
import os
import secrets
import time
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

AUTH_ENV = "BOOKRAG_ADMIN_PASSWORD"
REMOTE_OLLAMA_ENV = "BOOKRAG_ALLOW_REMOTE_OLLAMA"
OLLAMA_ALLOWLIST_ENV = "BOOKRAG_OLLAMA_ALLOWLIST"
CLEAR_PHRASE = "CLEAR ALL PDF DATA"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_QUERY_CHARS = 4000


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def require_auth() -> None:
    """Require an explicit admin password before exposing the application."""
    password = os.getenv(AUTH_ENV, "").strip()
    if len(password) < 12:
        st.error(
            "BookRAG is locked because BOOKRAG_ADMIN_PASSWORD is not configured "
            "with a password of at least 12 characters. Set it in the process environment "
            "before starting Streamlit."
        )
        st.stop()

    now = time.time()
    if st.session_state.get("bookrag_authenticated"):
        if now - float(st.session_state.get("bookrag_auth_at", 0)) <= 1800:
            return
        st.session_state.pop("bookrag_authenticated", None)

    st.markdown(
        "<div style='max-width:520px;margin:14vh auto 0;padding:28px;border:1px solid #273449;"
        "border-radius:18px;background:#111a2a;color:#f7f9fc'>"
        "<h2 style='margin:0 0 8px'>BookRAG Studio</h2>"
        "<p style='color:#8794a8;margin-bottom:18px'>Private local application — authentication required.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    attempt = st.text_input("Admin password", type="password", key="bookrag_login_password")
    if st.button("Unlock BookRAG", type="primary", key="bookrag_unlock"):
        if secrets.compare_digest(attempt, password):
            st.session_state["bookrag_authenticated"] = True
            st.session_state["bookrag_auth_at"] = now
            st.session_state["bookrag_failed_attempts"] = 0
            st.rerun()
        st.session_state["bookrag_failed_attempts"] = int(st.session_state.get("bookrag_failed_attempts", 0)) + 1
        st.error("Invalid password.")
        if st.session_state["bookrag_failed_attempts"] >= 5:
            st.warning("Too many failed attempts in this session. Restart Streamlit to reset the lock.")
            st.stop()
    st.stop()


def clear_confirmation_ui() -> None:
    """Render a typed confirmation and synchronize the legacy checkbox gate."""
    with st.sidebar:
        phrase = st.text_input(
            "Type CLEAR ALL PDF DATA to enable deletion",
            key="bookrag_clear_phrase",
            type="password",
            placeholder=CLEAR_PHRASE,
            label_visibility="collapsed",
        )
    st.session_state["exact_confirm"] = secrets.compare_digest(phrase, CLEAR_PHRASE)


def require_clear_confirmation() -> None:
    if not secrets.compare_digest(str(st.session_state.get("bookrag_clear_phrase", "")), CLEAR_PHRASE):
        raise PermissionError("Typed confirmation required before clearing PDF data.")


def validate_storage_path(project_root: Path, candidate: str | Path, label: str = "path") -> Path:
    root = Path(project_root).expanduser().resolve()
    target = Path(candidate).expanduser().resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must remain inside the BookRAG project directory.") from exc
    return target


def validate_ollama_url(value: str) -> str:
    raw = str(value or "").strip().rstrip("/")
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Ollama URL must be an HTTP(S) URL without embedded credentials.")
    host = parsed.hostname
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None

    allow_remote = _truthy(os.getenv(REMOTE_OLLAMA_ENV))
    allowlist = {h.strip().lower() for h in os.getenv(OLLAMA_ALLOWLIST_ENV, "").split(",") if h.strip()}
    if host.lower() in {"localhost", "ip6-localhost"} or ip and ip.is_loopback:
        return raw
    if host.lower() in allowlist:
        return raw
    if not allow_remote:
        raise ValueError(
            "Remote Ollama endpoints are disabled. Use 127.0.0.1/localhost or explicitly "
            "configure BOOKRAG_OLLAMA_ALLOWLIST / BOOKRAG_ALLOW_REMOTE_OLLAMA."
        )
    if ip and (ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified):
        raise ValueError("Ollama endpoint uses a disallowed network address.")
    return raw


def validate_query(value: str) -> str:
    question = str(value or "").strip()
    if len(question) > MAX_QUERY_CHARS:
        raise ValueError(f"Question is too long; maximum is {MAX_QUERY_CHARS} characters.")
    return question


def validate_pdf_payload(name: str, content: bytes) -> None:
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Uploaded file '{Path(name).name}' exceeds the 50 MB security limit.")
    if not content.startswith(b"%PDF-"):
        raise ValueError(f"Uploaded file '{Path(name).name}' is not a valid PDF payload.")
