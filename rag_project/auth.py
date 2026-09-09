from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path

import streamlit as st


_AUTHENTICATED_KEY = "bookrag_authenticated"
_AUTH_ERROR_KEY = "bookrag_auth_error"


def _load_auth_environment() -> None:
    """Load only authentication settings from the local project environment."""
    if os.getenv("BOOKRAG_ADMIN_PASSWORD") or os.getenv("BOOKRAG_ADMIN_PASSWORD_HASH"):
        return
    try:
        from dotenv import dotenv_values

        root = Path(__file__).resolve().parents[1]
        for candidate in (root / ".env", Path.home() / ".bookrag.env"):
            if not candidate.is_file():
                continue
            values = dotenv_values(candidate)
            for key in ("BOOKRAG_ADMIN_PASSWORD", "BOOKRAG_ADMIN_PASSWORD_HASH"):
                value = values.get(key)
                if value and not os.getenv(key):
                    os.environ[key] = value
            if os.getenv("BOOKRAG_ADMIN_PASSWORD") or os.getenv("BOOKRAG_ADMIN_PASSWORD_HASH"):
                return
    except (ImportError, OSError):
        return


def _verify_password(password: str) -> bool:
    plain = os.getenv("BOOKRAG_ADMIN_PASSWORD", "")
    if plain:
        return hmac.compare_digest(password, plain)

    encoded = os.getenv("BOOKRAG_ADMIN_PASSWORD_HASH", "").strip()
    if not encoded:
        return False
    try:
        algorithm, salt_hex, digest_hex, work_factor = encoded.split("$", 3)
        if algorithm != "scrypt-v1":
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
        n = int(work_factor)
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=n,
            r=8,
            p=1,
            dklen=len(expected),
        )
        return hmac.compare_digest(derived, expected)
    except (TypeError, ValueError):
        return False


def password_hash(password: str, *, salt: bytes | None = None, n: int = 2**14) -> str:
    """Create a portable scrypt-v1 password verifier string for BOOKRAG_ADMIN_PASSWORD_HASH."""
    if not isinstance(password, str) or not password:
        raise ValueError("Password cannot be empty.")
    if len(password) > 512:
        raise ValueError("Password is too long.")
    if n < 2**10 or n & (n - 1):
        raise ValueError("scrypt work factor must be a power of two >= 1024.")
    selected_salt = salt or os.urandom(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=selected_salt,
        n=n,
        r=8,
        p=1,
        dklen=32,
    )
    return f"scrypt-v1${selected_salt.hex()}${digest.hex()}${n}"


def require_auth() -> bool:
    """Require a valid local admin password before exposing the Streamlit workspace."""
    _load_auth_environment()
    if st.session_state.get(_AUTHENTICATED_KEY) is True:
        return True

    configured = bool(
        os.getenv("BOOKRAG_ADMIN_PASSWORD")
        or os.getenv("BOOKRAG_ADMIN_PASSWORD_HASH")
    )
    if not configured:
        st.error(
            "BookRAG is locked because no admin password is configured. "
            "Set BOOKRAG_ADMIN_PASSWORD or BOOKRAG_ADMIN_PASSWORD_HASH before starting the app."
        )
        st.stop()
        return False

    st.title("BookRAG Medical")
    st.subheader("Authentication required")
    with st.form("bookrag_authentication"):
        password = st.text_input("Admin password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        if _verify_password(password):
            st.session_state[_AUTHENTICATED_KEY] = True
            st.session_state.pop(_AUTH_ERROR_KEY, None)
            st.rerun()
        st.session_state[_AUTH_ERROR_KEY] = "Invalid admin password."

    error = st.session_state.get(_AUTH_ERROR_KEY)
    if error:
        st.error(str(error))
    st.stop()
    return False


__all__ = ["password_hash", "require_auth"]
