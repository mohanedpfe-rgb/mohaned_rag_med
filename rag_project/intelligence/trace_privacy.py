"""Privacy-safe telemetry normalization for internal RAG traces."""
from __future__ import annotations

import re
from typing import Any

_PHONENUMBER = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_LONG_ID = re.compile(r"\b\d{8,}\b")


def _redact_phone_match(match: re.Match[str]) -> str:
    candidate = match.group(0)
    if candidate.lstrip().startswith("+") or re.search(r"[\s().-]", candidate):
        return "[REDACTED_PHONE]"
    return candidate


def redact_sensitive_text(text: str) -> str:
    value = str(text or "")
    value = _EMAIL.sub("[REDACTED_EMAIL]", value)
    value = _PHONENUMBER.sub(_redact_phone_match, value)
    value = _LONG_ID.sub("[REDACTED_ID]", value)
    return value


def _redact_trace_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _redact_trace_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_trace_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_trace_value(item) for item in value)
    if isinstance(value, set):
        return {_redact_trace_value(item) for item in value}
    if isinstance(value, str):
        return redact_sensitive_text(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        redacted = redact_sensitive_text(str(value))
        return redacted if redacted != str(value) else value
    return value


def sanitize_trace(trace: dict[str, Any] | None) -> dict[str, Any]:
    """Return recursively redacted telemetry without changing user-facing answers."""
    if not trace:
        return {}
    return _redact_trace_value(dict(trace))


__all__ = ["redact_sensitive_text", "sanitize_trace"]
