from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import time

import pytest

from rag_project.api.med_evidence_api import _public_query_result
from rag_project.api.security import (
    APISettings,
    APISecurityError,
    FixedWindowRateLimiter,
    authorize,
    issue_test_jwt,
    principal_from_claims,
    verify_jwt,
)


def _settings(secret: str = "s" * 48) -> APISettings:
    return APISettings(
        auth_enabled=True,
        jwt_secret=secret,
        jwt_issuer="issuer",
        jwt_audience="audience",
        jwt_clock_skew_seconds=0,
        allowed_origins=("https://client.example",),
        max_body_bytes=2_000_000,
        max_query_chars=4000,
        rate_limit_per_minute=60,
        rate_limit_burst=10,
        trusted_proxies=(),
        environment="production",
    )


def _sign_claims(settings: APISettings, payload: dict) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":"), allow_nan=True).encode()).rstrip(b"=").decode()
    signed = f"{header}.{body}".encode("ascii")
    signature = hmac.new(settings.jwt_secret.encode(), signed, hashlib.sha256).digest()
    return f"{header}.{body}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode()}"


def test_issue_and_verify_jwt_round_trip():
    settings = _settings()
    token = issue_test_jwt(subject="alice", scopes=("query", "feedback"), secret=settings.jwt_secret, issuer=settings.jwt_issuer, audience=settings.jwt_audience)
    claims = verify_jwt(token, settings)
    assert claims["sub"] == "alice"
    assert set(claims["scope"].split()) == {"query", "feedback"}


def test_jwt_rejects_signature_tampering():
    settings = _settings()
    token = issue_test_jwt(secret=settings.jwt_secret, issuer=settings.jwt_issuer, audience=settings.jwt_audience)
    parts = token.split(".")
    parts[1] = parts[1][:-1] + ("A" if parts[1][-1] != "A" else "B")
    with pytest.raises(APISecurityError):
        verify_jwt(".".join(parts), settings)


def test_jwt_rejects_wrong_issuer_or_audience():
    settings = _settings()
    token = issue_test_jwt(secret=settings.jwt_secret, issuer="wrong", audience=settings.jwt_audience)
    with pytest.raises(APISecurityError):
        verify_jwt(token, settings)


def test_jwt_rejects_expired_token():
    settings = _settings()
    now = int(time.time())
    token = _sign_claims(settings, {"sub": "u", "iss": "issuer", "aud": "audience", "iat": now - 100, "nbf": now - 100, "exp": now - 1})
    with pytest.raises(APISecurityError):
        verify_jwt(token, settings)


def test_jwt_rejects_nan_timestamps():
    settings = _settings()
    token = _sign_claims(settings, {"sub": "u", "iss": "issuer", "aud": "audience", "iat": time.time() - 1, "nbf": time.time() - 1, "exp": math.nan})
    with pytest.raises(APISecurityError):
        verify_jwt(token, settings)


def test_jwt_rejects_oversized_token_and_invalid_base64():
    settings = _settings()
    with pytest.raises(APISecurityError):
        verify_jwt("A" * 25000, settings)
    valid = issue_test_jwt(secret=settings.jwt_secret, issuer=settings.jwt_issuer, audience=settings.jwt_audience)
    head, body, signature = valid.split(".")
    with pytest.raises(APISecurityError):
        verify_jwt(f"{head}.!invalid!.{signature}", settings)


def test_authorization_requires_scope_or_admin_role():
    principal = principal_from_claims({"sub": "alice", "scope": "query"})
    authorize(principal, "query")
    with pytest.raises(APISecurityError, match="Insufficient permissions"):
        authorize(principal, "ops")
    admin = principal_from_claims({"sub": "admin", "scope": "", "roles": ["admin"]})
    authorize(admin, "ops")


def test_rate_limiter_is_bounded_and_resets():
    limiter = FixedWindowRateLimiter(limit=2, window_seconds=10, max_keys=128)
    assert limiter.allow("a")
    assert limiter.allow("a")
    assert not limiter.allow("a")
    limiter._state["a"] = (0.0, 0)
    assert limiter.allow("a")


def test_production_settings_reject_wildcard_cors(monkeypatch):
    monkeypatch.setenv("MEDEVIDENCE_ENV", "production")
    monkeypatch.setenv("MEDEVIDENCE_AUTH_ENABLED", "true")
    monkeypatch.setenv("MEDEVIDENCE_JWT_SECRET", "s" * 48)
    monkeypatch.setenv("MEDEVIDENCE_CORS_ORIGINS", "*")
    with pytest.raises(RuntimeError, match="Wildcard CORS"):
        APISettings.from_env()


def test_production_settings_require_strong_jwt_secret(monkeypatch):
    monkeypatch.setenv("MEDEVIDENCE_ENV", "production")
    monkeypatch.setenv("MEDEVIDENCE_AUTH_ENABLED", "true")
    monkeypatch.setenv("MEDEVIDENCE_JWT_SECRET", "too-short")
    monkeypatch.setenv("MEDEVIDENCE_CORS_ORIGINS", "https://client.example")
    with pytest.raises(RuntimeError, match="at least 32 bytes"):
        APISettings.from_env()


def test_production_rejects_invalid_boolean(monkeypatch):
    monkeypatch.setenv("MEDEVIDENCE_ENV", "production")
    monkeypatch.setenv("MEDEVIDENCE_AUTH_ENABLED", "definitely-not-a-bool")
    monkeypatch.setenv("MEDEVIDENCE_JWT_SECRET", "s" * 48)
    monkeypatch.setenv("MEDEVIDENCE_CORS_ORIGINS", "https://client.example")
    with pytest.raises(RuntimeError, match="must be a boolean"):
        APISettings.from_env()


def test_production_rejects_invalid_numeric_configuration(monkeypatch):
    monkeypatch.setenv("MEDEVIDENCE_ENV", "production")
    monkeypatch.setenv("MEDEVIDENCE_AUTH_ENABLED", "true")
    monkeypatch.setenv("MEDEVIDENCE_JWT_SECRET", "s" * 48)
    monkeypatch.setenv("MEDEVIDENCE_CORS_ORIGINS", "https://client.example")
    monkeypatch.setenv("MEDEVIDENCE_MAX_BODY_BYTES", "not-an-int")
    with pytest.raises(RuntimeError, match="must be an integer"):
        APISettings.from_env()


def test_production_rejects_non_https_cors(monkeypatch):
    monkeypatch.setenv("MEDEVIDENCE_ENV", "production")
    monkeypatch.setenv("MEDEVIDENCE_AUTH_ENABLED", "true")
    monkeypatch.setenv("MEDEVIDENCE_JWT_SECRET", "s" * 48)
    monkeypatch.setenv("MEDEVIDENCE_CORS_ORIGINS", "http://example.com")
    with pytest.raises(RuntimeError, match="HTTPS"):
        APISettings.from_env()


def test_development_allows_only_local_http_cors(monkeypatch):
    monkeypatch.setenv("MEDEVIDENCE_ENV", "development")
    monkeypatch.setenv("MEDEVIDENCE_AUTH_ENABLED", "true")
    monkeypatch.setenv("MEDEVIDENCE_JWT_SECRET", "s" * 48)
    monkeypatch.setenv("MEDEVIDENCE_CORS_ORIGINS", "http://example.com")
    with pytest.raises(RuntimeError, match="Non-local HTTP"):
        APISettings.from_env()


def test_production_rejects_malformed_trusted_proxy(monkeypatch):
    monkeypatch.setenv("MEDEVIDENCE_ENV", "production")
    monkeypatch.setenv("MEDEVIDENCE_AUTH_ENABLED", "true")
    monkeypatch.setenv("MEDEVIDENCE_JWT_SECRET", "s" * 48)
    monkeypatch.setenv("MEDEVIDENCE_CORS_ORIGINS", "https://client.example")
    monkeypatch.setenv("MEDEVIDENCE_TRUSTED_PROXIES", "not-a-network")
    with pytest.raises(RuntimeError, match="Invalid trusted proxy"):
        APISettings.from_env()


def test_disabled_auth_is_allowed_for_nonproduction_only(monkeypatch):
    monkeypatch.setenv("MEDEVIDENCE_ENV", "development")
    monkeypatch.setenv("MEDEVIDENCE_AUTH_ENABLED", "false")
    monkeypatch.setenv("MEDEVIDENCE_CORS_ORIGINS", "http://localhost:8501")
    settings = APISettings.from_env()
    assert settings.auth_enabled is False


def test_production_auth_cannot_be_disabled(monkeypatch):
    monkeypatch.setenv("MEDEVIDENCE_ENV", "production")
    monkeypatch.setenv("MEDEVIDENCE_AUTH_ENABLED", "false")
    monkeypatch.setenv("MEDEVIDENCE_CORS_ORIGINS", "https://client.example")
    with pytest.raises(RuntimeError, match="cannot be disabled"):
        APISettings.from_env()


def test_production_app_rejects_disabled_auth(monkeypatch):
    from rag_project.api import med_evidence_api

    monkeypatch.setenv("MEDEVIDENCE_ENV", "production")
    monkeypatch.setenv("MEDEVIDENCE_AUTH_ENABLED", "false")
    monkeypatch.setenv("MEDEVIDENCE_CORS_ORIGINS", "https://client.example")
    monkeypatch.setenv("MEDEVIDENCE_JWT_SECRET", "s" * 48)
    with pytest.raises(RuntimeError, match="cannot be disabled"):
        med_evidence_api.create_app(object())


def test_upload_boundary_rejects_path_components_and_non_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    from rag_project.security import validate_pdf_payload

    doc = fitz.open()
    doc.new_page()
    payload = doc.tobytes()
    doc.close()
    validate_pdf_payload("ok.pdf", payload)
    with pytest.raises(ValueError):
        validate_pdf_payload("../ok.pdf", payload)
    with pytest.raises(ValueError):
        validate_pdf_payload("ok.txt", payload)
    with pytest.raises(ValueError):
        validate_pdf_payload("ok.pdf", b"%PDF-1.7\n")


def test_ui_upload_delegates_to_secure_pdf_validator(tmp_path, monkeypatch):
    import rag_project.app.bookrag_ui as ui

    called = []
    monkeypatch.setattr(ui, "validate_pdf_payload", lambda name, payload: called.append((name, payload)))
    payload = b"%PDF-1.7\nsecure-placeholder"
    digest = hashlib.sha256(payload).hexdigest()
    target = tmp_path / f"book_{digest[:12]}.pdf"
    target.write_bytes(payload)
    assert ui.save_pdf(tmp_path, "book.pdf", payload) == digest
    assert called == [("book.pdf", payload)]


def test_ui_upload_rejects_invalid_pdf_before_writing(tmp_path, monkeypatch):
    import rag_project.app.bookrag_ui as ui

    monkeypatch.setattr(ui, "validate_pdf_payload", lambda name, payload: (_ for _ in ()).throw(ValueError("blocked")))
    with pytest.raises(ValueError, match="blocked"):
        ui.save_pdf(tmp_path, "evil.pdf", b"%PDF-1.7\nnot-a-real-pdf")
    assert not list(tmp_path.iterdir())


def test_query_response_contains_no_internal_retrieval_or_verification_details():
    result = _public_query_result({
        "query_id": "q1",
        "status": "ok",
        "answer": "safe answer",
        "confidence": {"evidence_confidence": 0.9},
        "generation_path": "verified",
        "citations": [{"document_id": "doc-1", "page": 3}],
        "query_trace": {"timings_ms": {"total": 12.5}},
        "retrieval": {"full_text": "SECRET INTERNAL TEXT"},
        "verification": {"debug": "SECRET INTERNAL STATE"},
        "system_prompt": "SECRET",
    })
    assert result["body"] == "safe answer"
    assert result["citations"] == [{"document_id": "doc-1", "page": 3}]
    assert "retrieval" not in result
    assert "verification" not in result
    assert "system_prompt" not in result


def test_public_query_projection_is_type_bounded():
    result = _public_query_result({"answer": None, "citations": "bad", "confidence": "bad", "query_trace": "bad"})
    assert result["body"] == ""
    assert result["citations"] == []
    assert result["confidence"] == 0.0
    assert result["latency_ms"] == 0.0
