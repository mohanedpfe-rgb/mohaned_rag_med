from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable


class APISecurityError(Exception):
    def __init__(self, status_code: int, code: str, message: str = "Request rejected") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.public_message = message


@dataclass(frozen=True)
class APISettings:
    auth_enabled: bool
    jwt_secret: str
    jwt_issuer: str
    jwt_audience: str
    jwt_clock_skew_seconds: int
    allowed_origins: tuple[str, ...]
    max_body_bytes: int
    max_query_chars: int
    rate_limit_per_minute: int
    rate_limit_burst: int
    trusted_proxies: tuple[str, ...]
    environment: str

    @classmethod
    def from_env(cls) -> "APISettings":
        environment = os.getenv("MEDEVIDENCE_ENV", os.getenv("ENVIRONMENT", "development")).strip().lower()
        auth_enabled = _env_bool("MEDEVIDENCE_AUTH_ENABLED", True)
        secret = os.getenv("MEDEVIDENCE_JWT_SECRET", "")
        if auth_enabled and len(secret.encode("utf-8")) < 32:
            raise RuntimeError("MEDEVIDENCE_JWT_SECRET must contain at least 32 bytes when API authentication is enabled")
        origins = _csv("MEDEVIDENCE_CORS_ORIGINS")
        if not origins or "*" in origins:
            if environment == "production":
                raise RuntimeError("Production requires explicit MEDEVIDENCE_CORS_ORIGINS; wildcard CORS is forbidden")
            origins = ("http://127.0.0.1:8501", "http://localhost:8501")
        return cls(
            auth_enabled=auth_enabled,
            jwt_secret=secret,
            jwt_issuer=os.getenv("MEDEVIDENCE_JWT_ISSUER", "medevidence-api").strip() or "medevidence-api",
            jwt_audience=os.getenv("MEDEVIDENCE_JWT_AUDIENCE", "medevidence-clients").strip() or "medevidence-clients",
            jwt_clock_skew_seconds=_bounded_int("MEDEVIDENCE_JWT_CLOCK_SKEW_SECONDS", 30, 0, 300),
            allowed_origins=tuple(origins),
            max_body_bytes=_bounded_int("MEDEVIDENCE_MAX_BODY_BYTES", 2 * 1024 * 1024, 1024, 25 * 1024 * 1024),
            max_query_chars=_bounded_int("MEDEVIDENCE_MAX_QUERY_CHARS", 4000, 1, 10000),
            rate_limit_per_minute=_bounded_int("MEDEVIDENCE_RATE_LIMIT_PER_MINUTE", 60, 1, 6000),
            rate_limit_burst=_bounded_int("MEDEVIDENCE_RATE_LIMIT_BURST", 10, 1, 1000),
            trusted_proxies=tuple(_csv("MEDEVIDENCE_TRUSTED_PROXIES")),
            environment=environment,
        )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _bounded_int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def _csv(name: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(value + padding)
    except Exception as exc:
        raise APISecurityError(401, "invalid_token") from exc


def _json_segment(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(_b64url_decode(value).decode("utf-8"))
    except Exception as exc:
        raise APISecurityError(401, "invalid_token") from exc
    if not isinstance(decoded, dict):
        raise APISecurityError(401, "invalid_token")
    return decoded


def issue_test_jwt(*, subject: str = "test-user", scopes: tuple[str, ...] = ("query",), secret: str, issuer: str, audience: str, expires_in: int = 300) -> str:
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": subject,
        "scope": " ".join(scopes),
        "iss": issuer,
        "aud": audience,
        "iat": now,
        "nbf": now,
        "exp": now + expires_in,
        "jti": secrets.token_urlsafe(12),
    }
    head = _b64url_encode_json(header)
    body = _b64url_encode_json(payload)
    signing_input = f"{head}.{body}".encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{head}.{body}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode('ascii')}"


def _b64url_encode_json(value: dict[str, Any]) -> str:
    raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def verify_jwt(token: str, settings: APISettings) -> dict[str, Any]:
    parts = str(token or "").split(".")
    if len(parts) != 3:
        raise APISecurityError(401, "invalid_token")
    header = _json_segment(parts[0])
    payload = _json_segment(parts[1])
    if header.get("alg") != "HS256" or header.get("typ") not in {"JWT", "jwt"}:
        raise APISecurityError(401, "invalid_token")
    signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
    expected = hmac.new(settings.jwt_secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    supplied = _b64url_decode(parts[2])
    if not hmac.compare_digest(expected, supplied):
        raise APISecurityError(401, "invalid_token")

    now = int(time.time())
    skew = settings.jwt_clock_skew_seconds
    _require_str_claim(payload, "sub")
    if payload.get("iss") != settings.jwt_issuer or payload.get("aud") != settings.jwt_audience:
        raise APISecurityError(401, "invalid_token")
    for claim in ("exp", "nbf", "iat"):
        if not isinstance(payload.get(claim), (int, float)):
            raise APISecurityError(401, "invalid_token")
    if now > float(payload["exp"]) + skew or now + skew < float(payload["nbf"]):
        raise APISecurityError(401, "invalid_token")
    if float(payload["iat"]) > now + skew:
        raise APISecurityError(401, "invalid_token")
    return payload


def _require_str_claim(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise APISecurityError(401, "invalid_token")
    return value


def _scopes(claims: dict[str, Any]) -> set[str]:
    raw = claims.get("scope", "")
    if isinstance(raw, str):
        return {item for item in raw.split() if item}
    if isinstance(raw, list):
        return {str(item) for item in raw if str(item)}
    return set()


def principal_from_claims(claims: dict[str, Any]) -> dict[str, Any]:
    scopes = _scopes(claims)
    roles = claims.get("roles", [])
    if isinstance(roles, str):
        roles = [roles]
    if not isinstance(roles, list):
        roles = []
    return {"subject": str(claims.get("sub")), "scopes": scopes, "roles": {str(item) for item in roles}}


def authorize(principal: dict[str, Any] | None, required_scope: str) -> None:
    if principal is None:
        raise APISecurityError(401, "authentication_required", "Authentication required")
    if required_scope not in principal.get("scopes", set()) and "admin" not in principal.get("roles", set()):
        raise APISecurityError(403, "insufficient_scope", "Insufficient permissions")


class FixedWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0, max_keys: int = 8192) -> None:
        self.limit = max(1, int(limit))
        self.window_seconds = max(1.0, float(window_seconds))
        self.max_keys = max(128, int(max_keys))
        self._lock = threading.Lock()
        self._state: dict[str, tuple[float, int]] = {}

    def allow(self, key: str, *, cost: int = 1) -> bool:
        now = time.monotonic()
        cost = max(1, int(cost))
        with self._lock:
            start, count = self._state.get(key, (now, 0))
            if now - start >= self.window_seconds:
                start, count = now, 0
            if count + cost > self.limit:
                return False
            self._state[key] = (start, count + cost)
            self._prune(now)
            return True

    def _prune(self, now: float) -> None:
        if len(self._state) <= self.max_keys:
            return
        expired = [key for key, (start, _) in self._state.items() if now - start >= self.window_seconds]
        for key in expired[: len(expired) // 2 or 1]:
            self._state.pop(key, None)
        while len(self._state) > self.max_keys:
            self._state.pop(next(iter(self._state)))


def client_identity(request: Any, principal: dict[str, Any] | None, settings: APISettings) -> str:
    subject = principal.get("subject") if principal else None
    if subject:
        return f"sub:{subject}"
    host = getattr(getattr(request, "client", None), "host", "unknown")
    forwarded = request.headers.get("x-forwarded-for") if hasattr(request, "headers") else None
    if forwarded and _trusted_proxy(host, settings.trusted_proxies):
        candidate = forwarded.split(",", 1)[0].strip()
        try:
            ipaddress.ip_address(candidate)
            host = candidate
        except ValueError:
            pass
    return f"ip:{host}"


def _trusted_proxy(host: str, trusted: tuple[str, ...]) -> bool:
    if not trusted:
        return False
    try:
        host_ip = ipaddress.ip_address(host)
    except ValueError:
        return host in trusted
    for item in trusted:
        try:
            if host_ip in ipaddress.ip_network(item, strict=False):
                return True
        except ValueError:
            if host == item:
                return True
    return False


def request_id(request: Any) -> str:
    incoming = str(request.headers.get("x-request-id", "")).strip() if hasattr(request, "headers") else ""
    if 8 <= len(incoming) <= 128 and all(ch.isalnum() or ch in "-_." for ch in incoming):
        return incoming
    return uuid.uuid4().hex


def safe_log(logger: Any, level: str, event: str, *, request_id_value: str, **fields: Any) -> None:
    clean: dict[str, Any] = {"event": event, "request_id": request_id_value}
    for key, value in fields.items():
        if key.lower() in {"authorization", "token", "secret", "password", "api_key", "jwt"}:
            continue
        text = str(value)
        clean[key] = text[:240]
    getattr(logger, level, logger.info)("api_security %s", json.dumps(clean, ensure_ascii=False, separators=(",", ":")))


def bind_security(app: Any, *, engine: Any, settings: APISettings) -> dict[str, Any]:
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.middleware.cors import CORSMiddleware

    limiter = FixedWindowRateLimiter(settings.rate_limit_per_minute + settings.rate_limit_burst)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Retry-After"],
        max_age=600,
    )

    class SecurityMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next: Callable[..., Any]):
            rid = request_id(request)
            request.state.request_id = rid
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > settings.max_body_bytes:
                        return JSONResponse(status_code=413, content={"error": "request_too_large", "request_id": rid}, headers={"X-Request-ID": rid})
                except ValueError:
                    return JSONResponse(status_code=400, content={"error": "invalid_content_length", "request_id": rid}, headers={"X-Request-ID": rid})
            public_path = request.url.path in {"/health", "/docs", "/openapi.json", "/redoc"}
            if not public_path and not limiter.allow(client_identity(request, getattr(request.state, "principal", None), settings)):
                return JSONResponse(status_code=429, content={"error": "rate_limited", "request_id": rid}, headers={"Retry-After": "60", "X-Request-ID": rid})
            try:
                response = await call_next(request)
            except APISecurityError as exc:
                response = JSONResponse(status_code=exc.status_code, content={"error": exc.code, "request_id": rid, "message": exc.public_message})
            except Exception:
                response = JSONResponse(status_code=500, content={"error": "internal_error", "request_id": rid, "message": "An internal error occurred"})
            response.headers["X-Request-ID"] = rid
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
            response.headers["Cache-Control"] = "no-store"
            if settings.environment == "production":
                response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            return response

    app.add_middleware(SecurityMiddleware)

    @app.middleware("http")
    async def request_logging(request: Request, call_next: Callable[..., Any]):
        rid = getattr(request.state, "request_id", uuid.uuid4().hex)
        logger = getattr(engine, "logger", None)
        start = time.perf_counter()
        if logger is not None:
            safe_log(logger, "info", "request_start", request_id_value=rid, method=request.method, path=request.url.path)
        response = await call_next(request)
        if logger is not None:
            safe_log(logger, "info", "request_end", request_id_value=rid, method=request.method, path=request.url.path, status=response.status_code, latency_ms=round((time.perf_counter() - start) * 1000, 2))
        return response

    return {"limiter": limiter}


def auth_dependency(settings: APISettings):
    from fastapi import Request

    async def dependency(request: Request) -> dict[str, Any] | None:
        if not settings.auth_enabled:
            request.state.principal = {"subject": "anonymous", "scopes": {"query", "feedback", "ops"}, "roles": {"admin"}}
            return request.state.principal
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise APISecurityError(401, "authentication_required", "Authentication required")
        claims = verify_jwt(token.strip(), settings)
        principal = principal_from_claims(claims)
        request.state.principal = principal
        return principal

    return dependency


def scoped_dependency(settings: APISettings, scope: str):
    from fastapi import Depends

    base = auth_dependency(settings)

    async def dependency(principal: dict[str, Any] | None = Depends(base)) -> dict[str, Any]:
        authorize(principal, scope)
        return principal or {}

    return dependency
