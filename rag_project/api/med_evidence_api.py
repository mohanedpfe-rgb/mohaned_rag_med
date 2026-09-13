"""HTTP adapter implementing the MedEvidence Pro API contract.

The offline/Streamlit installation remains HTTP-free. When the optional FastAPI
service is enabled, operations use JWT authentication and explicit scopes,
request bodies are bounded, input models reject unknown fields, and responses
expose only intentionally public data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _public_query_result(result: dict[str, Any]) -> dict[str, Any]:
    confidence = result.get("confidence") if isinstance(result.get("confidence"), dict) else {}
    trace = result.get("query_trace") if isinstance(result.get("query_trace"), dict) else {}
    timings = trace.get("timings_ms") if isinstance(trace.get("timings_ms"), dict) else {}
    citations = result.get("citations") if isinstance(result.get("citations"), list) else []
    return {
        "query_id": str(result.get("query_id") or ""),
        "status": str(result.get("status") or ""),
        "body": str(result.get("answer") or ""),
        "confidence": float(confidence.get("evidence_confidence", 0.0) or 0.0),
        "path": str(result.get("generation_path") or ""),
        "citations": citations,
        "latency_ms": float(timings.get("total", 0.0) or 0.0),
    }


def create_app(engine: Any, store: Any | None = None):
    try:
        from fastapi import Depends, FastAPI, HTTPException
        from pydantic import BaseModel, ConfigDict, Field, field_validator
        from starlette.responses import Response
    except ImportError as exc:
        raise RuntimeError("FastAPI is optional; install it to use the HTTP API") from exc

    from rag_project.api.security import APISettings, APISecurityError, bind_security, scoped_dependency
    from rag_project.security import validate_query

    settings = APISettings.from_env()
    if settings.environment == "production" and not settings.auth_enabled:
        raise RuntimeError("Production MedEvidence API cannot start with authentication disabled.")
    if settings.environment == "production" and any(not origin.startswith("https://") for origin in settings.allowed_origins):
        raise RuntimeError("Production MedEvidence API requires HTTPS CORS origins.")

    app = FastAPI(
        title="MedEvidence Pro",
        version="2.1",
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        openapi_url="/openapi.json" if settings.environment != "production" else None,
    )
    bind_security(app, engine=engine, settings=settings)

    class QueryRequest(BaseModel):
        model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
        query: str = Field(min_length=1, max_length=settings.max_query_chars)
        metadata_filter: dict[str, Any] | None = Field(default=None, max_length=32)

        @field_validator("query")
        @classmethod
        def query_is_safe(cls, value: str) -> str:
            try:
                return validate_query(value)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc

        @field_validator("metadata_filter")
        @classmethod
        def filter_is_bounded(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
            if value is None:
                return None
            for key, item in value.items():
                if not isinstance(key, str) or not key or len(key) > 80:
                    raise ValueError("metadata_filter contains an invalid key")
                if isinstance(item, str) and len(item) > 500:
                    raise ValueError("metadata_filter value is too long")
                if isinstance(item, (dict, list, tuple, set)):
                    raise ValueError("metadata_filter values must be scalar")
            return value

    class FeedbackRequest(BaseModel):
        model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
        query_id: str = Field(min_length=1, max_length=200)
        feedback_type: str = Field(min_length=1, max_length=80)
        feedback_text: str | None = Field(default=None, max_length=2000)

        @field_validator("query_id", "feedback_type")
        @classmethod
        def identifiers_are_safe(cls, value: str) -> str:
            if any(ord(char) < 32 for char in value):
                raise ValueError("control characters are not allowed")
            return value

        @field_validator("feedback_type")
        @classmethod
        def feedback_type_is_allowed(cls, value: str) -> str:
            normalized = value.lower()
            if normalized not in {"positive", "negative", "correction"}:
                raise ValueError("unsupported feedback_type")
            return normalized

    query_scope = scoped_dependency(settings, "query")
    feedback_scope = scoped_dependency(settings, "feedback")
    ops_scope = scoped_dependency(settings, "ops")

    @app.exception_handler(APISecurityError)
    async def security_error_handler(request, exc: APISecurityError):
        from fastapi.responses import JSONResponse
        rid = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.code, "message": exc.public_message, "request_id": rid},
            headers={"X-Request-ID": rid},
        )

    @app.post("/query", response_model=dict[str, Any], dependencies=[Depends(query_scope)])
    def query(payload: QueryRequest):
        try:
            result = engine.answer(payload.query, payload.metadata_filter)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Invalid query request.") from exc
        except Exception as exc:
            logger = getattr(engine, "logger", None)
            if logger is not None:
                logger.exception("API query failed")
            raise HTTPException(status_code=500, detail="Query processing failed.") from exc
        return _public_query_result(result)

    @app.post("/feedback", response_model=dict[str, str], dependencies=[Depends(feedback_scope)])
    def feedback(payload: FeedbackRequest):
        if store is None:
            raise HTTPException(status_code=503, detail="Feedback store unavailable.")
        try:
            store.add_feedback(payload.query_id, payload.feedback_type, payload.feedback_text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Invalid feedback.") from exc
        except Exception as exc:
            logger = getattr(engine, "logger", None)
            if logger is not None:
                logger.exception("API feedback failed")
            raise HTTPException(status_code=500, detail="Feedback processing failed.") from exc
        return {"status": "logged"}

    @app.get("/health")
    def health():
        checks = {
            "router": True,
            "retrieval": getattr(engine, "retrieval", None) is not None,
            "llm": getattr(getattr(engine, "system", None), "llm", None) is not None,
            "kb": getattr(engine, "knowledge", None) is not None,
        }
        return {"status": "healthy" if all(checks.values()) else "degraded"}

    @app.get("/metrics", dependencies=[Depends(ops_scope)])
    def metrics():
        if store is None:
            raise HTTPException(status_code=503, detail="Operations store unavailable.")
        from rag_project.intelligence.production_ops_strict import MetricsService
        return MetricsService(store).snapshot()

    @app.get("/alerts", dependencies=[Depends(ops_scope)])
    def alerts():
        if store is None:
            raise HTTPException(status_code=503, detail="Operations store unavailable.")
        from rag_project.intelligence.production_ops_strict import MetricsService
        return {"alerts": MetricsService(store).alerts()}

    @app.get("/metrics/prometheus", dependencies=[Depends(ops_scope)])
    def prometheus_metrics():
        if store is None:
            raise HTTPException(status_code=503, detail="Operations store unavailable.")
        from rag_project.intelligence.production_ops_strict import MetricsService
        from rag_project.observability.prometheus_exporter import export_metrics
        service = MetricsService(store)
        return Response(content=export_metrics(service.snapshot(), service.alerts()), media_type="text/plain; version=0.0.4")

    @app.get("/human-test-readiness", dependencies=[Depends(ops_scope)])
    def human_test_readiness():
        root = Path(getattr(getattr(engine, "settings", None), "project_root", Path.cwd()))
        from rag_project.quality.human_test_readiness import HumanTestReadinessGate
        return HumanTestReadinessGate(root).evaluate().as_dict()

    return app
