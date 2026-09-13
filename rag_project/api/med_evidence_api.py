"""HTTP adapter implementing the MedEvidence Pro API contract.

The offline/Streamlit installation remains HTTP-free. When the optional FastAPI
service is enabled, every non-public operation is authenticated and scoped,
request bodies are bounded, input models reject unknown fields, and operational
endpoints return only intentionally public data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


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
    app = FastAPI(
        title="MedEvidence Pro",
        version="2.0",
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

    query_scope = scoped_dependency(settings, "query")
    feedback_scope = scoped_dependency(settings, "feedback")
    ops_scope = scoped_dependency(settings, "ops")

    @app.exception_handler(APISecurityError)
    async def security_error_handler(request, exc: APISecurityError):
        from fastapi.responses import JSONResponse
        rid = getattr(request.state, "request_id", "unknown")
        return JSONResponse(status_code=exc.status_code, content={"error": exc.code, "message": exc.public_message, "request_id": rid}, headers={"X-Request-ID": rid})

    @app.post("/query", response_model=dict[str, Any], dependencies=[Depends(query_scope)])
    def query(payload: QueryRequest):
        result = engine.answer(payload.query, payload.metadata_filter)
        return {
            "body": str(result.get("answer", "")),
            "confidence": result.get("confidence", {}).get("evidence_confidence", 0.0),
            "path": str(result.get("generation_path", "")),
            "citations": result.get("citations", []),
            "latency_ms": result.get("query_trace", {}).get("timings_ms", {}).get("total", 0.0),
            "metadata": {k: result.get(k) for k in ("query_id", "status", "route", "retrieval", "verification")},
        }

    @app.post("/feedback", response_model=dict[str, str], dependencies=[Depends(feedback_scope)])
    def feedback(payload: FeedbackRequest):
        if store is None:
            raise HTTPException(status_code=503, detail="feedback store unavailable")
        try:
            store.add_feedback(payload.query_id, payload.feedback_type, payload.feedback_text)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid feedback") from None
        return {"status": "logged"}

    @app.get("/health")
    def health():
        checks = {
            "router": True,
            "retrieval": getattr(engine, "retrieval", None) is not None,
            "llm": getattr(getattr(engine, "system", None), "llm", None) is not None,
            "kb": getattr(engine, "knowledge", None) is not None,
        }
        # Health is intentionally coarse: detailed dependency state is not public.
        return {"status": "healthy" if all(checks.values()) else "degraded"}

    @app.get("/metrics", dependencies=[Depends(ops_scope)])
    def metrics():
        if store is None:
            return {"accuracy_by_type": {}, "latency_percentiles": {}, "path_usage": {}, "error_rates": {}}
        from rag_project.intelligence.production_ops_strict import MetricsService
        return MetricsService(store).snapshot()

    @app.get("/alerts", dependencies=[Depends(ops_scope)])
    def alerts():
        if store is None:
            return {"alerts": ["operations_store_unavailable"]}
        from rag_project.intelligence.production_ops_strict import MetricsService
        return {"alerts": MetricsService(store).alerts()}

    @app.get("/metrics/prometheus", dependencies=[Depends(ops_scope)])
    def prometheus_metrics():
        if store is None:
            return Response(content="# MedEvidence Pro operations store unavailable\n", media_type="text/plain")
        from rag_project.intelligence.production_ops_strict import MetricsService
        from rag_project.observability.prometheus_exporter import export_metrics
        service = MetricsService(store)
        snapshot = service.snapshot()
        return Response(content=export_metrics(snapshot, service.alerts()), media_type="text/plain; version=0.0.4")

    @app.get("/human-test-readiness", dependencies=[Depends(ops_scope)])
    def human_test_readiness():
        root = Path(getattr(getattr(engine, "settings", None), "project_root", Path.cwd()))
        from rag_project.quality.human_test_readiness import HumanTestReadinessGate
        return HumanTestReadinessGate(root).evaluate().as_dict()

    return app
