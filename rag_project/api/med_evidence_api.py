"""HTTP adapter implementing the MedEvidence Pro API contract.

FastAPI remains optional so the offline/Streamlit installation does not require
an HTTP server. The adapter exposes operational readiness and Prometheus metrics
in addition to the plan's query/feedback/health/metrics endpoints.

The HTTP surface is intentionally fail-closed: an API key must be configured
before protected endpoints are exposed.
"""
from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Any

API_KEY_ENV = "MEDEVIDENCE_API_KEY"
MAX_QUERY_CHARS = 4000
MAX_QUERY_PAYLOAD_BYTES = 16 * 1024
MAX_FEEDBACK_CHARS = 2000
MAX_METADATA_KEYS = 32
MAX_METADATA_PAYLOAD_BYTES = 8192


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _validate_query_payload(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    question = str(payload.get("query") or "").strip()
    if not question:
        raise ValueError("query is required")
    if len(question) > MAX_QUERY_CHARS:
        raise ValueError(f"query exceeds the {MAX_QUERY_CHARS}-character limit")
    metadata_filter = _require_mapping(payload.get("metadata_filter"), "metadata_filter")
    if len(metadata_filter) > MAX_METADATA_KEYS:
        raise ValueError(f"metadata_filter exceeds the {MAX_METADATA_KEYS}-key limit")
    for key in metadata_filter:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metadata_filter keys must be non-empty strings")
    if len(repr(metadata_filter).encode("utf-8")) > MAX_METADATA_PAYLOAD_BYTES:
        raise ValueError("metadata_filter is too large")
    return question, metadata_filter


def _validate_feedback_payload(payload: dict[str, Any]) -> tuple[str, str, str | None]:
    query_id = str(payload.get("query_id") or "").strip()
    feedback_type = str(payload.get("feedback_type") or "").strip()
    feedback_text = payload.get("feedback_text")
    if not query_id or not feedback_type:
        raise ValueError("query_id and feedback_type are required")
    if len(query_id) > 256:
        raise ValueError("query_id is too long")
    if len(feedback_type) > 64:
        raise ValueError("feedback_type is too long")
    if feedback_text is not None:
        feedback_text = str(feedback_text)
        if len(feedback_text) > MAX_FEEDBACK_CHARS:
            raise ValueError(f"feedback_text exceeds the {MAX_FEEDBACK_CHARS}-character limit")
    return query_id, feedback_type, feedback_text


def create_app(engine: Any, store: Any | None = None):
    try:
        from fastapi import FastAPI, Header, HTTPException
    except ImportError as exc:
        raise RuntimeError("FastAPI is optional; install it to use the HTTP API") from exc

    api_key = os.getenv(API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(
            f"{API_KEY_ENV} must be set before starting the MedEvidence Pro HTTP API."
        )

    app = FastAPI(title="MedEvidence Pro", version="1.0")

    def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
        if not x_api_key or not hmac.compare_digest(x_api_key, api_key):
            raise HTTPException(status_code=401, detail="invalid API key")

    @app.post("/query", dependencies=[require_api_key])
    def query(payload: dict[str, Any]):
        try:
            question, metadata_filter = _validate_query_payload(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        result = engine.answer(question, metadata_filter)
        return {
            "body": result.get("answer", ""),
            "confidence": result.get("confidence", {}).get("evidence_confidence", 0.0),
            "path": result.get("generation_path", ""),
            "citations": result.get("citations", []),
            "latency_ms": result.get("query_trace", {}).get("timings_ms", {}).get("total", 0.0),
            "metadata": {k: result.get(k) for k in ("query_id", "status", "route", "retrieval", "verification")},
        }

    @app.post("/feedback", dependencies=[require_api_key])
    def feedback(payload: dict[str, Any]):
        if store is None:
            raise HTTPException(status_code=503, detail="feedback store unavailable")
        try:
            query_id, feedback_type, feedback_text = _validate_feedback_payload(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            store.add_feedback(query_id, feedback_type, feedback_text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "logged"}

    @app.get("/health")
    def health():
        checks = {
            "router": True,
            "retrieval": getattr(engine, "retrieval", None) is not None,
            "llm": getattr(getattr(engine, "system", None), "llm", None) is not None,
            "kb": getattr(engine, "knowledge", None) is not None,
        }
        return {"status": "healthy" if all(checks.values()) else "degraded", "components": checks}

    @app.get("/metrics", dependencies=[require_api_key])
    def metrics():
        if store is None:
            return {"accuracy_by_type": {}, "latency_percentiles": {}, "path_usage": {}, "error_rates": {}}
        from rag_project.intelligence.production_ops_strict import MetricsService
        return MetricsService(store).snapshot()

    @app.get("/alerts", dependencies=[require_api_key])
    def alerts():
        if store is None:
            return {"alerts": ["operations_store_unavailable"]}
        from rag_project.intelligence.production_ops_strict import MetricsService
        return {"alerts": MetricsService(store).alerts()}

    @app.get("/metrics/prometheus", dependencies=[require_api_key])
    def prometheus_metrics():
        if store is None:
            return "# MedEvidence Pro operations store unavailable\n"
        from rag_project.intelligence.production_ops_strict import MetricsService
        from rag_project.observability.prometheus_exporter import export_metrics
        service = MetricsService(store)
        snapshot = service.snapshot()
        return export_metrics(snapshot, service.alerts())

    @app.get("/human-test-readiness", dependencies=[require_api_key])
    def human_test_readiness():
        root = Path(getattr(getattr(engine, "settings", None), "project_root", Path.cwd()))
        from rag_project.quality.human_test_readiness import HumanTestReadinessGate
        return HumanTestReadinessGate(root).evaluate().as_dict()

    return app
