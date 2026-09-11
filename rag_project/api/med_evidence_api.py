"""HTTP adapter implementing the MedEvidence Pro API contract.

FastAPI remains optional so the offline/Streamlit installation does not require
an HTTP server. The adapter exposes operational readiness and Prometheus metrics
in addition to the plan's query/feedback/health/metrics endpoints.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def create_app(engine: Any, store: Any | None = None):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise RuntimeError("FastAPI is optional; install it to use the HTTP API") from exc

    app = FastAPI(title="MedEvidence Pro", version="1.0")

    @app.post("/query")
    def query(payload: dict[str, Any]):
        question = str(payload.get("query") or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="query is required")
        result = engine.answer(question, payload.get("metadata_filter"))
        return {
            "body": result.get("answer", ""),
            "confidence": result.get("confidence", {}).get("evidence_confidence", 0.0),
            "path": result.get("generation_path", ""),
            "citations": result.get("citations", []),
            "latency_ms": result.get("query_trace", {}).get("timings_ms", {}).get("total", 0.0),
            "metadata": {k: result.get(k) for k in ("query_id", "status", "route", "retrieval", "verification")},
        }

    @app.post("/feedback")
    def feedback(payload: dict[str, Any]):
        if store is None:
            raise HTTPException(status_code=503, detail="feedback store unavailable")
        query_id = str(payload.get("query_id") or "").strip()
        feedback_type = str(payload.get("feedback_type") or "").strip()
        if not query_id or not feedback_type:
            raise HTTPException(status_code=400, detail="query_id and feedback_type are required")
        try:
            store.add_feedback(query_id, feedback_type, payload.get("feedback_text"))
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

    @app.get("/metrics")
    def metrics():
        if store is None:
            return {"accuracy_by_type": {}, "latency_percentiles": {}, "path_usage": {}, "error_rates": {}}
        from rag_project.intelligence.production_ops_strict import MetricsService
        return MetricsService(store).snapshot()

    @app.get("/alerts")
    def alerts():
        if store is None:
            return {"alerts": ["operations_store_unavailable"]}
        from rag_project.intelligence.production_ops_strict import MetricsService
        return {"alerts": MetricsService(store).alerts()}

    @app.get("/metrics/prometheus")
    def prometheus_metrics():
        if store is None:
            return "# MedEvidence Pro operations store unavailable\n"
        from rag_project.observability.prometheus_exporter import render_prometheus
        return render_prometheus(store)

    @app.get("/human-test-readiness")
    def human_test_readiness():
        root = Path(getattr(getattr(engine, "settings", None), "project_root", Path.cwd()))
        from rag_project.quality.human_test_readiness import HumanTestReadinessGate
        return HumanTestReadinessGate(root).evaluate().as_dict()

    return app
