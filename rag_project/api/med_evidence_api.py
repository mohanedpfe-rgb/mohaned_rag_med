"""Optional HTTP adapter implementing the MedEvidence Pro API contract.

FastAPI is imported lazily so the existing offline/Streamlit installation remains
valid. Install the optional API requirements to expose /query, /feedback, /health
and /metrics.
"""
from __future__ import annotations
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
        store.add_feedback(str(payload.get("query_id") or ""), str(payload.get("feedback_type") or ""), payload.get("feedback_text"))
        return {"status": "logged"}

    @app.get("/health")
    def health():
        checks = {"router": True, "retrieval": getattr(engine, "retrieval", None) is not None,
                  "llm": getattr(getattr(engine, "system", None), "llm", None) is not None,
                  "kb": getattr(engine, "knowledge", None) is not None}
        return {"status": "healthy" if all(checks.values()) else "degraded", "components": checks}

    @app.get("/metrics")
    def metrics():
        if store is None:
            return {"accuracy_by_type": {}, "latency_percentiles": {}, "path_usage": {}, "error_rates": {}}
        from rag_project.intelligence.production_ops import MetricsService
        return MetricsService(store).snapshot()

    return app
