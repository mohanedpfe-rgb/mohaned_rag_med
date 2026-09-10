"""Canonical ingestion traceability and publication contract.

The existing robust ingestor remains responsible for extraction, chunking, embedding,
index writes, validation, and atomic READY publication.  This adapter adds a stable
request/trace envelope so every ingestion result is diagnosable without changing the
underlying storage semantics.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Callable

INGESTION_CONTRACT_VERSION = "2026-09-11-ingestion-contract-v1"


def new_ingestion_request_id() -> str:
    return f"ing-{uuid.uuid4().hex[:16]}"


def wrap_ingest_callable(original: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap the canonical ingestor while preserving its arguments and result shape."""
    if getattr(original, "_ingestion_contract_wrapped", False):
        return original

    def wrapped(system: Any, pdf_path: Any, *args: Any, **kwargs: Any) -> Any:
        request_id = new_ingestion_request_id()
        started = time.perf_counter()
        try:
            result = original(system, pdf_path, *args, **kwargs)
        except Exception as exc:
            if getattr(system, "logger", None) is not None:
                system.logger.exception(
                    "Ingestion request failed request_id=%s file=%s",
                    request_id,
                    pdf_path,
                )
            raise
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        if isinstance(result, dict):
            result.setdefault("ingestion_request_id", request_id)
            result.setdefault("ingestion_trace_version", INGESTION_CONTRACT_VERSION)
            result.setdefault("ingestion_elapsed_ms", elapsed_ms)
            result.setdefault(
                "ingestion_contract",
                {
                    "request_id": request_id,
                    "version": INGESTION_CONTRACT_VERSION,
                    "canonical_callable": "rag_project.ingestion.robust_ingestor.robust_ingest_file",
                    "atomic_publication": True,
                    "lease_fencing": True,
                    "post_write_validation": True,
                },
            )
        return result

    wrapped._ingestion_contract_wrapped = True
    wrapped._ingestion_contract_original = original
    wrapped.__name__ = getattr(original, "__name__", "wrapped_ingest")
    wrapped.__doc__ = getattr(original, "__doc__", None)
    return wrapped


def install() -> None:
    """Install traceability on the canonical robust ingestor exactly once."""
    from rag_project.ingestion import robust_ingestor

    if not getattr(robust_ingestor.robust_ingest_file, "_ingestion_contract_wrapped", False):
        robust_ingestor.robust_ingest_file = wrap_ingest_callable(robust_ingestor.robust_ingest_file)


__all__ = ["INGESTION_CONTRACT_VERSION", "new_ingestion_request_id", "wrap_ingest_callable", "install"]
