from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from rag_project.app import rag_system as rag_system_module
from rag_project.app.resilient_rag import ResilientRAGSystem
from rag_project.ingestion import versioned_ingestor
from rag_project.ingestion import robust_ingestor
from rag_project.intelligence.god_mode_100 import enhanced_god_answer
from rag_project.intelligence.medical_safety import apply_medical_safety_policy
from rag_project.intelligence.production_contract import validate_feature_contract
from rag_project.intelligence.production_contract_v2 import CONTRACT_VERSION as PRODUCTION_CONTRACT_VERSION
from rag_project.intelligence.retrieval_replay import record as record_replay
from rag_project.intelligence.runtime_safety import execute_with_runtime_safety
from rag_project.intelligence.trace_privacy import sanitize_trace
from rag_project.generation.latency_budget import elapsed, exhausted, request_budget

if TYPE_CHECKING:
    from rag_project.application import ANSWER_PIPELINE_AUTHORITY

ANSWER_PIPELINE_AUTHORITY = "rag_project.intelligence.top_level_pipeline.complete_phases"

_PRODUCTION_HARD_GATES = {
    "claim_evidence_matrix": "rag_project.intelligence.evidence_entailment.build_claim_evidence_matrix",
    "confidence_calibration": "rag_project.intelligence.confidence_calibration.calibrate_confidence",
    "medical_safety_policy": "rag_project.intelligence.medical_safety.apply_medical_safety_policy",
    "privacy_safe_trace": "rag_project.intelligence.trace_privacy.sanitize_trace",
    "canonical_ingestion": "rag_project.ingestion.versioned_ingestor.ingest_version_safely",
}

_FOLLOWUP_PATTERN = re.compile(r"\b(it|this|that|they|them|those|these|the latter|the former|what about|how about)\b|^(and|also|then|et|puis|و|ثم)\b|^و(?=\S)|\b(ça|cela|celui|celle|et le|et la)\b", re.I | re.UNICODE)


def _is_explicit_followup(question: str) -> bool:
    return bool(_FOLLOWUP_PATTERN.search(str(question or "").strip()))


def _safe_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except (TypeError, ValueError):
        return []


def _safe_logger(system: Any) -> Any:
    logger = getattr(system, "logger", None)
    return logger if logger is not None and callable(getattr(logger, "exception", None)) else logging.getLogger(__name__)


def _safe_exception_log(system: Any, message: str) -> None:
    try:
        _safe_logger(system).exception(message)
    except Exception:
        pass


def _safe_result(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _should_store_in_history(result: Any) -> bool:
    if not isinstance(result, dict) or not str(result.get("answer") or "").strip():
        return False
    return str(result.get("status") or "").strip().upper() in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}


def _norm_provenance_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().casefold()
    text = re.sub(r"\[(?:section|table|figure|source)\s*:[^\]]*\]", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def _extract_marker_lines(answer: str) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for raw in str(answer or "").splitlines():
        line = raw.strip()
        match = re.search(r"\[S(\d+)\]\s*$", line, re.I)
        if not match:
            continue
        text = re.sub(r"\s*\[S\d+\]\s*$", "", line, flags=re.I)
        text = re.sub(r"^[-*•\s]+", "", text).strip()
        if text:
            rows.append((text, int(match.group(1))))
    return rows


def _verify_extractive_provenance(answer: str, hits: list[Any]) -> dict[str, Any]:
    rows = _extract_marker_lines(answer)
    if not rows:
        return {"allow": False, "supported_ratio": 0.0, "method": "exact_extractive_provenance", "matched": 0, "total": 0}
    matched = 0
    details: list[dict[str, Any]] = []
    for text, source_no in rows:
        idx = source_no - 1
        if idx < 0 or idx >= len(hits):
            details.append({"source": source_no, "matched": False, "reason": "source_out_of_range"})
            continue
        source_text = _norm_provenance_text(getattr(hits[idx], "text", ""))
        needle = _norm_provenance_text(text)
        ok = bool(needle and len(needle) >= 12 and needle in source_text)
        matched += int(ok)
        details.append({"source": source_no, "matched": ok, "reason": "exact_source_substring" if ok else "source_mismatch"})
    ratio = matched / max(1, len(rows))
    return {"allow": matched == len(rows) and matched > 0, "supported_ratio": ratio, "method": "exact_extractive_provenance", "matched": matched, "total": len(rows), "details": details}


def _recovery_trace(question: str, hits: list[Any], exc: Exception, grounding: dict[str, Any], method: str) -> dict[str, Any]:
    return {"mode": "grounded_recovery", "question": str(question or "").strip(), "primary_pipeline_error": type(exc).__name__, "retrieval": {"status": "completed", "final_hits": len(hits)}, "generation": {"status": "extractive", "method": method}, "verification": {"status": "completed", "method": method, "supported_ratio": float(grounding.get("supported_ratio", 0.0) or 0.0)}}


def _record_answer_replay(system: Any, question: str, result: dict[str, Any]) -> None: