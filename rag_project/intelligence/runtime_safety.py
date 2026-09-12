"""Runtime safety adapters for cache freshness and deterministic generation recovery."""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from rag_project.intelligence.evidence_guard import grounding_decision, verify_claims
from rag_project.intelligence.final_answer_contract import verify_final_answer
from rag_project.intelligence.med_evidence_pro import RetrievalHit, SemanticCache

_SUCCESS = {"SUCCESS", "SUCCESS_WITH_WARNINGS"}


def ready_hits(system: Any, hits: Sequence[RetrievalHit]) -> list[RetrievalHit]:
    state_store = getattr(system, "state_store", None)
    if state_store is None:
        return []
    out: list[RetrievalHit] = []
    for hit in hits or ():
        meta = dict(getattr(hit, "metadata", {}) or {})
        if str(meta.get("index_state", "READY")).upper() != "READY":
            continue
        document_id = str(getattr(hit, "doc_id", "") or meta.get("document_id", ""))
        record = state_store.get_document(document_id) if document_id else None
        if not isinstance(record, dict) or not state_store.is_ready_status(record.get("status")):
            continue
        if str(record.get("index_state", "")).upper() != "READY":
            continue
        record_version = str(record.get("version_id") or "")
        hit_version = str(meta.get("version_id") or "")
        if record_version and hit_version and record_version != hit_version:
            continue
        out.append(hit)
    return out


def cached_result_is_fresh(system: Any, hits: Sequence[RetrievalHit]) -> bool:
    return bool(hits) and len(ready_hits(system, hits)) == len(hits)


def invalidate_stale_cache(system: Any, question: str) -> bool:
    settings = getattr(system, "settings", None)
    root = getattr(settings, "project_root", None)
    if root is None:
        return False
    cache = SemanticCache(root / "data" / "med_evidence_cache.sqlite3")
    payload = cache.get(question)
    if not payload:
        return False
    restored = cache.restore(payload)
    if cached_result_is_fresh(system, restored):
        return False
    cache.delete(question)
    return True


def _is_safe_success(result: dict[str, Any]) -> bool:
    """Require the public success envelope to contain verifiable evidence."""
    if str(result.get("status") or "").upper() not in _SUCCESS:
        return False
    if not str(result.get("answer") or "").strip():
        return False
    hits = list(result.get("hits") or [])
    if not hits:
        return False
    verification = result.get("verification") if isinstance(result.get("verification"), dict) else {}
    grounding = result.get("grounding") if isinstance(result.get("grounding"), dict) else {}
    if verification.get("allow") is False or grounding.get("allow") is False:
        return False
    markers = re.findall(r"\[S\d+\]", str(result.get("answer") or ""), flags=re.I)
    citations = result.get("citations") or []
    return bool(markers or citations)


def _force_safe_abstention(result: dict[str, Any], reason: str) -> dict[str, Any]:
    out = dict(result)
    out["status"] = "GENERATION_ABSTAIN"
    out["answer"] = "I could not safely verify the indexed evidence for this answer, so the answer was withheld."
    out["citations"] = []
    out["needs_review"] = True
    safety = dict(out.get("runtime_safety") or {})
    safety["rejected_success_contract"] = True
    safety["rejection_reason"] = reason
    out["runtime_safety"] = safety
    return out


def _verified_extractive_recovery(system: Any, result: dict[str, Any]) -> dict[str, Any] | None:
    original_verification = dict(result.get("verification") or {})
    # If the verifier blocked claims, this is an evidence-safety rejection, not
    # an infrastructure outage. Never convert it into a success automatically.
    if int(original_verification.get("blocked_claims", 0) or 0) > 0:
        return None
    hits = ready_hits(system, list(result.get("hits") or []))
    if not hits:
        return None
    compressed = str((result.get("evidence") or {}).get("compressed_context") or "").strip()
    if not compressed:
        return None
    lines = []
    for line in compressed.splitlines():
        cleaned = re.sub(r"^\s*[-*•]\s*", "", line).strip()
        if cleaned and re.search(r"\[S\d+\]", cleaned, flags=re.I):
            lines.append("- " + cleaned)
    fallback = "\n".join(lines[:6]).strip()
    if not fallback:
        return None
    blocks = [str(getattr(hit, "text", "") or "") for hit in hits]
    markers = [f"S{i+1}" for i in range(len(hits))]
    checks = list(verify_claims(fallback, blocks, markers))
    grounding = grounding_decision(checks, min_supported_ratio=0.70) if checks else {"allow": False, "supported_ratio": 0.0}
    try:
        final = dict(verify_final_answer(fallback, hits))
    except Exception:
        final = {"allow": False, "checked": True}
    if not grounding.get("allow") or not final.get("allow"):
        return None
    citations = []
    try:
        manager = getattr(system, "citation_manager", None)
        built = manager.build(hits) if manager else []
        citations = manager.validate(built, hits) if manager else []
    except Exception:
        citations = []
    recovered = dict(result)
    recovered["status"] = "SUCCESS_WITH_WARNINGS"
    recovered["answer"] = fallback
    recovered["hits"] = hits
    recovered["citations"] = citations
    recovered["generation_path"] = "PATH_A_VERIFIED_FALLBACK"
    recovered["generation_meta"] = {"attempted": True, "fallback": True, "recovered_from": "GENERATION_ABSTAIN"}
    recovered["verification"] = {
        "allow": True,
        "checked": True,
        "grounding": dict(grounding),
        "final_answer": final,
        "supported_ratio": float(grounding.get("supported_ratio", 0.0) or 0.0),
        "claim_count": len(checks),
        "blocked_claims": 0,
        "numeric_mismatch": False,
        "contradiction": dict(original_verification.get("contradiction") or {}),
    }
    recovered["grounding"] = dict(grounding)
    recovered["final_verification"] = final
    recovered["recovery"] = {"attempted": True, "pipeline_error": "RuntimeError", "grounded_extractive_fallback": True, "verification": "semantic_claim_verification"}
    phases = dict(recovered.get("phase_implementation") or {})
    phases["degraded_to_recovery"] = True
    recovered["phase_implementation"] = phases
    trace = dict(recovered.get("query_trace") or {})
    generation = dict(trace.get("generation") or {})
    generation.update({"status": "extractive", "path": "PATH_A_VERIFIED_FALLBACK"})
    trace["generation"] = generation
    recovered["query_trace"] = trace
    recovered["needs_review"] = True
    return recovered


def execute_with_runtime_safety(system: Any, question: str, answer_fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    invalidated = invalidate_stale_cache(system, question)
    result = dict(answer_fn() or {})
    hits = list(result.get("hits") or [])
    if hits and not cached_result_is_fresh(system, hits):
        invalidate_stale_cache(system, question)
        result = dict(answer_fn() or {})
        hits = list(result.get("hits") or [])
    if not cached_result_is_fresh(system, hits):
        result["hits"] = ready_hits(system, hits)
        if str(result.get("status") or "").upper() in _SUCCESS:
            result = _force_safe_abstention(result, "stale_or_non_ready_evidence")
    if not _is_safe_success(result) and str(result.get("status") or "").upper() in _SUCCESS:
        result = _force_safe_abstention(result, "incomplete_success_envelope")
    if str(result.get("status") or "").upper() == "GENERATION_ABSTAIN":
        recovered = _verified_extractive_recovery(system, result)
        if recovered is not None:
            result = recovered
    if str(result.get("status") or "").upper() in _SUCCESS and not _is_safe_success(result):
        result = _force_safe_abstention(result, "recovery_did_not_prove_success")
    result.setdefault("runtime_safety", {})
    result["runtime_safety"].update({"ready_evidence_enforced": True, "stale_cache_invalidated": invalidated, "success_contract_enforced": True})
    return result


__all__ = ["ready_hits", "cached_result_is_fresh", "invalidate_stale_cache", "execute_with_runtime_safety"]
