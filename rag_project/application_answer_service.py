from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from rag_project.canonical_runtime import ANSWER_AUTHORITY
from rag_project.intelligence.cloud_hybrid import CloudConfig, create_hybrid_router
from rag_project.intelligence.entity_coverage import score_entity_coverage
from rag_project.intelligence.med_evidence_pro import enhanced_med_evidence_answer
from rag_project.intelligence.production_contract_v2 import apply_contract, build_request_context
from rag_project.intelligence.production_ops_strict import OperationsStore
from rag_project.intelligence.runtime_safety import execute_with_runtime_safety
from rag_project.intelligence.semantic_cache import install as install_semantic_cache
from rag_project.retrieval.query_rewriter import QueryRewriter
from rag_project.retrieval.ready_only_retriever import ReadyOnlyRetriever

ACTIVE_ANSWER_PIPELINE_AUTHORITY = ANSWER_AUTHORITY


def detect_answer_language(question: str) -> tuple[str, float]:
    text = str(question or "")
    arabic = len(re.findall(r"[\u0600-\u06ff]", text))
    latin = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]", text))
    french_markers = sum(
        1
        for marker in (
            "qu'est-ce", "quelle", "quel", "quels", "quelles", "diabète",
            "traitement", "mécanisme", "contre-indication", "fréquence", "définition",
        )
        if re.search(rf"\b{re.escape(marker)}\b", text.casefold())
    )
    if arabic > 0 and arabic >= max(2, latin):
        return "ar", round(min(1.0, 0.75 + arabic / max(20, len(text)) * 0.25), 3)
    if french_markers:
        return "fr", min(1.0, 0.80 + 0.05 * min(french_markers, 4))
    if latin:
        return "en", 0.90
    return "unknown", 0.10


def normalize_public_answer_path(result: dict[str, Any]) -> dict[str, Any]:
    if str(result.get("generation_path") or "").upper() != "PATH_HYBRID_FALLBACK":
        return result
    verification = result.get("verification") if isinstance(result.get("verification"), dict) else {}
    if str(result.get("status") or "").upper() not in {"SUCCESS", "SUCCESS_WITH_WARNINGS"} or verification.get("allow") is not True:
        return result
    normalized = dict(result)
    normalized["generation_path"] = "PATH_A_VERIFIED_FALLBACK"
    metadata = dict(normalized.get("generation_meta") or {})
    metadata.update({"attempted": True, "fallback": True, "recovered": True, "internal_path": "PATH_HYBRID_FALLBACK"})
    normalized["generation_meta"] = metadata
    recovery = dict(normalized.get("recovery") or {})
    recovery.update({"attempted": True, "grounded_extractive_fallback": True, "verification": "semantic_claim_verification"})
    normalized["recovery"] = recovery
    phases = dict(normalized.get("phase_implementation") or {})
    phases["degraded_to_recovery"] = True
    normalized["phase_implementation"] = phases
    trace = dict(normalized.get("query_trace") or {})
    generation = dict(trace.get("generation") or {})
    generation.update({"path": "PATH_A_VERIFIED_FALLBACK", "internal_path": "PATH_HYBRID_FALLBACK"})
    trace["generation"] = generation
    normalized["query_trace"] = trace
    return normalized


def _seed_answer_contract(result: dict[str, Any], question: str, metadata_filter: dict[str, Any] | None):
    verification = result.get("verification") if isinstance(result.get("verification"), dict) else {}
    retrieval = result.get("retrieval") if isinstance(result.get("retrieval"), dict) else {}
    route = dict(result.get("route") or {}) if isinstance(result.get("route"), dict) else {}
    evidence = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
    contradiction = verification.get("contradiction") if isinstance(verification.get("contradiction"), dict) else {}
    detected_language, language_confidence = detect_answer_language(question)
    route.setdefault("language", detected_language)
    route.setdefault("language_confidence", language_confidence)
    result["route"] = route
    result.setdefault("query_analysis", route)
    result.setdefault("phase_plan", route)
    result.setdefault("rewritten_question", str(question or "").strip())
    result.setdefault("answer_plan", {"selected_path": result.get("generation_path", "")})
    result.setdefault("retrieval_quality", {
        "tier": retrieval.get("tier"), "candidate_count": retrieval.get("candidate_count", len(result.get("hits") or [])),
        "final_hits": len(result.get("hits") or []), "early_exit": retrieval.get("early_exit", False),
        "cache_hit": retrieval.get("cache_hit", False), "evidence_coverage": verification.get("supported_ratio", 0.0),
        "self_corrections": 0,
    })
    result.setdefault("grounding", verification.get("grounding", {}))
    result.setdefault("final_verification", verification.get("final_answer", {}))
    result.setdefault("claims", result.get("provenance", {}).get("claims", []))
    result.setdefault("contradiction_report", contradiction)
    if "entity_coverage" not in result:
        result["entity_coverage"] = score_entity_coverage(str(question or ""), list(result.get("hits") or []), route.get("entities", []) or [])
    result.setdefault("confidence_calibration", {})
    result.setdefault("adaptive_retrieval_budget", {"tier": retrieval.get("tier"), "cache_hit": retrieval.get("cache_hit", False)})
    recovery = result.get("recovery") if isinstance(result.get("recovery"), dict) else {}
    if str(result.get("status") or "").upper() == "SUCCESS_WITH_WARNINGS" and recovery.get("grounded_extractive_fallback"):
        result["generation_path"] = "PATH_A_VERIFIED_FALLBACK"
        generation_meta = dict(result.get("generation_meta") or {})
        generation_meta.update({"attempted": True, "fallback": True, "recovered": True})
        result["generation_meta"] = generation_meta
    return result, verification, retrieval, route, evidence, detected_language, language_confidence


def _apply_execution_visibility(result: dict[str, Any], metadata_filter: dict[str, Any] | None, verification: dict[str, Any], retrieval: dict[str, Any], detected_language: str, language_confidence: float) -> None:
    recovery = result.get("recovery") if isinstance(result.get("recovery"), dict) else {}
    result["canonical_pipeline_executed"] = bool(result.get("canonical_pipeline_executed", not bool(recovery.get("attempted"))))
    result["evidence_first"] = bool(result.get("evidence_first", bool(result.get("hits"))))
    result["document_aware"] = bool(result.get("document_aware", bool(metadata_filter) or bool((result.get("retrieval") or {}).get("document_aware"))))
    result["god_mode_100"] = False
    result["pipeline_authority"] = ACTIVE_ANSWER_PIPELINE_AUTHORITY
    result["implementation_authority"] = ACTIVE_ANSWER_PIPELINE_AUTHORITY
    phases = result.get("phases") if isinstance(result.get("phases"), dict) else {}
    result["phase_implementation"] = {
        "phase_0_safety_gate": {"status": phases.get("phase_0_safety_gate", "complete")},
        "phase_1_query_understanding": {"status": phases.get("phase_1_query_router", "complete"), "authority": ACTIVE_ANSWER_PIPELINE_AUTHORITY},
        "phase_2_retrieval_precision": {"status": phases.get("phase_2a_multi_tier_retrieval", retrieval.get("tier", "complete")), "hits": len(result.get("hits") or []), "authority": ACTIVE_ANSWER_PIPELINE_AUTHORITY},
        "phase_3_two_stage_generation": {"status": phases.get("phase_4_answer_cascade", result.get("generation_path", "complete")), "generation_path": result.get("generation_path"), "authority": ACTIVE_ANSWER_PIPELINE_AUTHORITY},
        "phase_4_verification": {"status": phases.get("phase_5_active_verification", "complete"), "checked": verification.get("checked", False), "final_answer_checked": bool(result.get("final_verification", {}).get("checked", verification.get("checked", False))), "claim_count": verification.get("claim_count", 0), "blocked_claims": verification.get("blocked_claims", 0), "authority": ACTIVE_ANSWER_PIPELINE_AUTHORITY},
        "phase_5_intelligence_visibility": {"status": "complete", "authority": ACTIVE_ANSWER_PIPELINE_AUTHORITY, "canonical_answer_authority": ACTIVE_ANSWER_PIPELINE_AUTHORITY, "implementation": "rag_project.intelligence.med_evidence_pro.enhanced_med_evidence_answer", "signals_present": bool(result.get("canonical_pipeline_executed") and result.get("pipeline_authority")), "canonical_executed": bool(result.get("canonical_pipeline_executed"))},
        "degraded_to_recovery": bool(recovery.get("grounded_extractive_fallback")),
    }
    trace = result.get("query_trace") if isinstance(result.get("query_trace"), dict) else {}
    trace["pipeline_authority"] = ACTIVE_ANSWER_PIPELINE_AUTHORITY
    trace["language"] = detected_language
    trace["language_confidence"] = language_confidence
    if result.get("generation_path"):
        generation = dict(trace.get("generation") or {})
        generation["path"] = result["generation_path"]
        trace["generation"] = generation
    result["query_trace"] = trace
    result.setdefault("evidence_summary", {"claim_count": result.get("evidence", {}).get("claim_count", 0)})
    result["runtime_safety"] = {**dict(result.get("runtime_safety") or {}), "ready_evidence_enforced": True}


def _history_pairs(memory: Any) -> list[tuple[str, str]]:
    history = list(getattr(memory, "history", []) or []) if memory is not None else []
    pairs: list[tuple[str, str]] = []
    for item in history:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.append((str(item[0] or ""), str(item[1] or "")))
        elif isinstance(item, dict):
            question = str(item.get("question") or item.get("user") or item.get("query") or "").strip()
            answer = str(item.get("answer") or item.get("assistant") or item.get("response") or "").strip()
            if question:
                pairs.append((question, answer))
    return pairs


def _rewrite_followup(system: Any, question: str) -> tuple[str, bool]:
    memory = getattr(system, "conversation_memory", None)
    clean = str(question or "").strip()
    if memory is None or not clean or not re.match(r"^(?:what about|how about|and|also|then|it|this|that|these|those|they|them|its|their|et|puis|et puis|ou|و|ثم)\b", clean, flags=re.I | re.UNICODE):
        return clean, False
    history = _history_pairs(memory)
    if not history:
        return clean, False
    try:
        rewritten = QueryRewriter.rewrite(clean, history=history, llm=None).strip()
    except Exception:
        rewritten = clean
    return rewritten or clean, rewritten.casefold() != clean.casefold()


def _set_active_scope(system: Any, metadata_filter: dict[str, Any] | None) -> None:
    setattr(system, "_active_metadata_filter", dict(metadata_filter or {}))


def _clear_active_scope(system: Any) -> None:
    try:
        delattr(system, "_active_metadata_filter")
    except AttributeError:
        setattr(system, "_active_metadata_filter", None)


def _invalidate_cache_on_corpus_change(system: Any) -> None:
    settings = getattr(system, "settings", None)
    root = Path(getattr(settings, "project_root", Path.cwd()))
    state_db = root / "data" / "ingestion.sqlite3"
    try:
        stat = state_db.stat()
        marker = (int(stat.st_mtime_ns), int(stat.st_size))
    except OSError:
        marker = (0, 0)
    previous = getattr(system, "_answer_service_corpus_generation", None)
    cache_db = root / "data" / "med_evidence_cache.sqlite3"
    if previous is None or marker != previous:
        try:
            from rag_project.intelligence.semantic_cache import SemanticRetrievalCache
            SemanticRetrievalCache(cache_db).delete_all()
        except Exception:
            pass
        setattr(system, "_answer_service_corpus_generation", marker)


def answer(system: Any, question: str, metadata_filter: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    clean_question = str(question or "").strip()
    memory = getattr(system, "conversation_memory", None)
    context = build_request_context(
        clean_question,
        history=_history_pairs(memory),
        metadata_filter=metadata_filter,
    )
    canonical_question = context.canonical_question or clean_question
    _set_active_scope(system, metadata_filter)
    try:
        _invalidate_cache_on_corpus_change(system)
        result = execute_with_runtime_safety(
            system,
            canonical_question,
            lambda: enhanced_med_evidence_answer(system, canonical_question, metadata_filter),
        )
    finally:
        _clear_active_scope(system)
    result = normalize_public_answer_path(result)
    result, verification, retrieval, _route, _evidence, detected_language, language_confidence = _seed_answer_contract(
        result, clean_question, metadata_filter
    )
    result["rewritten_question"] = canonical_question
    result["route"]["is_follow_up"] = bool(context.is_followup)
    _apply_execution_visibility(result, metadata_filter, verification, retrieval, detected_language, language_confidence)
    # This is the only place where the request/evidence/answer contract is applied.
    # No runtime installer is allowed to wrap the production answer path.
    result = apply_contract(result, context)
    memory = getattr(system, "conversation_memory", None)
    result_status = str(result.get("status") or "").upper()
    if memory is not None and result_status in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
        try:
            memory.add(clean_question, result)
        except Exception:
            pass
    try:
        settings = getattr(system, "settings", None)
        root = getattr(settings, "project_root", None)
        if root is not None:
            OperationsStore(root / "data" / "med_evidence_ops.sqlite3").record_result(str(result.get("query_id") or f"q-{time.time_ns()}"), clean_question, result, (time.perf_counter() - started) * 1000.0)
    except Exception:
        pass
    return result


answer._final_followup_guard = True


def install_runtime_adapters(system: Any) -> Any:
    retriever = getattr(system, "retriever", None)
    if retriever is not None and not isinstance(retriever, ReadyOnlyRetriever):
        system.retriever = ReadyOnlyRetriever(retriever)
    install_semantic_cache(system)
    try:
        system.cloud_hybrid = create_hybrid_router(CloudConfig.from_env(), getattr(system.settings, "project_root", Path.cwd()))
    except Exception:
        system.cloud_hybrid = None
    return system


__all__ = ["ACTIVE_ANSWER_PIPELINE_AUTHORITY", "answer", "detect_answer_language", "install_runtime_adapters", "normalize_public_answer_path"]
