from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict

from rag_project.app import rag_system as rag_system_module
from rag_project.app.resilient_rag import ResilientRAGSystem
from rag_project.ingestion import robust_ingestor
from rag_project.intelligence.god_mode import audit_god_mode_index
from rag_project.intelligence.god_mode_100 import enhanced_god_answer
from rag_project.intelligence.medical_safety import apply_medical_safety_policy
from rag_project.intelligence.production_contract import sanitize_trace, validate_feature_contract
from rag_project.intelligence.retrieval_replay import record as record_replay
from rag_project.generation.latency_budget import request_budget, exhausted, elapsed

# Keep the legacy production module self-contained.  Importing this constant
# from rag_project.application created a cycle because application.py defines
# MedEvidenceProductionRAGSystem by dynamically importing this module.
ANSWER_PIPELINE_AUTHORITY = "rag_project.intelligence.top_level_pipeline.complete_phases"

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
    return str(result.get("status") or "").strip().upper() not in {"ANSWER_UNAVAILABLE", "SYSTEM_NOT_READY", "NOT_SUPPORTED", "REASONING_ABSTAIN"}


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
    try:
        root = getattr(getattr(system, "settings", None), "project_root", None)
        if not root:
            return
        trace = result.get("query_trace") or {}
        status = str(result.get("status") or "").upper()
        replay = {
            "success": status in {"SUCCESS", "SUCCESS_WITH_WARNINGS"},
            "question": str(question or "")[:1200],
            "status": status,
            "failure_stage": "" if status in {"SUCCESS", "SUCCESS_WITH_WARNINGS"} else (str((trace.get("verification") or {}).get("status") or "") or "runtime_or_grounding"),
            "routing": trace.get("routing") or {},
            "retrieval": trace.get("retrieval") or {},
            "generation": trace.get("generation") or {},
            "verification": trace.get("verification") or {},
            "latency_ms": trace.get("timings_ms", {}).get("total") if isinstance(trace.get("timings_ms"), dict) else None,
        }
        record_replay(root, replay)
    except Exception:
        pass


class ProductionRAGSystem(ResilientRAGSystem):
    _certified_god_answer = enhanced_god_answer

    def __init__(self, settings=None):
        super().__init__(settings)
        self._production_feature_contract = validate_feature_contract()

    def _new_cancel_flag(self, document_id):
        flag = rag_system_module._IngestCancelFlag()
        with rag_system_module._INGEST_LOCK:
            rag_system_module._INGEST_CANCEL_FLAGS[document_id] = flag
        return flag

    def _remove_cancel_flag(self, document_id):
        with rag_system_module._INGEST_LOCK:
            rag_system_module._INGEST_CANCEL_FLAGS.pop(document_id, None)

    def _archive_duplicate_upload(self, pdf_path, document_id, result):
        source = Path(pdf_path)
        incoming_dir = getattr(self.settings, "incoming_dir", None)
        if incoming_dir is None:
            return result
        try:
            source.resolve().relative_to(Path(incoming_dir).resolve())
        except ValueError:
            return result
        if not source.is_file():
            return result
        try:
            digest = self._hash_file(source)
            archive = Path(self.settings.archive_dir)
            archive.mkdir(parents=True, exist_ok=True)
            target = archive / source.name
            if target.exists():
                target = archive / f"{source.stem}-duplicate-{digest[:12]}{source.suffix}"
            if target.exists():
                target = archive / f"{source.stem}-duplicate-{digest[:12]}-{document_id[:8]}{source.suffix}"
            source.replace(target)
            out = dict(result)
            out["archived_duplicate"] = str(target)
            return out
        except OSError as exc:
            out = dict(result)
            out["archive_warning"] = f"Duplicate was skipped but could not be archived: {type(exc).__name__}"
            return out

    def ingest_file(self, pdf_path):
        result = robust_ingestor.robust_ingest_file(self, pdf_path)
        if str((result or {}).get("status") or "").lower() == "skipped":
            result = self._archive_duplicate_upload(pdf_path, str((result or {}).get("document_id") or "unknown"), result)
        return result

    def ingest_directory(self, directory=None):
        source = Path(directory) if directory else self.settings.incoming_dir
        source.mkdir(parents=True, exist_ok=True)
        return [self.ingest_file(p) for p in sorted(source.glob("*.pdf"))]

    def cancel_all_ingests(self):
        with rag_system_module._INGEST_LOCK:
            count = 0
            for flag in rag_system_module._INGEST_CANCEL_FLAGS.values():
                if not flag.cancelled:
                    flag.cancel()
                    count += 1
            return count

    def _recovery_answer(self, question: str, metadata_filter: Dict[str, Any] | None, exc: Exception) -> dict[str, Any]:
        try:
            from rag_project.retrieval.metadata_filter import MetadataFilter
            where = MetadataFilter.build(metadata_filter)
        except Exception:
            where = None
        try:
            hits = _safe_list(self.retriever.retrieve(str(question or "").strip(), top_k=max(1, min(int(getattr(self.settings, "top_k", 8)), 12)), where=where))
            hits = [h for h in hits if h is not None]
        except Exception as retrieve_exc:
            _safe_exception_log(self, "Recovery retrieval failed")
            return {"status": "ANSWER_UNAVAILABLE", "answer": "I could not safely produce an answer from the indexed evidence right now.", "citations": [], "hits": [], "confidence": {"level": "none", "evidence_confidence": 0.0}, "recovery": {"attempted": True, "retrieval_failed": type(retrieve_exc).__name__, "pipeline_error": type(exc).__name__}, "pipeline_authority": ANSWER_PIPELINE_AUTHORITY}
        if not hits:
            return {"status": "NOT_SUPPORTED", "answer": "I could not find sufficient evidence in the indexed documents to answer this question.", "citations": [], "hits": [], "confidence": {"level": "none", "evidence_confidence": 0.0}, "recovery": {"attempted": True, "pipeline_error": type(exc).__name__}, "pipeline_authority": ANSWER_PIPELINE_AUTHORITY}
        try:
            from rag_project.intelligence.god_mode import _simple_extractive_answer
            answer = str(_simple_extractive_answer(str(question or ""), hits, max_sentences=6) or "").strip()
            answer_error = None
        except Exception as answer_exc:
            _safe_exception_log(self, "Recovery extractive answer failed")
            answer = ""
            answer_error = type(answer_exc).__name__
        if not answer:
            return {"status": "ANSWER_UNAVAILABLE", "answer": "The indexed evidence was retrieved, but it could not be safely converted into a grounded answer.", "citations": [], "hits": hits, "confidence": {"level": "low", "evidence_confidence": 0.0}, "recovery": {"attempted": True, "pipeline_error": type(exc).__name__, "extractive_failed": answer_error}, "pipeline_authority": ANSWER_PIPELINE_AUTHORITY}
        provenance = _verify_extractive_provenance(answer, hits)
        try:
            built = self.citation_manager.build(hits) or []
            citations = self.citation_manager.validate(built, hits) or []
        except Exception:
            citations = []
        if provenance.get("allow"):
            grounding = {"allow": True, "supported_ratio": 1.0, "method": "exact_extractive_provenance", "verified_items": provenance.get("details", [])}
            return {"status": "SUCCESS_WITH_WARNINGS", "answer": answer, "citations": citations, "hits": hits, "confidence": {"level": "high", "evidence_confidence": 1.0}, "grounding": grounding, "claims": provenance.get("details", []), "recovery": {"attempted": True, "pipeline_error": type(exc).__name__, "grounded_extractive_fallback": True, "verification": "exact_extractive_provenance"}, "query_trace": _recovery_trace(question, hits, exc, grounding, "exact_extractive_provenance"), "phase_implementation": {"canonical_pipeline_executed": False, "degraded_to_recovery": True, "phase_1": "preserved_from_primary_failure", "phase_2": "retrieval_completed", "phase_3": "extractive_fallback", "phase_4": "exact_source_provenance_verified", "phase_5": "visibility_preserved"}, "pipeline_authority": ANSWER_PIPELINE_AUTHORITY}
        try:
            from rag_project.intelligence.evidence_guard import verify_claims, grounding_decision
            blocks = [str(getattr(hit, "text", "") or "") for hit in hits]
            checks = _safe_list(verify_claims(answer, blocks, [f"S{i + 1}" for i in range(len(hits))]))
            ground = grounding_decision(checks, min_supported_ratio=0.60) if checks else {"allow": False, "supported_ratio": 0.0}
        except Exception:
            checks = []
            ground = {"allow": False, "supported_ratio": 0.0}
        if checks and ground.get("allow", False):
            ground = dict(ground)
            ground["method"] = "semantic_claim_verification"
            return {"status": "SUCCESS_WITH_WARNINGS", "answer": answer, "citations": citations, "hits": hits, "confidence": {"level": "medium", "evidence_confidence": float(ground.get("supported_ratio", 0.0) or 0.0)}, "grounding": ground, "claims": [getattr(c, "to_dict", lambda: {"claim": str(getattr(c, "claim", ""))})() for c in checks], "recovery": {"attempted": True, "pipeline_error": type(exc).__name__, "grounded_extractive_fallback": True, "verification": "semantic_claim_verification"}, "query_trace": _recovery_trace(question, hits, exc, ground, "semantic_claim_verification"), "phase_implementation": {"canonical_pipeline_executed": False, "degraded_to_recovery": True, "phase_1": "preserved_from_primary_failure", "phase_2": "retrieval_completed", "phase_3": "extractive_fallback", "phase_4": "semantic_claim_verification", "phase_5": "visibility_preserved"}, "pipeline_authority": ANSWER_PIPELINE_AUTHORITY}
        return {"status": "ANSWER_UNAVAILABLE", "answer": "The indexed evidence was retrieved, but the answer could not pass the grounding check safely.", "citations": [], "hits": hits, "confidence": {"level": "low", "evidence_confidence": float(ground.get("supported_ratio", 0.0) or 0.0)}, "recovery": {"attempted": True, "pipeline_error": type(exc).__name__, "grounding_failed": True, "provenance": provenance}, "query_trace": _recovery_trace(question, hits, exc, ground, "semantic_claim_verification"), "phase_implementation": {"canonical_pipeline_executed": False, "degraded_to_recovery": True, "phase_1": "primary_failed", "phase_2": "retrieval_completed", "phase_3": "extractive_fallback", "phase_4": "grounding_failed", "phase_5": "visibility_preserved"}, "pipeline_authority": ANSWER_PIPELINE_AUTHORITY}

    def answer(self, question: str, metadata_filter: Dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._production_feature_contract["all_resolved"]:
            return {"status": "SYSTEM_NOT_READY", "answer": "The production feature contract is incomplete; a grounded answer is disabled.", "citations": [], "hits": [], "confidence": {"level": "none", "evidence_confidence": 0.0}, "production_contract": self._production_feature_contract}
        memory = getattr(self, "conversation_memory", None)
        original_question = str(question or "").strip()
        if _is_explicit_followup(original_question) and memory is not None:
            try:
                question = memory.rewrite(original_question)
            except Exception:
                question = original_question
        else:
            question = original_question
        try:
            result = _safe_result(self._certified_god_answer(self, question, metadata_filter))
        except Exception as exc:
            _safe_exception_log(self, "Primary answer pipeline failed")
            result = self._recovery_answer(question, metadata_filter, exc)
        result = apply_medical_safety_policy(question, result, self.settings)
        try:
            result["query_trace"] = sanitize_trace(result.get("query_trace") or {})
        except Exception:
            pass
        if _should_store_in_history(result) and memory is not None:
            try:
                memory.add(original_question, result)
            except Exception:
                pass
        _record_answer_replay(self, original_question, result)
        return result
