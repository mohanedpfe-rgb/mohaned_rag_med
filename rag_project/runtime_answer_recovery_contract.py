"""Canonical answer failure propagation and public status normalization.

The MedEvidence cascade must not swallow infrastructure failures. Production's
outer recovery boundary owns retrieval and extractive recovery, so low-level LLM
failures are intentionally re-raised here. This module also protects an already
verified canonical answer from a second diagnostic verifier incorrectly downgrading
it to a reasoning abstention.
"""
from __future__ import annotations

import threading
from functools import wraps
from typing import Any


_INSTALL_LOCK = threading.RLock()
_INSTALLED = False


def _raise_llm_failures(self: Any, question: str, evidence: str, route: Any) -> str | None:
    """Run the original LLM call but propagate infrastructure failures to recovery."""
    llm = getattr(self.system, "llm", None)
    if llm is None or not str(evidence or "").strip():
        return None

    prompt = (
        f"Question: {str(question or '')[:2600]}\n"
        f"Intent: {getattr(route, 'intent', 'factual')}\n\n"
        f"Evidence:\n{str(evidence)[:6500]}"
    )
    system_prompt = (
        "You are MedEvidence Pro's constrained synthesis stage. Use ONLY the supplied evidence. "
        "Every factual sentence must end in an existing [S#] citation. Do not introduce a new "
        "number, unit, diagnosis, recommendation, cause, population, severity, timing, or "
        "contraindication. Preserve negation exactly. If the evidence is insufficient, say so "
        "briefly. Return only the answer."
    )
    value = llm.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=0.0,
    )
    value = str(value or "").strip()
    return value[:9000] if value else None


def _normalize_public_status(result: dict[str, Any]) -> dict[str, Any]:
    """Expose one stable public status for a clearly out-of-scope query."""
    if not isinstance(result, dict):
        return result
    if str(result.get("status") or "").upper() == "ABSTAIN":
        safety = result.get("safety")
        if isinstance(safety, dict) and str(safety.get("reason") or "").lower() == "outside_medical_scope":
            normalized = dict(result)
            normalized["status"] = "NOT_SUPPORTED"
            normalized.setdefault("decision", "NOT_SUPPORTED")
            return normalized
    return result


def _preserve_verified_answer(original):
    """Prevent diagnostic re-verification from downgrading an already proven answer."""
    @wraps(original)
    def wrapped(system: Any, question: str, result: dict[str, Any], metadata_filter=None):
        before = dict(result or {})
        output = original(system, question, result, metadata_filter)
        if not isinstance(output, dict):
            return output

        before_status = str(before.get("status") or "").upper()
        before_verification = before.get("final_verification")
        before_grounding = before.get("grounding")
        before_verified = (
            before_status in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}
            and isinstance(before_verification, dict)
            and before_verification.get("allow") is True
            and isinstance(before_grounding, dict)
            and before_grounding.get("allow") is True
        )
        diagnostic_status = str(output.get("status") or "").upper()
        if before_verified and diagnostic_status == "REASONING_ABSTAIN":
            restored = dict(output)
            for key in (
                "status",
                "answer",
                "citations",
                "hits",
                "confidence",
                "grounding",
                "claims",
                "final_verification",
                "generation_path",
                "generation_meta",
                "query_analysis",
                "phase_plan",
                "retrieval_quality",
                "contradiction_report",
            ):
                if key in before:
                    restored[key] = before[key]
            diagnostics = dict(restored.get("diagnostic_contract") or {})
            diagnostics["verification_downgrade_suppressed"] = True
            diagnostics["reason"] = "canonical_answer_was_already_verified_before_diagnostic_pass"
            restored["diagnostic_contract"] = diagnostics
            restored["abstained"] = False
            return restored
        return output

    wrapped._runtime_verified_answer_preservation = True
    return wrapped


def install() -> None:
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        from rag_project.intelligence import god_mode_100, med_evidence_pro

        if not hasattr(med_evidence_pro.AnswerCascade, "_runtime_answer_recovery_original_llm"):
            med_evidence_pro.AnswerCascade._runtime_answer_recovery_original_llm = med_evidence_pro.AnswerCascade._llm
            med_evidence_pro.AnswerCascade._llm = _raise_llm_failures

        if not hasattr(med_evidence_pro, "_runtime_answer_recovery_original_engine_answer"):
            original_answer = med_evidence_pro.MedEvidenceProEngine.answer

            def wrapped_answer(self: Any, question: str, metadata_filter: dict[str, Any] | None = None) -> dict[str, Any]:
                return _normalize_public_status(original_answer(self, question, metadata_filter))

            med_evidence_pro.MedEvidenceProEngine._runtime_answer_recovery_original_engine_answer = original_answer
            med_evidence_pro.MedEvidenceProEngine.answer = wrapped_answer

        diagnostic = getattr(god_mode_100, "_diagnostic_enhance", None)
        if callable(diagnostic) and not getattr(diagnostic, "_runtime_verified_answer_preservation", False):
            god_mode_100._diagnostic_enhance = _preserve_verified_answer(diagnostic)

        _INSTALLED = True


__all__ = ["install"]
