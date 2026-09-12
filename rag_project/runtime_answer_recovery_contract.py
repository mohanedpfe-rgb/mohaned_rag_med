"""Canonical answer failure propagation and public status normalization.

The MedEvidence cascade must not swallow infrastructure failures.  Production's
outer recovery boundary (ProductionRAGSystem._primary_answer) owns retrieval and
extractive recovery, so low-level LLM failures are intentionally re-raised here.
This also normalizes an out-of-scope safety decision to the public NOT_SUPPORTED
contract used by the high-level behavior suite.
"""
from __future__ import annotations

import threading
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


def install() -> None:
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        from rag_project.intelligence import med_evidence_pro

        if not hasattr(med_evidence_pro.AnswerCascade, "_runtime_answer_recovery_original_llm"):
            med_evidence_pro.AnswerCascade._runtime_answer_recovery_original_llm = med_evidence_pro.AnswerCascade._llm
            med_evidence_pro.AnswerCascade._llm = _raise_llm_failures

        if not hasattr(med_evidence_pro, "_runtime_answer_recovery_original_engine_answer"):
            original_answer = med_evidence_pro.MedEvidenceProEngine.answer

            def wrapped_answer(self: Any, question: str, metadata_filter: dict[str, Any] | None = None) -> dict[str, Any]:
                return _normalize_public_status(original_answer(self, question, metadata_filter))

            med_evidence_pro.MedEvidenceProEngine._runtime_answer_recovery_original_engine_answer = original_answer
            med_evidence_pro.MedEvidenceProEngine.answer = wrapped_answer

        _INSTALLED = True


__all__ = ["install"]
