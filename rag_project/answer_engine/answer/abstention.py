"""Abstention handling for deterministic medical answers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class AbstentionReason(str, Enum):
    """Reasons for abstaining from answering."""
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNSUPPORTED_CLAIMS = "unsupported_claims"
    CONTRADICTION = "contradiction"
    SAFETY_CONCERNS = "safety_concerns"
    AMBIGUOUS_QUESTION = "ambiguous_question"
    UNKNOWN_ENTITY = "unknown_entity"
    NOT_MEDICAL = "not_medical"
    OUT_OF_SCOPE = "out_of_scope"
    LANGUAGE_MISMATCH = "language_mismatch"
    TEMPORAL_EXCLUSION = "temporal_exclusion"
    ENTITY_EXCLUSION = "entity_exclusion"
    SECTION_EXCLUSION = "section_exclusion"
    VERSION_MISMATCH = "version_mismatch"
    CONFIDENCE_BELOW_THRESHOLD = "confidence_below_threshold"
    INCOMPLETE_DECOMPOSITION = "incomplete_decomposition"
    CANNOT_DETERMINE = "cannot_determine"
    NO_VALID_VECTORS = "no_valid_vectors"
    QUERY_BLOCKED = "query_blocked"
    EVIDENCE_BLOCKED = "evidence_blocked"
    CLAIM_BLOCKED = "claim_blocked"
    REASONING_FAILED = "reasoning_failed"
    COMPILATION_FAILED = "compilation_failed"


@dataclass(frozen=True)
class AbstentionDecision:
    """Decision to abstain from answering."""
    should_abstain: bool
    reason: Optional[AbstentionReason]
    explanation: str
    details: Dict[str, Any]


def check_abstention(
    answer_data: Dict[str, Any],
    question: str,
    evidence: List[Dict[str, Any]],
    claims: List[Dict[str, Any]],
    confidence: float,
    confidence_threshold: float = 0.60,
) -> AbstentionDecision:
    """Check if answer should be abstained."""
    
    # Check for explicit abstention flag
    if answer_data.get("abstained"):
        return AbstentionDecision(
            should_abstain=True,
            reason=AbstentionReason(answer_data.get("abstention_reason", "unknown")),
            explanation=answer_data.get("abstention_reason", "Abstained"),
            details={},
        )
    
    # Check for no valid vectors
    if not evidence:
        return AbstentionDecision(
            should_abstain=True,
            reason=AbstentionReason.NO_VALID_VECTORS,
            explanation="No valid vectors found in the database",
            details={"vector_count": 0},
        )
    
    # Check for insufficient evidence
    if len(evidence) < 1:
        return AbstentionDecision(
            should_abstain=True,
            reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
            explanation="Insufficient evidence to answer",
            details={"evidence_count": len(evidence)},
        )
    
    # Check confidence threshold
    if confidence < confidence_threshold:
        return AbstentionDecision(
            should_abstain=True,
            reason=AbstentionReason.CONFIDENCE_BELOW_THRESHOLD,
            explanation=f"Confidence ({confidence:.2f}) below threshold ({confidence_threshold:.2f})",
            details={
                "confidence": confidence,
                "threshold": confidence_threshold,
            },
        )
    
    # Check for unsupported claims
    unsupported_claims = [
        c for c in claims
        if c.get("status") == "UNSUPPORTED" or not c.get("evidence")
    ]
    if len(unsupported_claims) > len(claims) * 0.5:
        return AbstentionDecision(
            should_abstain=True,
            reason=AbstentionReason.UNSUPPORTED_CLAIMS,
            explanation=f"More than 50% of claims unsupported",
            details={
                "unsupported_count": len(unsupported_claims),
                "total_claims": len(claims),
            },
        )
    
    # Check for contradictions
    contradictory_claims = [
        c for c in claims
        if c.get("contradiction")
    ]
    if contradictory_claims:
        return AbstentionDecision(
            should_abstain=True,
            reason=AbstentionReason.CONTRADICTION,
            explanation="Contradictory evidence found",
            details={
                "contradictory_claims": [c.get("id", "") for c in contradictory_claims],
            },
        )
    
    # Check question clarity
    if _is_ambiguous_question(question):
        return AbstentionDecision(
            should_abstain=True,
            reason=AbstentionReason.AMBIGUOUS_QUESTION,
            explanation="Question is too ambiguous to answer",
            details={"question": question},
        )
    
    # Check for safety concerns
    if _has_safety_concerns(claims):
        return AbstentionDecision(
            should_abstain=True,
            reason=AbstentionReason.SAFETY_CONCERNS,
            explanation="Answer may involve safety concerns",
            details={"claims_with_safety": [c.get("id", "") for c in claims if _is_safety_critical(c)]},
        )
    
    # Check for out of scope
    if _is_out_of_scope(question):
        return AbstentionDecision(
            should_abstain=True,
            reason=AbstentionReason.OUT_OF_SCOPE,
            explanation="Question is out of scope",
            details={"question": question},
        )
    
    # Check for non-medical question
    if not _is_medical_question(question):
        return AbstentionDecision(
            should_abstain=True,
            reason=AbstentionReason.NOT_MEDICAL,
            explanation="Question is not medical",
            details={"question": question},
        )
    
    # Answer appears sufficient
    return AbstentionDecision(
        should_abstain=False,
        reason=None,
        explanation="Answer has sufficient evidence",
        details={
            "evidence_count": len(evidence),
            "claims_count": len(claims),
            "confidence": confidence,
        },
    )


def _is_ambiguous_question(question: str) -> bool:
    """Check if question is too ambiguous."""
    if not question or len(question.strip()) < 5:
        return True
    
    # Check for vague terms
    vague_patterns = [
        r"^(what|how|which|who)\s+.*\?$",
        r"^(this|it|that|they)\s+.*\?$",
        r"^(can|could|would|will|do)\s+.*\?$",
    ]
    
    lowered = question.lower().strip()
    
    for pattern in vague_patterns:
        if re.match(pattern, lowered, re.I):
            # But check if it's actually specific
            if any(term in lowered for term in ("diabetes", "hypertension", "cancer", "heart", "blood")):
                return False
            return True
    
    return False


def _has_safety_concerns(claims: List[Dict[str, Any]]) -> bool:
    """Check if claims have safety concerns."""
    safety_keywords = [
        "fatal", "deadly", "lethal", "dangerous", "hazardous",
        "contraindicated", "should not", "avoid", "caution",
        "warning", "warning", "precaution", "emergency",
    ]
    
    for claim in claims:
        text = claim.get("text", "").lower()
        if any(keyword in text for keyword in safety_keywords):
            return True
    
    return False


def _is_safety_critical(claim: Dict[str, Any]) -> bool:
    """Check if a claim is safety-critical."""
    text = claim.get("text", "").lower()
    
    safety_indicators = [
        "fatal", "deadly", "lethal", "dangerous", "hazardous",
        "contraindicated", "should not", "avoid", "emergency",
        "immediate", "urgent", "critical",
    ]
    
    return any(indicator in text for indicator in safety_indicators)


def _is_out_of_scope(question: str) -> bool:
    """Check if question is out of scope."""
    out_of_scope_patterns = [
        r"price|cost|insurance|coverage|bill",
        r"law|legal|attorney|court|lawsuit",
        r"buy|purchase|order|avail",
        r"recommend|opinion|personal",
        r"my|your|mine|yours",
        r"friend|family|relative",
        r"chat|talk|conversation",
    ]
    
    lowered = question.lower()
    
    for pattern in out_of_scope_patterns:
        if re.search(pattern, lowered, re.I):
            return True
    
    return False


def _is_medical_question(question: str) -> bool:
    """Check if question appears to be medical."""
    medical_terms = [
        "disease", "condition", "symptom", "diagnosis",
        "treatment", "therapy", "medication", "drug",
        "doctor", "physician", "nurse", "hospital",
        "clinic", "patient", "medical", "health",
        "pain", "fever", "headache", "nausea",
        "cancer", "diabetes", "heart", "blood",
        "surgery", "operation", "procedure",
        "symptoms", "treatments", "medications",
        "cause", "cause", "mechanism", "pathophysiology",
        "prognosis", "outcome", "survival",
        "risk", "factor", "prevention", "preventive",
    ]
    
    lowered = question.lower()
    
    # Count medical terms
    medical_count = sum(1 for term in medical_terms if term in lowered)
    
    # Consider medical if at least 2 terms found
    return medical_count >= 2


def build_abstention_answer(
    reason: AbstentionReason,
    explanation: str,
    question: str = "",
) -> Dict[str, Any]:
    """Build an abstention answer structure."""
    return {
        "abstained": True,
        "abstention_reason": reason.value,
        "abstention_explanation": explanation,
        "question": question,
        "direct_answer": "",
        "claims": [],
        "evidence": [],
        "citations": [],
        "reasoning_trace": [],
        "confidence": 0.0,
        "confidence_components": {},
        "certainty": "insufficient",
        "contradictions": [],
        "unsupported_parts": [],
        "safety_status": "safe",
    }


def format_abstention_message(
    reason: AbstentionReason,
    explanation: str,
    language: str = "en",
) -> str:
    """Format a user-friendly abstention message."""
    messages = {
        AbstentionReason.INSUFFICIENT_EVIDENCE: {
            "en": "Insufficient evidence found to answer this question.",
            "fr": "Preuve insuffisante trouvée pour répondre à cette question.",
            "ar": "لم يتم العثور على أدلة كافية للإجابة على هذا السؤال.",
        },
        AbstentionReason.UNSUPPORTED_CLAIMS: {
            "en": "The available evidence does not support the claims needed to answer.",
            "fr": "Les preuves disponibles ne soutiennent pas les revendications nécessaires pour répondre.",
            "ar": "الأدلة المتاحة لا تدعم المطالبات اللازمة للإجابة.",
        },
        AbstentionReason.CONTRADICTION: {
            "en": "Contradictory evidence was found, making it impossible to determine a clear answer.",
            "fr": "Des preuves contradictoires ont été trouvées, rendant impossible la détermination d'une réponse claire.",
            "ar": "تم العثور على أدلة متناقضة، مما يجعل من المستحيل تحديد إجابة واضحة.",
        },
        AbstentionReason.SAFETY_CONCERNS: {
            "en": "The answer may involve safety concerns and requires professional medical consultation.",
            "fr": "La réponse peut impliquer des préoccupations de sécurité et nécessite une consultation médicale professionnelle.",
            "ar": "قد تتضمن الإجابة مخاوف تتعلق بالسلامة وتتطلب استشارة طبية مهنية.",
        },
        AbstentionReason.CONFIDENCE_BELOW_THRESHOLD: {
            "en": "The answer confidence is below the required threshold for reliable information.",
            "fr": "Confiance de la réponse en dessous du seuil requis pour des informations fiables.",
            "ar": "ثقة الإجابة أقل من الحد المطلوب للمعلومات الموثوقة.",
        },
        AbstentionReason.AMBIGUOUS_QUESTION: {
            "en": "The question is too ambiguous to provide a specific answer.",
            "fr": "La question est trop ambiguë pour fournir une réponse spécifique.",
            "ar": "السؤال غامض جداً لتوفير إجابة محددة.",
        },
        AbstentionReason.NO_VALID_VECTORS: {
            "en": "No valid medical evidence was found in the database.",
            "fr": "Aucune preuve médicale valide n'a été trouvée dans la base de données.",
            "ar": "لم يتم العثور على أي أدلة طبية صالحة في قاعدة البيانات.",
        },
    }
    
    language_messages = messages.get(reason, {})
    if language in language_messages:
        return language_messages[language]
    
    return language_messages.get("en", "Unable to provide an answer at this time.")