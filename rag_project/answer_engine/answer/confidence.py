"""Confidence calculation for deterministic medical answers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass(frozen=True)
class ConfidenceComponents:
    """Components of answer confidence."""
    evidence_quality: float  # 0-1
    claim_support: float  # 0-1
    consistency: float  # 0-1
    clarity: float  # 0-1
    completeness: float  # 0-1
    specificity: float  # 0-1


@dataclass(frozen=True)
class ConfidenceResult:
    """Complete confidence assessment."""
    overall: float  # 0-1
    components: ConfidenceComponents
    level: str  # high, moderate, low, insufficient
    confidence_reasoning: List[str]


def calculate_answer_confidence(
    answer_data: Dict[str, Any],
) -> ConfidenceResult:
    """Calculate overall confidence for an answer."""
    claims = answer_data.get("claims", [])
    evidence = answer_data.get("evidence", [])
    reasoning_trace = answer_data.get("reasoning_trace", [])
    question = answer_data.get("question", "")
    answer = answer_data.get("direct_answer", "")
    
    # Calculate each component
    evidence_quality = _calculate_evidence_quality(evidence)
    claim_support = _calculate_claim_support(claims)
    consistency = _calculate_consistency(claims, evidence)
    clarity = _calculate_clarity(question, answer)
    completeness = _calculate_completeness(claims, evidence, question)
    specificity = _calculate_specificity(answer, evidence)
    
    # Calculate overall (weighted average)
    overall = (
        0.25 * evidence_quality +
        0.25 * claim_support +
        0.20 * consistency +
        0.10 * clarity +
        0.15 * completeness +
        0.05 * specificity
    )
    
    # Determine level
    if overall >= 0.75:
        level = "high"
    elif overall >= 0.50:
        level = "moderate"
    elif overall >= 0.30:
        level = "low"
    else:
        level = "insufficient"
    
    # Build reasoning
    reasoning = _build_confidence_reasoning(
        evidence_quality, claim_support, consistency, clarity, completeness, specificity
    )
    
    return ConfidenceResult(
        overall=overall,
        components=ConfidenceComponents(
            evidence_quality=evidence_quality,
            claim_support=claim_support,
            consistency=consistency,
            clarity=clarity,
            completeness=completeness,
            specificity=specificity,
        ),
        level=level,
        confidence_reasoning=reasoning,
    )


def _calculate_evidence_quality(evidence: List[Dict[str, Any]]) -> float:
    """Calculate evidence quality score."""
    if not evidence:
        return 0.05  # FIX: Return low but non-zero score to avoid full abstention
    
    scores = []
    for e in evidence:
        # Get quality indicators
        semantic = e.get("semantic_relevance", 0.0)
        lexical = e.get("lexical_relevance", 0.0)
        quality = e.get("page_quality", 0.0)
        extraction_quality = e.get("text_extraction_quality", 0.0)
        duplicate_penalty = e.get("duplicate_penalty", 0.0)
        contradiction_penalty = e.get("contradiction_penalty", 0.0)
        
        # Calculate composite score
        composite = (
            0.4 * semantic +
            0.3 * lexical +
            0.2 * quality +
            0.1 * extraction_quality
        )
        composite *= (1.0 - duplicate_penalty) * (1.0 - contradiction_penalty)
        
        scores.append(composite)
    
    return sum(scores) / len(scores) if scores else 0.0


def _calculate_claim_support(claims: List[Dict[str, Any]]) -> float:
    """Calculate claim support score."""
    if not claims:
        return 0.0
    
    supported_claims = 0
    total_claims = 0
    
    for claim in claims:
        status = claim.get("status", "UNSUPPORTED")
        support_ratio = claim.get("support_ratio", 0.0)
        
        # Check if claim is verifiable
        evidence = claim.get("evidence", [])
        if not evidence:
            continue
        
        total_claims += 1
        
        # Check support
        if status == "SUPPORTED" and support_ratio >= 0.8:
            supported_claims += 1
        elif status in {"SUPPORTED", "PARTIALLY_SUPPORTED"} and support_ratio >= 0.5:
            supported_claims += 0.5
        elif status == "UNSUPPORTED":
            continue
        else:
            supported_claims += 0.25
    
    return min(1.0, supported_claims / max(total_claims, 1))


def _calculate_consistency(claims: List[Dict[str, Any]], evidence: List[Dict[str, Any]]) -> float:
    """Calculate consistency score."""
    if not claims:
        return 1.0
    
    contradictions = sum(
        1 for claim in claims
        if claim.get("contradiction")
    )
    
    # Check evidence for conflicts
    evidence_conflicts = sum(
        1 for e in evidence
        if e.get("contradiction_penalty", 0.0) > 0.3
    )
    
    total_issues = contradictions + evidence_conflicts
    total_items = len(claims) + len(evidence)
    
    return max(0.0, 1.0 - (total_issues / max(total_items, 1)))


def _calculate_clarity(question: str, answer: str) -> float:
    """Calculate clarity score."""
    if not question or not answer:
        return 0.5
    
    # Check question clarity
    question_words = len(question.split())
    question_clarity = 0.5
    if question_words >= 3:
        question_clarity = 0.8
    if question_words >= 5:
        question_clarity = 1.0
    
    # Check answer clarity
    answer_words = len(answer.split())
    answer_clarity = 0.5
    if 10 <= answer_words <= 500:
        answer_clarity = 1.0
    elif answer_words < 10:
        answer_clarity = 0.3
    elif answer_words > 1000:
        answer_clarity = 0.7
    
    return (question_clarity + answer_clarity) / 2


def _calculate_completeness(
    claims: List[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
    question: str,
) -> float:
    """Calculate completeness score."""
    if not question:
        return 0.5
    
    # Check question type
    question_lower = question.lower()
    
    if any(term in question_lower for term in ("list", "all", "enumerate")):
        # Expect multiple items
        expected_items = 3
        actual_items = len(claims)
        return min(1.0, actual_items / expected_items)
    
    elif any(term in question_lower for term in ("compare", "vs", "versus")):
        # Expect comparison
        return min(1.0, len(claims) / 2)
    
    elif any(term in question_lower for term in ("cause", "mechanism", "pathway")):
        # Expect multiple steps
        expected_steps = 3
        actual_steps = sum(
            1 for c in claims if c.get("claim_type") in {"causal", "mechanism"}
        )
        return min(1.0, actual_steps / expected_steps)
    
    else:
        # General question
        if not claims:
            return 0.0
        
        # Check if answer addresses question
        supported_claims = sum(
            1 for c in claims
            if c.get("status") in {"SUPPORTED", "PARTIALLY_SUPPORTED"}
        )
        
        if supported_claims == 0:
            return 0.0
        
        return min(1.0, supported_claims / len(claims))


def _calculate_specificity(answer: str, evidence: List[Dict[str, Any]]) -> float:
    """Calculate specificity score."""
    if not answer:
        return 0.0
    
    # Check for specificity indicators
    specificity_indicators = [
        r"\d+(\.\d+)?\s*(mg|ml|g|kg)",
        r"\d+(\.\d+)?\s*(bpm|°c|°f)",
        r"\d+(\.\d+)?\s*(years?|months?|weeks?|days?)",
        r"\d+(\.\d+)?\s*%-",
        r"\d+(\.\d+)?-\d+(\.\d+)?",
        r"\[.*?\]",
    ]
    
    found_specific = 0
    for pattern in specificity_indicators:
        if re.search(pattern, answer, re.I):
            found_specific += 1
    
    # Check evidence for numeric/quantitative data
    has_quantitative_evidence = any(
        e.get("numeric_match") or e.get("table_match") or e.get("figure_match")
        for e in evidence
    )
    
    specificity_score = found_specific / len(specificity_indicators)
    
    if has_quantitative_evidence:
        specificity_score = max(specificity_score, 0.7)
    
    return specificity_score


def _build_confidence_reasoning(
    evidence_quality: float,
    claim_support: float,
    consistency: float,
    clarity: float,
    completeness: float,
    specificity: float,
) -> List[str]:
    """Build human-readable confidence reasoning."""
    reasoning = []
    
    if evidence_quality >= 0.7:
        reasoning.append("Strong evidence quality")
    elif evidence_quality >= 0.5:
        reasoning.append("Moderate evidence quality")
    else:
        reasoning.append("Limited evidence quality")
    
    if claim_support >= 0.7:
        reasoning.append("Claims well supported")
    elif claim_support >= 0.5:
        reasoning.append("Claims partially supported")
    else:
        reasoning.append("Claims insufficiently supported")
    
    if consistency >= 0.8:
        reasoning.append("High consistency")
    elif consistency >= 0.6:
        reasoning.append("Some inconsistencies")
    else:
        reasoning.append("Many inconsistencies")
    
    if clarity >= 0.7:
        reasoning.append("Clear question and answer")
    else:
        reasoning.append("Question or answer unclear")
    
    if completeness >= 0.7:
        reasoning.append("Comprehensive answer")
    elif completeness >= 0.5:
        reasoning.append("Partially comprehensive")
    else:
        reasoning.append("Answer may be incomplete")
    
    if specificity >= 0.7:
        reasoning.append("Specific information provided")
    else:
        reasoning.append("Answer may lack specificity")
    
    return reasoning


def determine_confidence_level(overall: float) -> str:
    """Determine confidence level from overall score."""
    if overall >= 0.75:
        return "high"
    elif overall >= 0.50:
        return "moderate"
    elif overall >= 0.30:
        return "low"
    else:
        return "insufficient"


def confidence_threshold_for_answer_type(answer_type: str) -> float:
    """Get minimum confidence threshold for answer type."""
    thresholds = {
        "definition": 0.50,
        "fact": 0.50,
        "numeric": 0.60,
        "dosage": 0.65,
        "list": 0.50,
        "comparison": 0.55,
        "causal": 0.60,
        "mechanism": 0.60,
        "management": 0.65,
        "diagnostic_criteria": 0.60,
        "prognosis": 0.55,
        "multi_hop": 0.60,
        "summary": 0.50,
    }
    
    return thresholds.get(answer_type, 0.60)


def is_answer_sufficient(
    overall_confidence: float,
    answer_type: str,
) -> bool:
    """Check if answer confidence meets threshold for answer type."""
    threshold = confidence_threshold_for_answer_type(answer_type)
    return overall_confidence >= threshold