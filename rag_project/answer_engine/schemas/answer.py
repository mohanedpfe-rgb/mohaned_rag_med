"""Answer envelope schema for deterministic medical answers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class AnswerType(str, Enum):
    DEFINITION = "definition"
    FACT = "fact"
    NUMERIC = "numeric"
    LIST = "list"
    TABLE = "table"
    COMPARISON = "comparison"
    RELATIONSHIP = "relationship"
    CAUSE = "cause"
    MECHANISM = "mechanism"
    DIAGNOSTIC_CRITERIA = "diagnostic_criteria"
    MANAGEMENT = "management"
    PROGNOSIS = "prognosis"
    MULTI_HOP = "multi_hop"
    SUMMARY = "summary"
    FOLLOW_UP = "follow_up"
    CROSS_REFERENCE = "cross_reference"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class AnswerEnvelope:
    """Complete answer with structured metadata and provenance."""
    question: str
    intent: str
    answer_type: AnswerType
    direct_answer: str
    claims: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    reasoning_trace: List[Dict[str, Any]]
    confidence: float
    confidence_components: Dict[str, float]
    certainty: ConfidenceLevel
    contradictions: List[Dict[str, Any]]
    unsupported_parts: List[Dict[str, Any]]
    safety_status: str
    abstained: bool = False
    abstention_reason: Optional[str] = None
    question_decomposition: Optional[List[Dict[str, Any]]] = None
    subanswers: Optional[Dict[str, Any]] = None
    entity_resolution: Optional[Dict[str, Any]] = None
    document_version: Optional[str] = None
    answer_version: str = "1.0"
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "intent": self.intent,
            "answer_type": self.answer_type,
            "direct_answer": self.direct_answer,
            "claims": self.claims,
            "evidence": self.evidence,
            "citations": self.citations,
            "reasoning_trace": self.reasoning_trace,
            "confidence": self.confidence,
            "confidence_components": self.confidence_components,
            "certainty": self.certainty,
            "contradictions": self.contradictions,
            "unsupported_parts": self.unsupported_parts,
            "safety_status": self.safety_status,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "question_decomposition": self.question_decomposition,
            "subanswers": self.subanswers,
            "entity_resolution": self.entity_resolution,
            "document_version": self.document_version,
            "answer_version": self.answer_version,
        }
    
    @property
    def has_sufficient_evidence(self) -> bool:
        """Check if answer has sufficient supporting evidence."""
        if self.abstained:
            return False
        supported_claims = sum(1 for c in self.claims if c.get("status") == "SUPPORTED")
        return supported_claims > 0 and self.confidence >= 0.60
