"""
Deterministic Medical Answer Engine

A zero-LLM, fully deterministic answer engine for medical PDF RAG.
Uses structured reasoning, claim verification, evidence graphs, and controlled language.
"""

from rag_project.answer_engine.schemas.answer import AnswerEnvelope, AnswerType, ConfidenceLevel
from rag_project.answer_engine.schemas.claims import Claim, ClaimStatus, ClaimEvidence
from rag_project.answer_engine.schemas.evidence import EvidenceItem, EvidenceQuality, SourceLocation
from rag_project.answer_engine.engine import DeterministicAnswerEngine

__all__ = [
    "AnswerEnvelope",
    "AnswerType",
    "ConfidenceLevel",
    "Claim",
    "ClaimStatus",
    "ClaimEvidence",
    "EvidenceItem",
    "EvidenceQuality",
    "SourceLocation",
    "DeterministicAnswerEngine",
]
