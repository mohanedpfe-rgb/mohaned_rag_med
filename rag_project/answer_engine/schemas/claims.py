"""Claim schemas for deterministic medical reasoning."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class ClaimStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNCERTAIN = "UNCERTAIN"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class ClaimEvidence:
    """Evidence supporting a claim with provenance."""
    source_id: str
    text: str
    document_id: str
    document_version: str
    page: int
    page_numbers: List[int]
    section: str
    heading_hierarchy: List[str]
    source_type: str  # text, table, figure
    source_span: str  # exact text span
    normalized_text: str
    quality_score: float
    relevance_score: float
    support_strength: float  # 0-1 strength of support
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "text": self.text,
            "document_id": self.document_id,
            "document_version": self.document_version,
            "page": self.page,
            "page_numbers": self.page_numbers,
            "section": self.section,
            "heading_hierarchy": self.heading_hierarchy,
            "source_type": self.source_type,
            "source_span": self.source_span,
            "normalized_text": self.normalized_text,
            "quality_score": self.quality_score,
            "relevance_score": self.relevance_score,
            "support_strength": self.support_strength,
        }


@dataclass(frozen=True)
class Claim:
    """A verifiable medical claim with evidence."""
    id: str
    text: str
    normalized_text: str
    claim_type: str  # definition, fact, numeric, relationship, etc.
    entities: List[str]
    qualifiers: List[str]  # may, usually, typically, only, except, etc.
    context: Dict[str, Any]  # population, age, stage, etc.
    evidence: List[ClaimEvidence]
    status: ClaimStatus = ClaimStatus.UNSUPPORTED
    support_ratio: float = 0.0  # fraction of evidence that supports
    numeric_value: Optional[float] = None
    numeric_unit: Optional[str] = None
    numeric_range: Optional[tuple] = None
    contradiction: Optional[Dict[str, Any]] = None
    source_span: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "claim_type": self.claim_type,
            "entities": self.entities,
            "qualifiers": self.qualifiers,
            "context": self.context,
            "evidence": [e.to_dict() for e in self.evidence],
            "status": self.status,
            "support_ratio": self.support_ratio,
            "numeric_value": self.numeric_value,
            "numeric_unit": self.numeric_unit,
            "numeric_range": self.numeric_range,
            "contradiction": self.contradiction,
            "source_span": self.source_span,
        }
    
    @property
    def is_verifiable(self) -> bool:
        """Check if claim can be verified."""
        return len(self.evidence) > 0 and self.status != ClaimStatus.UNSUPPORTED
    
    @property
    def is_safely_verifiable(self) -> bool:
        """Check if claim can be safely used in answer."""
        if self.status == ClaimStatus.CONTRADICTED:
            return False
        if self.status == ClaimStatus.UNSUPPORTED:
            return False
        if self.support_ratio < 0.50:
            return False
        if self.contradiction:
            return False
        return True
