"""Evidence schemas for deterministic medical answers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class EvidenceQuality(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class SourceLocation:
    """Location of evidence in document hierarchy."""
    document_id: str
    document_version: str
    page: int
    page_numbers: List[int]
    printed_page: Optional[int] = None
    section: str = ""
    section_id: str = ""
    heading_hierarchy: List[str] = field(default_factory=list)
    paragraph_id: str = ""
    chunk_id: str = ""
    source_type: str = "text"  # text, table, figure
    table_id: Optional[str] = None
    figure_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_version": self.document_version,
            "page": self.page,
            "page_numbers": self.page_numbers,
            "printed_page": self.printed_page,
            "section": self.section,
            "section_id": self.section_id,
            "heading_hierarchy": self.heading_hierarchy,
            "paragraph_id": self.paragraph_id,
            "chunk_id": self.chunk_id,
            "source_type": self.source_type,
            "table_id": self.table_id,
            "figure_id": self.figure_id,
        }


@dataclass(frozen=True)
class EvidenceItem:
    """A piece of evidence with full provenance and quality scoring."""
    id: str
    text: str
    normalized_text: str
    source_location: SourceLocation
    retrieval_scores: Dict[str, float]  # vector, lexical, combined
    semantic_relevance: float  # 0-1
    lexical_relevance: float  # 0-1
    query_intent_match: float  # 0-1
    entity_match: bool
    section_relevance: float  # 0-1 based on heading relevance
    document_authority: float  # 0-1 based on version/edition
    proximity_score: float  # 0-1 based on distance from query
    numeric_match: bool
    table_match: bool
    figure_match: bool
    page_quality: float  # extraction quality
    text_extraction_quality: float
    duplicate_penalty: float = 0.0
    contradiction_penalty: float = 0.0
    cross_source_agreement: float = 1.0
    evidence_diversity: float = 1.0
    
    @property
    def composite_score(self) -> float:
        """Calculate composite evidence quality score."""
        base = (
            0.25 * self.semantic_relevance +
            0.20 * self.lexical_relevance +
            0.15 * self.query_intent_match +
            0.10 * self.entity_match +
            0.10 * self.section_relevance +
            0.05 * self.document_authority +
            0.05 * self.proximity_score +
            0.05 * self.numeric_match +
            0.05 * self.table_match +
            0.05 * self.figure_match
        )
        adjusted = (
            base *
            (1.0 - self.duplicate_penalty) *
            (1.0 - self.contradiction_penalty) *
            self.cross_source_agreement *
            self.evidence_diversity
        )
        return min(1.0, max(0.0, adjusted))
    
    @property
    def quality(self) -> EvidenceQuality:
        """Determine evidence quality level."""
        score = self.composite_score
        if score >= 0.75:
            return EvidenceQuality.HIGH
        elif score >= 0.50:
            return EvidenceQuality.MEDIUM
        elif score >= 0.30:
            return EvidenceQuality.LOW
        else:
            return EvidenceQuality.INSUFFICIENT
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "source_location": self.source_location.to_dict(),
            "retrieval_scores": self.retrieval_scores,
            "semantic_relevance": self.semantic_relevance,
            "lexical_relevance": self.lexical_relevance,
            "query_intent_match": self.query_intent_match,
            "entity_match": self.entity_match,
            "section_relevance": self.section_relevance,
            "document_authority": self.document_authority,
            "proximity_score": self.proximity_score,
            "numeric_match": self.numeric_match,
            "table_match": self.table_match,
            "figure_match": self.figure_match,
            "page_quality": self.page_quality,
            "text_extraction_quality": self.text_extraction_quality,
            "duplicate_penalty": self.duplicate_penalty,
            "contradiction_penalty": self.contradiction_penalty,
            "cross_source_agreement": self.cross_source_agreement,
            "evidence_diversity": self.evidence_diversity,
            "composite_score": self.composite_score,
            "quality": self.quality,
        }
