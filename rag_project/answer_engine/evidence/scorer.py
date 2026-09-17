"""Evidence scorer for deterministic medical answers."""
from __future__ import annotations

import re
from typing import List, Dict, Any
from dataclasses import dataclass

from rag_project.answer_engine.evidence.normalizer import NormalizedEvidence, EvidenceNormalizer
from rag_project.utils.text_utils import meaningful_tokens


@dataclass
class ScoreBreakdown:
    """Detailed scoring breakdown for evidence."""
    semantic_relevance: float
    lexical_relevance: float
    query_intent_match: float
    entity_match: float
    section_relevance: float
    document_authority: float
    numeric_match: float
    table_match: float
    figure_match: float
    structure_bonus: float
    quality_score: float
    total: float


class EvidenceScorer:
    """Multi-dimensional evidence scoring."""
    
    def __init__(self, query: str = ""):
        self.query = query
        self.query_tokens = set(meaningful_tokens(query.lower())) if query else set()
        self.normalizer = EvidenceNormalizer()
    
    def score(self, evidence: NormalizedEvidence) -> ScoreBreakdown:
        """Calculate comprehensive score for evidence."""
        norm_text = evidence.normalized_text
        norm_lower = norm_text.lower()
        tokens = set(meaningful_tokens(norm_lower))
        
        # Semantic relevance (based on query overlap)
        semantic = len(self.query_tokens & tokens) / max(1, len(self.query_tokens)) if self.query_tokens else 0.5
        
        # Lexical relevance (exact word matches)
        query_words = set(self.query.lower().split())
        word_overlap = len(query_words & set(norm_lower.split())) / max(1, len(query_words)) if query_words else 0.5
        
        # Entity match
        entity_match = 1.0 if evidence.entities else 0.0
        
        # Section relevance (based on section headers)
        section_relevance = self._score_section(norm_text, evidence.evidence_item.source_location)
        
        # Document authority (based on metadata)
        doc_authority = evidence.evidence_item.document_authority
        
        # Numeric match
        numeric_match = 1.0 if evidence.numeric_values else 0.0
        
        # Table/figure match
        table_match = 1.0 if evidence.evidence_item.table_match else 0.0
        figure_match = 1.0 if evidence.evidence_item.figure_match else 0.0
        
        # Structure bonus
        structure_bonus = self._calculate_structure_bonus(norm_text, evidence.evidence_item)
        
        # Quality score
        quality = evidence.evidence_item.page_quality * 0.3 + evidence.evidence_item.text_extraction_quality * 0.3
        
        # Calculate total (weighted)
        total = (
            0.25 * semantic +
            0.20 * word_overlap +
            0.15 * entity_match +
            0.10 * section_relevance +
            0.05 * doc_authority +
            0.05 * numeric_match +
            0.05 * table_match +
            0.05 * figure_match +
            0.05 * structure_bonus +
            0.05 * quality
        )
        
        return ScoreBreakdown(
            semantic_relevance=semantic,
            lexical_relevance=word_overlap,
            query_intent_match=word_overlap * 1.2,
            entity_match=entity_match,
            section_relevance=section_relevance,
            document_authority=doc_authority,
            numeric_match=numeric_match,
            table_match=table_match,
            figure_match=figure_match,
            structure_bonus=structure_bonus,
            quality_score=quality,
            total=min(1.0, total),
        )
    
    def _score_section(self, text: str, location: Any) -> float:
        """Score based on section relevance to query."""
        section = str(location.section or "").lower()
        section_id = str(location.section_id or "").lower()
        
        section_keywords = {
            "diagnosis": ["diagnosis", "diagnostic", "criteria", "test", "evaluate"],
            "treatment": ["treatment", "therapy", "drug", "medication", "management"],
            "pathophysiology": ["pathophysiology", "mechanism", "pathway"],
            "epidemiology": ["epidemiology", "prevalence", "incidence"],
            "prognosis": ["prognosis", "outcome", "survival"],
        }
        
        for section_type, keywords in section_keywords.items():
            if section_type in section or any(kw in section for kw in keywords):
                return 0.9
        
        return 0.5
    
    def _calculate_structure_bonus(self, text: str, evidence_item: Any) -> float:
        """Calculate bonus for structured evidence."""
        bonus = 0.0
        
        # Table bonus
        if evidence_item.table_match:
            bonus += 0.2
        
        # Figure bonus
        if evidence_item.figure_match:
            bonus += 0.15
        
        # Chapter/section headers in text
        if "[RAG-STRUCTURE" in text or "[section:" in text:
            bonus += 0.1
        
        # Heading markers
        if "## " in text or "### " in text:
            bonus += 0.1
        
        return min(0.4, bonus)
    
    def score_batch(self, evidence: List[NormalizedEvidence]) -> List[ScoreBreakdown]:
        """Score multiple evidence items."""
        return [self.score(item) for item in evidence]
    
    def rank_by_score(self, evidence: List[NormalizedEvidence]) -> List[NormalizedEvidence]:
        """Rank evidence by score."""
        scored = [(item, self.score(item).total) for item in evidence]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scored]


def score_evidence(evidence: NormalizedEvidence, query: str = "") -> ScoreBreakdown:
    """Convenience function to score evidence."""
    scorer = EvidenceScorer(query)
    return scorer.score(evidence)
