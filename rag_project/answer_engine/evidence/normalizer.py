"""Evidence normalizer for deterministic medical answers."""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from rag_project.answer_engine.schemas.evidence import EvidenceItem, SourceLocation, EvidenceQuality
from rag_project.retrieval.hybrid_retriever import RetrievalHit
from rag_project.answer_engine.query.entity_resolver import normalize_medical_text


@dataclass
class NormalizedEvidence:
    """Normalized evidence with full provenance."""
    original_hit: RetrievalHit
    evidence_item: EvidenceItem
    normalized_text: str
    entities: List[str]
    numeric_values: List[Dict[str, Any]]


class EvidenceNormalizer:
    """Normalizes evidence for consistent processing."""
    
    def __init__(self):
        self._cached_normalizations: Dict[str, NormalizedEvidence] = {}
    
    def normalize(self, hit: RetrievalHit) -> NormalizedEvidence:
        """Normalize a single evidence hit."""
        cache_key = hit.id if hasattr(hit, "id") else str(hit.doc_id) + hit.text[:100]
        
        if cache_key in self._cached_normalizations:
            return self._cached_normalizations[cache_key]
        
        # Extract metadata
        metadata = hit.metadata or {}
        page_numbers = metadata.get("page_numbers", [1])
        page = page_numbers[0] if page_numbers else 1
        
        # Build source location
        source_location = SourceLocation(
            document_id=str(metadata.get("document_id", hit.doc_id)),
            document_version=str(metadata.get("version_id", "")),
            page=page,
            page_numbers=page_numbers,
            printed_page=metadata.get("printed_page_number"),
            section=str(metadata.get("section", "")),
            section_id=str(metadata.get("section_id", "")),
            heading_hierarchy=metadata.get("hierarchy_path", []),
            paragraph_id=str(metadata.get("chunk_id", "")),
            chunk_id=str(metadata.get("chunk_id", "")),
            source_type=str(metadata.get("source_type", "text")),
            table_id=metadata.get("table_id"),
            figure_id=metadata.get("figure_id"),
        )
        
        # Build evidence item
        text = str(hit.text or "")
        normalized_text = normalize_medical_text(text)
        
        evidence_item = EvidenceItem(
            id=str(metadata.get("chunk_id", hit.doc_id)),
            text=text,
            normalized_text=normalized_text,
            source_location=source_location,
            retrieval_scores={
                "vector": hit.vector_score or 0.0,
                "lexical": hit.lexical_score or 0.0,
                "combined": hit.score or 0.0,
            },
            semantic_relevance=hit.vector_score or 0.0,
            lexical_relevance=hit.lexical_score or 0.0,
            query_intent_match=min(1.0, (hit.score or 0.0) * 0.8),
            entity_match=bool(hit.metadata.get("entities")),
            section_relevance=0.7,
            document_authority=0.8,
            proximity_score=0.8,
            numeric_match=False,
            table_match=bool(hit.metadata.get("table_id")),
            figure_match=bool(hit.metadata.get("figure_id")),
            page_quality=metadata.get("quality_score", 0.9),
            text_extraction_quality=1.0 if metadata.get("ocr_status") != "failed" else 0.5,
            duplicate_penalty=0.0,
            contradiction_penalty=0.0,
            cross_source_agreement=1.0,
            evidence_diversity=1.0,
        )
        
        normalized = NormalizedEvidence(
            original_hit=hit,
            evidence_item=evidence_item,
            normalized_text=normalized_text,
            entities=[],
            numeric_values=[],
        )
        
        self._cached_normalizations[cache_key] = normalized
        return normalized
    
    def normalize_batch(self, hits: List[RetrievalHit]) -> List[NormalizedEvidence]:
        """Normalize multiple evidence hits."""
        return [self.normalize(hit) for hit in hits]
    
    def clear_cache(self):
        """Clear normalization cache."""
        self._cached_normalizations.clear()


def normalize_evidence(hit: RetrievalHit) -> NormalizedEvidence:
    """Convenience function to normalize evidence."""
    normalizer = EvidenceNormalizer()
    return normalizer.normalize(hit)
