"""Evidence deduplicator for deterministic medical answers."""
from __future__ import annotations

import re
from typing import List, Dict, Any
from dataclasses import dataclass

from rag_project.answer_engine.evidence.normalizer import NormalizedEvidence


@dataclass
class DeduplicationResult:
    """Result of deduplication."""
    unique_evidence: List[NormalizedEvidence]
    duplicates: List[Dict[str, Any]]
    removed_count: int
    similarity_threshold: float


class EvidenceDeduplicator:
    """Deduplicates evidence while preserving unique information."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
    
    def deduplicate(self, evidence: List[NormalizedEvidence]) -> DeduplicationResult:
        """Remove duplicate evidence while preserving unique information."""
        if not evidence:
            return DeduplicationResult(
                unique_evidence=[],
                duplicates=[],
                removed_count=0,
                similarity_threshold=self.similarity_threshold,
            )
        
        unique: List[NormalizedEvidence] = []
        duplicates: List[Dict[str, Any]] = []
        normalized_texts = []
        
        for item in evidence:
            norm_text = item.normalized_text
            normalized = re.sub(r"\s+", " ", norm_text.lower()).strip()
            
            is_duplicate = False
            for existing in normalized_texts:
                if self._similarity(normalized, existing) >= self.similarity_threshold:
                    is_duplicate = True
                    duplicates.append({
                        "text": norm_text[:100],
                        "similar_to": existing[:100],
                        "similarity": self._similarity(normalized, existing),
                    })
                    break
            
            if not is_duplicate:
                unique.append(item)
                normalized_texts.append(normalized)
        
        return DeduplicationResult(
            unique_evidence=unique,
            duplicates=duplicates,
            removed_count=len(evidence) - len(unique),
            similarity_threshold=self.similarity_threshold,
        )
    
    def _similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using normalized overlap."""
        if not text1 or not text2:
            return 0.0
        
        if text1 == text2:
            return 1.0
        
        tokens1 = set(text1.split())
        tokens2 = set(text2.split())
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def group_similar(self, evidence: List[NormalizedEvidence]) -> List[List[NormalizedEvidence]]:
        """Group similar evidence together."""
        if not evidence:
            return []
        
        groups: List[List[NormalizedEvidence]] = []
        
        for item in evidence:
            norm_text = re.sub(r"\s+", " ", item.normalized_text.lower()).strip()
            added = False
            
            for group in groups:
                group_norm = re.sub(r"\s+", " ", group[0].normalized_text.lower()).strip()
                if self._similarity(norm_text, group_norm) >= self.similarity_threshold:
                    group.append(item)
                    added = True
                    break
            
            if not added:
                groups.append([item])
        
        return groups


def deduplicate_evidence(evidence: List[NormalizedEvidence], threshold: float = 0.85) -> DeduplicationResult:
    """Convenience function to deduplicate evidence."""
    deduplicator = EvidenceDeduplicator(threshold)
    return deduplicator.deduplicate(evidence)
