"""Evidence grouper for deterministic medical answers."""
from __future__ import annotations

import re
from typing import List, Dict, Any, Set
from dataclasses import dataclass
from collections import defaultdict

from rag_project.answer_engine.evidence.normalizer import NormalizedEvidence


@dataclass
class EvidenceGroup:
    """Group of related evidence."""
    group_id: str
    topic: str
    evidence: List[NormalizedEvidence]
    scores: List[float]
    quality: str


class EvidenceGrouper:
    """Groups evidence by topic, section, and relationship."""
    
    def __init__(self):
        self._topic_patterns = {
            "definition": ["what is", "definition", "means"],
            "causal": ["cause", "causes", "risk factor", "due to"],
            "treatment": ["treat", "treatment", "therapy", "drug", "medication"],
            "diagnosis": ["diagnose", "diagnosis", "test", "criteria"],
            "mechanism": ["mechanism", "pathway", "pathophysiology"],
            "prognosis": ["prognosis", "outcome", "survival"],
        }
    
    def group_by_section(self, evidence: List[NormalizedEvidence]) -> Dict[str, List[NormalizedEvidence]]:
        """Group evidence by section."""
        groups: Dict[str, List[NormalizedEvidence]] = defaultdict(list)
        
        for item in evidence:
            section = item.evidence_item.source_location.section or "unknown"
            groups[section].append(item)
        
        return dict(groups)
    
    def group_by_document(self, evidence: List[NormalizedEvidence]) -> Dict[str, List[NormalizedEvidence]]:
        """Group evidence by document."""
        groups: Dict[str, List[NormalizedEvidence]] = defaultdict(list)
        
        for item in evidence:
            doc_id = item.evidence_item.source_location.document_id
            groups[doc_id].append(item)
        
        return dict(groups)
    
    def group_by_topic(self, evidence: List[NormalizedEvidence]) -> List[EvidenceGroup]:
        """Group evidence by detected topic."""
        groups: Dict[str, EvidenceGroup] = {}
        
        for item in evidence:
            topic = self._detect_topic(item.normalized_text)
            
            if topic not in groups:
                groups[topic] = EvidenceGroup(
                    group_id=f"group_{topic}",
                    topic=topic,
                    evidence=[],
                    scores=[],
                )
            
            groups[topic].evidence.append(item)
            groups[topic].scores.append(item.evidence_item.composite_score)
        
        return list(groups.values())
    
    def _detect_topic(self, text: str) -> str:
        """Detect the topic of text."""
        text_lower = text.lower()
        
        for topic, patterns in self._topic_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return topic
        
        return "general"
    
    def group_by_relationship(self, evidence: List[NormalizedEvidence]) -> Dict[str, List[NormalizedEvidence]]:
        """Group evidence by relationship types."""
        groups: Dict[str, List[NormalizedEvidence]] = defaultdict(list)
        
        for item in evidence:
            norm_text = item.normalized_text.lower()
            
            if "vs" in norm_text or "versus" in norm_text:
                groups["comparison"].append(item)
            elif "before" in norm_text or "after" in norm_text:
                groups["temporal"].append(item)
            elif "cause" in norm_text or "causes" in norm_text:
                groups["causal"].append(item)
            elif "mechanism" in norm_text or "pathway" in norm_text:
                groups["mechanism"].append(item)
            else:
                groups["general"].append(item)
        
        return dict(groups)
    
    def merge_related(self, evidence: List[NormalizedEvidence]) -> List[NormalizedEvidence]:
        """Merge related evidence into comprehensive items."""
        if not evidence:
            return []
        
        # Simple implementation: keep unique items
        seen = set()
        merged = []
        
        for item in evidence:
            key = item.evidence_item.source_location.chunk_id
            if key not in seen:
                seen.add(key)
                merged.append(item)
        
        return merged


def group_evidence(evidence: List[NormalizedEvidence]) -> Dict[str, Any]:
    """Convenience function to group evidence."""
    grouper = EvidenceGrouper()
    
    return {
        "by_section": grouper.group_by_section(evidence),
        "by_document": grouper.group_by_document(evidence),
        "by_topic": grouper.group_by_topic(evidence),
        "by_relationship": grouper.group_by_relationship(evidence),
    }
