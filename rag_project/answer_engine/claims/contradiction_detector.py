"""Contradiction detector for deterministic medical answers."""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

from rag_project.answer_engine.schemas.claims import Claim, ClaimStatus
from rag_project.answer_engine.evidence.normalizer import NormalizedEvidence
from rag_project.answer_engine.schemas.reasoning import Conflict, RelationshipType


@dataclass
class ContradictionResult:
    """Result of contradiction detection."""
    contradictions: List[Conflict]
    confidence: float
    resolved: List[str]
    unresolved: List[str]


class ContradictionDetector:
    """Detects contradictions between claims and evidence."""
    
    def __init__(self):
        self._opposites = {
            "should": "should not",
            "must": "must not",
            "is": "is not",
            "are": "are not",
            "increases": "decreases",
            "decreases": "increases",
            "higher": "lower",
            "lower": "higher",
            "better": "worse",
            "worse": "better",
            "recommended": "not recommended",
            "contraindicated": "indicated",
            "benefit": "risk",
            "risk": "benefit",
        }
        self._negation_words = {"not", "no", "never", "none", "without", "avoid", "contraindicated"}
    
    def detect(self, claims: List[Claim], evidence: List[NormalizedEvidence]) -> ContradictionResult:
        """Detect contradictions in claims and evidence."""
        contradictions: List[Conflict] = []
        resolved: List[str] = []
        unresolved: List[str] = []
        
        # Check claim-to-claim contradictions
        for i, claim1 in enumerate(claims):
            for claim2 in claims[i+1:]:
                if self._are_contradictory(claim1, claim2):
                    conflict = self._create_conflict(claim1, claim2, "claim-to-claim")
                    contradictions.append(conflict)
                    if self._resolve_conflict(conflict):
                        resolved.append(conflict.id)
                    else:
                        unresolved.append(conflict.id)
        
        # Check claim-to-evidence contradictions
        for claim in claims:
            for item in evidence:
                if self._evidence_contradicts_claim(claim, item):
                    conflict = self._create_conflict(claim, None, "claim-to-evidence", evidence=item)
                    contradictions.append(conflict)
                    if self._resolve_conflict(conflict):
                        resolved.append(conflict.id)
                    else:
                        unresolved.append(conflict.id)
        
        confidence = self._estimate_confidence(contradictions)
        
        return ContradictionResult(
            contradictions=contradictions,
            confidence=confidence,
            resolved=resolved,
            unresolved=unresolved,
        )
    
    def _are_contradictory(self, claim1: Claim, claim2: Claim) -> bool:
        """Check if two claims are contradictory."""
        text1 = claim1.normalized_text.lower()
        text2 = claim2.normalized_text.lower()
        
        # Check for direct opposites
        for word1, word2 in self._opposites.items():
            if word1 in text1 and word2 in text2:
                return True
            if word2 in text1 and word1 in text2:
                return True
        
        # Check for conflicting statements
        if self._has_negation(text1) and self._has_negation(text2):
            if self._similar_subject(text1, text2):
                return True
        
        return False
    
    def _has_negation(self, text: str) -> bool:
        """Check if text contains negation."""
        return any(word in text for word in self._negation_words)
    
    def _similar_subject(self, text1: str, text2: str) -> bool:
        """Check if texts have similar subjects."""
        # Extract subject (first noun phrase)
        subject1 = self._extract_subject(text1)
        subject2 = self._extract_subject(text2)
        
        return bool(subject1 and subject2 and subject1 == subject2)
    
    def _extract_subject(self, text: str) -> Optional[str]:
        """Extract subject from text."""
        # Simple subject extraction
        match = re.match(r"^\s*(?:the|a|an|this|that)\s+([a-zA-Z][a-zA-Z\s]+?)(?:\s+(?:is|are|was|were|causes|leads))", text)
        if match:
            return match.group(1).strip()
        
        return None
    
    def _evidence_contradicts_claim(self, claim: Claim, evidence: NormalizedEvidence) -> bool:
        """Check if evidence contradicts the claim."""
        claim_text = claim.normalized_text.lower()
        evidence_text = evidence.normalized_text.lower()
        
        # If claim says "not" but evidence doesn't
        if self._has_negation(claim_text):
            if self._has_significant_overlap(claim_text, evidence_text):
                return True
        
        # Check opposites
        for word1, word2 in self._opposites.items():
            if word1 in claim_text and word2 in evidence_text:
                return True
        
        return False
    
    def _has_significant_overlap(self, text1: str, text2: str) -> bool:
        """Check if texts have significant overlap."""
        tokens1 = set(re.findall(r"\b\w+\b", text1))
        tokens2 = set(re.findall(r"\b\w+\b", text2))
        
        if not tokens1 or not tokens2:
            return False
        
        overlap = len(tokens1 & tokens2)
        return overlap >= 3
    
    def _create_conflict(self, claim1: Claim, claim2: Optional[Claim], 
                         conflict_type: str, evidence: Optional[NormalizedEvidence] = None) -> Conflict:
        """Create a conflict object."""
        claim_text = claim1.normalized_text
        
        if claim2:
            evidence_a = [c.evidence_item.id for c in claim1.evidence]
            evidence_b = [c.evidence_item.id for c in claim2.evidence]
            
            return Conflict(
                id=f"conflict_{hash(claim_text + claim2.normalized_text) % 10000}",
                claim_a=claim_text,
                claim_b=claim2.normalized_text,
                evidence_a=evidence_a,
                evidence_b=evidence_b,
                source_versions={
                    claim1.evidence[0].document_version if claim1.evidence else "unknown",
                    claim2.evidence[0].document_version if claim2.evidence else "unknown",
                },
                confidence=0.85,
                resolution=None,
                resolution_reason=None,
            )
        else:
            evidence_a = [c.evidence_item.id for c in claim1.evidence]
            evidence_b = [evidence.evidence_item.id] if evidence else []
            
            return Conflict(
                id=f"conflict_{hash(claim_text + str(evidence.evidence_item.id if evidence else '')) % 10000}",
                claim_a=claim_text,
                claim_b=evidence.normalized_text if evidence else "",
                evidence_a=evidence_a,
                evidence_b=evidence_b,
                source_versions={
                    claim1.evidence[0].document_version if claim1.evidence else "unknown",
                    evidence.evidence_item.source_location.document_version if evidence else "unknown",
                },
                confidence=0.75,
                resolution=None,
                resolution_reason=None,
            )
    
    def _resolve_conflict(self, conflict: Conflict) -> bool:
        """Attempt to resolve a conflict."""
        # Simple resolution: check if one source is more specific
        # In production, would use version dates, document authority, etc.
        
        # If conflicts involve different entities, they may not truly contradict
        if self._different_contexts(conflict):
            conflict.resolution = "different_context"
            conflict.resolution_reason = "Claims apply to different contexts"
            return True
        
        # Cannot resolve automatically
        return False
    
    def _different_contexts(self, conflict: Conflict) -> bool:
        """Check if conflicts involve different contexts."""
        # Check if entities differ significantly
        return False  # Simplified
    
    def _estimate_confidence(self, contradictions: List[Conflict]) -> float:
        """Estimate confidence in detected contradictions."""
        if not contradictions:
            return 0.0
        
        avg_confidence = sum(c.confidence for c in contradictions) / len(contradictions)
        
        # Penalty for unresolved
        unresolved_ratio = len([c for c in contradictions if not c.resolution]) / len(contradictions)
        
        return avg_confidence * (1.0 - unresolved_ratio * 0.5)


def detect_contradictions(claims: List[Claim], evidence: List[NormalizedEvidence]) -> ContradictionResult:
    """Convenience function to detect contradictions."""
    detector = ContradictionDetector()
    return detector.detect(claims, evidence)
