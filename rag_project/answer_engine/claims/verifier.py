"""Claim verifier for deterministic medical answers."""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from rag_project.answer_engine.schemas.claims import Claim, ClaimStatus
from rag_project.answer_engine.evidence.normalizer import NormalizedEvidence


@dataclass
class VerificationResult:
    """Result of claim verification."""
    claim: Claim
    status: ClaimStatus
    support_ratio: float
    supported_by: List[str]
    contradicted_by: List[str]
    evidence_items: List[NormalizedEvidence]


class ClaimVerifier:
    """Verifies claims against evidence."""
    
    def __init__(self):
        self._support_threshold = 0.50
        self._partial_threshold = 0.30
    
    def verify(self, claim: Claim, evidence: List[NormalizedEvidence]) -> VerificationResult:
        """Verify a claim against evidence."""
        if not claim.evidence:
            return VerificationResult(
                claim=claim,
                status=ClaimStatus.UNSUPPORTED,
                support_ratio=0.0,
                supported_by=[],
                contradicted_by=[],
                evidence_items=[],
            )
        
        # Extract claim text for matching
        claim_text = claim.normalized_text.lower()
        claim_tokens = set(re.findall(r"\b\w+\b", claim_text))
        
        # Score evidence
        supported_by: List[str] = []
        contradicted_by: List[str] = []
        scores: List[float] = []
        matching_evidence: List[NormalizedEvidence] = []
        
        for item in evidence:
            evidence_text = item.normalized_text.lower()
            evidence_tokens = set(re.findall(r"\b\w+\b", evidence_text))
            
            # Calculate overlap
            overlap = len(claim_tokens & evidence_tokens)
            total = len(claim_tokens | evidence_tokens)
            
            if total == 0:
                score = 0.0
            else:
                score = overlap / total
            
            # Check for contradiction
            if self._detect_contradiction(claim_text, evidence_text):
                contradicted_by.append(item.evidence_item.id)
                score = max(0.0, score - 0.5)
            
            if score >= self._support_threshold:
                supported_by.append(item.evidence_item.id)
            
            scores.append(score)
            matching_evidence.append(item)
        
        # Calculate support ratio
        support_ratio = sum(1 for s in scores if s >= self._support_threshold) / max(1, len(scores))
        
        # Determine status
        if contradicted_by:
            status = ClaimStatus.CONTRADICTED
        elif support_ratio >= self._support_threshold:
            status = ClaimStatus.SUPPORTED
        elif support_ratio >= self._partial_threshold:
            status = ClaimStatus.PARTIALLY_SUPPORTED
        elif any(s > 0.1 for s in scores):
            status = ClaimStatus.UNCERTAIN
        else:
            status = ClaimStatus.UNSUPPORTED
        
        # Filter evidence items
        filtered_evidence = [e for e, s in zip(matching_evidence, scores) if s >= 0.1]
        
        return VerificationResult(
            claim=claim,
            status=status,
            support_ratio=support_ratio,
            supported_by=supported_by,
            contradicted_by=contradicted_by,
            evidence_items=filtered_evidence,
        )
    
    def _detect_contradiction(self, claim: str, evidence: str) -> bool:
        """Detect if evidence contradicts the claim."""
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()
        
        # Direct contradictions
        if "not" in claim_lower and "not" not in evidence_lower:
            if self._has_significant_overlap(claim_lower, evidence_lower):
                return True
        
        if "should not" in claim_lower and "should" in evidence_lower:
            return True
        
        if "avoid" in claim_lower and "recommended" in evidence_lower:
            return True
        
        return False
    
    def _has_significant_overlap(self, text1: str, text2: str) -> bool:
        """Check if two texts have significant overlap."""
        tokens1 = set(re.findall(r"\b\w+\b", text1))
        tokens2 = set(re.findall(r"\b\w+\b", text2))
        
        if not tokens1 or not tokens2:
            return False
        
        overlap = len(tokens1 & tokens2)
        return overlap >= 2
    
    def verify_batch(self, claims: List[Claim], evidence: List[NormalizedEvidence]) -> List[VerificationResult]:
        """Verify multiple claims."""
        return [self.verify(c, evidence) for c in claims]


def verify_claim(claim: Claim, evidence: List[NormalizedEvidence]) -> VerificationResult:
    """Convenience function to verify a claim."""
    verifier = ClaimVerifier()
    return verifier.verify(claim, evidence)
