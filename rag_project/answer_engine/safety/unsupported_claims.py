"""Unsupported claims detector for safety."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass(frozen=True)
class UnsupportedClaim:
    """A claim that is unsupported or potentially unsafe."""
    claim_id: str
    text: str
    normalized_text: str
    claim_type: str
    status: str
    evidence_count: int
    support_ratio: float
    has_contradiction: bool
    is_safety_critical: bool
    issues: List[str]


class UnsupportedClaimsDetector:
    """Detects unsupported and potentially unsafe claims."""
    
    # Safety-critical terms
    SAFETY_CRITICAL_PHRASES = [
        "first-line", "first line", "preferred", "recommended", "only",
        "always", "never", "must", "should", "shall", "required",
        "superior", "best", "most effective", "gold standard",
        "first choice", "initial treatment", "first-line therapy",
    ]
    
    # Absolute terms that require strong evidence
    ABSOLUTE_TERMS = [
        "always", "never", "everyone", "nobody", "all", "none",
        "completely", "entirely", "perfectly", "definitely",
        "cure", "guarantee", "guaranteed", "guarantees",
    ]
    
    def __init__(self, min_support_ratio: float = 0.50):
        self.min_support_ratio = min_support_ratio
    
    def detect_unsupported_claims(
        self,
        claims: List[Dict[str, Any]],
    ) -> List[UnsupportedClaim]:
        """Detect unsupported claims from a list."""
        unsupported = []
        
        for claim in claims:
            unsupported_claim = self._analyze_claim(claim)
            if unsupported_claim:
                unsupported.append(unsupported_claim)
        
        return unsupported
    
    def _analyze_claim(self, claim: Dict[str, Any]) -> Optional[UnsupportedClaim]:
        """Analyze a single claim."""
        claim_id = claim.get("id", "")
        text = claim.get("text", "")
        normalized_text = claim.get("normalized_text", text)
        claim_type = claim.get("claim_type", "")
        status = claim.get("status", "")
        evidence = claim.get("evidence", [])
        support_ratio = claim.get("support_ratio", 0.0)
        has_contradiction = bool(claim.get("contradiction"))
        
        issues = []
        
        # Check if unsupported
        if status == "UNSUPPORTED":
            issues.append("Claim is unsupported")
        elif status == "CONTRADICTED":
            issues.append("Claim is contradicted")
        
        # Check evidence count
        evidence_count = len(evidence) if isinstance(evidence, list) else 0
        if evidence_count == 0:
            issues.append("No evidence provided")
        elif evidence_count < 2:
            issues.append("Limited evidence (less than 2 sources)")
        
        # Check support ratio
        if support_ratio < self.min_support_ratio:
            issues.append(f"Support ratio ({support_ratio:.2f}) below threshold ({self.min_support_ratio})")
        
        # Check for safety-critical terms
        if self._contains_safety_critical(text):
            if support_ratio < 0.8:
                issues.append("Safety-critical claim with insufficient support")
        
        # Check for absolute terms
        if self._contains_absolute_terms(text):
            if support_ratio < 0.9:
                issues.append("Absolute term with insufficient support")
        
        # Determine if this is a significant issue
        is_safety_critical = self._contains_safety_critical(text)
        
        if issues:
            return UnsupportedClaim(
                claim_id=claim_id,
                text=text,
                normalized_text=normalized_text,
                claim_type=claim_type,
                status=status,
                evidence_count=evidence_count,
                support_ratio=support_ratio,
                has_contradiction=has_contradiction,
                is_safety_critical=is_safety_critical,
                issues=issues,
            )
        
        return None
    
    def _contains_safety_critical(self, text: str) -> bool:
        """Check if text contains safety-critical phrases."""
        text_lower = text.lower()
        
        for phrase in self.SAFETY_CRITICAL_PHRASES:
            if phrase in text_lower:
                return True
        
        return False
    
    def _contains_absolute_terms(self, text: str) -> bool:
        """Check if text contains absolute terms."""
        text_lower = text.lower()
        
        for term in self.ABSOLUTE_TERMS:
            if re.search(r'\b' + term + r'\b', text_lower):
                return True
        
        return False
    
    def get_critical_claims(
        self,
        unsupported_claims: List[UnsupportedClaim],
    ) -> List[UnsupportedClaim]:
        """Get only the critical unsupported claims."""
        return [
            uc for uc in unsupported_claims
            if uc.is_safety_critical or uc.status == "CONTRADICTED"
        ]
    
    def get_statistics(
        self,
        unsupported_claims: List[UnsupportedClaim],
    ) -> Dict[str, Any]:
        """Get statistics about unsupported claims."""
        total = len(unsupported_claims)
        safety_critical = sum(1 for uc in unsupported_claims if uc.is_safety_critical)
        contradictions = sum(1 for uc in unsupported_claims if uc.has_contradiction)
        no_evidence = sum(1 for uc in unsupported_claims if uc.evidence_count == 0)
        
        support_ratios = [uc.support_ratio for uc in unsupported_claims]
        avg_support_ratio = sum(support_ratios) / len(support_ratios) if support_ratios else 0.0
        
        return {
            "total_unsupported": total,
            "safety_critical": safety_critical,
            "contradicted": contradictions,
            "no_evidence": no_evidence,
            "average_support_ratio": avg_support_ratio,
        }


def detect_unsupported_claims(
    claims: List[Dict[str, Any]],
    min_support_ratio: float = 0.50,
) -> List[UnsupportedClaim]:
    """Convenience function to detect unsupported claims."""
    detector = UnsupportedClaimsDetector(min_support_ratio=min_support_ratio)
    return detector.detect_unsupported_claims(claims)


def filter_supported_claims(
    claims: List[Dict[str, Any]],
    min_support_ratio: float = 0.50,
) -> List[Dict[str, Any]]:
    """Filter claims to only supported ones."""
    detector = UnsupportedClaimsDetector(min_support_ratio=min_support_ratio)
    unsupported = detector.detect_unsupported_claims(claims)
    unsupported_ids = {uc.claim_id for uc in unsupported}
    
    return [
        claim for claim in claims
        if claim.get("id") not in unsupported_ids
    ]


def get_unsupported_issues(claim: Dict[str, Any]) -> List[str]:
    """Get issues for a single claim."""
    detector = UnsupportedClaimsDetector()
    unsupported = detector._analyze_claim(claim)
    
    if unsupported:
        return unsupported.issues
    
    return []