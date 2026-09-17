"""Support matrix for deterministic medical answers."""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from rag_project.answer_engine.schemas.claims import Claim, ClaimStatus
from rag_project.answer_engine.evidence.normalizer import NormalizedEvidence


@dataclass
class SupportMatrixEntry:
    """Single entry in the support matrix."""
    claim_id: str
    evidence_id: str
    status: ClaimStatus
    support_strength: float
    reasons: List[str]


@dataclass
class SupportMatrix:
    """Complete support matrix for claims vs evidence."""
    claims: List[Claim]
    evidence: List[NormalizedEvidence]
    matrix: List[SupportMatrixEntry]
    claim_status_summary: Dict[str, ClaimStatus]
    overall_support_ratio: float


class SupportMatrixBuilder:
    """Builds support matrices from claims and evidence."""
    
    def __init__(self):
        self._min_support_threshold = 0.50
    
    def build(self, claims: List[Claim], evidence: List[NormalizedEvidence]) -> SupportMatrix:
        """Build support matrix from claims and evidence."""
        matrix: List[SupportMatrixEntry] = []
        
        for claim in claims:
            for item in evidence:
                entry = self._build_entry(claim, item)
                if entry:
                    matrix.append(entry)
        
        # Calculate claim status summary
        claim_status_summary = self._calculate_claim_status(claims, matrix)
        
        # Calculate overall support ratio
        supported_claims = sum(1 for s in claim_status_summary.values() if s == ClaimStatus.SUPPORTED)
        overall_ratio = supported_claims / max(1, len(claims))
        
        return SupportMatrix(
            claims=claims,
            evidence=evidence,
            matrix=matrix,
            claim_status_summary=claim_status_summary,
            overall_support_ratio=overall_ratio,
        )
    
    def _build_entry(self, claim: Claim, evidence_item: NormalizedEvidence) -> Optional[SupportMatrixEntry]:
        """Build a single matrix entry."""
        claim_text = claim.normalized_text.lower()
        evidence_text = evidence_item.normalized_text.lower()
        
        # Calculate overlap
        claim_tokens = set(re.findall(r"\b\w+\b", claim_text))
        evidence_tokens = set(re.findall(r"\b\w+\b", evidence_text))
        
        if not claim_tokens:
            return None
        
        overlap = len(claim_tokens & evidence_tokens)
        union = len(claim_tokens | evidence_tokens)
        
        support_strength = overlap / max(1, union)
        
        # Determine status
        reasons = self._determine_reasons(claim_text, evidence_text, overlap)
        
        if overlap < 2:
            status = ClaimStatus.UNSUPPORTED
        elif support_strength >= self._min_support_threshold:
            status = ClaimStatus.SUPPORTED
        elif overlap >= 1:
            status = ClaimStatus.PARTIALLY_SUPPORTED
        else:
            status = ClaimStatus.UNSUPPORTED
        
        return SupportMatrixEntry(
            claim_id=claim.id,
            evidence_id=evidence_item.evidence_item.id,
            status=status,
            support_strength=support_strength,
            reasons=reasons,
        )
    
    def _determine_reasons(self, claim_text: str, evidence_text: str, overlap: int) -> List[str]:
        """Determine reasons for support status."""
        reasons = []
        
        if overlap >= 3:
            reasons.append("strong_entity_overlap")
        elif overlap >= 1:
            reasons.append("partial_entity_overlap")
        
        if self._has_qualifier_overlap(claim_text, evidence_text):
            reasons.append("qualifier_match")
        
        if self._has_numeric_match(claim_text, evidence_text):
            reasons.append("numeric_match")
        
        if not reasons:
            reasons.append("no_matching_evidence")
        
        return reasons
    
    def _has_qualifier_overlap(self, claim_text: str, evidence_text: str) -> bool:
        """Check for qualifier overlap."""
        qualifiers = {"may", "can", "usually", "typically", "often", "rarely", "not", "does not"}
        
        claim_lower = claim_text.lower()
        evidence_lower = evidence_text.lower()
        
        claim_qualifiers = {q for q in qualifiers if q in claim_lower}
        evidence_qualifiers = {q for q in qualifiers if q in evidence_lower}
        
        return bool(claim_qualifiers & evidence_qualifiers)
    
    def _has_numeric_match(self, claim_text: str, evidence_text: str) -> bool:
        """Check for numeric value match."""
        claim_nums = re.findall(r"\d+(?:\.\d+)?", claim_text)
        evidence_nums = re.findall(r"\d+(?:\.\d+)?", evidence_text)
        
        return bool(set(claim_nums) & set(evidence_nums))
    
    def _calculate_claim_status(self, claims: List[Claim], matrix: List[SupportMatrixEntry]) -> Dict[str, ClaimStatus]:
        """Calculate status for each claim."""
        claim_status: Dict[str, ClaimStatus] = {}
        
        for claim in claims:
            claim_entries = [e for e in matrix if e.claim_id == claim.id]
            
            if not claim_entries:
                claim_status[claim.id] = ClaimStatus.UNSUPPORTED
                continue
            
            statuses = [e.status for e in claim_entries]
            
            if ClaimStatus.CONTRADICTED in statuses:
                claim_status[claim.id] = ClaimStatus.CONTRADICTED
            elif all(s == ClaimStatus.SUPPORTED for s in statuses):
                claim_status[claim.id] = ClaimStatus.SUPPORTED
            elif any(s == ClaimStatus.SUPPORTED for s in statuses):
                claim_status[claim.id] = ClaimStatus.PARTIALLY_SUPPORTED
            elif ClaimStatus.UNSUPPORTED in statuses:
                claim_status[claim.id] = ClaimStatus.UNSUPPORTED
            else:
                claim_status[claim.id] = ClaimStatus.UNCERTAIN
        
        return claim_status
    
    def filter_by_status(self, matrix: SupportMatrix, status: ClaimStatus) -> List[SupportMatrixEntry]:
        """Filter matrix entries by status."""
        return [e for e in matrix.matrix if e.status == status]
    
    def get_supporting_evidence(self, matrix: SupportMatrix, claim_id: str) -> List[SupportMatrixEntry]:
        """Get supporting evidence for a claim."""
        return [e for e in matrix.matrix if e.claim_id == claim_id and e.status == ClaimStatus.SUPPORTED]


def build_support_matrix(claims: List[Claim], evidence: List[NormalizedEvidence]) -> SupportMatrix:
    """Convenience function to build support matrix."""
    builder = SupportMatrixBuilder()
    return builder.build(claims, evidence)
