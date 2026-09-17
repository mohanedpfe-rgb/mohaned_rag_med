"""Claim extractor from evidence for deterministic medical answers."""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from rag_project.answer_engine.evidence.normalizer import NormalizedEvidence
from rag_project.answer_engine.schemas.claims import Claim, ClaimEvidence, ClaimStatus


@dataclass
class ExtractionResult:
    """Result of claim extraction."""
    claims: List[Claim]
    extracted_from: List[NormalizedEvidence]
    confidence: float


class ClaimExtractor:
    """Extracts claims from normalized evidence."""
    
    def __init__(self):
        self._sentence_patterns = [
            r"([^.!?]+[.!?])",
            r"([^\n]+[\n])",
        ]
        self._claim_indicators = [
            "is", "are", "was", "were", "does", "do", "causes", "leads to",
            "associated with", "related to", "results in", "increases risk",
            "decreases risk", "improves", "worsens", "prevents", "treats",
            "diagnoses", "indicates", "suggests", "demonstrates", "shows",
            "may", "can", "usually", "typically", "often", "sometimes",
        ]
    
    def extract(self, evidence: List[NormalizedEvidence]) -> ExtractionResult:
        """Extract claims from evidence."""
        claims: List[Claim] = []
        extracted_from: List[NormalizedEvidence] = []
        
        for item in evidence:
            norm_text = item.normalized_text
            
            # Skip if too short or lacks claim indicators
            if len(norm_text) < 20:
                continue
            
            if not any(ind in norm_text.lower() for ind in self._claim_indicators):
                continue
            
            # Extract sentences
            sentences = self._split_into_sentences(norm_text)
            
            for sentence in sentences:
                claim = self._create_claim(sentence, item)
                if claim:
                    claims.append(claim)
                    extracted_from.append(item)
        
        # Deduplicate claims
        unique_claims = self._deduplicate_claims(claims)
        
        confidence = self._estimate_confidence(unique_claims, evidence)
        
        return ExtractionResult(
            claims=unique_claims,
            extracted_from=extracted_from,
            confidence=confidence,
        )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _create_claim(self, text: str, evidence_item: NormalizedEvidence) -> Optional[Claim]:
        """Create a claim from text."""
        if not text or len(text) < 10:
            return None
        
        # Clean text
        claim_text = re.sub(r"^\s*[-*•]\s*", "", text).strip()
        
        # Extract entities
        entities = self._extract_entities(claim_text)
        
        # Detect qualifiers
        qualifiers = self._extract_qualifiers(claim_text)
        
        # Extract numeric values
        numeric_value, numeric_unit, numeric_range = self._extract_numeric(claim_text)
        
        # Create claim evidence
        claim_evidence = ClaimEvidence(
            source_id=evidence_item.evidence_item.id,
            text=claim_text,
            document_id=evidence_item.evidence_item.source_location.document_id,
            document_version=evidence_item.evidence_item.source_location.document_version,
            page=evidence_item.evidence_item.source_location.page,
            page_numbers=evidence_item.evidence_item.source_location.page_numbers,
            section=evidence_item.evidence_item.source_location.section,
            heading_hierarchy=evidence_item.evidence_item.source_location.heading_hierarchy,
            source_type=str(evidence_item.evidence_item.source_location.source_type),
            source_span=claim_text,
            normalized_text=evidence_item.normalized_text,
            quality_score=evidence_item.evidence_item.composite_score,
            relevance_score=evidence_item.evidence_item.semantic_relevance,
            support_strength=0.7,
        )
        
        return Claim(
            id=f"claim_{len(claim_text)}_{hash(claim_text) % 10000}",
            text=claim_text,
            normalized_text=re.sub(r"\s+", " ", claim_text.lower()).strip(),
            claim_type=self._detect_claim_type(claim_text),
            entities=entities,
            qualifiers=qualifiers,
            context={},
            evidence=[claim_evidence],
            status=ClaimStatus.UNSUPPORTED,
            support_ratio=0.0,
            numeric_value=numeric_value,
            numeric_unit=numeric_unit,
            numeric_range=numeric_range,
            contradiction=None,
            source_span=claim_text,
        )
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from claim text."""
        entities = []
        text_lower = text.lower()
        
        # Simple entity extraction
        disease_keywords = ["diabetes", "hypertension", "obesity", "cardiovascular", "cancer"]
        drug_keywords = ["metformin", "lisinopril", "simvastatin", "insulin", "aspirin"]
        test_keywords = ["hba1c", "creatinine", "cholesterol", "bmi"]
        
        for kw in disease_keywords:
            if kw in text_lower:
                entities.append(kw)
        
        return list(set(entities))
    
    def _extract_qualifiers(self, text: str) -> List[str]:
        """Extract qualifiers from claim text."""
        qualifiers = []
        text_lower = text.lower()
        
        qualifier_patterns = [
            ("may", "possibility"),
            ("can", "possibility"),
            ("usually", "frequency"),
            ("typically", "frequency"),
            ("often", "frequency"),
            ("sometimes", "frequency"),
            ("rarely", "frequency"),
            ("only", "exclusivity"),
            ("except", "exclusion"),
            ("unless", "condition"),
            ("not", "negation"),
            ("does not", "negation"),
        ]
        
        for q, q_type in qualifier_patterns:
            if q in text_lower:
                qualifiers.append(q_type)
        
        return qualifiers
    
    def _extract_numeric(self, text: str) -> tuple[Optional[float], Optional[str], Optional[tuple]]:
        """Extract numeric values from claim text."""
        # Pattern for exact value
        match = re.search(r"(\d+(?:\.\d+)?)\s*(mg|mcg|ml|g|mmhg|mmol/l|%|bpm)", text, re.I)
        if match:
            return float(match.group(1)), match.group(2).lower(), None
        
        # Pattern for range
        range_match = re.search(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*(mg|ml|mmhg)", text, re.I)
        if range_match:
            return None, range_match.group(3).lower(), (
                float(range_match.group(1)),
                float(range_match.group(2)),
            )
        
        return None, None, None
    
    def _detect_claim_type(self, text: str) -> str:
        """Detect the type of claim."""
        text_lower = text.lower()
        
        if any(w in text_lower for w in ["definition", "means", "refers to"]):
            return "definition"
        if any(w in text_lower for w in ["treat", "therapy", "drug", "medication"]):
            return "management"
        if any(w in text_lower for w in ["diagnose", "diagnosis", "test"]):
            return "diagnostic"
        if any(w in text_lower for w in ["cause", "causes", "risk factor"]):
            return "causal"
        if any(w in text_lower for w in ["mechanism", "pathway", "pathophysiology"]):
            return "mechanism"
        if any(w in text_lower for w in ["prognosis", "outcome", "survival"]):
            return "prognosis"
        if any(w in text_lower for w in ["vs", "versus", "compare", "difference"]):
            return "comparison"
        if any(w in text_lower for w in ["list", "all", "every"]):
            return "list"
        
        return "fact"
    
    def _deduplicate_claims(self, claims: List[Claim]) -> List[Claim]:
        """Deduplicate similar claims."""
        seen_texts = set()
        unique = []
        
        for claim in claims:
            norm_text = re.sub(r"\s+", " ", claim.normalized_text.lower()).strip()
            
            if norm_text not in seen_texts:
                seen_texts.add(norm_text)
                unique.append(claim)
        
        return unique
    
    def _estimate_confidence(self, claims: List[Claim], evidence: List[NormalizedEvidence]) -> float:
        """Estimate confidence in extracted claims."""
        if not claims or not evidence:
            return 0.0
        
        avg_evidence_score = sum(e.evidence_item.composite_score for e in evidence) / len(evidence)
        claim_count_bonus = min(0.2, len(claims) * 0.05)
        
        return min(1.0, avg_evidence_score + claim_count_bonus)


def extract_claims(evidence: List[NormalizedEvidence]) -> ExtractionResult:
    """Convenience function to extract claims."""
    extractor = ClaimExtractor()
    return extractor.extract(evidence)
