"""Inference rules for deterministic medical answers."""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from rag_project.answer_engine.schemas.claims import Claim, ClaimStatus


@dataclass
class InferenceRule:
    """An inference rule for deriving new claims."""
    name: str
    pattern: str  # Regex pattern
    conclusion: str  # Conclusion template
    confidence: float


class InferenceRules:
    """Collection of inference rules for medical reasoning."""
    
    def __init__(self):
        self._rules = [
            InferenceRule(
                name="treatment_for_condition",
                pattern=r"treat(s|ment)?\s+(diabetes|hypertension|obesity)",
                conclusion="This is a treatment for {condition}",
                confidence=0.85,
            ),
            InferenceRule(
                name="drug_for_condition",
                pattern=r"(metformin|lisinopril|simvastatin)\s+(used|indicated)",
                conclusion="{drug} is indicated for the condition",
                confidence=0.9,
            ),
            InferenceRule(
                name="diagnostic_test",
                pattern=r"hba1c|creatinine|cholesterol.*measur",
                conclusion="{test} is a diagnostic test",
                confidence=0.8,
            ),
            InferenceRule(
                name="risk_factor",
                pattern=r"risk factor|associated with|linked to",
                conclusion="{factor} is a risk factor",
                confidence=0.75,
            ),
            InferenceRule(
                name="contraindication",
                pattern=r"contraindication|contraindicated|should not",
                conclusion="This is contraindicated",
                confidence=0.95,
            ),
            InferenceRule(
                name="mechanism",
                pattern=r"mechanism|pathway|pathophysiology",
                conclusion="This describes the mechanism",
                confidence=0.7,
            ),
            InferenceRule(
                name="dose_response",
                pattern=r"dose.*increas|increas.*dose",
                conclusion="There is a dose-response relationship",
                confidence=0.8,
            ),
        ]
    
    def apply(self, claim_text: str) -> List[Dict[str, Any]]:
        """Apply inference rules to a claim."""
        results = []
        
        for rule in self._rules:
            if re.search(rule.pattern, claim_text, re.I):
                conclusion = self._fill_conclusion(rule.conclusion, claim_text)
                results.append({
                    "rule": rule.name,
                    "pattern": rule.pattern,
                    "conclusion": conclusion,
                    "confidence": rule.confidence,
                })
        
        return results
    
    def _fill_conclusion(self, conclusion: str, claim_text: str) -> str:
        """Fill in conclusion template with extracted values."""
        # Extract values from claim
        claims = {
            "diabetes": "diabetes",
            "hypertension": "hypertension",
            "obesity": "obesity",
            "metformin": "metformin",
            "lisinopril": "lisinopril",
            "simvastatin": "simvastatin",
            "hba1c": "HbA1c",
            "creatinine": "creatinine",
            "cholesterol": "cholesterol",
        }
        
        result = conclusion
        for key, value in claims.items():
            result = result.replace(f"{{{key}}}", value)
        
        return result
    
    def get_applicable_rules(self, claim: Claim) -> List[InferenceRule]:
        """Get rules that apply to a claim."""
        applicable = []
        
        for rule in self._rules:
            if re.search(rule.pattern, claim.normalized_text, re.I):
                applicable.append(rule)
        
        return applicable
    
    def derive_new_claims(self, claim: Claim) -> List[Claim]:
        """Derive new claims from existing ones using inference rules."""
        new_claims = []
        
        rules = self.get_applicable_rules(claim)
        
        for rule in rules:
            conclusion = self._fill_conclusion(rule.conclusion, claim.normalized_text)
            
            new_claim = Claim(
                id=f"inferred_{rule.name}_{hash(conclusion) % 10000}",
                text=conclusion,
                normalized_text=conclusion.lower(),
                claim_type="inferred",
                entities=claim.entities,
                qualifiers=claim.qualifiers,
                context=claim.context,
                evidence=claim.evidence,
                status=ClaimStatus.SUPPORTED,
                support_ratio=rule.confidence,
            )
            
            new_claims.append(new_claim)
        
        return new_claims


def apply_inference_rules(claim_text: str) -> List[Dict[str, Any]]:
    """Convenience function to apply inference rules."""
    rules = InferenceRules()
    return rules.apply(claim_text)
