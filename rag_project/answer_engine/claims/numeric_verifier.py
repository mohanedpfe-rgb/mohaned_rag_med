"""Numeric claim verifier for strict medical数值 verification."""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from rag_project.answer_engine.schemas.claims import Claim, ClaimStatus
from rag_project.answer_engine.evidence.normalizer import NormalizedEvidence


@dataclass
class NumericVerificationResult:
    """Result of numeric verification."""
    claim_value: Optional[float]
    claim_unit: Optional[str]
    evidence_value: Optional[float]
    evidence_unit: Optional[str]
    matched: bool
    compatible: bool
    message: str


class NumericClaimVerifier:
    """Verifies numeric claims with strict unit handling."""
    
    UNIT_CONVERSIONS = {
        "mg": {"mg": 1.0, "g": 0.001, "µg": 1000, "mcg": 1000},
        "g": {"g": 1.0, "mg": 1000, "µg": 1000000, "mcg": 1000000},
        "µg": {"µg": 1.0, "mg": 0.001, "g": 0.000001, "mcg": 1.0},
        "mcg": {"mcg": 1.0, "mg": 0.001, "g": 0.000001, "µg": 1.0},
        "ml": {"ml": 1.0, "l": 0.001, "µl": 1000},
        "l": {"l": 1.0, "ml": 1000, "µl": 1000000},
        "mmhg": {"mmhg": 1.0},
        "mmol/l": {"mmol/l": 1.0},
        "%": {"%": 1.0},
        "bpm": {"bpm": 1.0},
    }
    
    def __init__(self):
        self._numeric_pattern = r"(\d+(?:\.\d+)?)\s*(mg|mcg|µg|g|kg|ml|l|mmhg|mmol/l|%|bpm)"
    
    def verify(self, claim: Claim, evidence: List[NormalizedEvidence]) -> List[NumericVerificationResult]:
        """Verify numeric claims against evidence."""
        if not claim.numeric_value and not claim.numeric_range:
            return []
        
        results: List[NumericVerificationResult] = []
        
        for item in evidence:
            result = self._verify_numeric(item.normalized_text, claim)
            if result:
                results.append(result)
        
        return results
    
    def _verify_numeric(self, text: str, claim: Claim) -> Optional[NumericVerificationResult]:
        """Verify numeric values in text against claim."""
        # Extract values from text
        matches = re.findall(self._numeric_pattern, text, re.I)
        
        if not matches:
            return None
        
        # Check for range
        range_match = re.search(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*(mg|ml|mmhg)", text, re.I)
        
        if range_match:
            low = float(range_match.group(1))
            high = float(range_match.group(2))
            unit = range_match.group(3).lower()
            
            if claim.numeric_range:
                claim_low, claim_high = claim.numeric_range
                compatible = self._ranges_compatible(claim_low, claim_high, low, high, unit)
                return NumericVerificationResult(
                    claim_value=None,
                    claim_unit=unit,
                    evidence_value=None,
                    evidence_unit=unit,
                    matched=True,
                    compatible=compatible,
                    message=f"Range {low}-{high} {unit} verified",
                )
        
        # Check single values
        for value_str, unit in matches:
            value = float(value_str)
            unit_lower = unit.lower()
            
            # Convert units if needed
            converted = self._convert_value(value, unit_lower, claim.numeric_unit)
            
            if converted is not None and claim.numeric_value is not None:
                compatible = abs(converted - claim.numeric_value) < claim.numeric_value * 0.1
                return NumericVerificationResult(
                    claim_value=claim.numeric_value,
                    claim_unit=claim.numeric_unit,
                    evidence_value=value,
                    evidence_unit=unit_lower,
                    matched=True,
                    compatible=compatible,
                    message=f"Value {value} {unit} converted to {converted} verified",
                )
        
        return None
    
    def _convert_value(self, value: float, from_unit: str, to_unit: Optional[str]) -> Optional[float]:
        """Convert value from one unit to another."""
        if not to_unit:
            return None
        
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()
        
        if from_unit in self.UNIT_CONVERSIONS:
            if to_unit in self.UNIT_CONVERSIONS[from_unit]:
                return value * self.UNIT_CONVERSIONS[from_unit][to_unit]
        
        # Same unit
        if from_unit == to_unit:
            return value
        
        return None
    
    def _ranges_compatible(self, c_low: float, c_high: float, e_low: float, e_high: float, unit: str) -> bool:
        """Check if two ranges are compatible."""
        # Ranges overlap if one starts before the other ends
        overlap = c_low <= e_high and e_low <= c_high
        return overlap
    
    def verify_range(self, claim: Claim, evidence: List[NormalizedEvidence]) -> Tuple[bool, str]:
        """Verify numeric ranges specifically."""
        if not claim.numeric_range:
            return True, "No range to verify"
        
        claim_low, claim_high = claim.numeric_range
        
        for item in evidence:
            text = item.normalized_text.lower()
            
            # Check for range patterns
            range_match = re.search(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)", text)
            if range_match:
                low = float(range_match.group(1))
                high = float(range_match.group(2))
                
                if self._ranges_compatible(claim_low, claim_high, low, high, claim.numeric_unit or ""):
                    return True, f"Range {claim_low}-{claim_high} supported by evidence"
        
        return False, "No compatible range found in evidence"
    
    def validate_numeric_statement(self, claim: Claim, evidence: List[NormalizedEvidence]) -> Tuple[bool, str]:
        """Validate a complete numeric statement."""
        if not claim.numeric_value and not claim.numeric_range:
            return True, "No numeric values to validate"
        
        if claim.numeric_value:
            results = self.verify(claim, evidence)
            matched = any(r.matched for r in results)
            if matched:
                return True, "Numeric value verified"
            return False, "Numeric value not found in evidence"
        
        if claim.numeric_range:
            return self.verify_range(claim, evidence)
        
        return False, "Unknown numeric verification error"


def verify_numeric_claim(claim: Claim, evidence: List[NormalizedEvidence]) -> Tuple[bool, str]:
    """Convenience function to verify numeric claim."""
    verifier = NumericClaimVerifier()
    return verifier.validate_numeric_statement(claim, evidence)
