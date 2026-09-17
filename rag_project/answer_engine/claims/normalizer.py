"""Claim normalizer for deterministic medical answers."""
from __future__ import annotations

import re
from typing import List, Dict, Any
from dataclasses import dataclass

from rag_project.answer_engine.schemas.claims import Claim, ClaimStatus


@dataclass
class NormalizedClaim:
    """Normalized claim with standardized format."""
    original_claim: Claim
    normalized_text: str
    canonical_form: str
    entities_normalized: List[str]
    qualifiers_preserved: List[str]
    numeric_normalized: Dict[str, Any]


class ClaimNormalizer:
    """Normalizes claims for consistent processing."""
    
    def __init__(self):
        self._common_normalizations = {
            "diabetes mellitus": "diabetes",
            "hta": "hypertension",
            "dm": "diabetes",
            "cvd": "cardiovascular disease",
            "ckd": "chronic kidney disease",
        }
    
    def normalize(self, claim: Claim) -> NormalizedClaim:
        """Normalize a single claim."""
        # Normalize text
        norm_text = re.sub(r"\s+", " ", claim.normalized_text.lower()).strip()
        
        # Apply common normalizations
        canonical = norm_text
        for pattern, replacement in self._common_normalizations.items():
            canonical = re.sub(r"\b" + re.escape(pattern) + r"\b", replacement, canonical, flags=re.I)
        
        # Normalize entities
        entities_normalized = [self._normalize_entity(e) for e in claim.entities]
        
        # Preserve qualifiers
        qualifiers_preserved = claim.qualifiers
        
        # Normalize numeric values
        numeric_normalized = {}
        if claim.numeric_value:
            numeric_normalized = {
                "value": claim.numeric_value,
                "unit": self._normalize_unit(claim.numeric_unit),
                "normalized_value": self._normalize_to_base_unit(claim.numeric_value, claim.numeric_unit),
            }
        
        return NormalizedClaim(
            original_claim=claim,
            normalized_text=norm_text,
            canonical_form=canonical,
            entities_normalized=entities_normalized,
            qualifiers_preserved=qualifiers_preserved,
            numeric_normalized=numeric_normalized,
        )
    
    def normalize_batch(self, claims: List[Claim]) -> List[NormalizedClaim]:
        """Normalize multiple claims."""
        return [self.normalize(c) for c in claims]
    
    def _normalize_entity(self, entity: str) -> str:
        """Normalize an entity name."""
        return re.sub(r"\s+", " ", entity.lower().strip())
    
    def _normalize_unit(self, unit: Optional[str]) -> Optional[str]:
        """Normalize a unit to standard form."""
        if not unit:
            return None
        
        unit_map = {
            "mcg": "µg",
            "mg": "mg",
            "ml": "mL",
            "g": "g",
            "kg": "kg",
            "mmhg": "mmHg",
            "mmol/l": "mmol/L",
        }
        
        return unit_map.get(unit.lower(), unit.lower())
    
    def _normalize_to_base_unit(self, value: float, unit: Optional[str]) -> Optional[float]:
        """Convert to base unit for comparison."""
        if not unit:
            return value
        
        unit_lower = unit.lower()
        
        if unit_lower == "µg" or unit_lower == "mcg":
            return value / 1000  # Convert to mg
        if unit_lower == "g":
            return value * 1000  # Convert to mg
        if unit_lower == "mg":
            return value
        
        return value


def normalize_claim(claim: Claim) -> NormalizedClaim:
    """Convenience function to normalize a claim."""
    normalizer = ClaimNormalizer()
    return normalizer.normalize(claim)
