"""Comparison engine for deterministic medical answers."""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ComparisonResult:
    """Result of comparing two items."""
    item_a: str
    item_b: str
    feature: str
    a_value: Optional[str]
    b_value: Optional[str]
    difference: Optional[str]
    confidence: float


class ComparatorEngine:
    """Deterministic comparison engine for medical facts."""
    
    def __init__(self):
        self._comparison_patterns = {
            "greater": [r"\bgreater than\b", r"\bhigher than\b", r"\bmore than\b", r"\babove\b"],
            "lesser": [r"\bless than\b", r"\blower than\b", r"\bfewer than\b", r"\bbelow\b"],
            "equal": [r"\bequal to\b", r"\bsame as\b", r"\bidentical\b"],
            "different": [r"\bdifferent from\b", r"\bunlike\b"],
        }
    
    def compare(self, item_a: str, item_b: str, feature: str, 
                evidence_a: str, evidence_b: str) -> ComparisonResult:
        """Compare two items on a specific feature."""
        a_lower = evidence_a.lower()
        b_lower = evidence_b.lower()
        feature_lower = feature.lower()
        
        # Extract values
        a_value = self._extract_value(evidence_a, feature_lower)
        b_value = self._extract_value(evidence_b, feature_lower)
        
        # Determine difference
        difference = None
        confidence = 0.0
        
        if a_value and b_value:
            # Compare values
            diff = self._compare_values(a_value, b_value)
            difference = diff
            confidence = 0.85 if diff != "unknown" else 0.5
        
        return ComparisonResult(
            item_a=item_a,
            item_b=item_b,
            feature=feature,
            a_value=a_value,
            b_value=b_value,
            difference=difference,
            confidence=confidence,
        )
    
    def _extract_value(self, text: str, feature: str) -> Optional[str]:
        """Extract a value for a feature from text."""
        # Check for numeric values
        match = re.search(r"(\d+(?:\.\d+)?)\s*(mg|ml|mmhg|mmol/l|%|bpm)", text, re.I)
        if match:
            return f"{match.group(1)} {match.group(2)}"
        
        # Check for categorical values
        for category in ["increased", "decreased", "normal", "elevated", "low", "high"]:
            if category in text.lower():
                return category
        
        return None
    
    def _compare_values(self, val_a: str, val_b: str) -> str:
        """Compare two values and return difference."""
        # Extract numeric parts
        match_a = re.search(r"(\d+(?:\.\d+)?)", val_a)
        match_b = re.search(r"(\d+(?:\.\d+)?)", val_b)
        
        if match_a and match_b:
            num_a = float(match_a.group(1))
            num_b = float(match_b.group(1))
            
            if abs(num_a - num_b) < 0.01:
                return "equal"
            elif num_a > num_b:
                return f"{num_a - num_b} greater"
            else:
                return f"{num_b - num_a} lesser"
        
        # Text comparison
        if val_a == val_b:
            return "equal"
        
        return "different"
    
    def compare_multiple_features(self, item_a: str, item_b: str, 
                                   features: List[str], 
                                   evidence_a: str, evidence_b: str) -> List[ComparisonResult]:
        """Compare items on multiple features."""
        results = []
        for feature in features:
            result = self.compare(item_a, item_b, feature, evidence_a, evidence_b)
            results.append(result)
        return results
    
    def detect_differences(self, text_a: str, text_b: str) -> List[str]:
        """Detect differences between two texts."""
        differences = []
        
        # Check length difference
        if len(text_a) != len(text_b):
            differences.append("length_difference")
        
        # Check keyword presence
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        
        unique_a = words_a - words_b
        unique_b = words_b - words_a
        
        if unique_a:
            differences.append(f"unique_to_a: {list(unique_a)[:3]}")
        if unique_b:
            differences.append(f"unique_to_b: {list(unique_b)[:3]}")
        
        return differences


def compare_items(item_a: str, item_b: str, feature: str, 
                  evidence_a: str, evidence_b: str) -> ComparisonResult:
    """Convenience function to compare items."""
    engine = ComparatorEngine()
    return engine.compare(item_a, item_b, feature, evidence_a, evidence_b)
