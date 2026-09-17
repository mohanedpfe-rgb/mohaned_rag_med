"""Causal reasoning for medical answers."""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class CausalRelationship:
    """Causal relationship between factors."""
    cause: str
    effect: str
    strength: float  # 0-1
    evidence: List[str]
    confidence: float


class CausalReasoner:
    """Reason about causal relationships in medical evidence."""
    
    def __init__(self):
        self._causal_keywords = [
            "causes", "cause", "caused by", "due to", "leads to", "results in",
            "increases risk", "decreases risk", "predisposes", "predispose",
            "contributes to", "associated with", "linked to",
        ]
    
    def extract_causal_relationships(self, text: str) -> List[CausalRelationship]:
        """Extract causal relationships from text."""
        relationships: List[CausalRelationship] = []
        
        text_lower = text.lower()
        
        # Pattern: X causes Y
        for kw in self._causal_keywords:
            if kw in text_lower:
                # Extract subject and object
                parts = text_lower.split(kw)
                if len(parts) >= 2:
                    cause = parts[0].strip()
                    effect = parts[1].strip()
                    
                    # Clean up
                    cause = re.sub(r"^\s*(?:the|a|an)\s+", "", cause)
                    effect = re.sub(r"^\s*(?:the|a|an)\s+", "", effect)
                    
                    if cause and effect:
                        relationships.append(CausalRelationship(
                            cause=cause,
                            effect=effect,
                            strength=0.8 if kw in ["causes", "leads to", "results in"] else 0.6,
                            evidence=[text],
                            confidence=0.85,
                        ))
        
        return relationships
    
    def verify_causal_claim(self, cause: str, effect: str, evidence: List[str]) -> Dict[str, Any]:
        """Verify a causal claim against evidence."""
        result = {
            "cause": cause,
            "effect": effect,
            "supported": False,
            "evidence_matches": [],
            "confidence": 0.0,
            "reason": "",
        }
        
        for ev in evidence:
            ev_lower = ev.lower()
            
            # Check for causal keywords with both entities
            if cause.lower() in ev_lower and effect.lower() in ev_lower:
                result["evidence_matches"].append(ev[:200])
                
                # Check for causal language
                for kw in self._causal_keywords:
                    if kw in ev_lower:
                        result["supported"] = True
                        result["confidence"] = max(result["confidence"], 0.85)
                        result["reason"] = f"Found causal language: {kw}"
                        break
        
        if not result["supported"]:
            result["reason"] = "No causal evidence found"
        
        return result
    
    def infer_causal_chain(self, events: List[str]) -> Optional[List[str]]:
        """Infer a causal chain from events."""
        if len(events) < 2:
            return None
        
        # Simple chain: A -> B -> C
        return events


def extract_causal_relationships(text: str) -> List[CausalRelationship]:
    """Convenience function to extract causal relationships."""
    reasoner = CausalReasoner()
    return reasoner.extract_causal_relationships(text)
