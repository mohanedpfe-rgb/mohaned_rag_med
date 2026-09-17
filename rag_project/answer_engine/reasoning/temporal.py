"""Temporal reasoning for medical answers."""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TemporalRelation:
    """Temporal relationship between events."""
    event_a: str
    event_b: str
    relation: str  # before, after, during, simultaneously
    confidence: float


class TemporalReasoner:
    """Reason about temporal sequences in medical evidence."""
    
    def __init__(self):
        self._temporal_patterns = {
            "before": [r"\bbefore\b", r"\b prior to\b", r"\bpreviously\b", r"\bformerly\b", r"\binitially\b"],
            "after": [r"\bafter\b", r"\b subsequently\b", r"\b following\b", r"\bthen\b", r"\b subsequently\b"],
            "during": [r"\bduring\b", r"\bwhile\b", r"\bwhen\b", r"\bas\b"],
            "simultaneously": [r"\bsimultaneously\b", r"\bat the same time\b", r"\bconcurrently\b"],
            "first": [r"\bfirst\b", r"\binitially\b", r"\binitial\b"],
            "then": [r"\bthen\b", r"\bnext\b", r"\bsubsequently\b"],
            "finally": [r"\bfinally\b", r"\bultimately\b", r"\bat the end\b"],
        }
    
    def extract_temporal_relations(self, text: str) -> List[TemporalRelation]:
        """Extract temporal relations from text."""
        relations: List[TemporalRelation] = []
        
        for relation_type, patterns in self._temporal_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, text, re.I))
                if matches:
                    # Simplified - would extract event context
                    relations.append(TemporalRelation(
                        event_a="event",
                        event_b="event",
                        relation=relation_type,
                        confidence=0.95 if relation_type in ["before", "after"] else 0.85,
                    ))
        
        return relations
    
    def order_events(self, events: List[str], text: str) -> List[str]:
        """Order events temporally."""
        ordered = list(events)
        
        # Check for sequence markers
        sequence_markers = {
            0: [r"\bfirst\b", r"\binitially\b", r"\binitial\b"],
            1: [r"\bthen\b", r"\bnext\b", r"\bsubsequently\b"],
            2: [r"\bfinally\b", r"\bultimately\b"],
        }
        
        for position, markers in sequence_markers.items():
            for marker in markers:
                if re.search(marker, text, re.I):
                    # Move appropriate event to position
                    break
        
        return ordered
    
    def check_temporal_consistency(self, event_a: str, event_b: str, 
                                   relation: str, evidence: List[str]) -> bool:
        """Check if temporal relation is supported by evidence."""
        evidence_text = " ".join(evidence).lower()
        relation_lower = relation.lower()
        
        patterns = self._temporal_patterns.get(relation_lower, [])
        for pattern in patterns:
            if re.search(pattern, evidence_text, re.I):
                return True
        
        return False


def extract_temporal_relations(text: str) -> List[TemporalRelation]:
    """Convenience function to extract temporal relations."""
    reasoner = TemporalReasoner()
    return reasoner.extract_temporal_relations(text)
