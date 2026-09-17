"""Relationship reasoning for medical answers."""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RelationshipFact:
    """A relationship between entities."""
    entity_a: str
    entity_b: str
    relationship_type: str
    evidence: str
    confidence: float


class RelationshipReasoner:
    """Reason about relationships between medical entities."""
    
    def __init__(self):
        self._relationship_patterns = {
            "treatment": [r"treat", r"therapy", r"medication", r"drug"],
            "diagnosis": [r"diagnose", r"diagnosis", r"test", r"criteria"],
            "symptom": [r"symptom", r"sign", r"manifestation"],
            "cause": [r"cause", r"causes", r"risk factor"],
            "contraindication": [r"contraindication", r"contraindicated"],
            "interaction": [r"interaction", r"interact"],
            "mechanism": [r"mechanism", r"pathway", r"pathophysiology"],
        }
    
    def extract_relationships(self, text: str) -> List[RelationshipFact]:
        """Extract relationships from text."""
        relationships: List[RelationshipFact] = []
        
        text_lower = text.lower()
        
        # Extract entity pairs
        entities = self._extract_entities(text)
        
        for entity_a in entities:
            for entity_b in entities:
                if entity_a == entity_b:
                    continue
                
                # Check for relationship patterns
                for rel_type, patterns in self._relationship_patterns.items():
                    for pattern in patterns:
                        if pattern in text_lower:
                            # Check if entities appear near each other
                            if self._are_related(entity_a, entity_b, text):
                                relationships.append(RelationshipFact(
                                    entity_a=entity_a,
                                    entity_b=entity_b,
                                    relationship_type=rel_type,
                                    evidence=text[:500],
                                    confidence=0.7,
                                ))
                                break
        
        return relationships
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text."""
        # Simplified - would use proper NER
        common_entities = [
            "diabetes", "hypertension", "obesity", "cardiovascular", "cancer",
            "metformin", "lisinopril", "simvastatin", "insulin", "aspirin",
            "hba1c", "creatinine", "cholesterol", "bmi",
        ]
        
        text_lower = text.lower()
        found = [e for e in common_entities if e in text_lower]
        
        return list(set(found))
    
    def _are_related(self, entity_a: str, entity_b: str, text: str) -> bool:
        """Check if two entities appear related in text."""
        # Check proximity
        pos_a = text.lower().find(entity_a.lower())
        pos_b = text.lower().find(entity_b.lower())
        
        if pos_a == -1 or pos_b == -1:
            return False
        
        # Within 100 characters
        return abs(pos_a - pos_b) < 100
    
    def find_relationship(self, entity_a: str, entity_b: str, 
                          relationship_type: str, evidence: List[str]) -> Dict[str, Any]:
        """Check if a relationship exists between entities."""
        result = {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "relationship_type": relationship_type,
            "found": False,
            "evidence_found": [],
            "confidence": 0.0,
        }
        
        for ev in evidence:
            ev_lower = ev.lower()
            
            if entity_a.lower() in ev_lower and entity_b.lower() in ev_lower:
                if self._relationship_type_matches(relationship_type, ev):
                    result["found"] = True
                    result["evidence_found"].append(ev[:300])
                    result["confidence"] = max(result["confidence"], 0.85)
        
        return result
    
    def _relationship_type_matches(self, rel_type: str, evidence: str) -> bool:
        """Check if evidence supports the relationship type."""
        patterns = self._relationship_patterns.get(rel_type, [])
        evidence_lower = evidence.lower()
        
        return any(p in evidence_lower for p in patterns)


def extract_relationships(text: str) -> List[RelationshipFact]:
    """Convenience function to extract relationships."""
    reasoner = RelationshipReasoner()
    return reasoner.extract_relationships(text)
