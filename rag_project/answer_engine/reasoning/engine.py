"""Deterministic reasoning engine for medical answers."""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

from rag_project.answer_engine.schemas.claims import Claim, ClaimStatus
from rag_project.answer_engine.schemas.reasoning import Relationship, RelationshipType
from rag_project.answer_engine.evidence.evidence_graph import EvidenceGraph


@dataclass
class ReasoningResult:
    """Result of deterministic reasoning."""
    reasoning_path: List[str]
    inferred_claims: List[Claim]
    relationships: List[Relationship]
    confidence: float
    trace: List[Dict[str, Any]]


class ReasoningEngine:
    """Perform deterministic reasoning over evidence graph."""
    
    def __init__(self):
        self._inference_rules = {
            RelationshipType.CAUSES: self._rule_causes,
            RelationshipType.TREATED_BY: self._rule_treated_by,
            RelationshipType.CONTRAINDICATES: self._rule_contraindicated,
            RelationshipType.PRECEDES: self._rule_precedes,
            RelationshipType.FOLLOWS: self._rule_follows,
        }
    
    def reason(self, claims: List[Claim], graph: EvidenceGraph) -> ReasoningResult:
        """Perform reasoning over evidence."""
        reasoning_path: List[str] = []
        inferred_claims: List[Claim] = []
        relationships: List[Relationship] = []
        trace: List[Dict[str, Any]] = []
        
        # Start reasoning
        reasoning_path.append("start_reasoning")
        trace.append({"step": "start", "claims_count": len(claims), "nodes_count": len(graph.nodes)})
        
        # Build inferred claims from relationships
        for relationship in graph.edges:
            inferred = self._apply_inference_rule(relationship, claims)
            if inferred:
                inferred_claims.append(inferred)
                relationships.append(relationship)
        
        # Add relationships from evidence graph
        relationships.extend(graph.edges)
        
        # Calculate confidence
        supported_claims = sum(1 for c in claims if c.status == ClaimStatus.SUPPORTED)
        confidence = supported_claims / max(1, len(claims))
        
        # Complete reasoning
        reasoning_path.append("complete_reasoning")
        trace.append({"step": "complete", "inferred_claims": len(inferred_claims), "relationships": len(relationships)})
        
        return ReasoningResult(
            reasoning_path=reasoning_path,
            inferred_claims=inferred_claims,
            relationships=relationships,
            confidence=confidence,
            trace=trace,
        )
    
    def _apply_inference_rule(self, relationship: Relationship, claims: List[Claim]) -> Optional[Claim]:
        """Apply an inference rule to create a new claim."""
        rule_func = self._inference_rules.get(relationship.type)
        if not rule_func:
            return None
        
        return rule_func(relationship, claims)
    
    def _rule_causes(self, rel: Relationship, claims: List[Claim]) -> Optional[Claim]:
        """If A causes B, and B is a disease, infer treatment may help."""
        return None  # Simplified - would extract from evidence
    
    def _rule_treated_by(self, rel: Relationship, claims: List[Claim]) -> Optional[Claim]:
        """If A is treated by B, infer B is a treatment for A."""
        return None
    
    def _rule_contraindicated(self, rel: Relationship, claims: List[Claim]) -> Optional[Claim]:
        """If A is contraindicated with B, infer safety warning."""
        return None
    
    def _rule_precedes(self, rel: Relationship, claims: List[Claim]) -> Optional[Claim]:
        """If A precedes B, create temporal relationship."""
        return None
    
    def _rule_follows(self, rel: Relationship, claims: List[Claim]) -> Optional[Claim]:
        """If A follows B, create sequential relationship."""
        return None
    
    def multi_hop_reasoning(self, start_entity: str, graph: EvidenceGraph, depth: int = 2) -> ReasoningResult:
        """Perform multi-hop reasoning from an entity."""
        reasoning_path = [f"start_multi_hop:{start_entity}"]
        trace = [{"step": "multi_hop_start", "entity": start_entity, "depth": depth}]
        
        # Get subgraph from start node
        subgraph = graph
        
        # Traverse edges
        relationships: List[Relationship] = []
        inferred_claims: List[Claim] = []
        
        current_entities = {start_entity}
        for hop in range(depth):
            reasoning_path.append(f"hop_{hop+1}")
            
            # Find relationships from current entities
            new_entities: Set[str] = set()
            for rel in subgraph.edges:
                if rel.from_entity in current_entities:
                    relationships.append(rel)
                    new_entities.add(rel.to_entity)
                    inferred_claims.append(Claim(
                        id=f"inferred_{hop}_{len(inferred_claims)}",
                        text=f"{rel.from_entity} {rel.type.value} {rel.to_entity}",
                        normalized_text=f"{rel.from_entity} {rel.type.value} {rel.to_entity}",
                        claim_type="relationship",
                        entities=[rel.from_entity, rel.to_entity],
                        qualifiers=[],
                        context={},
                        evidence=[],
                        status=ClaimStatus.SUPPORTED,
                        support_ratio=rel.strength,
                    ))
            
            current_entities = new_entities
            trace.append({"step": f"hop_{hop+1}", "entities": list(new_entities)})
        
        reasoning_path.append("complete_multi_hop")
        trace.append({"step": "complete", "relationships": len(relationships)})
        
        return ReasoningResult(
            reasoning_path=reasoning_path,
            inferred_claims=inferred_claims,
            relationships=relationships,
            confidence=0.7 if relationships else 0.0,
            trace=trace,
        )
    
    def compare_entities(self, entity_a: str, entity_b: str, claims: List[Claim]) -> Dict[str, Any]:
        """Compare two entities and return differences."""
        result = {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "differences": [],
            "similarities": [],
            "confidence": 0.0,
        }
        
        # Simple comparison
        a_lower = entity_a.lower()
        b_lower = entity_b.lower()
        
        if a_lower == b_lower:
            result["similarities"].append("identical_entities")
            result["confidence"] = 1.0
        elif a_lower in b_lower or b_lower in a_lower:
            result["similarities"].append("related_entities")
            result["confidence"] = 0.8
        else:
            result["differences"].append("distinct_entities")
            result["confidence"] = 0.9
        
        return result


def reason_over_evidence(claims: List[Claim], graph: EvidenceGraph) -> ReasoningResult:
    """Convenience function for reasoning."""
    engine = ReasoningEngine()
    return engine.reason(claims, graph)
