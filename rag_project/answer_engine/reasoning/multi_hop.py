"""Multi-hop reasoning for medical answers."""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

from rag_project.answer_engine.schemas.claims import Claim, ClaimStatus


@dataclass
class HopPath:
    """A multi-hop reasoning path."""
    start: str
    hops: List[str]
    end: str
    confidence: float
    evidence: List[str]


class MultiHopReasoner:
    """Perform multi-hop reasoning across evidence."""
    
    def __init__(self, max_hops: int = 3):
        self.max_hops = max_hops
    
    def reason(self, question: str, claims: List[Claim], evidence: List[str]) -> Optional[HopPath]:
        """Perform multi-hop reasoning to answer question."""
        # Identify query entities
        query_entities = self._extract_query_entities(question)
        
        if not query_entities:
            return None
        
        # Build graph
        graph = self._build_evidence_graph(claims, evidence)
        
        # Find path
        path = self._find_path(query_entities[0], query_entities[-1], graph)
        
        if path:
            return HopPath(
                start=query_entities[0],
                hops=path["hops"],
                end=query_entities[-1],
                confidence=path["confidence"],
                evidence=path["evidence"],
            )
        
        return None
    
    def _extract_query_entities(self, question: str) -> List[str]:
        """Extract entities from question."""
        # Common medical entities
        entities = {
            "diabetes", "hypertension", "obesity", "cardiovascular", "cancer",
            "metformin", "lisinopril", "simvastatin", "insulin", "aspirin",
            "hba1c", "creatinine", "cholesterol", "bmi",
        }
        
        question_lower = question.lower()
        found = [e for e in entities if e in question_lower]
        
        return found
    
    def _build_evidence_graph(self, claims: List[Claim], 
                               evidence: List[str]) -> Dict[str, List[str]]:
        """Build a graph from claims and evidence."""
        graph: Dict[str, List[str]] = {}
        
        all_text = " ".join([c.normalized_text for c in claims] + evidence).lower()
        
        # Extract entities
        entities = self._extract_query_entities(all_text)
        
        # Build adjacency
        for entity in entities:
            graph[entity] = []
            for other in entities:
                if entity != other and other in all_text:
                    graph[entity].append(other)
        
        return graph
    
    def _find_path(self, start: str, end: str, 
                   graph: Dict[str, List[str]]) -> Optional[Dict[str, Any]]:
        """Find a path from start to end using BFS."""
        if start == end:
            return {"hops": [], "confidence": 1.0, "evidence": []}
        
        # BFS
        queue = [(start, [start], [])]
        visited = {start}
        
        while queue:
            current, path, evidence = queue.pop(0)
            
            if current == end:
                return {"hops": path, "confidence": 0.7, "evidence": evidence}
            
            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor], evidence))
        
        return None
    
    def chain_claims(self, claims: List[Claim]) -> List[Claim]:
        """Chain related claims together."""
        chained: List[Claim] = []
        
        # Sort claims by confidence
        sorted_claims = sorted(claims, key=lambda c: c.support_ratio, reverse=True)
        
        # Build chains
        for claim in sorted_claims[:5]:  # Top 5 claims
            if claim.status == ClaimStatus.SUPPORTED:
                # Check if this claim can extend an existing chain
                extended = False
                for existing in chained:
                    if self._claims_related(claim, existing):
                        # Merge claims
                        existing.evidence.extend(claim.evidence)
                        existing.support_ratio = max(existing.support_ratio, claim.support_ratio)
                        extended = True
                        break
                
                if not extended:
                    chained.append(claim)
        
        return chained
    
    def _claims_related(self, claim_a: Claim, claim_b: Claim) -> bool:
        """Check if two claims are related."""
        entities_a = set(claim_a.entities)
        entities_b = set(claim_b.entities)
        
        return bool(entities_a & entities_b)


def multi_hop_reason(question: str, claims: List[Claim], 
                     evidence: List[str]) -> Optional[HopPath]:
    """Convenience function for multi-hop reasoning."""
    reasoner = MultiHopReasoner()
    return reasoner.reason(question, claims, evidence)
