"""Reasoning schemas for deterministic medical answers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class RelationshipType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    QUALIFIES = "qualifies"
    DEPENDS_ON = "depends_on"
    CAUSES = "causes"
    ASSOCIATED_WITH = "associated_with"
    MEASURED_BY = "measured_by"
    TREATED_BY = "treated_by"
    CONTRAINDICATES = "contraindicates"
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    SUBTYPE_OF = "subtype_of"
    PART_OF = "part_of"
    SAME_AS = "same_as"


@dataclass(frozen=True)
class Relationship:
    """Relationship between two entities or claims."""
    type: RelationshipType
    from_entity: str
    to_entity: str
    evidence: List[str]  # source IDs supporting this relationship
    strength: float  # 0-1
    context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "from_entity": self.from_entity,
            "to_entity": self.to_entity,
            "evidence": self.evidence,
            "strength": self.strength,
            "context": self.context,
        }


@dataclass(frozen=True)
class Conflict:
    """A conflict between two pieces of evidence."""
    id: str
    claim_a: str
    claim_b: str
    evidence_a: List[str]
    evidence_b: List[str]
    source_versions: Dict[str, str]
    confidence: float  # how certain the conflict is
    resolution: Optional[str] = None  # how it was resolved
    resolution_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "claim_a": self.claim_a,
            "claim_b": self.claim_b,
            "evidence_a": self.evidence_a,
            "evidence_b": self.evidence_b,
            "source_versions": self.source_versions,
            "confidence": self.confidence,
            "resolution": self.resolution,
            "resolution_reason": self.resolution_reason,
        }


@dataclass(frozen=True)
class ReasoningTrace:
    """A trace of reasoning steps."""
    step_id: str
    step_type: str  # retrieval, extraction, verification, reasoning, compilation
    input_data: Dict[str, Any]
    process: str
    output_data: Dict[str, Any]
    confidence: float
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "input_data": self.input_data,
            "process": self.process,
            "output_data": self.output_data,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }
