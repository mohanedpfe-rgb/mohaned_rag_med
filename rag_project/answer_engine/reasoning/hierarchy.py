"""Hierarchical reasoning for medical information."""
from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class HierarchyResult:
    """Result of hierarchical reasoning."""
    level: int
    entity: str
    parent: Optional[str]
    children: List[str]
    relationships: List[Dict[str, Any]]
    confidence: float


class HierarchyReasoner:
    """Reasoner for hierarchical medical concepts."""
    
    CONDITION_HIERARCHY = {
        "diabetes": ["diabetes_mellitus", "diabetes_insipidus"],
        "diabetes_mellitus": ["type_1_diabetes", "type_2_diabetes"],
        "hypertension": ["essential_hypertension", "secondary_hypertension"],
        "cancer": ["carcinoma", "sarcoma", "lymphoma", "leukemia"],
        "cardiovascular_disease": ["heart_disease", "stroke", "hypertension"],
    }
    
    def __init__(self):
        self.parent_map = {}
        for parent, children in self.CONDITION_HIERARCHY.items():
            for child in children:
                self.parent_map[child] = parent
    
    def get_parent(self, entity: str) -> Optional[str]:
        """Get parent of an entity."""
        return self.parent_map.get(entity)
    
    def get_children(self, entity: str) -> List[str]:
        """Get children of an entity."""
        return self.CONDITION_HIERARCHY.get(entity, [])
    
    def get_ancestors(self, entity: str) -> List[str]:
        """Get all ancestors of an entity."""
        ancestors = []
        current = entity
        while current in self.parent_map:
            parent = self.parent_map[current]
            ancestors.append(parent)
            current = parent
        return ancestors
    
    def get_descendants(self, entity: str) -> List[str]:
        """Get all descendants of an entity."""
        descendants = []
        stack = [entity]
        while stack:
            current = stack.pop()
            children = self.get_children(current)
            descendants.extend(children)
            stack.extend(children)
        return descendants
    
    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """Check if ancestor is an ancestor of descendant."""
        return descendant in self.get_descendants(ancestor)
    
    def is_descendant(self, descendant: str, ancestor: str) -> bool:
        """Check if descendant is a descendant of ancestor."""
        return ancestor in self.get_descendants(descendant)
    
    def find_common_ancestor(self, entities: List[str]) -> Optional[str]:
        """Find common ancestor of multiple entities."""
        if not entities:
            return None
        if len(entities) == 1:
            return entities[0]
        ancestors = set(self.get_ancestors(entities[0]))
        ancestors.add(entities[0])
        for entity in entities[1:]:
            entity_ancestors = set(self.get_ancestors(entity))
            entity_ancestors.add(entity)
            ancestors = ancestors & entity_ancestors
        if ancestors:
            return max(ancestors, key=lambda a: len(self.get_ancestors(a)))
        return None
    
    def reason_over_entities(
        self,
        entities: List[str],
    ) -> List[HierarchyResult]:
        """Perform hierarchical reasoning over entities."""
        results = []
        for entity in entities:
            parent = self.get_parent(entity)
            children = self.get_children(entity)
            ancestors = self.get_ancestors(entity)
            descendants = self.get_descendants(entity)
            relationships = []
            if parent:
                relationships.append({"type": "is_a", "target": parent, "confidence": 0.95})
            for child in children:
                relationships.append({"type": "has_subtype", "target": child, "confidence": 0.9})
            for ancestor in ancestors[:3]:
                relationships.append({"type": "is_subtype_of", "target": ancestor, "confidence": 0.85})
            for descendant in descendants[:3]:
                relationships.append({"type": "has_supertype", "target": descendant, "confidence": 0.85})
            depth = len(ancestors)
            confidence = max(0.5, 0.95 - (depth * 0.05))
            results.append(HierarchyResult(
                level=depth,
                entity=entity,
                parent=parent,
                children=children,
                relationships=relationships,
                confidence=confidence,
            ))
        return results
    
    def infer_from_hierarchy(
        self,
        entity: str,
        property_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Infer properties from hierarchy."""
        ancestors = self.get_ancestors(entity)
        for ancestor in ancestors:
            if ancestor in self.CONDITION_HIERARCHY:
                return {
                    "entity": ancestor,
                    "property": property_name,
                    "inferred": True,
                    "confidence": 0.8,
                    "reason": f"Inherited from ancestor {ancestor}",
                }
        return None


def reason_hierarchy(entities: List[str]) -> List[Dict[str, Any]]:
    """Convenience function for hierarchical reasoning."""
    reasoner = HierarchyReasoner()
    results = reasoner.reason_over_entities(entities)
    return [r.__dict__ for r in results]