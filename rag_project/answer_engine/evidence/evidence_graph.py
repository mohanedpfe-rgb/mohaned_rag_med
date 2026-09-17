"""Evidence graph for deterministic medical reasoning."""
from __future__ import annotations

from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from rag_project.answer_engine.evidence.normalizer import NormalizedEvidence
from rag_project.answer_engine.schemas.reasoning import Relationship, RelationshipType


@dataclass
class EvidenceNode:
    """Node in the evidence graph."""
    id: str
    text: str
    type: str  # concept, claim, fact, entity, etc.
    evidence: List[NormalizedEvidence]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceGraph:
    """Graph representation of evidence."""
    nodes: Dict[str, EvidenceNode]
    edges: List[Relationship]
    root_nodes: List[str]
    leaf_nodes: List[str]


class EvidenceGraphBuilder:
    """Builds evidence graphs from normalized evidence."""
    
    def __init__(self):
        self._entity_patterns = {
            "disease": ["diabetes", "hypertension", "obesity", "cardiovascular"],
            "drug": ["metformin", "lisinopril", "simvastatin", "insulin"],
            "test": ["hba1c", "creatinine", "cholesterol", "bmi"],
            "symptom": ["pain", "fever", "headache", "nausea"],
            "procedure": ["surgery", "biopsy", "angiography", "echocardiography"],
        }
    
    def build(self, evidence: List[NormalizedEvidence]) -> EvidenceGraph:
        """Build evidence graph from normalized evidence."""
        nodes: Dict[str, EvidenceNode] = {}
        edges: List[Relationship] = []
        entity_map: Dict[str, Set[str]] = defaultdict(set)  # entity -> node_ids
        
        # Create nodes for each evidence item
        for item in evidence:
            node_id = item.evidence_item.id
            
            # Detect node type
            node_type = self._detect_node_type(item.normalized_text)
            
            # Create node
            node = EvidenceNode(
                id=node_id,
                text=item.normalized_text,
                type=node_type,
                evidence=[item],
                metadata={
                    "section": item.evidence_item.source_location.section,
                    "page": item.evidence_item.source_location.page,
                    "document_id": item.evidence_item.source_location.document_id,
                },
            )
            nodes[node_id] = node
            
            # Extract entities and map to nodes
            entities = self._extract_entities(item.normalized_text)
            for entity in entities:
                entity_map[entity].add(node_id)
        
        # Create edges between related nodes
        edges = self._create_edges(nodes, entity_map)
        
        # Identify root and leaf nodes
        root_nodes = self._find_root_nodes(nodes, edges)
        leaf_nodes = self._find_leaf_nodes(nodes, edges)
        
        return EvidenceGraph(
            nodes=nodes,
            edges=edges,
            root_nodes=root_nodes,
            leaf_nodes=leaf_nodes,
        )
    
    def _detect_node_type(self, text: str) -> str:
        """Detect the type of node."""
        text_lower = text.lower()
        
        if any(d in text_lower for d in self._entity_patterns["disease"]):
            return "disease"
        if any(d in text_lower for d in self._entity_patterns["drug"]):
            return "drug"
        if any(d in text_lower for d in self._entity_patterns["test"]):
            return "test"
        if any(d in text_lower for d in self._entity_patterns["symptom"]):
            return "symptom"
        if any(d in text_lower for d in self._entity_patterns["procedure"]):
            return "procedure"
        
        return "fact"
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text."""
        entities = []
        text_lower = text.lower()
        
        for entity_type, patterns in self._entity_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    entities.append(pattern)
        
        return list(set(entities))
    
    def _create_edges(self, nodes: Dict[str, EvidenceNode], entity_map: Dict[str, Set[str]]) -> List[Relationship]:
        """Create relationships between nodes."""
        edges: List[Relationship] = []
        
        # Create edges based on shared entities
        for entity, node_ids in entity_map.items():
            node_list = list(node_ids)
            for i, node1_id in enumerate(node_list):
                for node2_id in node_list[i+1:]:
                    edges.append(Relationship(
                        type=RelationshipType.ASSOCIATED_WITH,
                        from_entity=entity,
                        to_entity=node1_id,
                        evidence=[node2_id],
                        strength=0.7,
                    ))
        
        # Create edges based on section relationships
        section_nodes: Dict[str, List[str]] = defaultdict(list)
        for node_id, node in nodes.items():
            section = node.metadata.get("section", "")
            if section:
                section_nodes[section].append(node_id)
        
        for section, node_ids in section_nodes.items():
            for i, node1_id in enumerate(node_ids):
                for node2_id in node_ids[i+1:]:
                    edges.append(Relationship(
                        type=RelationshipType.PART_OF,
                        from_entity=node1_id,
                        to_entity=section,
                        evidence=[],
                        strength=0.8,
                    ))
        
        return edges
    
    def _find_root_nodes(self, nodes: Dict[str, EvidenceNode], edges: List[Relationship]) -> List[str]:
        """Find nodes with no incoming edges."""
        has_incoming: Set[str] = set()
        
        for edge in edges:
            has_incoming.add(edge.to_entity)
        
        return [node_id for node_id in nodes if node_id not in has_incoming]
    
    def _find_leaf_nodes(self, nodes: Dict[str, EvidenceNode], edges: List[Relationship]) -> List[str]:
        """Find nodes with no outgoing edges."""
        has_outgoing: Set[str] = set()
        
        for edge in edges:
            has_outgoing.add(edge.from_entity)
        
        return [node_id for node_id in nodes if node_id not in has_outgoing]
    
    def get_subgraph(self, graph: EvidenceGraph, start_node: str, depth: int = 2) -> EvidenceGraph:
        """Get subgraph starting from a node."""
        visited: Set[str] = set()
        nodes: Dict[str, EvidenceNode] = {}
        edges: List[Relationship] = []
        
        def traverse(node_id: str, current_depth: int):
            if current_depth > depth or node_id in visited:
                return
            
            visited.add(node_id)
            
            if node_id in graph.nodes:
                nodes[node_id] = graph.nodes[node_id]
            
            # Add outgoing edges
            for edge in graph.edges:
                if edge.from_entity == node_id:
                    edges.append(edge)
                    traverse(edge.to_entity, current_depth + 1)
        
        traverse(start_node, 0)
        
        return EvidenceGraph(
            nodes=nodes,
            edges=edges,
            root_nodes=[start_node] if start_node in nodes else [],
            leaf_nodes=list(visited),
        )


def build_evidence_graph(evidence: List[NormalizedEvidence]) -> EvidenceGraph:
    """Convenience function to build evidence graph."""
    builder = EvidenceGraphBuilder()
    return builder.build(evidence)
