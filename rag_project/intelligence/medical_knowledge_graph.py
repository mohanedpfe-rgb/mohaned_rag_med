"""Deterministic medical knowledge graph built from indexed chunk metadata/text.

This is intentionally lightweight. It does not claim to be a clinical ontology;
it records document-derived entities and explicit lexical relations so multi-hop
retrieval can expand from what the book actually contains.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable

_RELATIONS = (
    ("causes", re.compile(r"\b(?:causes?|leads? to|results? in|responsible for)\b", re.I)),
    ("associated_with", re.compile(r"\b(?:associated with|related to|linked to)\b", re.I)),
    ("diagnosed_by", re.compile(r"\b(?:diagnosed by|diagnosis by|confirmed by)\b", re.I)),
    ("treated_with", re.compile(r"\b(?:treated with|managed with|therapy with)\b", re.I)),
    ("symptom_of", re.compile(r"\b(?:symptom of|manifestation of|characteristic of)\b", re.I)),
)


def _entities(meta: dict[str, Any], text: str) -> list[str]:
    raw = meta.get("entities") or meta.get("entity_names") or ()
    values = [str(value).strip() for value in raw if str(value).strip()]
    if not values:
        # Conservative fallback: preserve capitalized medical-looking phrases and abbreviations.
        values.extend(re.findall(r"\b[A-Z][A-Za-z0-9-]{2,}(?:\s+[A-Z][A-Za-z0-9-]{2,}){0,2}\b", text))
        values.extend(re.findall(r"\b[A-Z]{2,6}\b", text))
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key not in seen and len(value) >= 2:
            seen.add(key)
            unique.append(value)
    return unique[:24]


def build_graph(hits: Iterable[Any]) -> dict[str, Any]:
    graph: dict[str, set[str]] = defaultdict(set)
    relations: list[dict[str, Any]] = []
    nodes: set[str] = set()
    for hit in hits:
        meta = dict(getattr(hit, "metadata", {}) or {})
        text = str(getattr(hit, "text", "") or "")
        entities = _entities(meta, text)
        for entity in entities:
            nodes.add(entity.casefold())
        lowered = text.casefold()
        for relation_name, pattern in _RELATIONS:
            for match in pattern.finditer(lowered):
                left = [entity for entity in entities if entity.casefold() in lowered[max(0, match.start() - 220):match.start()]]
                right = [entity for entity in entities if entity.casefold() in lowered[match.end():match.end() + 220]]
                if left and right:
                    for source in left[:2]:
                        for target in right[:2]:
                            if source.casefold() == target.casefold():
                                continue
                            graph[source.casefold()].add(target.casefold())
                            relations.append({"source": source, "relation": relation_name, "target": target})
    return {"nodes": sorted(nodes), "edges": relations[:200], "adjacency": {key: sorted(values) for key, values in graph.items()}, "node_count": len(nodes), "edge_count": len(relations[:200])}


def expand_from_query(query: str, graph: dict[str, Any], *, limit: int = 4) -> list[str]:
    lowered = str(query or "").casefold()
    additions: list[str] = []
    adjacency = graph.get("adjacency") if isinstance(graph, dict) else {}
    if not isinstance(adjacency, dict):
        return additions
    for node, neighbors in adjacency.items():
        if node in lowered:
            values = [str(value) for value in neighbors if str(value)]
            additions.append(f"{query} {' '.join(values[:3])}".strip())
    return additions[:max(1, int(limit))]


__all__ = ["build_graph", "expand_from_query"]
