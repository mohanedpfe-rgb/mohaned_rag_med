"""Lightweight persistent-free document intelligence derived from chunk metadata.

The ingestion layer already records chapter/section/parent/page metadata. This
module turns those fields into a deterministic document map and structural
signals without requiring another model or database migration.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Iterable


@dataclass(frozen=True)
class DocumentNode:
    node_id: str
    document_id: str
    node_type: str
    title: str
    page: str
    parent_id: str = ""
    child_count: int = 0
    chunk_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _page(meta: dict[str, Any]) -> str:
    value = meta.get("page_numbers") or meta.get("page_number") or meta.get("page") or "?"
    if isinstance(value, (list, tuple)):
        return str(value[0] if value else "?")
    return str(value)


def _doc(meta: dict[str, Any], hit: Any) -> str:
    return str(meta.get("document_id") or getattr(hit, "doc_id", ""))


def build_document_map(hits: Iterable[Any]) -> dict[str, Any]:
    nodes: dict[str, DocumentNode] = {}
    child_counts: Counter[str] = Counter()
    section_chunks: Counter[str] = Counter()
    chapters: dict[str, set[str]] = defaultdict(set)
    documents: dict[str, dict[str, Any]] = {}

    for hit in hits:
        meta = dict(getattr(hit, "metadata", {}) or {})
        document_id = _doc(meta, hit)
        page = _page(meta)
        chapter = str(meta.get("chapter") or "").strip()
        section = str(meta.get("section") or "").strip()
        section_id = str(meta.get("section_id") or "")
        parent_id = str(meta.get("parent_id") or "")
        if not document_id:
            continue
        doc = documents.setdefault(document_id, {"document_id": document_id, "file_name": str(meta.get("file_name") or ""), "pages": set(), "chunks": 0})
        doc["pages"].add(page)
        doc["chunks"] += 1
        if chapter:
            chapters[document_id].add(chapter)
            node_id = f"{document_id}:chapter:{chapter.casefold()}"
            nodes.setdefault(node_id, DocumentNode(node_id, document_id, "chapter", chapter, page, ""))
        if section or section_id:
            node_id = f"{document_id}:section:{section_id or section.casefold()}"
            parent = parent_id or (f"{document_id}:chapter:{chapter.casefold()}" if chapter else "")
            nodes.setdefault(node_id, DocumentNode(node_id, document_id, "section", section or "Unnamed section", page, parent))
            section_chunks[node_id] += 1
            if parent:
                child_counts[parent] += 1

    finalized: list[dict[str, Any]] = []
    for node_id, node in nodes.items():
        finalized.append(DocumentNode(node.node_id, node.document_id, node.node_type, node.title, node.page, node.parent_id, child_counts[node_id], section_chunks[node_id]).to_dict())
    finalized.sort(key=lambda item: (item["document_id"], 0 if item["node_type"] == "chapter" else 1, item["page"], item["title"]))

    doc_rows = []
    for item in sorted(documents.values(), key=lambda row: row["document_id"]):
        doc_rows.append({"document_id": item["document_id"], "file_name": item["file_name"], "page_count_observed": len(item["pages"]), "chunk_count_observed": item["chunks"], "chapter_count": len(chapters[item["document_id"]])})
    return {"documents": doc_rows, "nodes": finalized, "node_count": len(finalized), "document_count": len(doc_rows)}


def structural_score(meta: dict[str, Any], *, prefer: tuple[str, ...] = ()) -> float:
    evidence_types = {str(x).casefold() for x in (meta.get("evidence_types") or ())}
    score = 0.0
    if meta.get("chapter"):
        score += 0.10
    if meta.get("section"):
        score += 0.16
    if meta.get("parent_id"):
        score += 0.08
    if meta.get("table_id") or "table" in evidence_types:
        score += 0.12 if "table" in prefer else 0.02
    if meta.get("figure_id") or "figure" in evidence_types:
        score += 0.10 if "figure" in prefer else 0.02
    if meta.get("quality_score") is not None:
        try:
            score += 0.10 * max(0.0, min(1.0, float(meta.get("quality_score"))))
        except (TypeError, ValueError):
            pass
    return min(1.0, score)


def section_coverage(hits: Iterable[Any]) -> dict[str, Any]:
    sections: Counter[str] = Counter()
    pages: Counter[str] = Counter()
    documents: Counter[str] = Counter()
    for hit in hits:
        meta = dict(getattr(hit, "metadata", {}) or {})
        document_id = str(meta.get("document_id") or getattr(hit, "doc_id", ""))
        section = str(meta.get("section_id") or meta.get("section") or "")
        page = _page(meta)
        if document_id:
            documents[document_id] += 1
        if section:
            sections[f"{document_id}:{section}"] += 1
        pages[f"{document_id}:{page}"] += 1
    return {
        "distinct_documents": len(documents),
        "distinct_sections": len(sections),
        "distinct_pages": len(pages),
        "documents": dict(documents),
        "sections": dict(sections),
    }


__all__ = ["DocumentNode", "build_document_map", "structural_score", "section_coverage"]
