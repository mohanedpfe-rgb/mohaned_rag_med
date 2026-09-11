from __future__ import annotations

from collections import defaultdict
from hashlib import sha1
from math import sqrt
from typing import Any, Iterable, Sequence


def _unit_mean(vectors: Sequence[Sequence[float]]) -> list[float]:
    usable = [list(map(float, v)) for v in vectors if v]
    if not usable:
        return []
    dim = len(usable[0])
    usable = [v for v in usable if len(v) == dim]
    if not usable:
        return []
    result = [sum(v[i] for v in usable) / len(usable) for i in range(dim)]
    norm = sqrt(sum(value * value for value in result))
    return [value / norm for value in result] if norm > 1e-12 else []


def _stable_id(value: str) -> str:
    return sha1(value.encode("utf-8")).hexdigest()[:20]


def build_aggregate_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_doc: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_chapter: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_section: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        meta = dict(record.get("metadata") or {})
        if meta.get("record_type", "chunk") != "chunk" or str(meta.get("index_state", "READY")).upper() != "READY":
            continue
        key = (str(meta.get("document_id", "")), str(meta.get("version_id", "")))
        if key[0]:
            by_doc[key].append(record)
        chapter = str(meta.get("chapter_id") or meta.get("chapter") or "")
        section = str(meta.get("section_id") or meta.get("section") or "")
        if chapter:
            by_chapter[(key[0], key[1], chapter)].append(record)
        if section:
            by_section[(key[0], key[1], section)].append(record)
    output: list[dict[str, Any]] = []
    for (document_id, version_id), items in by_doc.items():
        vector = _unit_mean([item["embedding"] for item in items])
        if vector:
            output.append({"id": f"{document_id}-{_stable_id(version_id)}-book", "embedding": vector, "document": f"Book overview: {items[0]['metadata'].get('file_name', document_id)}", "metadata": {"document_id": document_id, "version_id": version_id, "record_type": "book", "hierarchy_level": "book", "file_name": items[0]["metadata"].get("file_name"), "page_numbers": sorted({p for item in items for p in item["metadata"].get("page_numbers", [])}), "index_state": "READY"}})
    for (document_id, version_id, chapter_id), items in by_chapter.items():
        vector = _unit_mean([item["embedding"] for item in items])
        if vector:
            meta = items[0]["metadata"]
            output.append({"id": f"{document_id}-{_stable_id(version_id + ':chapter:' + chapter_id)}", "embedding": vector, "document": f"Chapter: {meta.get('chapter') or chapter_id}", "metadata": {"document_id": document_id, "version_id": version_id, "record_type": "chapter", "hierarchy_level": "chapter", "chapter": meta.get("chapter"), "chapter_id": chapter_id, "page_numbers": sorted({p for item in items for p in item["metadata"].get("page_numbers", [])}), "index_state": "READY"}})
    for (document_id, version_id, section_id), items in by_section.items():
        vector = _unit_mean([item["embedding"] for item in items])
        if vector:
            meta = items[0]["metadata"]
            output.append({"id": f"{document_id}-{_stable_id(version_id + ':section:' + section_id)}", "embedding": vector, "document": f"Section: {meta.get('section') or section_id}", "metadata": {"document_id": document_id, "version_id": version_id, "record_type": "section", "hierarchy_level": "section", "chapter": meta.get("chapter"), "chapter_id": meta.get("chapter_id"), "section": meta.get("section"), "section_id": section_id, "parent_id": meta.get("parent_id"), "page_numbers": sorted({p for item in items for p in item["metadata"].get("page_numbers", [])}), "index_state": "READY"}})
    return output
