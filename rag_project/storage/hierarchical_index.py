from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Any, Iterable, Sequence


def _mean_vectors(vectors: Sequence[Sequence[float]]) -> list[float]:
    vectors = [list(map(float, v)) for v in vectors if v]
    if not vectors:
        return []
    dim = len(vectors[0])
    usable = [v for v in vectors if len(v) == dim]
    if not usable:
        return []
    result = [sum(v[i] for v in usable) / len(usable) for i in range(dim)]
    norm = sqrt(sum(x * x for x in result))
    return [x / norm for x in result] if norm > 1e-12 else []


def build_hierarchical_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build deterministic book/chapter/section aggregate representations from child chunks."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        meta = dict(record.get("metadata") or {})
        if str(meta.get("index_state", "READY")).upper() == "READY":
            groups[(str(meta.get("document_id", "")), str(meta.get("version_id", "")), str(meta.get("chapter_id") or meta.get("chapter") or "__document__"))].append(record)

    output: list[dict[str, Any]] = []
    for (document_id, version_id, chapter_key), chapter_records in groups.items():
        if not document_id or chapter_key == "__document__":
            continue
        chapter_vectors = [r.get("embedding") for r in chapter_records]
        chapter_vector = _mean_vectors(chapter_vectors)
        if chapter_vector:
            first = chapter_records[0]
            meta = dict(first.get("metadata") or {})
            chapter = str(meta.get("chapter") or chapter_key)
            output.append({
                "id": f"{document_id}-{version_id}-hierarchy-chapter-{abs(hash(chapter_key))}",
                "embedding": chapter_vector,
                "document": f"Chapter: {chapter}",
                "metadata": {
                    "document_id": document_id,
                    "version_id": version_id,
                    "record_type": "chapter",
                    "chapter": chapter,
                    "chapter_id": meta.get("chapter_id") or chapter_key,
                    "page_numbers": sorted({p for r in chapter_records for p in (r.get("metadata") or {}).get("page_numbers", [])}),
                    "index_state": "BUILDING",
                },
            })
    return output
