from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Sequence


@dataclass(frozen=True)
class EvidenceHierarchy:
    document_id: str
    section_id: str
    paragraph_id: str
    sentence_id: str
    fact_id: str
    entity_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    page_numbers: tuple[Any, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def build_evidence_hierarchy(hits: Sequence[Any]) -> tuple[EvidenceHierarchy, ...]:
    """Project retrieved chunks into stable document/section/paragraph/sentence/fact identifiers.

    This is metadata-first and deterministic; it never manufactures clinical facts.
    """
    records: list[EvidenceHierarchy] = []
    for index, hit in enumerate(hits):
        meta = getattr(hit, "metadata", {}) or {}
        document_id = str(meta.get("document_id") or getattr(hit, "doc_id", "") or f"doc_{index + 1}")
        section_id = str(meta.get("section_id") or meta.get("section_title") or f"section_{index + 1}")
        paragraph_id = str(meta.get("paragraph_id") or meta.get("chunk_id") or f"paragraph_{index + 1}")
        text = _norm(getattr(hit, "text", ""))
        sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
        for sentence_index, sentence in enumerate(s for s in sentences if s):
            sentence_id = f"{paragraph_id}:s{sentence_index + 1}"
            fact_id = f"{sentence_id}:f1"
            entity_ids = tuple(str(x) for x in (meta.get("entity_ids") or meta.get("entities") or ()) if x)[:16]
            relation_ids = tuple(str(x) for x in (meta.get("relation_ids") or meta.get("relations") or ()) if x)[:16]
            records.append(EvidenceHierarchy(document_id, section_id, paragraph_id, sentence_id, fact_id, entity_ids, relation_ids, tuple(meta.get("page_numbers") or ())))
    return tuple(records)


def select_context_levels(records: Sequence[EvidenceHierarchy], *, include_parent: bool = True) -> dict[str, int]:
    return {
        "documents": len({r.document_id for r in records}),
        "sections": len({(r.document_id, r.section_id) for r in records}),
        "paragraphs": len({(r.document_id, r.paragraph_id) for r in records}),
        "sentences": len(records),
        "facts": len({r.fact_id for r in records}),
        "parents_enabled": int(bool(include_parent)),
    }
