from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Sequence

from rag_project.storage.vector_store import VectorStore, _clean_metadata, _unit_mean


class EnhancedVectorStore(VectorStore):
    """Vector store with durable document hierarchy records."""

    def _coerce_metadata(self, value: Any) -> Dict[str, Any]:
        return dict(value or {}) if isinstance(value, dict) else {}

    @staticmethod
    def _stable_id(value: str) -> str:
        return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]

    def _enrich_metadata(self, metadata: Dict[str, Any], document: str, item_id: str) -> Dict[str, Any]:
        enriched = dict(metadata or {})
        if "page_numbers" not in enriched and enriched.get("page_start") is not None:
            enriched["page_numbers"] = [enriched["page_start"]]
        if "source_pages" not in enriched and enriched.get("page_numbers"):
            enriched["source_pages"] = list(enriched["page_numbers"])
        for key in ("document_id", "version_id", "chapter_id", "section_id", "parent_id", "record_type", "hierarchy_level", "representation_type", "table_id", "figure_id"):
            value = enriched.get(key)
            if isinstance(value, str) and not value.strip():
                enriched.pop(key, None)
        if values := {k: enriched.get(k) for k in ("chapter", "section") if enriched.get(k)}:
            if values.get("chapter"):
                enriched["chapter"] = values["chapter"]
            if values.get("section"):
                enriched["section"] = values["section"]
        if "[TABLE]" in (document or "") and not enriched.get("table_id"):
            enriched["table_id"] = f"{enriched.get('document_id', 'document')}:table:{item_id}"
            enriched["representation_type"] = "table"
            enriched["evidence_types"] = ["table"]
        if "[FIGURE CAPTION]" in (document or "") and not enriched.get("figure_id"):
            enriched["figure_id"] = f"{enriched.get('document_id', 'document')}:figure:{item_id}"
            enriched["representation_type"] = "figure_caption"
            enriched["evidence_types"] = ["figure"]
        enriched.setdefault("record_type", "chunk")
        enriched.setdefault("hierarchy_level", "chunk")
        enriched.setdefault("structure_version", 3)
        return _clean_metadata(enriched)

    def add_documents(self, documents: Sequence[str], metadatas: Sequence[Dict[str, Any]], embeddings: Sequence[Sequence[float]], ids: Sequence[str]) -> None:
        normalized = [
            self._enrich_metadata(dict(meta or {}), str(document), str(item_id))
            for meta, document, item_id in zip(metadatas, documents, ids, strict=True)
        ]
        super().add_documents(documents, normalized, embeddings, ids)

    def _publish_hierarchy(self, document_id: str, version_id: str) -> None:
        records = self.collection.get(where={"document_id": document_id}, include=["documents", "metadatas", "embeddings"])
        ids = list(records.get("ids") or [])
        documents = list(records.get("documents") or [])
        metadatas = list(records.get("metadatas") or [])
        raw_embeddings = records.get("embeddings")
        embeddings = [] if raw_embeddings is None else list(raw_embeddings)
        rows: list[dict[str, Any]] = []
        for index, metadata in enumerate(metadatas):
            meta = self._coerce_metadata(metadata)
            if str(meta.get("version_id")) != str(version_id) or str(meta.get("record_type", "chunk")) != "chunk" or index >= len(embeddings):
                continue
            rows.append({"id": str(ids[index]), "document": str(documents[index] if index < len(documents) else ""), "metadata": meta, "embedding": embeddings[index]})
        if not rows:
            return

        aggregate_rows: list[dict[str, Any]] = []
        document_vector = _unit_mean([row["embedding"] for row in rows])
        if document_vector:
            first_meta = rows[0]["metadata"]
            pages = [p for row in rows for p in row["metadata"].get("page_numbers", [])]
            aggregate_rows.append({"id": f"{document_id}-{self._stable_id(str(version_id))}-book", "document": f"Book overview: {first_meta.get('file_name', document_id)}", "embedding": document_vector, "metadata": _clean_metadata({"document_id": document_id, "version_id": version_id, "record_type": "book", "hierarchy_level": "book", "file_name": first_meta.get("file_name"), "page_start": min(pages, default=0), "page_end": max(pages, default=0), "index_state": "READY"})})

        chapter_groups: dict[str, list[dict[str, Any]]] = {}
        section_groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            meta = row["metadata"]
            chapter_id = str(meta.get("chapter_id") or meta.get("chapter") or "").strip()
            section_id = str(meta.get("section_id") or meta.get("section") or "").strip()
            if chapter_id:
                chapter_groups.setdefault(chapter_id, []).append(row)
            if section_id:
                section_groups.setdefault(section_id, []).append(row)

        for chapter_id, group in chapter_groups.items():
            vector = _unit_mean([row["embedding"] for row in group])
            if not vector:
                continue
            page_numbers = [p for row in group for p in row["metadata"].get("page_numbers", [])]
            first = group[0]["metadata"]
            aggregate_rows.append({"id": f"{document_id}-{self._stable_id(chapter_id)}-chapter", "document": f"Chapter: {first.get('chapter') or chapter_id}", "embedding": vector, "metadata": _clean_metadata({"document_id": document_id, "version_id": version_id, "record_type": "chapter", "hierarchy_level": "chapter", "chapter_id": chapter_id, "chapter": first.get("chapter"), "page_start": min(page_numbers, default=0), "page_end": max(page_numbers, default=0), "index_state": "READY"})})

        for section_id, group in section_groups.items():
            vector = _unit_mean([row["embedding"] for row in group])
            if not vector:
                continue
            page_numbers = [p for row in group for p in row["metadata"].get("page_numbers", [])]
            first = group[0]["metadata"]
            aggregate_rows.append({"id": f"{document_id}-{self._stable_id(section_id)}-section", "document": f"Section: {first.get('section') or section_id}", "embedding": vector, "metadata": _clean_metadata({"document_id": document_id, "version_id": version_id, "record_type": "section", "hierarchy_level": "section", "section_id": section_id, "section": first.get("section"), "chapter_id": first.get("chapter_id"), "page_start": min(page_numbers, default=0), "page_end": max(page_numbers, default=0), "index_state": "READY"})})

        if aggregate_rows:
            existing_ids = set(str(x) for x in (self.collection.get(ids=[row["id"] for row in aggregate_rows]).get("ids") or []))
            pending = [row for row in aggregate_rows if row["id"] not in existing_ids]
            if pending:
                self.collection.add(
                    ids=[row["id"] for row in pending],
                    documents=[row["document"] for row in pending],
                    metadatas=[row["metadata"] for row in pending],
                    embeddings=[row["embedding"] for row in pending],
                )

    def set_version_index_state(self, document_id: str, version_id: str, state: str) -> None:
        super().set_version_index_state(document_id, version_id, state)
        if str(state).upper() == "READY":
            self._publish_hierarchy(document_id, version_id)
