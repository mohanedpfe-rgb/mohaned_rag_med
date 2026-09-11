from __future__ import annotations

import re
from hashlib import sha1
from math import sqrt
from typing import Any, Dict, Sequence

from rag_project.storage.vector_store import VectorStore


_STRUCTURE_RE = re.compile(
    r"\[RAG-STRUCTURE\s+chapter_id=(?P<chapter_id>[^;\]]*);\s*chapter=(?P<chapter>[^;\]]*);\s*"
    r"section_id=(?P<section_id>[^;\]]*);\s*section=(?P<section>[^;\]]*);\s*"
    r"parent_id=(?P<parent_id>[^;\]]*);\s*quality=(?P<quality>[^;\]]*);\s*"
    r"page_type=(?P<page_type>[^;\]]*);\s*ocr_status=(?P<ocr_status>[^;\]]*);\s*"
    r"table_id=(?P<table_id>[^;\]]*);\s*figure_id=(?P<figure_id>[^;\]]*)\]"
)


def _stable_id(value: str) -> str:
    return sha1(value.encode("utf-8")).hexdigest()[:20]


def _unit_mean(vectors: Sequence[Sequence[float]]) -> list[float]:
    usable = [list(map(float, vector)) for vector in vectors if vector]
    if not usable:
        return []
    dimension = len(usable[0])
    usable = [vector for vector in usable if len(vector) == dimension]
    if not usable:
        return []
    result = [sum(vector[index] for vector in usable) / len(usable) for index in range(dimension)]
    norm = sqrt(sum(value * value for value in result))
    return [value / norm for value in result] if norm > 1e-12 else []


class EnhancedVectorStore(VectorStore):
    """VectorStore with durable structure metadata and a separate hierarchy index."""

    def __init__(self, persist_directory: str, collection_name: str = "rag_documents"):
        super().__init__(persist_directory, collection_name)
        self.hierarchy_collection = self.client.get_or_create_collection(
            name=f"{collection_name}_hierarchy",
            metadata={"hnsw:space": "cosine", "index_role": "document_hierarchy"},
        )

    @staticmethod
    def _enrich_metadata(metadata: Dict[str, Any], document: str) -> Dict[str, Any]:
        enriched = dict(metadata)
        match = _STRUCTURE_RE.search(document or "")
        if match:
            values = match.groupdict()
            enriched["chapter_id"] = values.get("chapter_id") or None
            enriched["chapter"] = values.get("chapter") or None
            enriched["section_id"] = values.get("section_id") or None
            enriched["section"] = values.get("section") or None
            enriched["parent_id"] = values.get("parent_id") or None
            try:
                enriched["quality_score"] = float(values.get("quality") or 0.0)
            except (TypeError, ValueError):
                pass
            enriched["page_type"] = values.get("page_type") or enriched.get("page_type")
            enriched["ocr_status"] = values.get("ocr_status") or enriched.get("ocr_status")
            enriched["table_id"] = values.get("table_id") or None
            enriched["figure_id"] = values.get("figure_id") or None
        enriched.setdefault("record_type", "chunk")
        enriched.setdefault("hierarchy_level", "chunk")
        return enriched

    def add_documents(
        self,
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
        embeddings: Sequence[Sequence[float]],
        ids: Sequence[str],
    ) -> None:
        normalized = [
            self._enrich_metadata(dict(meta or {}), str(document))
            for meta, document in zip(metadatas, documents, strict=True)
        ]
        super().add_documents(documents, normalized, embeddings, ids)

    def _publish_hierarchy(self, document_id: str, version_id: str) -> None:
        records = self.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas", "embeddings"],
        )
        ids = list(records.get("ids") or [])
        documents = list(records.get("documents") or [])
        metadatas = list(records.get("metadatas") or [])
        embeddings = list(records.get("embeddings") or [])
        rows: list[dict[str, Any]] = []
        for index, metadata in enumerate(metadatas):
            meta = self._coerce_metadata(metadata)
            if str(meta.get("version_id")) != str(version_id):
                continue
            if str(meta.get("record_type", "chunk")) != "chunk":
                continue
            if index >= len(embeddings):
                continue
            rows.append(
                {
                    "id": str(ids[index]),
                    "document": str(documents[index] if index < len(documents) else ""),
                    "metadata": meta,
                    "embedding": embeddings[index],
                }
            )
        if not rows:
            return

        aggregate_rows: list[dict[str, Any]] = []
        document_vector = _unit_mean([row["embedding"] for row in rows])
        if document_vector:
            first_meta = rows[0]["metadata"]
            pages = [p for row in rows for p in row["metadata"].get("page_numbers", [])]
            aggregate_rows.append(
                {
                    "id": f"{document_id}-{_stable_id(str(version_id))}-book",
                    "document": f"Book overview: {first_meta.get('file_name', document_id)}",
                    "embedding": document_vector,
                    "metadata": {
                        "document_id": document_id,
                        "version_id": version_id,
                        "record_type": "book",
                        "hierarchy_level": "book",
                        "file_name": first_meta.get("file_name"),
                        "page_start": min(pages, default=0),
                        "page_end": max(pages, default=0),
                        "index_state": "READY",
                    },
                }
            )

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
            meta = group[0]["metadata"]
            pages = [p for row in group for p in row["metadata"].get("page_numbers", [])]
            aggregate_rows.append(
                {
                    "id": f"{document_id}-{_stable_id(str(version_id) + ':chapter:' + chapter_id)}",
                    "document": f"Chapter: {meta.get('chapter') or chapter_id}",
                    "embedding": vector,
                    "metadata": {
                        "document_id": document_id,
                        "version_id": version_id,
                        "record_type": "chapter",
                        "hierarchy_level": "chapter",
                        "chapter": meta.get("chapter"),
                        "chapter_id": chapter_id,
                        "page_start": min(pages, default=0),
                        "page_end": max(pages, default=0),
                        "index_state": "READY",
                    },
                }
            )

        for section_id, group in section_groups.items():
            vector = _unit_mean([row["embedding"] for row in group])
            if not vector:
                continue
            meta = group[0]["metadata"]
            pages = [p for row in group for p in row["metadata"].get("page_numbers", [])]
            aggregate_rows.append(
                {
                    "id": f"{document_id}-{_stable_id(str(version_id) + ':section:' + section_id)}",
                    "document": f"Section: {meta.get('section') or section_id}",
                    "embedding": vector,
                    "metadata": {
                        "document_id": document_id,
                        "version_id": version_id,
                        "record_type": "section",
                        "hierarchy_level": "section",
                        "chapter": meta.get("chapter"),
                        "chapter_id": meta.get("chapter_id"),
                        "section": meta.get("section"),
                        "section_id": section_id,
                        "parent_id": meta.get("parent_id"),
                        "page_start": min(pages, default=0),
                        "page_end": max(pages, default=0),
                        "index_state": "READY",
                    },
                }
            )

        if aggregate_rows:
            self.hierarchy_collection.upsert(
                ids=[row["id"] for row in aggregate_rows],
                documents=[row["document"] for row in aggregate_rows],
                metadatas=[row["metadata"] for row in aggregate_rows],
                embeddings=[row["embedding"] for row in aggregate_rows],
            )
            metadata = dict(self.hierarchy_collection.metadata or {})
            metadata["dimension"] = len(aggregate_rows[0]["embedding"])
            metadata["source_collection"] = self.collection_name
            self.hierarchy_collection.modify(metadata=metadata)

    def set_version_index_state(self, document_id: str, version_id: str, state: str) -> None:
        super().set_version_index_state(document_id, version_id, state)
        if str(state).upper() == "READY":
            self._publish_hierarchy(document_id, version_id)
        else:
            matches = self.hierarchy_collection.get(
                where={"document_id": document_id}, include=["metadatas"]
            )
            remove_ids = [
                str(item_id)
                for item_id, meta in zip(matches.get("ids") or [], matches.get("metadatas") or [])
                if str((meta or {}).get("version_id")) == str(version_id)
            ]
            if remove_ids:
                self.hierarchy_collection.delete(ids=remove_ids)

    def delete_version(self, document_id: str, version_id: str) -> None:
        matches = self.hierarchy_collection.get(
            where={"document_id": document_id}, include=["metadatas"]
        )
        remove_ids = [
            str(item_id)
            for item_id, meta in zip(matches.get("ids") or [], matches.get("metadatas") or [])
            if str((meta or {}).get("version_id")) == str(version_id)
        ]
        if remove_ids:
            self.hierarchy_collection.delete(ids=remove_ids)
        super().delete_version(document_id, version_id)

    def hierarchy_count(self) -> int:
        try:
            return int(self.hierarchy_collection.count())
        except Exception:
            return 0

    def search_hierarchy(
        self,
        embedding: Sequence[float],
        n_results: int = 5,
        where: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if not embedding:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        return self.hierarchy_collection.query(
            query_embeddings=[list(map(float, embedding))],
            n_results=max(1, int(n_results)),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
