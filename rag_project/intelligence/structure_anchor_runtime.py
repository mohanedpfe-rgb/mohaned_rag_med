from __future__ import annotations

import hashlib
import math
import sqlite3
import threading
from typing import Any

from rag_project.storage.vector_store import VectorStore

_LOCK = threading.RLock()
_INSTALLED = False


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dimension = len(vectors[0])
    total = [0.0] * dimension
    count = 0
    for vector in vectors:
        if len(vector) != dimension:
            continue
        for index, value in enumerate(vector):
            total[index] += float(value)
        count += 1
    if not count:
        return []
    mean = [value / count for value in total]
    norm = math.sqrt(sum(value * value for value in mean))
    if norm <= 1e-12:
        return []
    return [value / norm for value in mean]


def _stable_anchor_id(document_id: str, kind: str, key: str) -> str:
    digest = hashlib.sha256(f"{document_id}|{kind}|{key}".encode("utf-8")).hexdigest()[:20]
    return f"__structure__:{kind}:{digest}"


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return

        original_add = VectorStore.add_documents
        original_search = VectorStore.search
        original_set_state = VectorStore.set_version_index_state
        original_clear_all = VectorStore.clear_all

        def structure_collection(self: VectorStore):
            collection = getattr(self, "_structure_collection", None)
            if collection is None:
                collection = self.client.get_or_create_collection(
                    name="rag_structure",
                    metadata={"hnsw:space": "cosine", "structure_schema_version": 3},
                )
                self._structure_collection = collection
            return collection

        VectorStore._deep_structure_collection = structure_collection

        def rebuild_anchors(self: VectorStore, document_id: str, section_ids: set[str], chapter_ids: set[str]) -> None:
            if not section_ids and not chapter_ids:
                return
            collection = structure_collection(self)
            expected_ids: set[str] = set()
            identity = self.expected_identity
            base_meta = self.collection.metadata or {}
            for kind, keys in (("section", section_ids), ("chapter", chapter_ids)):
                for key in keys:
                    if not key:
                        continue
                    rows = self.collection.get(
                        where={"document_id": document_id, "section_id" if kind == "section" else "chapter_id": key},
                        include=["documents", "embeddings", "metadatas"],
                    )
                    vectors = []
                    metas = []
                    docs = []
                    for vector, meta, doc in zip(
                        rows.get("embeddings", []) or [],
                        rows.get("metadatas", []) or [],
                        rows.get("documents", []) or [],
                        strict=False,
                    ):
                        if isinstance(meta, dict) and str(meta.get("index_state", "READY")).upper() == "BUILDING":
                            vectors.append(list(vector))
                            metas.append(meta)
                            docs.append(str(doc))
                    mean = _mean_vector(vectors)
                    if not mean:
                        continue
                    first = metas[0] if metas else {}
                    title = str(first.get("section") or first.get("chapter") or key)
                    anchor_id = _stable_anchor_id(document_id, kind, key)
                    expected_ids.add(anchor_id)
                    path = first.get("hierarchy_path") or []
                    if isinstance(path, str):
                        path_text = path
                    else:
                        path_text = " > ".join(str(item) for item in path)
                    anchor_text = (
                        f"[{kind.upper()} ANCHOR]\n"
                        f"Document: {first.get('file_name', '')}\n"
                        f"Chapter: {first.get('chapter', '')}\n"
                        f"Section: {first.get('section', '')}\n"
                        f"Hierarchy: {path_text}\n"
                        f"Representative evidence:\n{docs[0][:1800] if docs else title}"
                    )
                    metadata = {
                        "document_id": document_id,
                        "chunk_id": anchor_id,
                        "file_name": first.get("file_name", ""),
                        "page_numbers": first.get("page_numbers", []),
                        "chunk_index": -1,
                        "document_type": first.get("document_type", "unknown"),
                        "language": first.get("language", "unknown"),
                        "index_state": "BUILDING",
                        "version_id": first.get("version_id", ""),
                        "content_hash": first.get("content_hash", ""),
                        "build_id": first.get("build_id", ""),
                        "structure_version": 3,
                        "representation_type": f"{kind}_anchor",
                        "parent_id": first.get("parent_id") or f"{document_id}:document",
                        "section_id": first.get("section_id") or first.get("parent_id"),
                        "chapter_id": first.get("chapter_id"),
                        "chapter": first.get("chapter"),
                        "section": first.get("section"),
                        "hierarchy_path": path,
                        "quality_score": max(float(item.get("quality_score", 0.0) or 0.0) for item in metas) if metas else 0.0,
                        "ocr_status": "aggregated",
                        "routing_decision": "structure_anchor",
                    }
                    collection.upsert(ids=[anchor_id], documents=[anchor_text], metadatas=[metadata], embeddings=[mean])
                    if identity is not None:
                        md = dict(collection.metadata or {})
                        md.update(
                            {
                                "provider": getattr(identity, "provider", "unknown"),
                                "model": getattr(identity, "model", "unknown"),
                                "dimension": int(getattr(identity, "dimension", len(mean)) or len(mean)),
                                "embedding_fingerprint": getattr(identity, "fingerprint", ""),
                                "structure_schema_version": 3,
                            }
                        )
                        collection.modify(metadata=md)

            # Remove structure anchors for sections/chapters no longer present in the build.
            rows = collection.get(include=["ids", "metadatas"])
            stale = []
            for item_id, raw in zip(rows.get("ids", []) or [], rows.get("metadatas", []) or [], strict=False):
                if not isinstance(raw, dict) or str(raw.get("document_id") or "") != str(document_id):
                    continue
                if str(item_id) not in expected_ids:
                    stale.append(str(item_id))
            if stale:
                collection.delete(ids=stale)

        VectorStore._deep_rebuild_structure_anchors = rebuild_anchors

        def add_documents(self: VectorStore, documents, metadatas, embeddings, ids):
            original_add(self, documents, metadatas, embeddings, ids)
            document_ids = {str((meta or {}).get("document_id") or "") for meta in metadatas if isinstance(meta, dict)}
            for document_id in document_ids:
                if not document_id:
                    continue
                section_ids = {str((meta or {}).get("section_id") or "") for meta in metadatas if isinstance(meta, dict) and (meta or {}).get("document_id") == document_id}
                chapter_ids = {str((meta or {}).get("chapter_id") or "") for meta in metadatas if isinstance(meta, dict) and (meta or {}).get("document_id") == document_id}
                rebuild_anchors(self, document_id, {item for item in section_ids if item}, {item for item in chapter_ids if item})

        VectorStore.add_documents = add_documents

        def search(self: VectorStore, embedding, n_results=5, where=None):
            base = original_search(self, embedding, n_results, where)
            collection = structure_collection(self)
            try:
                structure_where = where or None
                result = collection.query(
                    query_embeddings=[list(map(float, embedding))],
                    n_results=max(1, int(n_results)),
                    where=structure_where,
                    include=["documents", "metadatas", "distances"],
                )
                ids = list((result.get("ids") or [[]])[0])
                docs = list((result.get("documents") or [[]])[0])
                metas = list((result.get("metadatas") or [[]])[0])
                distances = list((result.get("distances") or [[]])[0])
                existing_ids = set((base.get("ids") or [[]])[0])
                for item_id, doc, meta, distance in zip(ids, docs, metas, distances, strict=False):
                    if item_id in existing_ids:
                        continue
                    base.setdefault("ids", [[]])[0].append(str(item_id))
                    base.setdefault("documents", [[]])[0].append(str(doc))
                    base.setdefault("metadatas", [[]])[0].append(dict(meta or {}))
                    base.setdefault("distances", [[]])[0].append(float(distance))
            except Exception:
                # Structure anchors are an enhancement; the canonical child index remains authoritative.
                pass
            return base

        VectorStore.search = search

        def set_version_index_state(self: VectorStore, document_id: str, version_id: str, state: str):
            result = original_set_state(self, document_id, version_id, state)
            collection = getattr(self, "_structure_collection", None)
            if collection is not None and str(state).upper() in {"READY", "FAILED"}:
                rows = collection.get(where={"document_id": document_id}, include=["metadatas"])
                ids = []
                metas = []
                for item_id, meta in zip(rows.get("ids", []) or [], rows.get("metadatas", []) or [], strict=False):
                    if isinstance(meta, dict):
                        if str(meta.get("version_id") or "") == str(version_id) or str(meta.get("content_hash") or "") == str(version_id):
                            meta["index_state"] = str(state).upper()
                            ids.append(str(item_id)); metas.append(meta)
                if ids:
                    collection.update(ids=ids, metadatas=metas)
            return result

        VectorStore.set_version_index_state = set_version_index_state

        def clear_all(self: VectorStore):
            result = original_clear_all(self)
            collection = getattr(self, "_structure_collection", None)
            if collection is not None:
                ids = [str(item_id) for item_id in (collection.get(include=["metadatas"]).get("ids", []) or [])]
                if ids:
                    collection.delete(ids=ids)
            return result

        VectorStore.clear_all = clear_all
        _INSTALLED = True


__all__ = ["install"]
