from __future__ import annotations

import json
import math
import re
import sqlite3
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import chromadb


class IndexCompatibilityError(RuntimeError):
    """Raised when the vector index is incompatible with the expected embedding identity."""


def _metadata_dict(metadata: Any) -> Dict[str, Any]:
    if isinstance(metadata, dict):
        return dict(metadata)
    if metadata is None:
        return {}
    return {"value": metadata}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except (TypeError, ValueError):
        return []


class VectorStore:
    def __init__(self, persist_directory: str | Path, collection_name: str = "rag_documents"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.lexical_database = self.persist_directory / "lexical.sqlite3"
        self._initialize_lexical_index()
        self.expected_identity: Any | None = None

    def _initialize_lexical_index(self) -> None:
        with sqlite3.connect(self.lexical_database) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS lexical_documents (
                    id TEXT PRIMARY KEY,
                    document TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    index_state TEXT NOT NULL,
                    tokens TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_lexical_state ON lexical_documents(index_state)"
            )

    @staticmethod
    def _lexical_tokens(text: str) -> list[str]:
        return re.findall(r"\w+", str(text or "").casefold(), flags=re.UNICODE)

    def _upsert_lexical_records(
        self,
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
        ids: Sequence[str],
    ) -> None:
        with sqlite3.connect(self.lexical_database) as connection:
            connection.executemany(
                """
                INSERT INTO lexical_documents(id, document, metadata, index_state, tokens)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    document=excluded.document,
                    metadata=excluded.metadata,
                    index_state=excluded.index_state,
                    tokens=excluded.tokens
                """,
                [
                    (
                        str(item_id),
                        str(document),
                        json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                        str(metadata.get("index_state", "READY")).upper(),
                        json.dumps(self._lexical_tokens(str(document)), ensure_ascii=False),
                    )
                    for item_id, document, metadata in zip(
                        ids, documents, metadatas, strict=True
                    )
                ],
            )

    def set_expected_identity(self, identity: Any | None) -> None:
        self.expected_identity = identity

    def _collection_dim(self) -> int:
        try:
            return int(self.collection.metadata.get("dimension", 0) or 0)
        except Exception:
            return 0

    def _resolve_dimension(self, embeddings: Sequence[Sequence[float]] | None = None) -> int:
        if embeddings is not None:
            try:
                if len(embeddings) > 0:
                    first = embeddings[0]
                    try:
                        return len(first)
                    except TypeError:
                        pass
            except (TypeError, ValueError, IndexError):
                pass
        stored = self._collection_dim()
        if stored > 0:
            return stored
        return 32

    def _coerce_metadata(self, metadata: Any) -> Dict[str, Any]:
        base = _metadata_dict(metadata)
        base.setdefault("index_state", "READY")
        if "document_id" not in base and "doc_id" in base:
            base["document_id"] = base["doc_id"]
        if "chunk_id" not in base and "id" in base:
            base["chunk_id"] = base["id"]
        if "page_numbers" not in base:
            base["page_numbers"] = []
        if "version_id" not in base:
            base["version_id"] = base.get("document_id", "legacy")
        return base

    def _normalize_records(self, records: dict[str, Any]) -> list[dict[str, Any]]:
        ids = _as_list(records.get("ids"))
        documents = _as_list(records.get("documents"))
        metadatas = _as_list(records.get("metadatas"))
        results: list[dict[str, Any]] = []
        for index, item_id in enumerate(ids):
            metadata = self._coerce_metadata(
                metadatas[index] if index < len(metadatas) else {}
            )
            document = documents[index] if index < len(documents) else ""
            results.append(
                {"id": str(item_id), "document": str(document), "metadata": metadata}
            )
        return results

    def _read_collection_identity(self) -> Dict[str, Any] | None:
        metadata = self.collection.metadata or {}
        fp = metadata.get("embedding_fingerprint")
        if not fp:
            return None
        return {
            "provider": metadata.get("provider"),
            "model": metadata.get("model"),
            "model_version": metadata.get("model_version"),
            "dimension": metadata.get("dimension"),
            "fingerprint": fp,
            "embedding_id": metadata.get("embedding_id"),
        }

    def _update_collection_identity(self, identity: Any | None) -> None:
        if identity is None:
            return
        metadata = {
            key: value
            for key, value in (self.collection.metadata or {}).items()
            if key != "hnsw:space"
        }
        metadata.update(
            {
                "provider": getattr(identity, "provider", "unknown"),
                "model": getattr(identity, "model", "unknown"),
                "model_version": getattr(identity, "model_version", "unknown"),
                "dimension": int(getattr(identity, "dimension", 0) or 0),
                "embedding_fingerprint": getattr(
                    identity,
                    "fingerprint",
                    getattr(identity, "configuration_fingerprint", ""),
                ),
                "embedding_id": getattr(identity, "embedding_id", ""),
            }
        )
        self.collection.modify(metadata=metadata)

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception:
            return 0

    def lexical_count(self) -> int:
        with sqlite3.connect(self.lexical_database) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM lexical_documents WHERE index_state = 'READY'"
            ).fetchone()
        return int(row[0] if row else 0)

    def clear_all(self) -> None:
        records = self.collection.get(include=["metadatas"])
        ids = [str(item_id) for item_id in _as_list(records.get("ids"))]
        if ids:
            self.collection.delete(ids=ids)
        with sqlite3.connect(self.lexical_database) as connection:
            connection.execute("DELETE FROM lexical_documents")
            connection.commit()
            connection.execute("VACUUM")

    def get_documents(self, where: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if where:
            return self.collection.get(where=where, include=["documents", "metadatas"])
        return self.collection.get(include=["documents", "metadatas"])

    def set_document_index_state(self, document_id: str, state: str) -> None:
        if document_id == "*":
            matches = self.collection.get(include=["metadatas"])
            for item_id, metadata in zip(
                _as_list(matches.get("ids")),
                _as_list(matches.get("metadatas")),
                strict=False,
            ):
                meta = self._coerce_metadata(metadata)
                meta["index_state"] = state
                self.collection.update(ids=[str(item_id)], metadatas=[meta])
            with sqlite3.connect(self.lexical_database) as connection:
                connection.execute(
                    "UPDATE lexical_documents SET index_state = ?, "
                    "metadata = json_set(metadata, '$.index_state', ?)",
                    (str(state).upper(), str(state).upper()),
                )
            if str(state).upper() == "READY":
                self._nonready_lexical_ids = set()
            return
        matches = self.collection.get(
            where={"document_id": document_id}, include=["metadatas"]
        )
        for item_id, metadata in zip(
            _as_list(matches.get("ids")),
            _as_list(matches.get("metadatas")),
            strict=False,
        ):
            meta = self._coerce_metadata(metadata)
            meta["index_state"] = state
            self.collection.update(ids=[str(item_id)], metadatas=[meta])
        with sqlite3.connect(self.lexical_database) as connection:
            connection.execute(
                "UPDATE lexical_documents SET index_state = ?, "
                "metadata = json_set(metadata, '$.index_state', ?) "
                "WHERE json_extract(metadata, '$.document_id') = ?",
                (str(state).upper(), str(state).upper(), document_id),
            )
        if str(state).upper() == "READY":
            self._nonready_lexical_ids = set()

    def _document_lexical_rows(self, document_id: str, version_id: str | None = None) -> list[tuple[str, dict[str, Any]]]:
        with sqlite3.connect(self.lexical_database) as connection:
            rows = connection.execute(
                "SELECT id, metadata FROM lexical_documents WHERE json_extract(metadata, '$.document_id') = ?",
                (str(document_id),),
            ).fetchall()
        result: list[tuple[str, dict[str, Any]]] = []
        for item_id, raw_metadata in rows:
            try:
                metadata = json.loads(raw_metadata)
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {}
            metadata = self._coerce_metadata(metadata)
            if version_id is None or str(metadata.get("version_id") or "") == str(version_id):
                result.append((str(item_id), metadata))
        return result

    def set_version_index_state(
        self, document_id: str, version_id: str, state: str
    ) -> None:
        normalized_state = str(state).upper()
        records = self.collection.get(
            where={"document_id": document_id}, include=["metadatas"]
        )
        semantic_matches = []
        for item_id, metadata in zip(
            _as_list(records.get("ids")),
            _as_list(records.get("metadatas")),
            strict=False,
        ):
            meta = self._coerce_metadata(metadata)
            if str(meta.get("version_id") or "") == str(version_id):
                semantic_matches.append(str(item_id))
        lexical_matches = self._document_lexical_rows(document_id, version_id)
        if normalized_state == "READY":
            if not semantic_matches or len(semantic_matches) != len(lexical_matches):
                raise RuntimeError(
                    "READY publication contract requires semantic/lexical count parity: "
                    f"semantic={len(semantic_matches)}, lexical={len(lexical_matches)}."
                )
        for item_id in semantic_matches:
            metadata = next(
                self._coerce_metadata(meta)
                for current_id, meta in zip(
                    _as_list(records.get("ids")),
                    _as_list(records.get("metadatas")),
                    strict=False,
                )
                if str(current_id) == item_id
            )
            metadata["index_state"] = normalized_state
            self.collection.update(ids=[item_id], metadatas=[metadata])
        if lexical_matches:
            with sqlite3.connect(self.lexical_database) as connection:
                connection.executemany(
                    "UPDATE lexical_documents SET index_state = ?, metadata = json_set(metadata, '$.index_state', ?) WHERE id = ?",
                    [(normalized_state, normalized_state, item_id) for item_id, _ in lexical_matches],
                )
                connection.commit()

    def delete_version(self, document_id: str, version_id: str) -> None:
        matches = self.collection.get(
            where={"document_id": document_id}, include=["metadatas"]
        )
        removable = []
        for item_id, metadata in zip(
            _as_list(matches.get("ids")),
            _as_list(matches.get("metadatas")),
            strict=False,
        ):
            meta = self._coerce_metadata(metadata)
            if meta.get("version_id") == version_id:
                removable.append(str(item_id))
        if removable:
            self.collection.delete(ids=removable)
        with sqlite3.connect(self.lexical_database) as connection:
            connection.execute(
                """
                DELETE FROM lexical_documents
                WHERE json_extract(metadata, '$.document_id') = ?
                  AND json_extract(metadata, '$.version_id') = ?
                """,
                (str(document_id), str(version_id)),
            )
            connection.commit()

    def reconcile_index(self, document_id: str | None = None) -> Dict[str, Any]:
        query = {"document_id": document_id} if document_id else None
        if query is not None:
            records = self.collection.get(
                where=query, include=["metadatas", "documents"]
            )
        else:
            records = self.collection.get(include=["metadatas", "documents"])
        record_ids = _as_list(records.get("ids"))
        metadata_values = _as_list(records.get("metadatas"))
        document_values = _as_list(records.get("documents"))
        deduped: Dict[str, Dict[str, Any]] = {}
        removed = 0
        for item_id, metadata, document in zip(
            record_ids, metadata_values, document_values, strict=False
        ):
            meta = self._coerce_metadata(metadata)
            key = f"{meta.get('document_id', '')}::{meta.get('chunk_id', str(item_id))}"
            if key in deduped:
                removed += 1
                continue
            deduped[key] = {
                "id": str(item_id),
                "metadata": meta,
                "document": str(document),
            }
        if removed:
            old_ids = [entry["id"] for entry in deduped.values()]
            all_ids = [str(item_id) for item_id in record_ids]
            stale = [item_id for item_id in all_ids if item_id not in old_ids]
            if stale:
                self.collection.delete(ids=stale)
        return {
            "valid": True,
            "removed": removed,
            "count": len(deduped),
            "document_id": document_id,
        }

    def validate_document_index(
        self, document_id: str, version_id: str | None = None
    ) -> Dict[str, Any]:
        records = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas", "documents", "embeddings"],
        )
        ids = _as_list(records.get("ids"))
        metadatas_all = _as_list(records.get("metadatas"))
        if version_id is not None:
            keep = [
                index
                for index, metadata in enumerate(metadatas_all)
                if str(self._coerce_metadata(metadata).get("version_id") or "") == str(version_id)
            ]
            normalized_records = {
                key: _as_list(values) for key, values in records.items()
            }
            records = {
                key: [values[index] for index in keep]
                for key, values in normalized_records.items()
            }
            ids = _as_list(records.get("ids"))
            metadatas_all = _as_list(records.get("metadatas"))
        metadatas = metadatas_all
        issues: list[str] = []
        seen_chunk_ids: set[str] = set()
        for metadata in metadatas:
            meta = self._coerce_metadata(metadata)
            chunk_id = str(meta.get("chunk_id") or meta.get("id") or "")
            if not chunk_id:
                issues.append("missing chunk_id")
            elif chunk_id in seen_chunk_ids:
                issues.append(f"duplicate chunk_id: {chunk_id}")
            seen_chunk_ids.add(chunk_id)
            if meta.get("index_state") not in {"READY", "BUILDING"}:
                issues.append(f"unexpected index_state: {meta.get('index_state')}")
        embeddings = _as_list(records.get("embeddings"))
        for vector in embeddings:
            if not self._valid_vector(vector, self._collection_dim()):
                issues.append("invalid semantic embedding")

        lexical_rows = self._document_lexical_rows(document_id, version_id)
        semantic_count = len(ids)
        lexical_count = len(lexical_rows)
        if semantic_count != lexical_count:
            issues.append(
                f"semantic/lexical count mismatch: semantic={semantic_count}, lexical={lexical_count}"
            )
        valid = not issues and bool(ids) and semantic_count == lexical_count
        return {
            "document_id": document_id,
            "count": semantic_count,
            "semantic_count": semantic_count,
            "lexical_count": lexical_count,
            "valid": valid,
            "issues": issues,
        }

    @staticmethod
    def _valid_vector(vector: Sequence[float], expected_dimension: int = 0) -> bool:
        try:
            values = [float(value) for value in _as_list(vector)]
        except (TypeError, ValueError):
            return False
        return bool(
            values
            and (not expected_dimension or len(values) == expected_dimension)
            and all(math.isfinite(value) for value in values)
            and math.sqrt(sum(value * value for value in values)) > 1e-12
        )

    def _apply_collection_metadata(self, dim: int | None = None) -> None:
        current = dict(self.collection.metadata or {})
