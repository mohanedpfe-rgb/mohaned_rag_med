from __future__ import annotations

import json
import math
import re
import sqlite3
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

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

    def set_version_index_state(
        self, document_id: str, version_id: str, state: str
    ) -> None:
        matches = self.collection.get(
            where={"document_id": document_id}, include=["metadatas"]
        )
        for item_id, metadata in zip(
            _as_list(matches.get("ids")),
            _as_list(matches.get("metadatas")),
            strict=False,
        ):
            meta = self._coerce_metadata(metadata)
            if meta.get("version_id") == version_id:
                meta["index_state"] = state
                self.collection.update(ids=[str(item_id)], metadatas=[meta])
        with sqlite3.connect(self.lexical_database) as connection:
            rows = connection.execute("SELECT id, metadata FROM lexical_documents").fetchall()
            matching = [
                row[0]
                for row in rows
                if (
                    json.loads(row[1]).get("document_id") == document_id
                    and json.loads(row[1]).get("version_id") == version_id
                )
            ]
            if matching:
                connection.executemany(
                    "UPDATE lexical_documents SET index_state = ?, "
                    "metadata = json_set(metadata, '$.index_state', ?) WHERE id = ?",
                    [
                        (str(state).upper(), str(state).upper(), item_id)
                        for item_id in matching
                    ],
                )

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
                if self._coerce_metadata(metadata).get("version_id") == version_id
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
        valid = not issues and bool(ids)
        return {
            "document_id": document_id,
            "count": len(ids),
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
        metadata = {
            key: value for key, value in current.items() if key != "hnsw:space"
        }
        if dim is not None:
            metadata["dimension"] = int(dim)
        if current.get("dimension") == metadata.get("dimension"):
            return
        self.collection.modify(metadata=metadata)

    def add_documents(
        self,
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
        embeddings: Sequence[Sequence[float]],
        ids: Sequence[str],
    ) -> None:
        documents_list = _as_list(documents)
        metadata_list = _as_list(metadatas)
        embedding_list = _as_list(embeddings)
        ids_list = _as_list(ids)
        if not documents_list:
            return
        if (
            len(documents_list) != len(metadata_list)
            or len(documents_list) != len(embedding_list)
            or len(documents_list) != len(ids_list)
        ):
            raise ValueError(
                "documents, metadatas, embeddings, and ids must have the same length"
            )
        dim = self._resolve_dimension(embedding_list)
        self._apply_collection_metadata(dim)
        normalized = []
        for index, item in enumerate(documents_list):
            metadata = self._coerce_metadata(metadata_list[index])
            metadata.setdefault("document_id", "unknown")
            metadata.setdefault("chunk_id", ids_list[index])
            metadata.setdefault("index_state", "READY")
            metadata.setdefault("version_id", metadata.get("document_id", "legacy"))
            metadata.setdefault("page_numbers", [])
            normalized.append(metadata)
            if not self._valid_vector(embedding_list[index], dim):
                raise ValueError(f"Invalid semantic embedding at index {index}.")
        self.collection.add(
            ids=[str(item) for item in ids_list],
            documents=[str(item) for item in documents_list],
            metadatas=normalized,
            embeddings=[list(map(float, vector)) for vector in embedding_list],
        )
        # DIAGNOSTIC: immediate read-back with ID comparison
        first_doc_id = normalized[0].get("document_id")
        immediate_check = self.collection.get(
            where={"document_id": first_doc_id},
            include=["metadatas"],
        )
        print(
            f"DEBUG: Added {len(ids_list)} records. "
            f"Searched for document_id='{first_doc_id}'. "
            f"Immediate check found: {len(immediate_check.get('ids', []))} records"
        )
        self._update_collection_identity(self.expected_identity)
        self._upsert_lexical_records(documents_list, normalized, ids_list)

    def add_lexical_documents(
        self,
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
        ids: Sequence[str],
    ) -> None:
        documents_list = _as_list(documents)
        metadata_list = _as_list(metadatas)
        ids_list = _as_list(ids)
        if not documents_list:
            return
        if len(documents_list) != len(metadata_list) or len(documents_list) != len(ids_list):
            raise ValueError("documents, metadatas, and ids must have the same length")
        normalized = []
        for index, item in enumerate(documents_list):
            metadata = self._coerce_metadata(metadata_list[index])
            metadata.setdefault("document_id", "unknown")
            metadata.setdefault("chunk_id", ids_list[index])
            metadata.setdefault("index_state", "BUILDING")
            metadata.setdefault("version_id", metadata.get("document_id", "legacy"))
            metadata.setdefault("page_numbers", [])
            normalized.append(metadata)
        self._upsert_lexical_records(documents_list, normalized, ids_list)

    def compatibility_report(self, expected_identity: Any | None) -> Dict[str, Any]:
        stored = self._read_collection_identity()
        collection_dimension = self._collection_dim() or self._resolve_dimension()
        expected_dimension = (
            int(getattr(expected_identity, "dimension", 0) or 0)
            if expected_identity
            else collection_dimension
        )
        metadata_valid = True
        issues: list[str] = []
        is_empty = self.count() == 0
        if (
            not is_empty
            and expected_identity is not None
            and expected_dimension != 0
            and collection_dimension != 0
            and collection_dimension != expected_dimension
        ):
            metadata_valid = False
            issues.append(
                f"dimension mismatch: expected {expected_dimension}, found {collection_dimension}"
            )
        if not is_empty and expected_identity is not None and stored is not None:
            expected_fingerprint = getattr(
                expected_identity,
                "fingerprint",
                getattr(expected_identity, "configuration_fingerprint", None),
            )
            if expected_fingerprint and stored.get("fingerprint") != expected_fingerprint:
                metadata_valid = False
                issues.append("Index fingerprint does not match expected embedding profile.")
        if not is_empty and expected_identity is not None and stored is None:
            metadata_valid = False
            issues.append("Index metadata missing expected embedding profile.")
        valid = metadata_valid
        status = "READY"
        if not valid:
            status = "INDEX_MIGRATION_REQUIRED"
        if valid and is_empty:
            message = (
                "Index is ready for the expected embedding profile; "
                "no chunks have been indexed yet."
            )
        else:
            message = (
                "Index is compatible with the expected embedding profile."
                if valid
                else "; ".join(issues) or "Index compatibility check failed."
            )
        return {
            "status": status,
            "message": message,
            "valid": valid,
            "metadata_valid": metadata_valid,
            "expected_identity": getattr(
                expected_identity, "to_dict", lambda: expected_identity
            )(),
            "stored_identity": stored,
            "collection_dimension": collection_dimension,
            "expected_dimension": expected_dimension,
            "issues": issues,
        }

    def index_health_check(self, expected_identity: Any | None) -> Dict[str, Any]:
        report = self.compatibility_report(expected_identity)
        collection_count = self.count()
        metadata_issues = report.get("issues", [])
        issues = [*metadata_issues]
        if collection_count == 0:
            issues.append("Collection is empty.")
        try:
            records = self.collection.get(include=["embeddings"])
            embeddings = _as_list(records.get("embeddings"))
            invalid = sum(
                1
                for vector in embeddings
                if not self._valid_vector(
                    vector, report.get("collection_dimension") or 0
                )
            )
            if invalid:
                issues.append(f"{invalid} invalid semantic embeddings.")
        except Exception as exc:
            issues.append(
                f"Unable to validate stored embeddings: {type(exc).__name__}: {exc}"
            )
        valid = not issues
        return {
            "valid": valid,
            "metadata_valid": report.get("metadata_valid", False),
            "expected_identity": getattr(
                expected_identity, "to_dict", lambda: expected_identity
            )(),
            "stored_identity": report.get("stored_identity"),
            "collection_dimension": report.get("collection_dimension"),
            "expected_dimension": report.get("expected_dimension"),
            "metadata_issues": metadata_issues,
            "issues": issues,
            "vector_count": collection_count,
        }

    def _as_query_result(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        distances: list[float] | None = None,
    ) -> Dict[str, Any]:
        if distances is None:
            distances = [0.0] * len(ids)
        return {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    @staticmethod
    def _metadata_matches(
        meta: Dict[str, Any], where: Dict[str, Any] | None
    ) -> bool:
        if not where:
            return True
        if "$and" in where:
            return all(
                VectorStore._metadata_matches(meta, clause)
                for clause in where.get("$and") or []
            )
        if "$or" in where:
            return any(
                VectorStore._metadata_matches(meta, clause)
                for clause in where.get("$or") or []
            )
        return all(meta.get(key) == value for key, value in where.items())

    def search(
        self,
        embedding: Sequence[float],
        n_results: int = 5,
        where: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        expected = self.expected_identity
        report = self.compatibility_report(expected)
        if not report["valid"]:
            raise IndexCompatibilityError(report["message"])
        embedding_list = _as_list(embedding)
        if not embedding_list:
            return self._as_query_result([], [], [])
        collection_dim = self._collection_dim()
        if collection_dim and len(embedding_list) != collection_dim:
            raise IndexCompatibilityError(
                f"dimension mismatch: expected {collection_dim}, got {len(embedding_list)}"
            )
        if not self._valid_vector(embedding_list, collection_dim or 0):
            raise IndexCompatibilityError(
                "query embedding is not a valid finite non-zero vector"
            )
        results = self.collection.query(
            query_embeddings=[list(map(float, embedding_list))],
            n_results=max(1, int(n_results)),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        raw_ids = _as_list(results.get("ids"))
        raw_documents = _as_list(results.get("documents"))
        raw_metadatas = _as_list(results.get("metadatas"))
        raw_distances = _as_list(results.get("distances"))
        ids = (
            _as_list(raw_ids[0])
            if raw_ids and isinstance(raw_ids[0], (list, tuple))
            else raw_ids
        )
        documents = (
            _as_list(raw_documents[0])
            if raw_documents and isinstance(raw_documents[0], (list, tuple))
            else raw_documents
        )
        metadatas = (
            _as_list(raw_metadatas[0])
            if raw_metadatas and isinstance(raw_metadatas[0], (list, tuple))
            else raw_metadatas
        )
        distances = (
            _as_list(raw_distances[0])
            if raw_distances and isinstance(raw_distances[0], (list, tuple))
            else raw_distances
        )
        return self._as_query_result(
            [str(item) for item in ids],
            [str(item) for item in documents],
            [dict(item or {}) if isinstance(item, dict) else {} for item in metadatas],
            [float(item) for item in distances],
        )

    def search_lexical(
        self,
        query: str,
        n_results: int = 5,
        where: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return self._as_query_result([], [], [])
        tokens = {token for token in self._lexical_tokens(query) if token}
        if not tokens:
            return self._as_query_result([], [], [])
        with sqlite3.connect(self.lexical_database) as connection:
            records = connection.execute(
                "SELECT id, document, metadata, tokens FROM lexical_documents "
                "WHERE index_state = 'READY'"
            ).fetchall()
        corpus = [json.loads(row[3]) for row in records]
        document_count = len(records)
        document_frequency = {
            token: sum(token in row_tokens for row_tokens in corpus) for token in tokens
        }
        average_length = max(
            1.0,
            sum(len(item) for item in corpus) / max(1, document_count),
        )
        rank: list[tuple[float, dict[str, Any]]] = []
        for row, row_tokens in zip(records, corpus, strict=True):
            meta = self._coerce_metadata(json.loads(row[2]))
            if str(meta.get("index_state", "READY")).upper() != "READY":
                continue
            if not VectorStore._metadata_matches(meta, where):
                continue
            term_counts = {token: row_tokens.count(token) for token in tokens}
            length = max(1, len(row_tokens))
            score = 0.0
            for token, frequency in term_counts.items():
                if not frequency:
                    continue
                idf = math.log(
                    1.0
                    + (document_count - document_frequency[token] + 0.5)
                    / (document_frequency[token] + 0.5)
                )
                score += idf * (frequency * 2.2) / (
                    frequency + 1.2 * (0.75 + 0.25 * length / average_length)
                )
            if score <= 0.0:
                continue
            rank.append(
                (
                    score,
                    {
                        "id": str(row[0]),
                        "document": str(row[1]),
                        "metadata": meta,
                    },
                )
            )
        if not rank:
            return self._as_query_result([], [], [])
        ranked = sorted(rank, key=lambda item: item[0], reverse=True)[
            : max(1, int(n_results))
        ]
        ids = [entry["id"] for _, entry in ranked]
        documents = [entry["document"] for _, entry in ranked]
        metadatas = [entry["metadata"] for _, entry in ranked]
        distances = [1.0 / (1.0 + score) for score, _ in ranked]
        return self._as_query_result(ids, documents, metadatas, distances)

    def set_version_state(self, document_id: str, version_id: str, state: str) -> None:
        self.set_version_index_state(document_id, version_id, state)

    def verify_index(self, document_id: str | None = None) -> Dict[str, Any]:
        if document_id:
            return self.validate_document_index(document_id)
        return self.index_health_check(self.expected_identity)

    def rebuild_index(self, document_id: str | None = None) -> Dict[str, Any]:
        reconciled = self.reconcile_index(document_id)
        health = self.index_health_check(self.expected_identity)
        return {
            "status": "RECONCILED" if health.get("valid") else "NEEDS_ATTENTION",
            "reconciled": reconciled,
            "health": health,
        }

    def activate_staging_index(
        self,
        document_id: str | None = None,
        *,
        expected_identity: Any | None = None,
        staging_directory: str | Path | None = None,
    ) -> Dict[str, Any]:
        if expected_identity is not None:
            self.expected_identity = expected_identity
        if staging_directory is not None:
            staging_path = Path(staging_directory)
        else:
            parent = self.persist_directory.parent
            candidates = sorted(
                parent.glob(f"{self.persist_directory.name}.staging-*"),
                key=lambda path: path.stat().st_mtime,
            )
            if not candidates:
                raise RuntimeError(
                    f"No staging directory matching {self.persist_directory.name}.staging-* was found."
                )
            staging_path = candidates[-1]
        if not staging_path.exists():
            raise RuntimeError(f"Staging directory {staging_path} does not exist.")
        staging_store = VectorStore(staging_path)
        health = staging_store.index_health_check(self.expected_identity)
        if not health["valid"] or not health["metadata_valid"]:
            raise RuntimeError(json.dumps(health, default=str))
        active_path = self.persist_directory
        backup_path = active_path.parent / (
            f"{active_path.name}.backup-{int(time.time())}"
        )
        if active_path.exists():
            shutil.move(str(active_path), str(backup_path))
        shutil.move(str(staging_path), str(active_path))
        self.client = chromadb.PersistentClient(path=str(active_path))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.set_document_index_state(document_id or "*", "READY")
        return {
            "status": "ACTIVATED",
            "staging_directory": str(staging_path),
            "active_directory": str(active_path),
            "vector_count": self.collection.count(),
            "health": health,
            "message": "Staged index validated and activated.",
        }
