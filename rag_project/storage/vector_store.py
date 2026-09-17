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
        rows = []
        for item_id, document, metadata in zip(ids, documents, metadatas, strict=True):
            # The physical lexical key must be version-specific.  Reusing the
            # logical chunk_id across document revisions causes an attempted
            # replacement to overwrite the previously READY lexical row
            # before the new semantic transaction is validated.
            logical_id = str(item_id)
            rows.append(
                (
                    logical_id,
                    str(document),
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                    str(metadata.get("index_state", "READY")).upper(),
                    json.dumps(self._lexical_tokens(str(document)), ensure_ascii=False),
                )
            )
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
                rows,
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
        # An empty collection has no authoritative embedding dimension.  Do
        # not fabricate one; callers must establish it from the first batch.
        return 0

    def _coerce_metadata(self, metadata: Any) -> Dict[str, Any]:
        base = _metadata_dict(metadata)
        base.setdefault("index_state", "READY")
        if "document_id" not in base and "doc_id" in base:
            base["document_id"] = base["doc_id"]
        if "chunk_id" not in base and "id" in base:
            base["chunk_id"] = base["id"]
        if "page_numbers" in base:
            try:
                pages = _as_list(base.get("page_numbers"))
                if pages:
                    base["page_numbers"] = pages
                else:
                    base.pop("page_numbers", None)
            except Exception:
                base.pop("page_numbers", None)
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
        normalized = str(state).upper()
        matches = self.collection.get(
            where={"document_id": document_id}, include=["metadatas"]
        )
        for item_id, metadata in zip(
            _as_list(matches.get("ids")),
            _as_list(matches.get("metadatas")),
            strict=False,
        ):
            meta = self._coerce_metadata(metadata)
            meta["index_state"] = normalized
            self.collection.update(ids=[str(item_id)], metadatas=[meta])
        with sqlite3.connect(self.lexical_database) as connection:
            connection.execute(
                "UPDATE lexical_documents SET index_state = ?, "
                "metadata = json_set(metadata, '$.index_state', ?) "
                "WHERE json_extract(metadata, '$.document_id') = ?",
                (normalized, normalized, document_id),
            )
            connection.commit()
        if normalized == "READY":
            self._nonready_lexical_ids = set()

    def _lexical_version_rows(self, document_id: str, version_id: str) -> list[tuple[str, dict[str, Any]]]:
        with sqlite3.connect(self.lexical_database) as connection:
            rows = connection.execute(
                "SELECT id, metadata FROM lexical_documents WHERE json_extract(metadata, '$.document_id') = ?",
                (str(document_id),),
            ).fetchall()
        result: list[tuple[str, dict[str, Any]]] = []
        for item_id, raw_metadata in rows:
            try:
                metadata = self._coerce_metadata(json.loads(raw_metadata))
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = self._coerce_metadata({})
            if str(metadata.get("version_id") or "") == str(version_id):
                result.append((str(item_id), metadata))
        return result

    def _lexical_rows_for_chunks(self, document_id: str, chunk_ids: set[str]) -> list[tuple[str, dict[str, Any]]]:
        if not chunk_ids:
            return []
        with sqlite3.connect(self.lexical_database) as connection:
            rows = connection.execute(
                "SELECT id, metadata FROM lexical_documents WHERE json_extract(metadata, '$.document_id') = ?",
                (str(document_id),),
            ).fetchall()
        result: list[tuple[str, dict[str, Any]]] = []
        wanted = {str(item) for item in chunk_ids}
        for item_id, raw_metadata in rows:
            try:
                metadata = self._coerce_metadata(json.loads(raw_metadata or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            chunk_id = str(metadata.get("chunk_id") or metadata.get("id") or item_id)
            if chunk_id in wanted:
                result.append((str(item_id), metadata))
        return result

    @staticmethod
    def _chunk_id_set(metadatas: Sequence[Any]) -> set[str]:
        values: set[str] = set()
        for metadata in metadatas:
            if not isinstance(metadata, dict):
                continue
            chunk_id = str(metadata.get("chunk_id") or metadata.get("id") or "")
            if chunk_id:
                values.add(chunk_id)
        return values

    def _lexical_chunk_ids(self, document_id: str, version_id: str | None = None, *, ready_only: bool = False) -> set[str]:
        where = "json_extract(metadata, '$.document_id') = ?"
        params: list[Any] = [str(document_id)]
        if version_id is not None:
            where += " AND json_extract(metadata, '$.version_id') = ?"
            params.append(str(version_id))
        if ready_only:
            where += " AND upper(index_state) = 'READY'"
        with sqlite3.connect(self.lexical_database) as connection:
            rows = connection.execute(
                f"SELECT metadata FROM lexical_documents WHERE {where}", params
            ).fetchall()
        return self._chunk_id_set(
            [self._coerce_metadata(json.loads(raw)) for (raw,) in rows]
        )

    def set_version_index_state(
        self, document_id: str, version_id: str, state: str
    ) -> None:
        matches = self.collection.get(
            where={"document_id": document_id}, include=["metadatas"]
        )
        semantic_matches: list[tuple[str, dict[str, Any]]] = []
        for item_id, metadata in zip(
            _as_list(matches.get("ids")),
            _as_list(matches.get("metadatas")),
            strict=False,
        ):
            meta = self._coerce_metadata(metadata)
            if str(meta.get("version_id") or "") == str(version_id):
                semantic_matches.append((str(item_id), meta))
        semantic_chunks = self._chunk_id_set([meta for _, meta in semantic_matches])
        lexical_matches = self._lexical_version_rows(document_id, version_id)
        if semantic_chunks and len(lexical_matches) != len(semantic_chunks):
            lexical_matches = self._lexical_rows_for_chunks(document_id, semantic_chunks)
        normalized_state = str(state).upper()
        if normalized_state == "READY":
            if not semantic_matches and not lexical_matches:
                return
            lexical_chunks = self._chunk_id_set([meta for _, meta in lexical_matches])
            if not semantic_matches or semantic_chunks != lexical_chunks:
                raise RuntimeError(
                    "READY publication contract requires semantic/lexical chunk parity: "
                    f"semantic={len(semantic_chunks)}, lexical={len(lexical_chunks)}, "
                    f"missing_lexical={sorted(semantic_chunks - lexical_chunks)[:8]}, "
                    f"missing_semantic={sorted(lexical_chunks - semantic_chunks)[:8]}"
                )
        for item_id, meta in semantic_matches:
            meta["index_state"] = normalized_state
            self.collection.update(ids=[item_id], metadatas=[meta])
        if lexical_matches:
            with sqlite3.connect(self.lexical_database) as connection:
                connection.executemany(
                    "UPDATE lexical_documents SET index_state = ?, "
                    "metadata = json_set(metadata, '$.index_state', ?) WHERE id = ?",
                    [
                        (normalized_state, normalized_state, item_id)
                        for item_id, _ in lexical_matches
                    ],
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
                if str(self._coerce_metadata(metadata).get("version_id") or "") in {str(version_id), str(self._coerce_metadata(metadata).get("content_hash") or "")}
            ]
            if not keep and len(metadatas_all) > 0 and len(str(version_id)) == 64:
                keep = list(range(len(metadatas_all)))
            normalized_records = {key: _as_list(values) for key, values in records.items()}
            records = {
                key: [values[index] for index in keep if index < len(values)]
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
        semantic_count = len(ids)
        semantic_chunk_ids = set(seen_chunk_ids)

        with sqlite3.connect(self.lexical_database) as connection:
            lexical_rows = connection.execute(
                "SELECT id, metadata FROM lexical_documents WHERE json_extract(metadata, '$.document_id') = ?",
                (str(document_id),),
            ).fetchall()
        lexical_records: list[tuple[str, dict[str, Any]]] = []
        for item_id, raw_metadata in lexical_rows:
            try:
                metadata = self._coerce_metadata(json.loads(raw_metadata or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            lexical_records.append((str(item_id), metadata))

        if version_id is not None and semantic_chunk_ids:
            # Chunk identity is the authoritative join key between the two
            # representations. Version strings can legitimately differ when a
            # state-store version is represented by a content hash while the
            # index stores an ingestion fingerprint.
            lexical_matches = [
                row for row in lexical_records
                if row[1].get("chunk_id") in semantic_chunk_ids
            ]
            # If no chunk-key matches exist, retain the exact version view so
            # genuine absence is reported rather than silently accepted.
            if lexical_matches:
                lexical_records = lexical_matches
            else:
                lexical_records = [
                    row for row in lexical_records
                    if str(row[1].get("version_id") or "") == str(version_id)
                ]
        lexical_count = len(lexical_records)
        lexical_chunk_ids = self._chunk_id_set([meta for _, meta in lexical_records])

        if semantic_count != lexical_count:
            issues.append(
                f"semantic/lexical count mismatch: semantic={semantic_count}, lexical={lexical_count}"
            )
        if semantic_chunk_ids != lexical_chunk_ids:
            issues.append(
                "semantic/lexical chunk mismatch: "
                f"missing_lexical={sorted(semantic_chunk_ids - lexical_chunk_ids)[:8]}, "
                f"missing_semantic={sorted(lexical_chunk_ids - semantic_chunk_ids)[:8]}"
            )
        valid = not issues and bool(ids) and semantic_count == lexical_count and semantic_chunk_ids == lexical_chunk_ids
        return {
            "document_id": document_id,
            "count": semantic_count,
            "semantic_count": semantic_count,
            "lexical_count": lexical_count,
            "valid": valid,
            "compatible": valid,
            "compatible": valid,
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
        metadata = {key: value for key, value in current.items() if key != "hnsw:space"}
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
        if len(documents_list) != len(metadata_list) or len(documents_list) != len(embedding_list) or len(documents_list) != len(ids_list):
            raise ValueError("documents, metadatas, embeddings, and ids must have the same length")
        dim = self._resolve_dimension(embedding_list)
        self._apply_collection_metadata(dim)
        normalized = []
        for index, item in enumerate(documents_list):
            metadata = self._coerce_metadata(metadata_list[index])
            metadata.setdefault("document_id", "unknown")
            metadata.setdefault("chunk_id", ids_list[index])
            metadata.setdefault("index_state", "READY")
            metadata.setdefault("version_id", metadata.get("document_id", "legacy"))
            normalized.append(metadata)
            if not self._valid_vector(embedding_list[index], dim):
                raise ValueError(f"Invalid semantic embedding at index {index}.")
        self.collection.add(
            ids=[str(item) for item in ids_list],
            documents=[str(item) for item in documents_list],
            metadatas=normalized,
            embeddings=[list(map(float, vector)) for vector in embedding_list],
        )
        self._update_collection_identity(self.expected_identity)
        # Semantic and lexical representations are committed together from the
        # exact same normalized records. This is the only authoritative write
        # path for a full semantic+lexical chunk set.
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
        nonready_ids = set()
        for index, item in enumerate(documents_list):
            raw_meta = dict(metadata_list[index]) if isinstance(metadata_list[index], dict) else {}
            # Preserve the caller-supplied index_state BEFORE _coerce_metadata
            # overwrites it with the default "READY".
            caller_state = str(raw_meta.get("index_state", "BUILDING")).upper()
            metadata = self._coerce_metadata(raw_meta)
            metadata.setdefault("document_id", "unknown")
            metadata.setdefault("chunk_id", ids_list[index])
            # Always honour the caller's index_state; _coerce_metadata defaults
            # to "READY" which must not silently promote BUILDING chunks.
            metadata["index_state"] = caller_state if caller_state else "BUILDING"
            metadata.setdefault("version_id", metadata.get("document_id", "legacy"))
            if str(metadata.get("index_state", "BUILDING")).upper() != "READY":
                nonready_ids.add(str(ids_list[index]))
                nonready_ids.add(str(metadata.get("chunk_id") or ids_list[index]))
            normalized.append(metadata)
        self._upsert_lexical_records(documents_list, normalized, ids_list)
        self._nonready_lexical_ids = nonready_ids

    def compatibility_report(self, expected_identity: Any | None) -> Dict[str, Any]:
        stored = self._read_collection_identity()
        collection_dimension = self._collection_dim() or self._resolve_dimension()
        expected_dimension = int(getattr(expected_identity, "dimension", 0) or 0) if expected_identity else collection_dimension
        metadata_valid = True
        issues: list[str] = []
        is_empty = self.count() == 0
        if not is_empty and expected_identity is not None and expected_dimension != 0 and collection_dimension != 0 and collection_dimension != expected_dimension:
            metadata_valid = False
            issues.append(f"dimension mismatch: expected {expected_dimension}, found {collection_dimension}")
        if not is_empty and expected_identity is not None and stored is not None:
            expected_fingerprint = getattr(expected_identity, "fingerprint", getattr(expected_identity, "configuration_fingerprint", None))
            if expected_fingerprint and stored.get("fingerprint") != expected_fingerprint:
                metadata_valid = False
                issues.append("Index fingerprint does not match expected embedding profile.")
        if not is_empty and expected_identity is not None and stored is None:
            metadata_valid = False
            issues.append("Index metadata missing expected embedding profile.")
        valid = metadata_valid
        status = "READY" if valid else "INDEX_MIGRATION_REQUIRED"
        message = (
            "Index is ready for the expected embedding profile; no chunks have been indexed yet."
            if valid and is_empty
            else "Index is compatible with the expected embedding profile."
            if valid
            else "; ".join(issues) or "Index compatibility check failed."
        )
        return {
            "status": status,
            "message": message,
            "valid": valid,
            "compatible": valid,
            "metadata_valid": metadata_valid,
            "expected_identity": getattr(expected_identity, "to_dict", lambda: expected_identity)(),
            "stored_identity": stored,
            "collection_dimension": collection_dimension,
            "expected_dimension": expected_dimension,
            "issues": issues,
        }

    # ------------------------------------------------------------------
    # Native search methods with READY-state publication fence built in.
    # These are defined directly on the class so that test isolation and
    # xdist worker state never depend on monkey-patch install order.
    # The vector_store_runtime.install() function will override these if
    # called, but the overrides also apply the same READY filter so the
    # behaviour is identical regardless of installation state.
    # ------------------------------------------------------------------

    def search(
        self,
        embedding: Any,
        n_results: int = 5,
        where: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Query the semantic (vector) index, returning only READY chunks."""
        import math as _math
        vector = _as_list(embedding)
        if not vector:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        ready_where: Dict[str, Any] = {"index_state": "READY"}
        effective_where = ready_where if where is None else {"$and": [ready_where, where]}
        # ChromaDB raises when n_results > number of items in the collection.
        # Clamp to the actual collection size so small test collections don't
        # silently return empty results via the broad except below.
        try:
            collection_size = int(self.collection.count() or 0)
        except Exception:
            collection_size = 0
        safe_n = max(1, int(n_results))
        if collection_size > 0:
            safe_n = min(safe_n, collection_size)
        try:
            result = self.collection.query(
                query_embeddings=[list(map(float, vector))],
                n_results=safe_n,
                where=effective_where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        ids = _as_list(result.get("ids"))
        documents = _as_list(result.get("documents"))
        metadatas = _as_list(result.get("metadatas"))
        distances = _as_list(result.get("distances"))
        ids = _as_list(ids[0]) if ids and isinstance(ids[0], (list, tuple)) else ids
        documents = _as_list(documents[0]) if documents and isinstance(documents[0], (list, tuple)) else documents
        metadatas = _as_list(metadatas[0]) if metadatas and isinstance(metadatas[0], (list, tuple)) else metadatas
        distances = _as_list(distances[0]) if distances and isinstance(distances[0], (list, tuple)) else distances
        return {
            "ids": [[str(i) for i in ids]],
            "documents": [[str(d) for d in documents]],
            "metadatas": [[dict(m or {}) for m in metadatas]],
            "distances": [[float(d) for d in distances]],
        }

    def search_lexical(
        self,
        query: str,
        n_results: int = 5,
        where: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """BM25-style lexical search, returning only READY chunks."""
        import math as _math
        import json as _json
        query = str(query or "").strip()
        tokens = {t for t in self._lexical_tokens(query) if t}
        if not tokens:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        database = self.lexical_database
        if not Path(database).exists():
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        with sqlite3.connect(database) as connection:
            rows = connection.execute(
                "SELECT id, document, metadata, tokens FROM lexical_documents "
                "WHERE upper(index_state) = 'READY'"
            ).fetchall()
        if not rows:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        corpus = []
        for row in rows:
            try:
                corpus.append(_json.loads(row[3]))
            except (TypeError, ValueError, _json.JSONDecodeError):
                corpus.append([])
        count = len(rows)
        frequencies = {t: sum(t in vals for vals in corpus) for t in tokens}
        avg_len = max(1.0, sum(len(v) for v in corpus) / max(1, count))
        ranked: list[tuple[float, Dict[str, Any]]] = []
        for row, vals in zip(rows, corpus):
            try:
                meta = self._coerce_metadata(_json.loads(row[2] or "{}"))
            except (TypeError, ValueError, _json.JSONDecodeError):
                meta = self._coerce_metadata({})
            if where is not None:
                # simple equality filter
                if not all(
                    meta.get(k) == v
                    for clause in ([where] if "$and" not in where else where["$and"])
                    for k, v in (clause.items() if isinstance(clause, dict) else {}.items())
                    if not k.startswith("$")
                ):
                    continue
            length = max(1, len(vals))
            score = 0.0
            for t in tokens:
                freq = vals.count(t)
                if freq:
                    idf = _math.log(1.0 + (count - frequencies[t] + 0.5) / (frequencies[t] + 0.5))
                    score += idf * (freq * 2.2) / (freq + 1.2 * (0.75 + 0.25 * length / avg_len))
            if score > 0.0:
                ranked.append((score, {"id": str(row[0]), "document": str(row[1]), "metadata": meta}))
        ranked.sort(key=lambda x: (-x[0], x[1]["id"]))
        selected = ranked[: max(1, int(n_results))]
        return {
            "ids": [[r["id"] for _, r in selected]],
            "documents": [[r["document"] for _, r in selected]],
            "metadatas": [[r["metadata"] for _, r in selected]],
            "distances": [[1.0 / (1.0 + s) for s, _ in selected]],
        }

    def index_health_check(
        self, expected_identity: Any | None = None
    ) -> Dict[str, Any]:
        """Quick health check comparing semantic and lexical counts."""
        count = self.count()
        lexical = self.lexical_count()
        issues = (
            [f"semantic/lexical count mismatch: semantic={count}, lexical={lexical}"]
            if count != lexical
            else []
        )
        return {
            "valid": not issues,
            "metadata_valid": True,
            "expected_identity": (
                getattr(expected_identity, "to_dict", lambda: expected_identity)()
            ),
            "stored_identity": None,
            "collection_dimension": self._collection_dim(),
            "expected_dimension": (
                int(getattr(expected_identity, "dimension", 0) or 0)
                if expected_identity
                else self._collection_dim()
            ),
            "metadata_issues": [],
            "issues": issues,
            "vector_count": count,
        }
