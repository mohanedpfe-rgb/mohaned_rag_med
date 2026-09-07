from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import chromadb
import json
import re
import sqlite3


class IndexCompatibilityError(RuntimeError):
    def __init__(self, diagnostic: dict[str, Any]):
        self.diagnostic = diagnostic
        super().__init__(diagnostic["message"])


class VectorStore:
    def __init__(self, persist_directory: str | Path):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"},
        )
        self.manifest_path = self.persist_directory / "index_manifest.json"
        self._manifest = self._load_manifest()
        self.expected_identity: dict[str, Any] | None = None
        self.lexical_database = self.persist_directory / "lexical.sqlite3"
        self._initialize_lexical_index()

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Invalid vector index manifest: {self.manifest_path}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"Invalid vector index manifest: {self.manifest_path}")
        return value

    @property
    def collection_dimension(self) -> int | None:
        if self.collection.count() == 0:
            return None
        embeddings = self.collection.peek(1).get("embeddings")
        return len(embeddings[0]) if embeddings is not None and len(embeddings) else None

    @staticmethod
    def _coerce_identity(identity: Any) -> dict[str, Any] | None:
        if identity is None:
            return None
        if isinstance(identity, dict):
            return identity
        if hasattr(identity, "to_dict"):
            return identity.to_dict()
        return dict(identity)

    def set_expected_identity(self, identity: Any) -> None:
        self.expected_identity = self._coerce_identity(identity)

    def compatibility_report(self, expected_identity: Any | None = None) -> dict[str, Any]:
        expected = self._coerce_identity(expected_identity) or self.expected_identity
        collection_dimension = self.collection_dimension
        expected_dimension = expected.get("dimension") if expected else None
        reasons: list[str] = []
        if expected_dimension and collection_dimension and expected_dimension != collection_dimension:
            reasons.append(
                f"Embedding configuration changed from {collection_dimension} dimensions "
                f"to {expected_dimension} dimensions."
            )
        stored_identity = self._manifest.get("embedding")
        if expected and self.collection.count() and not stored_identity:
            reasons.append("Existing index has no embedding identity manifest.")
        elif expected and stored_identity:
            expected_fingerprint = (
                expected.get("configuration_fingerprint")
                or expected.get("fingerprint")
                or expected.get("embedding_id")
            )
            stored_fingerprint = (
                stored_identity.get("configuration_fingerprint")
                or stored_identity.get("fingerprint")
                or stored_identity.get("embedding_id")
            )
            if expected_fingerprint and stored_fingerprint and expected_fingerprint != stored_fingerprint:
                reasons.append("Index fingerprint does not match the current embedding profile.")
            for key in ("provider", "model", "model_name", "model_version", "dimension", "metric", "normalization"):
                if expected.get(key) is not None and stored_identity.get(key) != expected.get(key):
                    reasons.append(f"Index identity differs for {key}.")
        if expected and expected.get("metric") and self.collection.metadata.get("hnsw:space") != expected.get("metric"):
            reasons.append("Collection metric differs from the current embedding profile.")
        return {
            "status": "INDEX_MIGRATION_REQUIRED" if reasons else "READY",
            "valid": not reasons,
            "message": "Index is compatible." if not reasons else (
                " ".join(reasons) + " A rebuild/migration is required."
            ),
            "expected_dimension": expected_dimension,
            "collection_dimension": collection_dimension,
            "expected_identity": expected,
            "stored_identity": stored_identity,
            "metric": self.collection.metadata.get("hnsw:space", "cosine"),
            "vector_count": self.collection.count(),
        }

    def ensure_compatible(self, vector: list[float], *, operation: str) -> None:
        actual_dimension = len(vector)
        collection_dimension = self.collection_dimension
        if collection_dimension is not None and actual_dimension != collection_dimension:
            raise IndexCompatibilityError(
                {
                    "status": "INDEX_MIGRATION_REQUIRED",
                    "operation": operation,
                    "expected_dimension": collection_dimension,
                    "actual_dimension": actual_dimension,
                    "message": (
                        f"Embedding dimension mismatch before {operation}: index expects "
                        f"{collection_dimension}, received {actual_dimension}. "
                        "A rebuild/migration is required."
                    ),
                }
            )
        report = self.compatibility_report()
        if self.expected_identity and report["status"] != "READY":
            raise IndexCompatibilityError({**report, "operation": operation})

    def _lexical_connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.lexical_database, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def _initialize_lexical_index(self) -> None:
        with self._lexical_connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS lexical_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    document TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS lexical_chunks_fts
                USING fts5(document, content='lexical_chunks', content_rowid='rowid')
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS lexical_chunks_ai
                AFTER INSERT ON lexical_chunks BEGIN
                    INSERT INTO lexical_chunks_fts(rowid, document)
                    VALUES (new.rowid, new.document);
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS lexical_chunks_ad
                AFTER DELETE ON lexical_chunks BEGIN
                    INSERT INTO lexical_chunks_fts(lexical_chunks_fts, rowid, document)
                    VALUES ('delete', old.rowid, old.document);
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS lexical_chunks_au
                AFTER UPDATE OF document ON lexical_chunks BEGIN
                    INSERT INTO lexical_chunks_fts(lexical_chunks_fts, rowid, document)
                    VALUES ('delete', old.rowid, old.document);
                    INSERT INTO lexical_chunks_fts(rowid, document)
                    VALUES (new.rowid, new.document);
                END
                """
            )
            before_triggers = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='lexical_chunks'"
                ).fetchall()
            }
            lexical_count = connection.execute(
                "SELECT COUNT(*) FROM lexical_chunks"
            ).fetchone()[0]
            fts_count = connection.execute(
                "SELECT COUNT(*) FROM lexical_chunks_fts"
            ).fetchone()[0]
            after_triggers = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='lexical_chunks'"
                ).fetchall()
            }
            triggers_created_now = after_triggers - before_triggers
            if lexical_count == 0 and fts_count == 0:
                return
            fts_table_just_created = (
                before_triggers == set() and fts_count == 0 and triggers_created_now
            )
            if lexical_count > 0 and not fts_table_just_created:
                if lexical_count == fts_count:
                    return
            connection.execute(
                "INSERT INTO lexical_chunks_fts(lexical_chunks_fts) VALUES ('rebuild')"
            )

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]], ids: List[str]) -> None:
        if not documents:
            return
        dimensions = {len(vector) for vector in embeddings}
        if len(dimensions) != 1:
            raise ValueError("All embeddings in a batch must have the same dimension.")
        self.ensure_compatible(embeddings[0], operation="indexing")
        normalized_metadata: List[Dict[str, Any]] = []
        for metadata in metadatas:
            item = dict(metadata or {})
            item.setdefault("index_state", "READY")
            item.setdefault("version_id", item.get("document_id", "unknown"))
            normalized_metadata.append(item)
        self.collection.add(documents=documents, metadatas=normalized_metadata, embeddings=embeddings, ids=ids)
        if self.expected_identity:
            embedding_profile = dict(self.expected_identity)
            embedding_profile.setdefault("model_name", embedding_profile.get("model", "unknown"))
            embedding_profile.setdefault("normalized_model_identifier", f"{embedding_profile.get('provider', 'unknown')}:{embedding_profile.get('model_name', embedding_profile.get('model', 'unknown'))}:{embedding_profile.get('model_version', 'unknown')}")
            embedding_profile.setdefault("configuration_fingerprint", embedding_profile.get("fingerprint") or embedding_profile.get("configuration_fingerprint") or embedding_profile.get("embedding_id"))
            self._manifest = {
                "schema_version": 2,
                "metric": self.collection.metadata.get("hnsw:space", "cosine"),
                "embedding": embedding_profile,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self.manifest_path.write_text(json.dumps(self._manifest, indent=2), encoding="utf-8")
        with self._lexical_connect() as connection:
            connection.executemany(
                """
                INSERT INTO lexical_chunks(chunk_id, document_id, document, metadata)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    document_id=excluded.document_id,
                    document=excluded.document,
                    metadata=excluded.metadata
                """,
                [
                    (
                        item_id,
                        str(metadata.get("document_id", "")),
                        document,
                        json.dumps(metadata, default=str),
                    )
                    for item_id, document, metadata in zip(
                        ids, documents, normalized_metadata, strict=True
                    )
                ],
            )

    def add_lexical_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> None:
        if not documents:
            return
        with self._lexical_connect() as connection:
            connection.executemany(
                """
                INSERT INTO lexical_chunks(chunk_id, document_id, document, metadata)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    document_id=excluded.document_id,
                    document=excluded.document,
                    metadata=excluded.metadata
                """,
                [
                    (
                        item_id,
                        str(metadata.get("document_id", "")),
                        document,
                        json.dumps(metadata, default=str),
                    )
                    for item_id, document, metadata in zip(ids, documents, metadatas, strict=True)
                ],
            )

    @staticmethod
    def _fts_query(query: str) -> str:
        terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
        if not terms:
            return ""
        # Require all query terms for the precise lexical pass. The vector
        # pass still recovers semantically related wording.
        return " AND ".join(f'"{term}"' for term in terms)

    def search_lexical(
        self,
        query: str,
        n_results: int = 10,
        where: Dict[str, Any] | None = None,
    ) -> Dict[str, List[List[Any]]]:
        fts_query = self._fts_query(query)
        if not fts_query:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        sql = """
            SELECT c.chunk_id, c.document, c.metadata, bm25(lexical_chunks_fts) AS rank
            FROM lexical_chunks_fts
            JOIN lexical_chunks c ON c.rowid = lexical_chunks_fts.rowid
            WHERE lexical_chunks_fts MATCH ?
              AND json_extract(c.metadata, '$.index_state') = 'READY'
        """
        parameters: list[Any] = [fts_query]
        filters: dict[str, Any] = {}
        if where:
            clauses = where.get("$and", [where]) if "$and" in where else [where]
            for clause in clauses:
                if isinstance(clause, dict):
                    filters.update(
                        {
                            key: value
                            for key, value in clause.items()
                            if key not in {"index_state", "$and"} and value is not None
                        }
                    )
        for key, value in filters.items():
            if isinstance(value, (str, int, float, bool)):
                sql += f" AND json_extract(c.metadata, '$.{key}') = ?"
                parameters.append(value)
        sql += " ORDER BY rank LIMIT ?"
        parameters.append(max(1, n_results))
        with self._lexical_connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
            if not rows and len(re.findall(r"[\w]+", query, flags=re.UNICODE)) > 1:
                fallback = " OR ".join(
                    f'"{term}"' for term in re.findall(r"[\w]+", query, flags=re.UNICODE)
                )
                rows = connection.execute(sql, [fallback, *parameters[1:]]).fetchall()
        return {
            "ids": [[row["chunk_id"] for row in rows]],
            "documents": [[row["document"] for row in rows]],
            "metadatas": [[json.loads(row["metadata"]) for row in rows]],
            "distances": [[1.0 / (1.0 + max(0.0, -float(row["rank"]))) for row in rows]],
        }

    def search(self, query_embedding: List[float], n_results: int = 5, where: Dict[str, Any] | None = None):
        self.ensure_compatible(query_embedding, operation="query")
        if where is None:
            where = {"index_state": "READY"}
        elif "index_state" not in where:
            where = {"$and": [{"index_state": "READY"}, where]}
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def rebuild_index(self, document_id: str | None = None) -> Dict[str, Any]:
        """Rebuild the index by activating a staged directory.

        The external script `scripts/rebuild_index.py` creates a staged index
        directory. This method validates the staged index, swaps it with the
        active one, and updates the collection reference.
        """
        import time
        import shutil
        from pathlib import Path
        import chromadb
        import json

        # Determine staged directory path (expected to be created by the script)
        staging_path = Path(self.persist_directory.parent) / (
            f"{self.persist_directory.name}.staging-{time.strftime('%Y%m%d%H%M%S')}"
        )
        if not staging_path.exists():
            raise RuntimeError(f"Staging directory {staging_path} does not exist.")

        # Verify health of the staged index before activation
        health = self.index_health_check(self.expected_identity)
        if not health["valid"] or not health.get("metadata_valid", True):
            raise RuntimeError(json.dumps(health, default=str))

        # Atomically replace the active index directory with the staged one
        active_path = self.persist_directory
        backup_path = active_path.parent / f"{active_path.name}.backup-{int(time.time())}"
        if active_path.exists():
            shutil.move(str(active_path), str(backup_path))
        shutil.move(str(staging_path), str(active_path))

        # Re‑initialize the chromadb client and collection to point to the new index
        self.client = chromadb.PersistentClient(path=str(active_path))
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata=self.collection.metadata,
        )

        # Mark all documents as READY (or a specific document if provided)
        self.set_document_index_state(document_id or "*", "READY")

        return {
            "status": "ACTIVATED",
            "staging_directory": str(staging_path),
            "active_directory": str(active_path),
            "vector_count": self.collection.count(),
            "health": health,
            "message": "Staged index validated and activated.",
        }

    def verify_index(self, document_id: str | None = None) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"include": ["documents", "metadatas", "embeddings"]}
        if document_id is not None:
            kwargs["where"] = {"document_id": document_id}
        records = self.collection.get(**kwargs)
        ids = records.get("ids", [])
        documents = records.get("documents", [])
        metadatas = records.get("metadatas", [])
        embeddings = records.get("embeddings", [])
        issues: List[str] = []
        seen: set[str] = set()
        identity_mismatches: int = 0
        for index, item_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            document = documents[index] if index < len(documents) else ""
            item_embedding = embeddings[index] if index < len(embeddings) else None
            if not metadata:
                issues.append(f"{item_id}: missing metadata")
                continue
            required = ["document_id", "chunk_id", "page_numbers", "version_id", "index_state"]
            missing = [field for field in required if metadata.get(field) in (None, "", [])]
            if missing:
                issues.append(f"{item_id}: missing required metadata {missing}")
            metadata_chunk_id = str(metadata.get("chunk_id") or "")
            if metadata_chunk_id and metadata_chunk_id != item_id:
                identity_mismatches += 1
                issues.append(
                    f"{item_id}: Chroma primary key does not match metadata chunk_id "
                    f"({metadata_chunk_id!r}); provenance/cross-reference lookups will fail"
                )
            if metadata.get("index_state") not in {"READY", "PENDING", "FAILED", "BUILDING", "ACTIVE"}:
                issues.append(f"{item_id}: invalid index_state {metadata.get('index_state')!r}")
            if item_embedding is not None and len(item_embedding) == 0:
                issues.append(f"{item_id}: zero-length embedding")
            chunk_key = str(metadata.get("chunk_id") or metadata.get("document_id") or item_id)
            if chunk_key in seen:
                issues.append(f"{item_id}: duplicate chunk metadata for {chunk_key}")
            else:
                seen.add(chunk_key)
            if not document or not str(document).strip():
                issues.append(f"{item_id}: empty document text")
        return {
            "document_id": document_id,
            "count": len(ids),
            "valid": not issues,
            "issues": issues,
            "identity_mismatches": identity_mismatches,
        }

    def index_health_check(self, expected_identity: Any | None = None) -> Dict[str, Any]:
        compatibility = self.compatibility_report(expected_identity)
        verification = self.verify_index()
        return {
            **compatibility,
            "collection_name": self.collection.name,
            "metadata_valid": verification["valid"],
            "metadata_issues": verification["issues"],
            "vector_count": self.collection.count(),
        }

    def reconcile_index(self, document_id: str | None = None) -> Dict[str, Any]:
        result = self.verify_index(document_id)
        if result["valid"]:
            return result
        kwargs: Dict[str, Any] = {"include": ["documents", "metadatas"]}
        if document_id is not None:
            kwargs["where"] = {"document_id": document_id}
        records = self.collection.get(**kwargs)
        ids = records.get("ids", [])
        metadatas = records.get("metadatas", [])
        seen: set[str] = set()
        identity_repairs: int = 0
        for index, item_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            key = str(metadata.get("chunk_id") or metadata.get("document_id") or item_id)
            if key in seen:
                self.collection.delete(ids=[item_id])
                continue
            seen.add(key)
            metadata_chunk_id = str(metadata.get("chunk_id") or "")
            if metadata_chunk_id and metadata_chunk_id != item_id:
                metadata["chunk_id"] = item_id
                identity_repairs += 1
            if metadata.get("index_state") is None:
                metadata["index_state"] = "READY"
            if metadata.get("version_id") is None:
                metadata["version_id"] = metadata.get("document_id", "unknown")
            if metadata.get("page_numbers") is None:
                metadata["page_numbers"] = []
            self.collection.update(ids=[item_id], metadatas=[metadata])
        reconciled = self.verify_index(document_id)
        return {
            "document_id": document_id,
            "valid": reconciled["valid"],
            "issues": reconciled["issues"],
            "count": reconciled["count"],
            "identity_repairs": identity_repairs,
        }


        import time
        import shutil
        from pathlib import Path
        import chromadb
        import json

        # Determine staged directory path (expected to be created by the script)
        staging_path = Path(self.persist_directory.parent) / (
            f"{self.persist_directory.name}.staging-{time.strftime('%Y%m%d%H%M%S')}"
        )
        if not staging_path.exists():
            raise RuntimeError(f"Staging directory {staging_path} does not exist.")

        # Verify health of the staged index before activation
        health = self.index_health_check(self.expected_identity)
        if not health["valid"] or not health.get("metadata_valid", True):
            raise RuntimeError(json.dumps(health, default=str))

        # Atomically replace the active index directory with the staged one
        active_path = self.persist_directory
        backup_path = active_path.parent / f"{active_path.name}.backup-{int(time.time())}"
        if active_path.exists():
            shutil.move(str(active_path), str(backup_path))
        shutil.move(str(staging_path), str(active_path))

        # Re‑initialize the chromadb client and collection to point to the new index
        self.client = chromadb.PersistentClient(path=str(active_path))
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata=self.collection.metadata,
        )

            """
            import time
            import shutil
            from pathlib import Path
            import chromadb

            # Determine staged directory path (expected to be created by the script)
            staging_path = Path(self.persist_directory.parent) / (
                f"{self.persist_directory.name}.staging-{time.strftime('%Y%m%d%H%M%S')}"
            )
            if not staging_path.exists():
                raise RuntimeError(f"Staging directory {staging_path} does not exist.")

            # Re‑initialize the chromadb client and collection to point to the new index
            self.client = chromadb.PersistentClient(path=str(active_path))
            self.collection = self.client.get_or_create_collection(
                name=self.collection.name,
                metadata=self.collection.metadata,
            )

            # Mark all documents as READY (or a specific document if provided)
            self.set_document_index_state(document_id or "*", "READY")

            return {
                "status": "ACTIVATED",
                "staging_directory": str(staging_path),
                "active_directory": str(active_path),
                "vector_count": self.collection.count(),
                "health": health,
                "message": "Staged index validated and activated.",
            }
(self, document_id: str | None = None) -> Dict[str, Any]:
        """Rebuild the index by activating a staged directory.

        The external script `scripts/rebuild_index.py` creates a staged index
        directory. This method validates the staged index, swaps it with the
        active one, and updates the collection reference.
        """
        import time
        import shutil
        import os
        from pathlib import Path
        import chromadb

        # Determine staged directory path (expected to be created by the script)
        staging_path = Path(self.persist_directory.parent) / (
            f"{self.persist_directory.name}.staging-{time.strftime('%Y%m%d%H%M%S')}"
        )
        if not staging_path.exists():
            raise RuntimeError(f"Staging directory {staging_path} does not exist.")

        # Verify health of the staged index before activation
        health = self.index_health_check(self.expected_identity)
        if not health["valid"] or not health["metadata_valid"]:
            raise RuntimeError(json.dumps(health, default=str))

        # Atomically replace the active index directory with the staged one
        active_path = self.persist_directory
        backup_path = active_path.parent / f"{active_path.name}.backup-{int(time.time())}"
        if active_path.exists():
            shutil.move(str(active_path), str(backup_path))
        shutil.move(str(staging_path), str(active_path))

        # Re‑initialize the chromadb client and collection to point to the new index
        self.client = chromadb.PersistentClient(path=str(active_path))
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata=self.collection.metadata,
        )

        # Mark all documents as READY (or a specific document if provided)
        self.set_document_index_state(document_id or "*", "READY")

        return {
            "status": "ACTIVATED",
            "staging_directory": str(staging_path),
            "active_directory": str(active_path),
            "vector_count": self.collection.count(),
            "health": health,
            "message": "Staged index validated and activated.",
        }



        # This mirrors the behaviour described in the implementation plan.
        # The original method returned a MIGRATION_REQUIRED status without activation.
        # New flow:
        #   1. Build a staged index (handled by the external script).
        #   2. Validate health via `index_health_check`.
        #   3. If valid, atomically replace the active directory.
        #   4. Update index_state to READY for all documents.
        #   5. Return a status indicating activation.
        staging_path = Path(self.persist_directory.parent) / (
            f"{self.persist_directory.name}.staging-{time.strftime('%Y%m%d%H%M%S')}"
        )
        # The external script already creates the staged index and returns its path.
        # Here we assume the caller has moved the staged index to `staging_path` and
        # validated it, so we just promote it.
        if not staging_path.exists():
            raise RuntimeError(f"Staging directory {staging_path} does not exist.")
        # Verify health before activation
        health = self.index_health_check(self._expected_identity)
        if not health["valid"] or not health["metadata_valid"]:
            raise RuntimeError(json.dumps(health, default=str))
        # Atomically replace the active index directory
        active_path = self.persist_directory
        backup_path = active_path.parent / f"{active_path.name}.backup-{int(time.time())}"
        # Move current active to backup (if it exists)
        if active_path.exists():
            shutil.move(str(active_path), str(backup_path))
        # Move staged to active
        shutil.move(str(staging_path), str(active_path))
        # Update the in‑memory collection reference to point to the new index
        self.collection = self._load_collection(active_path)
        # Mark all documents as READY
        self.set_document_index_state(document_id or "*", "READY")
        return {
            "status": "ACTIVATED",
            "staging_directory": str(staging_path),
            "active_directory": str(active_path),
            "vector_count": self.collection.count(),
            "health": health,
            "message": "Staged index validated and activated.",
        }

    def validate_document_index(self, document_id: str) -> Dict[str, Any]:
        return self.verify_index(document_id)

    def count(self) -> int:
        return self.collection.count()

    def delete(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})
        with self._lexical_connect() as connection:
            connection.execute(
                "DELETE FROM lexical_chunks WHERE document_id = ?", (document_id,)
            )

    def delete_version(self, document_id: str, version_id: str) -> None:
        self.collection.delete(
            where={"$and": [{"document_id": document_id}, {"version_id": version_id}]}
        )
        with self._lexical_connect() as connection:
            connection.execute(
                """
                DELETE FROM lexical_chunks
                WHERE document_id = ?
                  AND json_extract(metadata, '$.version_id') = ?
                """,
                (document_id, version_id),
            )

    def set_document_index_state(self, document_id: str, state: str) -> None:
        """Atomically expose a completed document to both retrieval indexes."""
        if state not in {"BUILDING", "READY", "FAILED"}:
            raise ValueError(f"Unsupported index state: {state}")
        self.set_version_index_state(document_id, None, state)

    def set_version_index_state(
        self,
        document_id: str,
        version_id: str | None,
        state: str,
    ) -> None:
        """Update one document version, or all versions when version_id is None."""
        where: Dict[str, Any] = {"document_id": document_id}
        if version_id is not None:
            where = {"$and": [where, {"version_id": version_id}]}
        records = self.collection.get(where=where, include=["metadatas"])
        ids = records.get("ids", [])
        metadatas = records.get("metadatas", [])
        updated = []
        for metadata in metadatas:
            item = dict(metadata or {})
            item["index_state"] = state
            updated.append(item)
        if ids:
            self.collection.update(ids=ids, metadatas=updated)
        with self._lexical_connect() as connection:
            if version_id is None:
                connection.execute(
                    "UPDATE lexical_chunks SET metadata = "
                    "json_set(metadata, '$.index_state', ?) WHERE document_id = ?",
                    (state, document_id),
                )
            else:
                connection.execute(
                    "UPDATE lexical_chunks SET metadata = "
                    "json_set(metadata, '$.index_state', ?) "
                    "WHERE document_id = ? AND json_extract(metadata, '$.version_id') = ?",
                    (state, document_id, version_id),
                )

    def get_documents(self) -> dict:
        return self.collection.get(include=["documents", "metadatas"])
