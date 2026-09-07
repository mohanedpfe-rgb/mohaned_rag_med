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
                   for item_id, document, metadata in zip(ids, documents, metadatas, strict=True)
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
       if embeddings:
           first = next(iter(embeddings), None)
           if first is not None:
               return len(first)
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
       ids = list(records.get("ids", []) or [])
       documents = list(records.get("documents", []) or [])
       metadatas = list(records.get("metadatas", []) or [])
       results: list[dict[str, Any]] = []
       for index, item_id in enumerate(ids):
           metadata = self._coerce_metadata(metadatas[index] if index < len(metadatas) else {})
           document = documents[index] if index < len(documents) else ""
           results.append({"id": str(item_id), "document": str(document), "metadata": metadata})
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
       metadata = {key: value for key, value in (self.collection.metadata or {}).items() if key != "hnsw:space"}
       metadata.update(
           {
               "provider": getattr(identity, "provider", "unknown"),
               "model": getattr(identity, "model", "unknown"),
               "model_version": getattr(identity, "model_version", "unknown"),
               "dimension": int(getattr(identity, "dimension", 0) or 0),
               "embedding_fingerprint": getattr(identity, "fingerprint", getattr(identity, "configuration_fingerprint", "")),
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

   def get_documents(self, where: Dict[str, Any] | None = None) -> Dict[str, Any]:
       if where:
           return self.collection.get(where=where, include=["documents", "metadatas"])
       return self.collection.get(include=["documents", "metadatas"])

   def set_document_index_state(self, document_id: str, state: str) -> None:
      if document_id == "*":
        matches = self.collection.get(include=["metadatas"])
        for item_id, metadata in zip(matches.get("ids", []), matches.get("metadatas", []), strict=False):
           meta = self._coerce_metadata(metadata)
           meta["index_state"] = state
           self.collection.update(ids=[str(item_id)], metadatas=[meta])
        with sqlite3.connect(self.lexical_database) as connection:
           connection.execute(
              "UPDATE lexical_documents SET index_state = ?, metadata = json_set(metadata, '$.index_state', ?)",
              (str(state).upper(), str(state).upper()),
           )
        return
      matches = self.collection.get(where={"document_id": document_id}, include=["metadatas"])
      for item_id, metadata in zip(matches.get("ids", []), matches.get("metadatas", []), strict=False):
        meta = self._coerce_metadata(metadata)
        meta["index_state"] = state
        self.collection.update(ids=[str(item_id)], metadatas=[meta])
      with sqlite3.connect(self.lexical_database) as connection:
        connection.execute(
           "UPDATE lexical_documents SET index_state = ?, metadata = json_set(metadata, '$.index_state', ?) "
           "WHERE json_extract(metadata, '$.document_id') = ?",
           (str(state).upper(), str(state).upper(), document_id),
        )

   def set_version_index_state(self, document_id: str, version_id: str, state: str) -> None:
      matches = self.collection.get(where={"document_id": document_id}, include=["metadatas"])
      for item_id, metadata in zip(matches.get("ids", []), matches.get("metadatas", []), strict=False):
        meta = self._coerce_metadata(metadata)
        if meta.get("version_id") == version_id:
           meta["index_state"] = state
           self.collection.update(ids=[str(item_id)], metadatas=[meta])
      with sqlite3.connect(self.lexical_database) as connection:
        rows = connection.execute(
           "SELECT id, metadata FROM lexical_documents"
        ).fetchall()
        matching = [
           row[0] for row in rows
           if (json.loads(row[1]).get("document_id") == document_id
               and json.loads(row[1]).get("version_id") == version_id)
        ]
        if matching:
           connection.executemany(
              "UPDATE lexical_documents SET index_state = ?, metadata = json_set(metadata, '$.index_state', ?) WHERE id = ?",
              [(str(state).upper(), str(state).upper(), item_id) for item_id in matching],
           )

   def delete_version(self, document_id: str, version_id: str) -> None:
       matches = self.collection.get(where={"document_id": document_id}, include=["metadatas"])
       removable = []
       for item_id, metadata in zip(matches.get("ids", []), matches.get("metadatas", []), strict=False):
           meta = self._coerce_metadata(metadata)
           if meta.get("version_id") == version_id:
               removable.append(str(item_id))
       if removable:
           self.collection.delete(ids=removable)
       with sqlite3.connect(self.lexical_database) as connection:
           rows = connection.execute(
               "SELECT id, metadata FROM lexical_documents"
           ).fetchall()
           matching = [
               row[0] for row in rows
               if (json.loads(row[1]).get("document_id") == document_id
                   and json.loads(row[1]).get("version_id") == version_id)
           ]
           if matching:
               connection.executemany(
                   "DELETE FROM lexical_documents WHERE id = ?",
                   [(item_id,) for item_id in matching],
               )

   def reconcile_index(self, document_id: str | None = None) -> Dict[str, Any]:
       query = {"document_id": document_id} if document_id else None
       if query is not None:
           records = self.collection.get(where=query, include=["metadatas", "documents"])
       else:
           records = self.collection.get(include=["metadatas", "documents"])
       deduped: Dict[str, Dict[str, Any]] = {}
       removed = 0
       for item_id, metadata, document in zip(
           records.get("ids", []),
           records.get("metadatas", []),
           records.get("documents", []),
           strict=False,
       ):
           meta = self._coerce_metadata(metadata)
           key = f"{meta.get('document_id','')}::{meta.get('chunk_id', str(item_id))}"
           if key in deduped:
               removed += 1
               continue
           deduped[key] = {"id": str(item_id), "metadata": meta, "document": str(document)}
       if removed:
           old_ids = [entry["id"] for entry in deduped.values()]
           all_ids = [str(item_id) for item_id in records.get("ids", [])]
           stale = [item_id for item_id in all_ids if item_id not in old_ids]
           if stale:
               self.collection.delete(ids=stale)
       count = len(deduped)
       return {"valid": True, "removed": removed, "count": count, "document_id": document_id}

   def validate_document_index(
       self,
       document_id: str,
       version_id: str | None = None,
   ) -> Dict[str, Any]:
       records = self.collection.get(
           where={"document_id": document_id},
           include=["metadatas", "documents", "embeddings"],
       )
       if version_id is not None:
           keep = [
               index
               for index, metadata in enumerate(records.get("metadatas", []))
               if self._coerce_metadata(metadata).get("version_id") == version_id
           ]
           records = {
               key: [values[index] for index in keep]
               for key, values in records.items()
               if isinstance(values, list)
           }
       metadatas = records.get("metadatas", [])
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
       embeddings = records.get("embeddings")
       if embeddings is None:
           embeddings = []
       for vector in embeddings:
           if not self._valid_vector(vector, self._collection_dim()):
               issues.append("invalid semantic embedding")
       valid = not issues and bool(records.get("ids"))
       return {"document_id": document_id, "count": len(records.get("ids", [])), "valid": valid, "issues": issues}

   @staticmethod
   def _valid_vector(vector: Sequence[float], expected_dimension: int = 0) -> bool:
     try:
        values = [float(value) for value in vector]
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
       if not documents:
           return
       if len(documents) != len(metadatas) or len(documents) != len(embeddings) or len(documents) != len(ids):
           raise ValueError("documents, metadatas, embeddings, and ids must have the same length")
       dim = self._resolve_dimension(embeddings)
       self._apply_collection_metadata(dim)
       normalized = []
       for index, item in enumerate(documents):
           metadata = self._coerce_metadata(metadatas[index])
           metadata.setdefault("document_id", "unknown")
           metadata.setdefault("chunk_id", ids[index])
           metadata.setdefault("index_state", "READY")
           metadata.setdefault("version_id", metadata.get("document_id", "legacy"))
           metadata.setdefault("page_numbers", [])
           normalized.append(metadata)
           if not self._valid_vector(embeddings[index], dim):
               raise ValueError(f"Invalid semantic embedding at index {index}.")
       self.collection.add(
           ids=[str(item) for item in ids],
           documents=[str(item) for item in documents],
           metadatas=normalized,
           embeddings=[list(map(float, vector)) for vector in embeddings],
       )
       self._update_collection_identity(self.expected_identity)
       self._upsert_lexical_records(documents, normalized, ids)

   def add_lexical_documents(
       self,
       documents: Sequence[str],
       metadatas: Sequence[Dict[str, Any]],
       ids: Sequence[str],
   ) -> None:
       if not documents:
           return
       if len(documents) != len(metadatas) or len(documents) != len(ids):
           raise ValueError("documents, metadatas, and ids must have the same length")
       normalized = []
       for index, item in enumerate(documents):
           metadata = self._coerce_metadata(metadatas[index])
           metadata.setdefault("document_id", "unknown")
           metadata.setdefault("chunk_id", ids[index])
           metadata.setdefault("index_state", "BUILDING")
           metadata.setdefault("version_id", metadata.get("document_id", "legacy"))
           metadata.setdefault("page_numbers", [])
           normalized.append(metadata)
       self._upsert_lexical_records(documents, normalized, ids)

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
       elif not is_empty and expected_identity is not None and stored is None:
           metadata_valid = False
           issues.append("Index metadata missing expected embedding profile.")
       if not is_empty:
           records = self.collection.get(include=["embeddings"])
           embeddings = records.get("embeddings")
           if embeddings is None:
               embeddings = []
           invalid_count = sum(
               1 for vector in embeddings
               if not self._valid_vector(vector, collection_dimension)
           )
           if invalid_count:
               metadata_valid = False
               issues.append(f"Index contains {invalid_count} invalid semantic embeddings.")
       valid = metadata_valid
       status = "READY"
       if not valid:
           status = "INDEX_MIGRATION_REQUIRED"
       if valid and is_empty:
           message = "Index is ready for the expected embedding profile; no chunks have been indexed yet."
       else:
           message = "Index is compatible with the expected embedding profile." if valid else "; ".join(issues) or "Index compatibility check failed."
       return {
           "status": status,
           "message": message,
           "valid": valid,
           "metadata_valid": metadata_valid,
           "expected_identity": getattr(expected_identity, "to_dict", lambda: expected_identity)(),
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
           embeddings = records.get("embeddings")
           if embeddings is None:
               embeddings = []
           invalid = sum(
               1 for vector in embeddings
               if not self._valid_vector(vector, report.get("collection_dimension") or 0)
           )
           if invalid:
               issues.append(f"{invalid} invalid semantic embeddings.")
       except Exception as exc:
           issues.append(f"Unable to validate stored embeddings: {exc}")
       if not issues:
           valid = True
       else:
           valid = False
       return {
           "valid": valid,
           "metadata_valid": report.get("metadata_valid", False),
           "expected_identity": getattr(expected_identity, "to_dict", lambda: expected_identity)(),
           "stored_identity": report.get("stored_identity"),
           "collection_dimension": report.get("collection_dimension"),
           "expected_dimension": report.get("expected_dimension"),
           "metadata_issues": metadata_issues,
           "issues": issues,
           "vector_count": collection_count,
       }

   def _as_query_result(self, ids: list[str], documents: list[str], metadatas: list[dict], distances: list[float] | None = None) -> Dict[str, Any]:
       if distances is None:
           distances = [0.0] * len(ids)
       return {
           "ids": [ids],
           "documents": [documents],
           "metadatas": [metadatas],
           "distances": [distances],
       }

   def search(self, embedding: Sequence[float], n_results: int = 5, where: Dict[str, Any] | None = None) -> Dict[str, Any]:
       expected = self.expected_identity
       report = self.compatibility_report(expected)
       if not report["valid"]:
           if "dimension mismatch" in report["message"]:
               raise IndexCompatibilityError(report["message"])
           raise IndexCompatibilityError(report["message"])
       if not embedding:
           return self._as_query_result([], [], [])
       collection_dim = self._collection_dim()
       if collection_dim and len(embedding) != collection_dim:
           raise IndexCompatibilityError(f"dimension mismatch: expected {collection_dim}, got {len(embedding)}")
       results = self.collection.query(
           query_embeddings=[list(map(float, embedding))],
           n_results=max(1, int(n_results)),
           where=where,
           include=["documents", "metadatas", "distances"],
       )
       ids = results.get("ids", [[]])[0] if "ids" in results and results["ids"] else []
       documents = results.get("documents", [[]])[0] if "documents" in results and results["documents"] else []
       metadatas = results.get("metadatas", [[]])[0] if "metadatas" in results and results["metadatas"] else []
       distances = results.get("distances", [[]])[0] if "distances" in results and results["distances"] else []
       return self._as_query_result([str(item) for item in ids], [str(item) for item in documents], [dict(item or {}) for item in metadatas], [float(item) for item in distances])

   def search_lexical(self, query: str, n_results: int = 5, where: Dict[str, Any] | None = None) -> Dict[str, Any]:
       query = (query or "").strip()
       if not query:
           return self._as_query_result([], [], [])
       tokens = set(token for token in self._lexical_tokens(query) if token)
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
           token: sum(token in tokens for tokens in corpus) for token in tokens
       }
       rank: list[tuple[float, dict[str, Any]]] = []
       for row, row_tokens in zip(records, corpus, strict=True):
           meta = self._coerce_metadata(json.loads(row[2]))
           if str(meta.get("index_state", "READY")).upper() != "READY":
               continue
           if where and any(meta.get(key) != value for key, value in where.items()):
               continue
           term_counts = {token: row_tokens.count(token) for token in tokens}
           length = max(1, len(row_tokens))
           average_length = max(1.0, sum(len(item) for item in corpus) / max(1, document_count))
           score = 0.0
           for token, frequency in term_counts.items():
               if not frequency:
                   continue
               idf = math.log(1.0 + (document_count - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5))
               score += idf * (frequency * 2.2) / (frequency + 1.2 * (0.75 + 0.25 * length / average_length))
           if score <= 0.0:
               continue
           rank.append((score, {"id": str(row[0]), "document": str(row[1]), "metadata": meta}))
       if not rank:
           return self._as_query_result([], [], [])
       ranked = sorted(rank, key=lambda item: item[0], reverse=True)[: max(1, int(n_results))]
       ids = [entry["id"] for _, entry in ranked]
       documents = [entry["document"] for _, entry in ranked]
       metadatas = [entry["metadata"] for _, entry in ranked]
       distances = [1.0 / (1.0 + score) for score, _ in ranked]
       return self._as_query_result(ids, documents, metadatas, distances)

   def set_version_state(self, document_id: str, version_id: str, state: str) -> None:
       self.set_version_index_state(document_id, version_id, state)

   def activate_staging_index(self, document_id: str | None = None, *, expected_identity: Any | None = None) -> Dict[str, Any]:
       if expected_identity is not None:
           self.expected_identity = expected_identity
       staging_path = self.persist_directory.parent / (
           f"{self.persist_directory.name}.staging-{time.strftime('%Y%m%d%H%M%S')}"
       )
       if not staging_path.exists():
           raise RuntimeError(f"Staging directory {staging_path} does not exist.")
       health = self.index_health_check(self.expected_identity)
       if not health["valid"] or not health["metadata_valid"]:
           raise RuntimeError(json.dumps(health, default=str))
       active_path = self.persist_directory
       backup_path = active_path.parent / f"{active_path.name}.backup-{int(time.time())}"
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
