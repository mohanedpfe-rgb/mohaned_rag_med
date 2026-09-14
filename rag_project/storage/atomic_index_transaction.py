from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Sequence


# A SQLite write transaction is used as the cross-process mutex.  Thread locks
# avoid needless SQLite contention inside one Python process, while the
# BEGIN IMMEDIATE below also serializes independent pytest-xdist workers.
_LOCK_GUARD = threading.Lock()
_LOCKS: dict[str, threading.RLock] = {}
_TX_LOCAL = threading.local()
_INSTALLED = False


def _database_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _LOCK_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


def _active_connections() -> dict[int, sqlite3.Connection]:
    connections = getattr(_TX_LOCAL, "connections", None)
    if connections is None:
        connections = {}
        _TX_LOCAL.connections = connections
    return connections


def _upsert_lexical_on_connection(
    store: Any,
    connection: sqlite3.Connection,
    documents: Sequence[str],
    metadatas: Sequence[dict[str, Any]],
    ids: Sequence[str],
) -> None:
    rows = []
    for item_id, document, metadata in zip(ids, documents, metadatas, strict=True):
        logical_id = str(metadata.get("chunk_id") or item_id)
        rows.append(
            (
                logical_id,
                str(document),
                store._metadata_json(metadata),
                str(metadata.get("index_state", "READY")).upper(),
                store._tokens_json(str(document)),
            )
        )
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


def _install_lexical_transaction_bridge(VectorStore: type[Any]) -> None:
    if hasattr(VectorStore, "_atomic_tx_original_upsert_lexical_records"):
        return

    original = VectorStore._upsert_lexical_records
    VectorStore._atomic_tx_original_upsert_lexical_records = original

    def upsert_lexical_records(
        self: Any,
        documents: Sequence[str],
        metadatas: Sequence[dict[str, Any]],
        ids: Sequence[str],
    ) -> None:
        connection = _active_connections().get(id(self))
        if connection is None:
            return original(self, documents, metadatas, ids)

        rows = []
        for item_id, document, metadata in zip(ids, documents, metadatas, strict=True):
            logical_id = str(metadata.get("chunk_id") or item_id)
            rows.append(
                (
                    logical_id,
                    str(document),
                    __import__("json").dumps(metadata, ensure_ascii=False, sort_keys=True),
                    str(metadata.get("index_state", "READY")).upper(),
                    __import__("json").dumps(self._lexical_tokens(str(document)), ensure_ascii=False),
                )
            )
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

    upsert_lexical_records.__name__ = getattr(original, "__name__", "_upsert_lexical_records")
    upsert_lexical_records.__qualname__ = getattr(original, "__qualname__", "VectorStore._upsert_lexical_records")
    upsert_lexical_records.__module__ = getattr(original, "__module__", VectorStore.__module__)
    VectorStore._upsert_lexical_records = upsert_lexical_records


def _install_atomic_add_documents(VectorStore: type[Any]) -> None:
    if hasattr(VectorStore, "_atomic_tx_original_add_documents"):
        return

    original = VectorStore.add_documents
    VectorStore._atomic_tx_original_add_documents = original

    def add_documents(
        self: Any,
        documents: Sequence[str],
        metadatas: Sequence[dict[str, Any]],
        embeddings: Sequence[Sequence[float]],
        ids: Sequence[str],
    ) -> None:
        documents_list = list(documents or [])
        metadata_list = list(metadatas or [])
        embedding_list = list(embeddings or [])
        ids_list = [str(item) for item in ids or []]
        if not documents_list:
            return
        if not (
            len(documents_list)
            == len(metadata_list)
            == len(embedding_list)
            == len(ids_list)
        ):
            # Preserve the base implementation's public contract and error
            # message for malformed callers.
            return original(self, documents_list, metadata_list, embedding_list, ids_list)

        database = Path(self.lexical_database)
        database.parent.mkdir(parents=True, exist_ok=True)
        lock = _database_lock(database)
        connection: sqlite3.Connection | None = None
        active = _active_connections()
        preexisting_ids: set[str] = set()

        with lock:
            try:
                # SQLite's BEGIN IMMEDIATE obtains a process-wide write lock
                # before Chroma is touched.  This is the crucial ordering rule:
                # no worker can expose a semantic-only BUILDING batch while its
                # lexical counterpart is still being committed.
                connection = sqlite3.connect(database, timeout=60.0)
                connection.execute("BEGIN IMMEDIATE")

                try:
                    existing = self.collection.get(ids=ids_list, include=["metadatas"])
                    preexisting_ids = {
                        str(item_id) for item_id in existing.get("ids", []) or []
                    }
                except Exception:
                    # A failed preflight read must not make an otherwise valid
                    # ingestion fail solely because a Chroma diagnostic query
                    # is unsupported by a particular backend version.
                    preexisting_ids = set()

                active[id(self)] = connection
                try:
                    original(
                        self,
                        documents_list,
                        metadata_list,
                        embedding_list,
                        ids_list,
                    )
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise
                finally:
                    active.pop(id(self), None)
            except Exception:
                # The semantic store has no shared transaction with SQLite.
                # Compensate only IDs that were not present before this batch.
                # This leaves previously published data untouched.
                try:
                    current = self.collection.get(ids=ids_list, include=["metadatas"])
                    current_ids = {
                        str(item_id) for item_id in current.get("ids", []) or []
                    }
                    newly_created = sorted(current_ids - preexisting_ids)
                    if newly_created:
                        self.collection.delete(ids=newly_created)
                except Exception:
                    # The original exception is the actionable failure.  A
                    # compensation failure is logged by the caller's existing
                    # ingestion failure path rather than masking the root cause.
                    pass
                raise
            finally:
                active.pop(id(self), None)
                if connection is not None:
                    connection.close()

    add_documents.__name__ = getattr(original, "__name__", "add_documents")
    add_documents.__qualname__ = getattr(original, "__qualname__", "VectorStore.add_documents")
    add_documents.__module__ = getattr(original, "__module__", VectorStore.__module__)
    VectorStore.add_documents = add_documents


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from rag_project.storage.vector_store import VectorStore

    # Reuse the store's existing serializers/tokenizer instead of creating a
    # second representation format.  The transaction bridge calls the exact
    # same SQL shape as the production lexical writer.
    def _metadata_json(store: Any, metadata: dict[str, Any]) -> str:
        import json
        return json.dumps(metadata, ensure_ascii=False, sort_keys=True)

    def _tokens_json(store: Any, document: str) -> str:
        import json
        return json.dumps(store._lexical_tokens(document), ensure_ascii=False)

    if not hasattr(VectorStore, "_metadata_json"):
        VectorStore._metadata_json = _metadata_json
    if not hasattr(VectorStore, "_tokens_json"):
        VectorStore._tokens_json = _tokens_json

    _install_lexical_transaction_bridge(VectorStore)
    _install_atomic_add_documents(VectorStore)
    _INSTALLED = True


__all__ = ["install"]
