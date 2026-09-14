from __future__ import annotations

import contextlib
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any


_LOCK = threading.RLock()
_INSTALLED = False
_TARGET_DISTANCE = "cosine"
_ERROR_MARKER = "Changing the distance function of a collection once it is created is not supported"
_BATCH_SIZE = 500


def _is_distance_mismatch(exc: BaseException) -> bool:
    text = str(exc)
    return _ERROR_MARKER in text or (
        "distance function" in text.lower() and "collection" in text.lower()
    )


def _safe_collection_metadata(metadata: Any) -> dict[str, Any]:
    """Copy only Chroma-safe scalar metadata, excluding the immutable distance key."""
    safe: dict[str, Any] = {}
    for key, value in dict(metadata or {}).items():
        if key == "hnsw:space":
            continue
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            safe[str(key)] = value
    safe["hnsw:space"] = _TARGET_DISTANCE
    return safe


def _metric(collection: Any) -> str:
    metadata = getattr(collection, "metadata", {}) or {}
    return str(metadata.get("hnsw:space", "l2")).strip().lower()


@contextlib.contextmanager
def _migration_lock(persist_directory: Path):
    """Serialize metric migrations across pytest workers/processes on the same index."""
    lock_path = persist_directory / ".chroma_distance_migration.sqlite3"
    connection = sqlite3.connect(lock_path, timeout=120)
    try:
        connection.execute("PRAGMA busy_timeout = 120000")
        connection.execute("CREATE TABLE IF NOT EXISTS migration_lock (id INTEGER PRIMARY KEY, marker TEXT)")
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")
        yield
        connection.commit()
    finally:
        connection.close()


def _clear_chroma_process_cache() -> None:
    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        clear = getattr(SharedSystemClient, "clear_system_cache", None)
        if callable(clear):
            clear()
    except Exception:
        pass


def _open_collection_for_target_metric(client: Any, collection_name: str) -> Any:
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": _TARGET_DISTANCE},
    )


def _migrate_collection(client: Any, collection: Any, collection_name: str) -> Any:
    current_metric = _metric(collection)
    if current_metric == _TARGET_DISTANCE:
        return collection

    persist_directory = Path(getattr(client, "_system", None).settings.persist_directory) if getattr(getattr(client, "_system", None), "settings", None) is not None else None
    if persist_directory is None:
        raise RuntimeError(
            f"Chroma collection '{collection_name}' uses '{current_metric}', but the target is '{_TARGET_DISTANCE}'. "
            "Automatic migration requires a persistent Chroma directory."
        )

    with _migration_lock(persist_directory):
        # Another worker may have repaired the collection while we waited.
        try:
            current = client.get_collection(name=collection_name)
            if _metric(current) == _TARGET_DISTANCE:
                return current
            collection = current
        except Exception:
            collection = client.get_collection(name=collection_name)

        total = int(collection.count())
        metadata = _safe_collection_metadata(getattr(collection, "metadata", {}))

        if total == 0:
            client.delete_collection(name=collection_name)
            _clear_chroma_process_cache()
            return _open_collection_for_target_metric(client, collection_name)

        temporary_name = f"{collection_name}__distance_migration_{uuid.uuid4().hex[:12]}"
        temporary = None
        try:
            temporary = client.get_or_create_collection(
                name=temporary_name,
                metadata=metadata,
            )

            offset = 0
            while offset < total:
                records = collection.get(
                    limit=_BATCH_SIZE,
                    offset=offset,
                    include=["documents", "metadatas", "embeddings"],
                )
                ids = list(records.get("ids") or [])
                documents = list(records.get("documents") or [])
                metadatas = list(records.get("metadatas") or [])
                embeddings = list(records.get("embeddings") or [])
                if not ids:
                    break
                if not (
                    len(ids) == len(documents) == len(metadatas) == len(embeddings)
                ):
                    raise RuntimeError(
                        f"Chroma distance migration for '{collection_name}' found an inconsistent batch "
                        f"at offset {offset}: ids={len(ids)}, documents={len(documents)}, "
                        f"metadatas={len(metadatas)}, embeddings={len(embeddings)}."
                    )
                temporary.add(
                    ids=[str(value) for value in ids],
                    documents=[str(value) for value in documents],
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
                offset += len(ids)

            migrated_count = int(temporary.count())
            if migrated_count != total:
                raise RuntimeError(
                    f"Chroma distance migration verification failed for '{collection_name}': "
                    f"expected {total} records, migrated {migrated_count}."
                )

            # Collection.modify(name=...) is supported by Chroma and avoids a
            # second full vector copy after the verified migration.
            client.delete_collection(name=collection_name)
            _clear_chroma_process_cache()
            temporary = client.get_collection(name=temporary_name)
            temporary.modify(name=collection_name, metadata=metadata)
            _clear_chroma_process_cache()
            return client.get_collection(name=collection_name)
        except Exception:
            # Never destroy the temporary verified copy. It is the recovery source
            # if the final rename is the operation that fails.
            raise


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return

        from rag_project.storage.vector_store import VectorStore

        original_init = VectorStore.__init__
        if getattr(original_init, "_chroma_distance_fix", False):
            _INSTALLED = True
            return

        def hardened_init(
            self: Any,
            persist_directory: str | Path,
            collection_name: str = "rag_documents",
        ) -> None:
            try:
                original_init(self, persist_directory, collection_name)
                return
            except Exception as exc:
                if not _is_distance_mismatch(exc):
                    raise

            # Reconstruct the initialization state after the original constructor
            # failed exactly at get_or_create_collection().
            self.persist_directory = Path(persist_directory)
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self.collection_name = collection_name

            import chromadb

            client = chromadb.PersistentClient(path=str(self.persist_directory))
            existing = client.get_collection(name=self.collection_name)
            self.client = client
            self.collection = _migrate_collection(
                client,
                existing,
                self.collection_name,
            )
            self.lexical_database = self.persist_directory / "lexical.sqlite3"
            initialize_lexical_index = getattr(self, "_initialize_lexical_index")
            initialize_lexical_index()
            self.expected_identity = None

        hardened_init._chroma_distance_fix = True
        VectorStore.__init__ = hardened_init
        _INSTALLED = True


__all__ = [
    "install",
    "_is_distance_mismatch",
    "_safe_collection_metadata",
]
