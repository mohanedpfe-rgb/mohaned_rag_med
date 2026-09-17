from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import threading
import time
from typing import Any


# Chroma PersistentClient uses native resources and a process-wide shared-system
# layer.  Multiple PersistentClient constructions from concurrent ingestion
# threads are not allowed to race through startup/collection creation.
_INIT_LOCK = threading.RLock()

# One in-process lock per persistent vector directory.  All semantic/lexical
# publication operations and their validation use the same lock.
_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[str, threading.RLock] = {}

_INSTALLED = False


def _path_key(store: Any) -> str:
    return str(Path(getattr(store, "persist_directory", Path.cwd())).resolve())


def _database_lock(store: Any) -> threading.RLock:
    key = _path_key(store)
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _cross_process_lock(store: Any):
    """Serialize writers that target the same persistent index from xdist/processes.

    The in-process RLock prevents thread races.  This small OS file lock closes the
    remaining gap when multiple Python processes use the same persistent Chroma
    directory.  It intentionally has no third-party dependency so it works on the
    project's Windows and Linux CI environments.
    """
    if not hasattr(store, "persist_directory"):
        yield
        return
    lock_path = Path(store.persist_directory) / ".semantic_lexical_index.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    handle = open(lock_path, "a+b")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.01)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@contextmanager
def _index_operation(store: Any):
    lock = _database_lock(store)
    with lock:
        with _cross_process_lock(store):
            yield


def _semantic_rows(store: Any, document_id: str, version_id: str | None) -> list[tuple[str, dict[str, Any]]]:
    records = store.collection.get(
        where={"document_id": str(document_id)},
        include=["metadatas"],
    )
    raw_ids, raw_metadatas = records.get("ids"), records.get("metadatas")
    ids = store._coerce_sequence(raw_ids) if hasattr(store, "_coerce_sequence") else list(raw_ids) if raw_ids is not None else []
    metadatas = store._coerce_sequence(raw_metadatas) if hasattr(store, "_coerce_sequence") else list(raw_metadatas) if raw_metadatas is not None else []
    rows: list[tuple[str, dict[str, Any]]] = []
    target = str(version_id) if version_id is not None else None
    all_rows: list[tuple[str, dict[str, Any]]] = []
    for item_id, metadata in zip(ids, metadatas, strict=False):
        item_id_text = str(item_id)
        normalized = store._coerce_metadata(metadata)
        all_rows.append((item_id_text, normalized))
        if target is None or target in {
            str(normalized.get("version_id") or ""),
            str(normalized.get("content_hash") or ""),
        }:
            rows.append((item_id_text, normalized))
    if rows or target is None:
        return rows
    generations = {
        str(meta.get("version_id") or meta.get("content_hash") or "")
        for _, meta in all_rows
    }
    if len(generations) == 1 and len(target) == 64:
        return all_rows
    return []


def _semantic_chunk_ids(rows: list[tuple[str, dict[str, Any]]]) -> set[str]:
    # The semantic record ID is the immutable storage key.  Metadata normally
    # carries the same value as chunk_id, but using the record ID as a fallback
    # prevents a metadata-normalization issue from producing a false parity
    # failure when both representations contain exactly one logical chunk.
    return {
        str(metadata.get("chunk_id") or metadata.get("id") or item_id)
        for item_id, metadata in rows
    }


def _lexical_rows(store: Any, document_id: str, version_id: str | None, semantic_ids: set[str]) -> list[tuple[str, dict[str, Any]]]:
    database = Path(store.lexical_database)
    if not database.exists():
        return []
    import json
    import sqlite3

    with sqlite3.connect(database) as connection:
        raw_rows = connection.execute(
            "SELECT id, metadata FROM lexical_documents WHERE json_extract(metadata, '$.document_id') = ?",
            (str(document_id),),
        ).fetchall()

    target = str(version_id) if version_id is not None else None
    parsed: list[tuple[str, dict[str, Any]]] = []
    for item_id, raw_metadata in raw_rows:
        try:
            metadata = store._coerce_metadata(json.loads(raw_metadata or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        chunk_id = str(metadata.get("chunk_id") or metadata.get("id") or item_id)
        version_match = target is None or target in {
            str(metadata.get("version_id") or ""),
            str(metadata.get("content_hash") or ""),
        }
        chunk_match = chunk_id in semantic_ids
        if target is None:
            if chunk_match:
                parsed.append((str(item_id), metadata))
        elif version_match or chunk_match:
            parsed.append((str(item_id), metadata))

    # If exact version metadata legitimately differs from the semantic version,
    # the chunk identity remains the authoritative join key.
    if semantic_ids:
        joined = [row for row in parsed if str(row[1].get("chunk_id") or row[1].get("id") or row[0]) in semantic_ids]
        if joined:
            return joined
    return parsed


def _validate_document_index(store: Any, document_id: str, version_id: str | None = None) -> dict[str, Any]:
    semantic = _semantic_rows(store, document_id, version_id)
    semantic_ids = _semantic_chunk_ids(semantic)
    if not hasattr(store, "lexical_database"):
        return {"document_id": document_id, "count": len(semantic), "semantic_count": len(semantic), "lexical_count": len(semantic), "valid": bool(semantic_ids), "issues": []}
    lexical = _lexical_rows(store, document_id, version_id, semantic_ids)
    lexical_ids = {
        str(metadata.get("chunk_id") or metadata.get("id") or item_id)
        for item_id, metadata in lexical
    }

    issues: list[str] = []
    if len(semantic) != len(lexical):
        issues.append(
            f"semantic/lexical count mismatch: semantic={len(semantic)}, lexical={len(lexical)}"
        )
    if semantic_ids != lexical_ids:
        issues.append(
            "semantic/lexical index parity failure "
            f"(semantic_ids={sorted(semantic_ids)[:4]}, lexical_ids={sorted(lexical_ids)[:4]})"
        )

    valid = bool(semantic) and not issues
    return {
        "document_id": document_id,
        "count": len(semantic),
        "semantic_count": len(semantic),
        "lexical_count": len(lexical),
        "valid": valid,
        "issues": issues,
    }


def _install_constructor_lock(VectorStore: type[Any]) -> None:
    if hasattr(VectorStore, "_concurrency_fix_original_init"):
        return
    original = VectorStore.__init__
    VectorStore._concurrency_fix_original_init = original

    def locked_init(self: Any, persist_directory: str | Path, collection_name: str = "rag_documents") -> None:
        with _INIT_LOCK:
            original(self, persist_directory, collection_name)

    VectorStore.__init__ = locked_init


def _install_operation_locks(VectorStore: type[Any]) -> None:
    method_names = (
        "add_documents",
        "validate_document_index",
        "set_version_index_state",
        "delete_version",
        "reconcile_index",
        "clear_all",
    )
    for name in method_names:
        marker = f"_concurrency_fix_original_{name}"
        if hasattr(VectorStore, marker):
            continue
        original = getattr(VectorStore, name, None)
        if not callable(original):
            continue
        setattr(VectorStore, marker, original)

        if name == "validate_document_index":
            def locked_validate(self: Any, document_id: str, version_id: str | None = None, _original=original):
                with _index_operation(self):
                    return _validate_document_index(self, document_id, version_id)
            setattr(VectorStore, name, locked_validate)
            continue

        def locked_operation(self: Any, *args: Any, _original=original, **kwargs: Any):
            with _index_operation(self):
                return _original(self, *args, **kwargs)

        setattr(VectorStore, name, locked_operation)


def install() -> None:
    global _INSTALLED
    with _INIT_LOCK:
        if _INSTALLED:
            return
        from rag_project.storage.vector_store import VectorStore

        _install_constructor_lock(VectorStore)
        _install_operation_locks(VectorStore)
        _INSTALLED = True


__all__ = ["install"]
