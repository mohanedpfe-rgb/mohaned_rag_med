from __future__ import annotations

import json
import sqlite3
from typing import Any


def _safe_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        try:
            converted = value.tolist()
            if isinstance(converted, list):
                return converted
            if isinstance(converted, tuple):
                return list(converted)
            return [converted]
        except Exception:
            pass
    try:
        return list(value)
    except (TypeError, ValueError):
        return []


def install() -> None:
    """Keep lexical storage synchronized after semantic index reconciliation."""
    from rag_project.storage.vector_store import VectorStore

    original = VectorStore.reconcile_index
    if getattr(original, "_runtime_reconcile_parity_fix", False):
        return

    def reconcile(self: Any, document_id: str | None = None):
        result = original(self, document_id)

        query = {"document_id": document_id} if document_id else None
        if query is None:
            records = self.collection.get(include=["documents", "metadatas"])
        else:
            records = self.collection.get(
                where=query,
                include=["documents", "metadatas"],
            )

        ids = _safe_list(records.get("ids"))
        documents = _safe_list(records.get("documents"))
        metadatas = _safe_list(records.get("metadatas"))

        rows: dict[str, tuple[str, str, dict[str, Any]]] = {}
        for item_id, document, metadata in zip(ids, documents, metadatas, strict=False):
            meta = self._coerce_metadata(metadata)
            key = f"{meta.get('document_id', '')}::{meta.get('chunk_id', str(item_id))}"
            rows.setdefault(
                key,
                (str(item_id), str(document), meta),
            )

        target_documents = {str(document_id)} if document_id else {
            str(meta.get("document_id"))
            for _, _, meta in rows.values()
            if meta.get("document_id") is not None
        }

        with sqlite3.connect(self.lexical_database) as connection:
            if document_id is not None:
                connection.execute(
                    "DELETE FROM lexical_documents WHERE json_extract(metadata, '$.document_id') = ?",
                    (str(document_id),),
                )
            else:
                connection.execute("DELETE FROM lexical_documents")

            lexical_rows = []
            for _, document, metadata in rows.values():
                if str(metadata.get("document_id")) not in target_documents:
                    continue
                normalized = self._coerce_metadata(metadata)
                lexical_rows.append(
                    (
                        str(normalized.get("chunk_id") or metadata.get("id") or ""),
                        document,
                        json.dumps(normalized, ensure_ascii=False, sort_keys=True),
                        str(normalized.get("index_state", "READY")).upper(),
                        json.dumps(self._lexical_tokens(document), ensure_ascii=False),
                    )
                )

            if lexical_rows:
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
                    lexical_rows,
                )
            connection.commit()

        return {
            **dict(result or {}),
            "count": len(rows),
            "valid": True,
        }

    reconcile._runtime_reconcile_parity_fix = True
    reconcile.__name__ = "reconcile_index"
    reconcile.__qualname__ = "VectorStore.reconcile_index"
    reconcile.__module__ = VectorStore.__module__
    VectorStore.reconcile_index = reconcile


__all__ = ["install"]
