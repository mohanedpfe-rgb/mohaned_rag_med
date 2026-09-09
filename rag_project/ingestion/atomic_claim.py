from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any


def ensure_and_claim(store: Any, values: dict[str, Any], worker_id: str, lease_seconds: int) -> bool:
    """Atomically materialize a document row and claim its ingestion lease.

    The old pattern performed UPSERT and CLAIM as two independent transactions,
    which allowed two processes to overwrite the same row before either one won
    the lease. This helper does both operations in one SQLite transaction.
    """
    required = {
        "document_id", "content_hash", "file_path", "file_name", "file_size",
        "created_at", "modified_at", "ingestion_started_at", "current_stage",
        "current_page", "total_pages", "status", "parser_version", "ocr_config",
        "chunking_config", "embedding_model", "index_state", "version_id",
    }
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"Missing document fields for atomic claim: {', '.join(missing)}")

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=max(1, int(lease_seconds)))
    database_path = store.database_path
    with store._connect() as connection:
        row = connection.execute(
            "SELECT document_id, lease_owner, lease_expires_at FROM documents "
            "WHERE content_hash = ? OR document_id = ? LIMIT 1",
            (values["content_hash"], values["document_id"]),
        ).fetchone()
        if row:
            owner = row["lease_owner"]
            expires = row["lease_expires_at"]
            live = False
            if owner and expires:
                try:
                    stamp = datetime.fromisoformat(str(expires).replace("Z", "+00:00"))
                    if stamp.tzinfo is None:
                        stamp = stamp.replace(tzinfo=timezone.utc)
                    live = stamp.astimezone(timezone.utc) > now
                except ValueError:
                    live = False
            if live and str(owner) != str(worker_id):
                return False
            values = dict(values)
            values["document_id"] = row["document_id"]

        columns = [
            "document_id", "content_hash", "file_path", "file_name", "file_size",
            "created_at", "modified_at", "ingestion_started_at", "current_stage",
            "current_page", "total_pages", "status", "error", "parser_version",
            "ocr_config", "chunking_config", "embedding_model", "embedding_dimension",
            "index_state", "version_id", "lease_owner", "lease_expires_at",
            "heartbeat_at", "ingestion_metrics",
        ]
        row_values = {key: values.get(key) for key in columns if key in values}
        row_values.setdefault("error", None)
        row_values.setdefault("embedding_dimension", None)
        row_values.setdefault("ingestion_metrics", None)
        row_values["lease_owner"] = str(worker_id)
        row_values["lease_expires_at"] = expires_at.isoformat()
        row_values["heartbeat_at"] = now.isoformat()
        row_values["modified_at"] = now.isoformat()

        placeholders = ", ".join("?" for _ in row_values)
        assignments = ", ".join(
            f"{key}=excluded.{key}" for key in row_values if key != "document_id"
        )
        connection.execute(
            f"INSERT INTO documents ({', '.join(row_values)}) VALUES ({placeholders}) "
            f"ON CONFLICT(document_id) DO UPDATE SET {assignments}",
            tuple(row_values.values()),
        )
        return True
