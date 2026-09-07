from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IngestionStateStore:
    """Durable document and page checkpoints backed by SQLite."""

    READY_STATUSES = {"READY", "COMPLETED"}
    ACTIVE_STATUSES = {"RUNNING", "DISCOVERED", "VALIDATING", "EXTRACTING", "OCR", "CHUNKING", "EMBEDDING", "INDEXING", "VALIDATING_INDEX", "INTERRUPTED", "RECOVERING"}
    TERMINAL_STATUSES = {"FAILED", "QUARANTINED", "READY", "COMPLETED"}

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @staticmethod
    def normalize_status(status: str | None) -> str | None:
        if status is None:
            return None
        value = str(status).upper()
        if value == "COMPLETED":
            return "READY"
        return value

    @staticmethod
    def is_ready_status(status: str | None) -> bool:
        return IngestionStateStore.normalize_status(status) in IngestionStateStore.READY_STATUSES

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    modified_at TEXT NOT NULL,
                    ingestion_started_at TEXT,
                    ingestion_completed_at TEXT,
                    current_stage TEXT NOT NULL,
                    current_page INTEGER NOT NULL DEFAULT 0,
                    total_pages INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    error TEXT,
                    parser_version TEXT NOT NULL,
                    ocr_config TEXT NOT NULL,
                    chunking_config TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_dimension INTEGER,
                    index_state TEXT DEFAULT 'READY',
                    version_id TEXT,
                    lease_owner TEXT,
                    lease_expires_at TEXT,
                    heartbeat_at TEXT,
                    ingestion_metrics TEXT,
                    UNIQUE(content_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(file_path);
                CREATE TABLE IF NOT EXISTS pages (
                    document_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    extraction_status TEXT NOT NULL,
                    ocr_status TEXT NOT NULL,
                    extraction_method TEXT,
                    text TEXT,
                    cache_reference TEXT,
                    processing_error TEXT,
                    checksum TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(document_id, page_number),
                    FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS query_traces (
                    query_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(documents)")}
            for column_name, definition in {
                "version_id": "ALTER TABLE documents ADD COLUMN version_id TEXT",
                "index_state": "ALTER TABLE documents ADD COLUMN index_state TEXT DEFAULT 'READY'",
                "lease_owner": "ALTER TABLE documents ADD COLUMN lease_owner TEXT",
                "lease_expires_at": "ALTER TABLE documents ADD COLUMN lease_expires_at TEXT",
                "heartbeat_at": "ALTER TABLE documents ADD COLUMN heartbeat_at TEXT",
                "ingestion_metrics": "ALTER TABLE documents ADD COLUMN ingestion_metrics TEXT",
            }.items():
                if column_name not in columns:
                    connection.execute(definition)

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_all_documents(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
            ).fetchone()
        return dict(row) if row else None

    def get_by_path(self, file_path: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE file_path = ? ORDER BY modified_at DESC LIMIT 1",
                (file_path,),
            ).fetchone()
        return dict(row) if row else None

    def upsert_document(self, values: dict[str, Any]) -> None:
        if values.get("content_hash"):
            existing_hash = self.get_by_hash(values["content_hash"])
            if existing_hash and values.get("document_id") is None:
                values["document_id"] = existing_hash["document_id"]
            elif existing_hash and values.get("document_id") != existing_hash["document_id"]:
                values["document_id"] = existing_hash["document_id"]
        values.setdefault("document_id", str(uuid.uuid4()))
        columns = [
            "document_id", "content_hash", "file_path", "file_name", "file_size",
            "created_at", "modified_at", "ingestion_started_at", "current_stage",
            "current_page", "total_pages", "status", "error", "parser_version",
            "ocr_config", "chunking_config", "embedding_model", "embedding_dimension",
            "index_state", "version_id",
            "lease_owner", "lease_expires_at", "heartbeat_at",
            "ingestion_metrics",
        ]
        values = {key: values[key] for key in columns if key in values}
        values.setdefault("created_at", utc_now())
        values.setdefault("modified_at", values["created_at"])
        values.setdefault("ingestion_started_at", utc_now())
        values.setdefault("current_stage", "DISCOVERED")
        values.setdefault("current_page", 0)
        values.setdefault("total_pages", 0)
        values.setdefault("status", "RUNNING")
        values["status"] = self.normalize_status(values["status"])
        values.setdefault("error", None)
        values.setdefault("parser_version", "pdf-extractor-v2")
        values.setdefault("ocr_config", "{}")
        values.setdefault("chunking_config", "{}")
        values.setdefault("embedding_model", "unknown")
        values.setdefault("index_state", "READY" if self.is_ready_status(values.get("status")) else "PENDING")
        values.setdefault("version_id", values.get("content_hash", values.get("document_id")))
        placeholders = ", ".join("?" for _ in values)
        assignments = ", ".join(
            f"{key}=excluded.{key}" for key in values if key != "document_id"
        )
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO documents ({', '.join(values)}) VALUES ({placeholders}) "
                f"ON CONFLICT(document_id) DO UPDATE SET {assignments}",
                tuple(values.values()),
            )

    def update_document(self, document_id: str, **values: Any) -> None:
        if not values:
            return
        if "status" in values:
            values["status"] = self.normalize_status(values["status"])
        if "index_state" not in values and values.get("status") and self.is_ready_status(values.get("status")):
            values["index_state"] = "READY"
        values["modified_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE documents SET {assignments} WHERE document_id = ?",
                (*values.values(), document_id),
            )

    def claim_document(
        self,
        document_id: str,
        worker_id: str,
        lease_seconds: int = 900,
    ) -> bool:
        """Claim a document unless another live worker currently owns it."""
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=max(1, int(lease_seconds)))
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE documents
                SET lease_owner = ?, lease_expires_at = ?, heartbeat_at = ?,
                    modified_at = ?
                WHERE document_id = ?
                  AND (
                    lease_owner IS NULL
                    OR lease_owner = ?
                    OR lease_expires_at IS NULL
                    OR lease_expires_at <= ?
                  )
                """,
                (
                    worker_id,
                    expires_at.isoformat(),
                    now.isoformat(),
                    now.isoformat(),
                    document_id,
                    worker_id,
                    now.isoformat(),
                ),
            )
        return cursor.rowcount == 1

    def heartbeat_document(
        self,
        document_id: str,
        worker_id: str,
        lease_seconds: int = 900,
    ) -> bool:
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=max(1, int(lease_seconds)))
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE documents
                SET lease_expires_at = ?, heartbeat_at = ?, modified_at = ?
                WHERE document_id = ? AND lease_owner = ?
                """,
                (
                    expires_at.isoformat(),
                    now.isoformat(),
                    now.isoformat(),
                    document_id,
                    worker_id,
                ),
            )
        return cursor.rowcount == 1

    def release_document(self, document_id: str, worker_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE documents
                SET lease_owner = NULL, lease_expires_at = NULL,
                    heartbeat_at = NULL, modified_at = ?
                WHERE document_id = ? AND lease_owner = ?
                """,
                (utc_now(), document_id, worker_id),
            )
        return cursor.rowcount == 1

    def recover_stale_documents(self) -> int:
        """Make expired active jobs retryable without touching terminal records."""
        now = utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE documents
                SET status = 'INTERRUPTED', current_stage = 'INTERRUPTED',
                    index_state = 'FAILED', lease_owner = NULL,
                    lease_expires_at = NULL, heartbeat_at = NULL,
                    modified_at = ?
                WHERE status IN (
                    'RUNNING', 'DISCOVERED', 'VALIDATING', 'EXTRACTING',
                    'OCR', 'CHUNKING', 'EMBEDDING', 'INDEXING',
                    'VALIDATING_INDEX', 'INTERRUPTED', 'RECOVERING'
                )
                AND lease_expires_at IS NOT NULL
                AND lease_expires_at <= ?
                """,
                (now, now),
            )
        return cursor.rowcount

    def transition_document_state(self, document_id: str, new_stage: str, **values: Any) -> None:
        record = self.get_document(document_id)
        if not record:
            raise ValueError(f"Document {document_id!r} does not exist.")
        current_stage = str(record.get("current_stage", "DISCOVERED")).upper()
        new_stage_value = str(new_stage).upper()
        allowed = {
            "DISCOVERED": {"VALIDATING", "FAILED", "QUARANTINED"},
            "VALIDATING": {"EXTRACTING", "FAILED", "QUARANTINED"},
            "EXTRACTING": {"OCR", "CHUNKING", "EMBEDDING", "INDEXING", "FAILED", "QUARANTINED"},
            "OCR": {"CHUNKING", "FAILED", "QUARANTINED"},
            "CHUNKING": {"EMBEDDING", "FAILED", "QUARANTINED"},
            "EMBEDDING": {"INDEXING", "FAILED", "QUARANTINED"},
            "INDEXING": {"VALIDATING_INDEX", "FAILED", "QUARANTINED"},
            "VALIDATING_INDEX": {"READY", "FAILED", "QUARANTINED"},
            "READY": {"READY"},
            "FAILED": {"RECOVERING", "QUARANTINED"},
            "QUARANTINED": {"RECOVERING", "FAILED"},
            "INTERRUPTED": {"RECOVERING", "FAILED", "QUARANTINED"},
            "RECOVERING": {"VALIDATING", "EXTRACTING", "CHUNKING", "EMBEDDING", "INDEXING", "READY", "FAILED", "QUARANTINED"},
        }
        if new_stage_value not in allowed.get(current_stage, set()):
            raise ValueError(f"Invalid state transition: {current_stage} -> {new_stage_value}")
        values.setdefault("current_stage", new_stage_value)
        if new_stage_value in {"READY", "COMPLETED"}:
            values.setdefault("status", "READY")
            values.setdefault("index_state", "READY")
        elif new_stage_value == "FAILED":
            values.setdefault("status", "FAILED")
            values.setdefault("index_state", "FAILED")
        self.update_document(document_id, **values)

    def is_document_ready(self, document_id: str) -> bool:
        record = self.get_document(document_id)
        return bool(record and self.is_ready_status(record.get("status")))

    def upsert_page(self, document_id: str, page_number: int, **values: Any) -> None:
        values.setdefault("extraction_status", "PENDING")
        values.setdefault("ocr_status", "not_required")
        values.setdefault("updated_at", utc_now())
        columns = ["document_id", "page_number", *values]
        placeholders = ", ".join("?" for _ in columns)
        assignments = ", ".join(f"{key}=excluded.{key}" for key in values)
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO pages ({', '.join(columns)}) VALUES ({placeholders}) "
                f"ON CONFLICT(document_id, page_number) DO UPDATE SET {assignments}",
                (document_id, page_number, *values.values()),
            )

    def get_pages(self, document_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM pages WHERE document_id = ? ORDER BY page_number",
                (document_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_pages(self, document_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM pages WHERE document_id = ?", (document_id,))

    def record_query_trace(self, query_id: str, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO query_traces(query_id, created_at, payload) VALUES (?, ?, ?)",
                (query_id, utc_now(), json.dumps(payload, default=str)),
            )
