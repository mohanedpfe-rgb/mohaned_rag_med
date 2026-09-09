from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


# Columns accepted by update_document. Values remain parameterized; the whitelist
# prevents untrusted dictionary keys from becoming SQL identifiers.
_ALLOWED_DOCUMENT_UPDATE_KEYS = {
    "content_hash", "file_path", "file_name", "file_size", "created_at", "modified_at",
    "ingestion_started_at", "current_stage", "current_page", "total_pages", "status", "error",
    "parser_version", "ocr_config", "chunking_config", "embedding_model", "embedding_dimension",
    "index_state", "version_id", "lease_owner", "lease_expires_at", "heartbeat_at",
    "ingestion_metrics",
}


# The remainder of this file intentionally preserves the existing implementation.
