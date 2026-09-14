from __future__ import annotations

import json
import sqlite3
from functools import wraps
from typing import Any


def install() -> None:
    """Enforce the semantic/lexical publication contract at the final boundary."""
    from rag_project.storage.vector_store import VectorStore

    original_validate = VectorStore.validate_document_index
    if not getattr(original_validate, "_post_index_parity_contract", False):

        @wraps(original_validate)
        def validate(self, document_id: str, version_id: str | None = None):
            result = original_validate(self, document_id, version_id)
            issues = list(result.get("issues", []))
            semantic_count = int(result.get("count", 0) or 0)

            with sqlite3.connect(self.lexical_database) as connection:
                rows = connection.execute(
                    "SELECT id, metadata, index_state FROM lexical_documents "
                    "WHERE json_extract(metadata, '$.document_id') = ?",
                    (str(document_id),),
                ).fetchall()

            lexical_matching: list[tuple[str, dict[str, Any], str]] = []
            for row_id, raw_metadata, state in rows:
                try:
                    metadata = json.loads(raw_metadata or "{}")
                except (TypeError, ValueError, json.JSONDecodeError):
                    metadata = {}
                if version_id is None or str(metadata.get("version_id")) == str(version_id):
                    lexical_matching.append((str(row_id), metadata, str(state or "").upper()))

            lexical_count = len(lexical_matching)
            if semantic_count != lexical_count:
                issues.append(
                    f"semantic/lexical count mismatch: semantic={semantic_count}, lexical={lexical_count}"
                )

            semantic_ids = {
                str(metadata.get("chunk_id"))
                for metadata in (result.get("metadatas") or [])
                if isinstance(metadata, dict) and metadata.get("chunk_id")
            }
            lexical_ids = {
                str(metadata.get("chunk_id"))
                for _, metadata, _ in lexical_matching
                if metadata.get("chunk_id")
            }
            if semantic_ids and semantic_ids != lexical_ids:
                missing_lexical = sorted(semantic_ids - lexical_ids)
                extra_lexical = sorted(lexical_ids - semantic_ids)
                issues.append(
                    "semantic/lexical chunk-id mismatch: "
                    f"missing_lexical={missing_lexical[:5]}, extra_lexical={extra_lexical[:5]}"
                )

            result["semantic_count"] = semantic_count
            result["lexical_count"] = lexical_count
            result["issues"] = issues
            result["valid"] = bool(result.get("valid")) and not issues
            return result

        validate._post_index_parity_contract = True
        VectorStore.validate_document_index = validate

    original_set_version = VectorStore.set_version_index_state
    if not getattr(original_set_version, "_post_index_ready_contract", False):

        @wraps(original_set_version)
        def set_version_state(self, document_id: str, version_id: str, state: str) -> None:
            target_state = str(state).upper()
            result = original_set_version(self, document_id, version_id, state)
            if target_state != "READY":
                return result

            validation = self.validate_document_index(document_id, version_id)
            if not validation.get("valid"):
                issues = "; ".join(validation.get("issues", [])) or "unknown publication inconsistency"
                raise RuntimeError(
                    "READY publication contract failed after vector/lexical state update: " + issues
                )

            matches = self.collection.get(
                where={"document_id": document_id}, include=["metadatas"]
            )
            semantic_ready = sum(
                1
                for metadata in (matches.get("metadatas") or [])
                if isinstance(metadata, dict)
                and str(metadata.get("version_id")) == str(version_id)
                and str(metadata.get("index_state", "")).upper() == "READY"
            )

            with sqlite3.connect(self.lexical_database) as connection:
                lexical_ready = connection.execute(
                    "SELECT COUNT(*) FROM lexical_documents "
                    "WHERE json_extract(metadata, '$.document_id') = ? "
                    "AND json_extract(metadata, '$.version_id') = ? "
                    "AND upper(index_state) = 'READY'",
                    (str(document_id), str(version_id)),
                ).fetchone()[0]

            expected = int(validation.get("count", 0) or 0)
            if expected <= 0 or semantic_ready != expected or lexical_ready != expected:
                raise RuntimeError(
                    "READY publication contract produced an incomplete publication: "
                    f"expected={expected}, semantic_ready={semantic_ready}, lexical_ready={lexical_ready}"
                )
            return result

        set_version_state._post_index_ready_contract = True
        VectorStore.set_version_index_state = set_version_state


__all__ = ["install"]
