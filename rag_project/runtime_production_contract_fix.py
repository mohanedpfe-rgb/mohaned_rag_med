from __future__ import annotations

import json
import sqlite3
from typing import Any


def _install_page_checkpoint_fix() -> None:
    from rag_project.ingestion import state_store as module
    cls = module.IngestionStateStore
    if getattr(cls.upsert_page, "_production_fix", False):
        return
    allowed = module._ALLOWED_PAGE_UPDATE_KEYS
    utc_now = module.utc_now

    def upsert_page(self: Any, document_id: str, page_number: int, **values: Any) -> None:
        if not document_id:
            raise ValueError("document_id must be non-empty")
        try:
            page_number = int(page_number)
        except (TypeError, ValueError) as exc:
            raise ValueError("page_number must be an integer") from exc
        if page_number < 1:
            raise ValueError("page_number must be >= 1")
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unsupported page update field(s): {', '.join(sorted(unknown))}")
        if self.get_document(document_id) is None:
            raise ValueError(f"Document {document_id!r} does not exist.")
        page_values = dict(values)
        page_values.setdefault("extraction_status", "PENDING")
        page_values.setdefault("ocr_status", "PENDING")
        page_values.setdefault("updated_at", utc_now())
        selected = {"document_id": document_id, "page_number": page_number}
        selected.update({key: page_values[key] for key in allowed if key in page_values})
        placeholders = ", ".join("?" for _ in selected)
        assignments = ", ".join(f"{key}=excluded.{key}" for key in selected if key not in {"document_id", "page_number"})
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO pages ({', '.join(selected)}) VALUES ({placeholders}) "
                f"ON CONFLICT(document_id, page_number) DO UPDATE SET {assignments}",
                tuple(selected.values()),
            )

    upsert_page._production_fix = True
    cls.upsert_page = upsert_page


def _install_numeric_contract_fix() -> None:
    from rag_project.intelligence import evidence_guard
    current = getattr(evidence_guard, "numeric_consistency_details", None)
    if not callable(current) or getattr(current, "_production_fix", False):
        return

    def numeric_consistency(claim: Any, evidence: Any) -> dict[str, Any]:
        details = current(claim, evidence)
        return dict(details) if isinstance(details, dict) else {"mismatch": not bool(details), "consistent": bool(details), "claim": str(claim or "")}

    numeric_consistency._production_fix = True
    evidence_guard.numeric_consistency = numeric_consistency


def _install_lexical_contract_fix() -> None:
    from rag_project.storage.vector_store import VectorStore
    current = VectorStore.search_lexical
    if getattr(current, "_production_fix", False):
        return

    def search_lexical(self: Any, query: str, n_results: int = 5, where: dict[str, Any] | None = None):
        result = current(self, query, n_results=n_results, where=where)
        existing_ids = list((result.get("ids") or [[]])[0] or []) if isinstance(result, dict) else []
        blocked = set(getattr(self, "_nonready_lexical_ids", set()))
        with sqlite3.connect(self.lexical_database) as connection:
            blocked.update(str(row[0]) for row in connection.execute("SELECT json_extract(metadata, '$.chunk_id') FROM lexical_documents WHERE upper(json_extract(metadata, '$.index_state')) <> 'READY'").fetchall() if row[0])
        if blocked and existing_ids:
            keep = [i for i, item_id in enumerate(existing_ids) if str(item_id) not in blocked]
            for key in ("ids", "documents", "metadatas", "distances"):
                values = list((result.get(key) or [[]])[0] or [])
                result[key] = [[values[i] for i in keep]]
            existing_ids = [existing_ids[i] for i in keep]
        if existing_ids:
            with sqlite3.connect(self.lexical_database) as connection:
                states = {str(row[0]): str(row[1]).upper() for row in connection.execute("SELECT id, index_state FROM lexical_documents")}
            keep = [i for i, item_id in enumerate(existing_ids) if states.get(str(item_id), "READY") == "READY"]
            if len(keep) != len(existing_ids):
                for key in ("ids", "documents", "metadatas", "distances"):
                    values = list((result.get(key) or [[]])[0] or [])
                    result[key] = [[values[i] for i in keep]]
                existing_ids = [existing_ids[i] for i in keep]
        if existing_ids:
            return result
        tokens = {token for token in self._lexical_tokens(query) if token}
        if not tokens:
            return result
        try:
            with sqlite3.connect(self.lexical_database) as connection:
                rows = connection.execute("SELECT id, document, metadata, tokens FROM lexical_documents WHERE index_state = 'READY'").fetchall()
            ranked = []
            for item_id, document, raw_metadata, raw_tokens in rows:
                metadata = self._coerce_metadata(json.loads(raw_metadata or "{}"))
                if not self._metadata_matches(metadata, where):
                    continue
                row_tokens = json.loads(raw_tokens or "[]")
                overlap = sum(row_tokens.count(token) for token in tokens)
                if overlap > 0:
                    ranked.append((float(overlap), str(item_id), str(document), metadata))
            ranked.sort(key=lambda item: (-item[0], item[1]))
            ranked = ranked[: max(1, int(n_results))]
            return self._as_query_result([item[1] for item in ranked], [item[2] for item in ranked], [item[3] for item in ranked], [1.0 / (1.0 + item[0]) for item in ranked])
        except Exception:
            return result

    search_lexical._production_fix = True
    VectorStore.search_lexical = search_lexical


def install() -> None:
    _install_page_checkpoint_fix()
    _install_numeric_contract_fix()
    _install_lexical_contract_fix()


install()
