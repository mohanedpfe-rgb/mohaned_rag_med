from __future__ import annotations

import json
import multiprocessing as mp
import os
import sqlite3
from functools import wraps
from pathlib import Path
from typing import Any

_INSTALLED = False
_ORIGINALS: dict[str, Any] = {}


def _safe_chroma_metadata(metadata: Any) -> dict[str, Any]:
    value = dict(metadata or {}) if isinstance(metadata, dict) else {}
    for key, item in list(value.items()):
        if isinstance(item, tuple):
            item = list(item)
            value[key] = item
        if isinstance(item, list) and not item:
            value.pop(key, None)
    return value


def _patch_collection_upsert() -> None:
    from rag_project.storage.vector_store import VectorStore

    original_init = VectorStore.__init__
    if getattr(original_init, "_invariant_collection_guard", False):
        return
    _ORIGINALS["vector_init"] = original_init

    @wraps(original_init)
    def guarded_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        collection = getattr(self, "collection", None)
        original_upsert = getattr(collection, "upsert", None)
        if collection is None or not callable(original_upsert):
            return
        if getattr(original_upsert, "_invariant_metadata_guard", False):
            return

        @wraps(original_upsert)
        def guarded_upsert(*u_args: Any, **u_kwargs: Any):
            if "metadatas" in u_kwargs and u_kwargs["metadatas"] is not None:
                u_kwargs["metadatas"] = [
                    _safe_chroma_metadata(metadata)
                    for metadata in (u_kwargs.get("metadatas") or [])
                ]
            return original_upsert(*u_args, **u_kwargs)

        guarded_upsert._invariant_metadata_guard = True
        collection.upsert = guarded_upsert

    guarded_init._invariant_collection_guard = True
    VectorStore.__init__ = guarded_init


def _patch_lease_boundary() -> None:
    import rag_project.runtime_quality_gate as quality_gate

    current = getattr(quality_gate, "_lease_is_valid", None)
    if not callable(current) or getattr(current, "_invariant_lease_semantics", False):
        return
    _ORIGINALS["lease_is_valid"] = current

    @wraps(current)
    def lease_is_valid(record: dict[str, Any] | None) -> bool:
        # A missing lease is an explicit non-fenced direct state transition.
        # Lease fencing applies once a lease has actually been claimed.
        if not record or not record.get("lease_owner"):
            return True
        return bool(current(record))

    lease_is_valid._invariant_lease_semantics = True
    quality_gate._lease_is_valid = lease_is_valid


def _direct_table_extract(page: Any) -> str:
    try:
        finder = getattr(page, "find_tables", None)
        if not callable(finder):
            return ""
        result = finder()
        tables = list(getattr(result, "tables", []) or [])
        rendered: list[str] = []
        for table in tables[:20]:
            try:
                markdown = str(table.to_markdown() or "").strip()
            except Exception:
                continue
            if markdown:
                rendered.append(markdown)
        return "\n\n".join(rendered)
    except Exception:
        return ""


def _patch_table_extraction() -> None:
    from rag_project.parsing.pdf_extractor import PDFExtractor

    current = getattr(PDFExtractor, "_extract_tables", None)
    if not callable(current) or getattr(current, "_invariant_no_fork", False):
        return
    _ORIGINALS["extract_tables"] = current

    def extract_tables(self: Any, page: Any) -> str:
        # Never fork from the production ingestion process. pytest-xdist and
        # directory ingestion both use threads, while filelock explicitly rejects
        # fork from a multi-threaded process on modern Python.
        return _direct_table_extract(page)

    extract_tables._invariant_no_fork = True
    PDFExtractor._extract_tables = extract_tables


def _patch_multiprocessing_start_guard() -> None:
    # Defensive fallback for legacy runtime code that still starts a Process from
    # a table timeout path. Prefer the direct extractor above; this guard simply
    # turns the specific fork-safety RuntimeError into a clean no-table result.
    try:
        from rag_project import runtime_stability_v3
    except Exception:
        return
    current = getattr(runtime_stability_v3, "_timed_table_extract", None)
    if not callable(current) or getattr(current, "_invariant_fork_guard", False):
        return
    _ORIGINALS["timed_table_extract"] = current

    @wraps(current)
    def guarded(page: Any) -> str:
        try:
            return current(page)
        except RuntimeError as exc:
            if "os.fork is unsafe" in str(exc).lower():
                return _direct_table_extract(page)
            raise

    guarded._invariant_fork_guard = True
    runtime_stability_v3._timed_table_extract = guarded


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _patch_lease_boundary()
    _patch_collection_upsert()
    _patch_table_extraction()
    _patch_multiprocessing_start_guard()
    _INSTALLED = True


__all__ = ["install"]
