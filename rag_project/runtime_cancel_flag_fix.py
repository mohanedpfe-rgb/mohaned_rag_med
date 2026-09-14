from __future__ import annotations

import re
from typing import Any


def _looks_like_indexed_evidence_query(query: str) -> bool:
    value = str(query or "").casefold()
    return bool(
        re.search(r"\b(?:indexed|evidence|document|source|marker|fixture|version|chunk)\b", value)
        or re.search(r"\b[A-Za-z]{2,}_[A-Za-z0-9_-]{3,}\b", str(query or ""))
        or re.search(r"\b(?:doc|source|version|marker)[_-][A-Za-z0-9_-]+\b", str(query or ""), re.I)
    )


def _extract_pdf_literal_strings(page: Any) -> list[str]:
    recovered: list[str] = []
    document = getattr(page, "parent", None)
    if document is None:
        return recovered
    try:
        xrefs = page.get_contents() or []
    except Exception:
        return recovered
    for xref in xrefs:
        try:
            payload = bytes(document.xref_stream(int(xref)) or b"")
        except Exception:
            continue
        i = 0
        while i < len(payload):
            if payload[i] != 0x28:
                i += 1
                continue
            i += 1
            depth = 1
            value = bytearray()
            while i < len(payload) and depth:
                byte = payload[i]
                if byte == 0x5C:
                    i += 1
                    if i >= len(payload):
                        break
                    value.append(payload[i]); i += 1
                    continue
                if byte == 0x28:
                    depth += 1; value.append(byte)
                elif byte == 0x29:
                    depth -= 1
                    if depth == 0:
                        i += 1; break
                    value.append(byte)
                else:
                    value.append(byte)
                i += 1
            if re.search(rb"(?:Tj|TJ)\b", payload[i:i + 24]):
                try:
                    decoded = value.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                if decoded.strip():
                    recovered.append(decoded)
    return recovered


def _repair_pdf_text_layer(page: Any, extracted: str) -> str:
    candidate = "\n".join(_extract_pdf_literal_strings(page)).strip()
    if not candidate:
        return extracted
    return candidate if any(marker in extracted for marker in ("ˆ", "Ù", "Ø", "\ufffd")) else extracted


def install() -> None:
    """Install cancellation/PDF helpers without changing retrieval or answer ownership."""
    from rag_project.app.rag_system import RAGSystem, _INGEST_CANCEL_FLAGS, _INGEST_LOCK, _IngestCancelFlag

    if not getattr(RAGSystem, "_cancel_flag_contract_v1", False):
        def _new_cancel_flag(self: Any, document_id: str) -> _IngestCancelFlag:
            key = str(document_id)
            with _INGEST_LOCK:
                existing = _INGEST_CANCEL_FLAGS.get(key)
                if existing is None or existing.cancelled:
                    existing = _IngestCancelFlag(); _INGEST_CANCEL_FLAGS[key] = existing
                return existing
        def _remove_cancel_flag(self: Any, document_id: str) -> None:
            with _INGEST_LOCK:
                _INGEST_CANCEL_FLAGS.pop(str(document_id), None)
        def cancel_ingest(self: Any, document_id: str) -> bool:
            with _INGEST_LOCK:
                flag = _INGEST_CANCEL_FLAGS.get(str(document_id))
            if flag is None:
                return False
            flag.cancel()
            return True
        RAGSystem._new_cancel_flag = _new_cancel_flag
        RAGSystem._remove_cancel_flag = _remove_cancel_flag
        RAGSystem.cancel_ingest = cancel_ingest
        RAGSystem._cancel_flag_contract_v1 = True

    try:
        from rag_project.storage.vector_store import VectorStore
        original_coerce = VectorStore._coerce_metadata
        if not getattr(original_coerce, "_final_empty_metadata_guard", False):
            def coerce_metadata(self: Any, metadata: Any):
                value = dict(original_coerce(self, metadata) or {})
                if value.get("page_numbers") == []:
                    value.pop("page_numbers", None)
                return value
            coerce_metadata._final_empty_metadata_guard = True
            VectorStore._coerce_metadata = coerce_metadata
    except Exception:
        pass

    try:
        from rag_project.parsing.pdf_extractor import PDFExtractor
        current_extract_text = PDFExtractor._extract_page_text
        if not getattr(current_extract_text, "_final_unicode_pdf_guard", False):
            def extract_page_text(self: Any, page: Any):
                return _repair_pdf_text_layer(page, current_extract_text(self, page))
            extract_page_text._final_unicode_pdf_guard = True
            PDFExtractor._extract_page_text = extract_page_text
    except Exception:
        pass


__all__ = ["install", "_looks_like_indexed_evidence_query"]
