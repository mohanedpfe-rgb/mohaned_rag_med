from __future__ import annotations

from typing import Any, Dict, List


class CitationManager:
    @staticmethod
    def build(hits: List[Any]) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []
        for hit in hits:
            metadata = hit.metadata or {}
            citations.append(
                {
                    "document_id": metadata.get("document_id", hit.doc_id),
                    "version_id": metadata.get("version_id"),
                    "chunk_id": metadata.get("chunk_id"),
                    "file_name": metadata.get("file_name", "unknown.pdf"),
                    "page_numbers": metadata.get("page_numbers", metadata.get("source_pages", [])),
                    "preview": hit.text[:180],
                }
            )
        return citations

    @staticmethod
    def validate(citations: List[Dict[str, Any]], hits: List[Any]) -> List[Dict[str, Any]]:
        validated = []
        for citation in citations:
            expected_document = citation.get("document_id")
            expected_version = citation.get("version_id")
            expected_chunk = citation.get("chunk_id")
            expected_pages = tuple(citation.get("page_numbers", []) or [])
            for hit in hits:
                metadata = hit.metadata or {}
                document_match = expected_document is None or str(metadata.get("document_id", hit.doc_id)) == str(expected_document)
                version_match = expected_version is None or metadata.get("version_id") == expected_version
                chunk_match = expected_chunk is None or str(metadata.get("chunk_id", "")) == str(expected_chunk)
                hit_pages = tuple(metadata.get("page_numbers", metadata.get("source_pages", [])) or [])
                page_match = not expected_pages or bool(set(expected_pages) & set(hit_pages))
                if document_match and version_match and chunk_match and page_match:
                    item = dict(citation)
                    item["valid"] = True
                    validated.append(item)
                    break
        return validated
