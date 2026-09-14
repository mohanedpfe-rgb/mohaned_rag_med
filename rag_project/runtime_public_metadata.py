from __future__ import annotations

import json

_INSTALLED = False

_LIST_KEYS = {"page_numbers", "source_pages", "evidence_types", "entities", "headings", "number_forms"}


def _decode_value(key, value):
    if key not in _LIST_KEYS or not isinstance(value, str):
        return value
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value
    return decoded if isinstance(decoded, (list, dict)) else value


def _public_metadata(metadata):
    if not isinstance(metadata, dict):
        return {}
    return {key: _decode_value(key, value) for key, value in metadata.items()}


def _public_result(result):
    if not isinstance(result, dict):
        return result
    out = dict(result)
    raw = out.get("metadatas")
    if isinstance(raw, list):
        if raw and isinstance(raw[0], list):
            out["metadatas"] = [[_public_metadata(item) for item in group if isinstance(item, dict)] for group in raw]
        else:
            out["metadatas"] = [_public_metadata(item) for item in raw if isinstance(item, dict)]
    return out


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from rag_project.storage.vector_store import VectorStore
    original_search = VectorStore.search
    original_lexical = VectorStore.search_lexical
    original_get = getattr(VectorStore, "get_documents", None)
    if not getattr(original_search, "_public_metadata_wrapper", False):
        def search(self, *args, **kwargs):
            return _public_result(original_search(self, *args, **kwargs))
        search._public_metadata_wrapper = True
        VectorStore.search = search
    if not getattr(original_lexical, "_public_metadata_wrapper", False):
        def search_lexical(self, *args, **kwargs):
            return _public_result(original_lexical(self, *args, **kwargs))
        search_lexical._public_metadata_wrapper = True
        VectorStore.search_lexical = search_lexical
    if callable(original_get) and not getattr(original_get, "_public_metadata_wrapper", False):
        def get_documents(self, *args, **kwargs):
            return _public_result(original_get(self, *args, **kwargs))
        get_documents._public_metadata_wrapper = True
        VectorStore.get_documents = get_documents
    _INSTALLED = True
