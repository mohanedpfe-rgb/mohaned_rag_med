from __future__ import annotations


def install() -> None:
    """Install vector store runtime methods (search, search_lexical with READY filter, etc.)."""
    from rag_project.storage import vector_store_runtime
    vector_store_runtime.install()


__all__ = ["install"]
