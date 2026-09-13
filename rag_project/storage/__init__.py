"""Canonical storage package with one runtime hardening layer."""

from rag_project.storage.vector_store_runtime import install as install_vector_store_runtime
from rag_project.runtime_chroma_metadata_fix import install as install_chroma_metadata_fix

install_vector_store_runtime()
install_chroma_metadata_fix()

__all__ = ["install_vector_store_runtime", "install_chroma_metadata_fix"]
