"""Canonical storage package with one runtime hardening layer."""

from rag_project.storage.vector_store_runtime import install as install_vector_store_runtime

install_vector_store_runtime()

__all__ = ["install_vector_store_runtime"]
