"""Canonical storage package with one runtime hardening layer."""

from rag_project.storage.vector_store_runtime import install as install_vector_store_runtime
from rag_project.runtime_chroma_metadata_fix import install as install_chroma_metadata_fix
from rag_project.storage.reconcile_parity_fix import install as install_reconcile_parity_fix

install_vector_store_runtime()
install_chroma_metadata_fix()
install_reconcile_parity_fix()

__all__ = [
    "install_vector_store_runtime",
    "install_chroma_metadata_fix",
    "install_reconcile_parity_fix",
]
