"""Canonical storage package with one runtime hardening layer."""

from rag_project.storage.vector_store_runtime import install as install_vector_store_runtime
from rag_project.runtime_chroma_metadata_fix import install as install_chroma_metadata_fix
from rag_project.storage.atomic_index_transaction import install as install_atomic_index_transaction
from rag_project.storage.reconcile_parity_fix import install as install_reconcile_parity_fix

install_vector_store_runtime()
install_chroma_metadata_fix()
# RAGSystem tests and legacy callers instantiate VectorStore directly without
# going through rag_project.runtime.install(). The atomic bridge therefore has
# to be part of the storage bootstrap itself, not only the application runtime.
# It wraps the already-normalized add_documents path and binds the lexical
# SQLite write to the same transaction boundary as the semantic publication.
install_atomic_index_transaction()
install_reconcile_parity_fix()

__all__ = [
    "install_vector_store_runtime",
    "install_chroma_metadata_fix",
    "install_atomic_index_transaction",
    "install_reconcile_parity_fix",
]
