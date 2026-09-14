"""Canonical storage package with one runtime hardening layer."""

from rag_project.storage.vector_store_runtime import install as install_vector_store_runtime
from rag_project.runtime_chroma_metadata_fix import install as install_chroma_metadata_fix
from rag_project.storage.atomic_index_transaction import install as install_atomic_index_transaction
from rag_project.storage.concurrency_index_fix import install as install_concurrency_index_fix
from rag_project.storage.reconcile_parity_fix import install as install_reconcile_parity_fix

install_vector_store_runtime()
install_chroma_metadata_fix()
# RAGSystem tests and legacy callers instantiate VectorStore directly without
# going through rag_project.runtime.install(). The atomic bridge therefore has
# to be part of the storage bootstrap itself, not only the application runtime.
# It binds the lexical SQLite write to the semantic publication boundary.
install_atomic_index_transaction()
# Chroma's PersistentClient and native HNSW lifecycle must also be serialized;
# otherwise concurrent system construction and validation can race even when
# each ingestion has a distinct SQLite database.
install_concurrency_index_fix()
install_reconcile_parity_fix()

__all__ = [
    "install_vector_store_runtime",
    "install_chroma_metadata_fix",
    "install_atomic_index_transaction",
    "install_concurrency_index_fix",
    "install_reconcile_parity_fix",
]
