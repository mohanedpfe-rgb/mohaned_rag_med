"""Application package and runtime storage binding."""

# rag_system.py imports VectorStore from the storage module after this package
# initializer executes. Bind the enhanced implementation here so all production
# entry points receive the richer metadata + hierarchy index without duplicating
# construction logic throughout the application stack.
from rag_project.storage import vector_store as _vector_store_module
from rag_project.storage.enhanced_vector_store import EnhancedVectorStore

_vector_store_module.VectorStore = EnhancedVectorStore
