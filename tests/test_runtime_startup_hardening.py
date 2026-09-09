from __future__ import annotations

from rag_project.embeddings.embedding_service import EmbeddingService
from rag_project.runtime_final_gate import install


def test_embedding_identity_is_safe_descriptor_after_runtime_install():
    install()
    descriptor = getattr(EmbeddingService, "identity", None)
    assert descriptor is not None
    assert isinstance(descriptor, property)
    assert callable(descriptor.fget)


def test_runtime_install_is_idempotent():
    install()
    install()
