"""Embedding package."""

from .embedding_service import EmbeddingIdentity, EmbeddingProfile, EmbeddingService
from .embedding_runtime import install as install_embedding_runtime

install_embedding_runtime()

__all__ = [
    "EmbeddingIdentity",
    "EmbeddingProfile",
    "EmbeddingService",
    "install_embedding_runtime",
]
