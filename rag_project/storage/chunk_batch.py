"""Immutable canonical chunk batch shared by semantic and lexical indexes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CanonicalChunk:
    chunk_id: str
    document_id: str
    version_id: str
    text: str
    metadata: Mapping[str, Any]
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class CanonicalChunkBatch:
    chunks: tuple[CanonicalChunk, ...]

    @classmethod
    def build(
        cls,
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]],
        embeddings: Sequence[Sequence[float]],
        ids: Sequence[str],
        *,
        default_state: str = "BUILDING",
    ) -> "CanonicalChunkBatch":
        if not (len(documents) == len(metadatas) == len(embeddings) == len(ids)):
            raise ValueError("documents, metadatas, embeddings, and ids must have the same length")
        chunks: list[CanonicalChunk] = []
        for index, (document, raw_metadata, embedding, chunk_id) in enumerate(
            zip(documents, metadatas, embeddings, ids, strict=True)
        ):
            metadata = dict(raw_metadata or {})
            cid = str(metadata.get("chunk_id") or chunk_id)
            document_id = str(metadata.get("document_id") or "unknown")
            version_id = str(metadata.get("version_id") or document_id or "legacy")
            metadata["chunk_id"] = cid
            metadata["document_id"] = document_id
            metadata["version_id"] = version_id
            metadata.setdefault("index_state", default_state)
            try:
                values = tuple(float(value) for value in embedding)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid embedding at index {index}") from exc
            if not values:
                raise ValueError(f"Empty embedding at index {index}")
            chunks.append(CanonicalChunk(cid, document_id, version_id, str(document), metadata, values))
        return cls(tuple(chunks))

    @property
    def documents(self) -> tuple[str, ...]:
        return tuple(chunk.text for chunk in self.chunks)

    @property
    def metadatas(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(chunk.metadata) for chunk in self.chunks)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(chunk.chunk_id for chunk in self.chunks)

    @property
    def embeddings(self) -> tuple[tuple[float, ...], ...]:
        return tuple(chunk.embedding for chunk in self.chunks)

    def for_semantic_index(self) -> dict[str, list[Any]]:
        return {
            "ids": list(self.ids),
            "documents": list(self.documents),
            "metadatas": [dict(item) for item in self.metadatas],
            "embeddings": [list(item) for item in self.embeddings],
        }

    def for_lexical_index(self) -> dict[str, list[Any]]:
        return {
            "ids": list(self.ids),
            "documents": list(self.documents),
            "metadatas": [dict(item) for item in self.metadatas],
        }


__all__ = ["CanonicalChunk", "CanonicalChunkBatch"]
