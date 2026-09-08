from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List

from rag_project.embeddings.embedding_service import EmbeddingService
from rag_project.storage.vector_store import VectorStore
from rag_project.utils.text_utils import keyword_overlap_score, meaningful_tokens


_VALID_MODES = {"hybrid", "lexical", "vector"}


@dataclass
class RetrievalHit:
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    score: float
    vector_score: float = 0.0
    lexical_score: float = 0.0


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        *,
        lexical_mode: str = "hybrid",
        vector_weight: float = 0.7,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.set_mode(lexical_mode, vector_weight)

    def set_mode(self, lexical_mode: str, vector_weight: float) -> None:
        if lexical_mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid retrieval mode {lexical_mode!r}; expected one of {sorted(_VALID_MODES)}"
            )
        self.lexical_mode = lexical_mode
        self.vector_weight = max(0.0, min(1.0, float(vector_weight)))

    def set_embedding_service(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service

    @staticmethod
    def _rrf(rank: int, k: float = 60.0) -> float:
        return 1.0 / (k + float(max(1, int(rank))))

    def retrieve(self, query: str, top_k: int = 6, where: Dict[str, Any] | None = None) -> List[RetrievalHit]:
        original_query = (query or "").strip()
        lexical_query = " ".join(meaningful_tokens(original_query))
        candidate_count = max(top_k * 5, 20)
        vector_results: dict[str, list[list[Any]]] = {
            "ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]
        }
        lexical_results: dict[str, list[list[Any]]] = {
            "ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]
        }
        mode = self.lexical_mode
        vector_weight = self.vector_weight
        lexical_weight = 1.0 - vector_weight
        with ThreadPoolExecutor(max_workers=2) as executor:
            lexical_future = None
            vector_future = None
            if mode in {"hybrid", "lexical"}:
                lexical_future = executor.submit(
                    self.vector_store.search_lexical, lexical_query, candidate_count, where
                )
            if mode in {"hybrid", "vector"}:
                try:
                    query_embedding = self.embedding_service.embed_query(original_query)
                    vector_future = executor.submit(
                        self.vector_store.search, query_embedding, candidate_count, where
                    )
                except (RuntimeError, ValueError, TypeError):
                    if mode == "vector":
                        mode = "lexical"
            if vector_future is not None:
                try:
                    vector_results = vector_future.result()
                except (RuntimeError, ValueError, TypeError):
                    if mode == "vector":
                        mode = "lexical"
                    elif mode == "hybrid":
                        mode = "lexical"
            if lexical_future is not None:
                lexical_results = lexical_future.result()

        if mode == "vector" and not vector_results.get("ids", [[]])[0]:
            mode = "lexical"
        elif mode == "lexical" and not lexical_results.get("ids", [[]])[0]:
            mode = "vector"

        hits_by_id: dict[str, RetrievalHit] = {}
        vector_ids = vector_results.get("ids", [[]])[0] or []
        vector_documents = vector_results.get("documents", [[]])[0] or []
        vector_metadatas = vector_results.get("metadatas", [[]])[0] or []
        vector_distances = vector_results.get("distances", [[]])[0] or []
        for index, document in enumerate(vector_documents):
            metadata = vector_metadatas[index] if index < len(vector_metadatas) else {}
            distance = vector_distances[index] if index < len(vector_distances) else 1e9
            try:
                distance_value = max(0.0, float(distance))
            except (TypeError, ValueError):
                distance_value = 1e9
            result_id = vector_ids[index]
            hits_by_id[result_id] = RetrievalHit(
                doc_id=str(metadata.get("document_id", "unknown")),
                text=str(document),
                metadata=dict(metadata),
                score=0.0,
                vector_score=math.exp(-distance_value),
                lexical_score=0.0,
            )

        lexical_ids = lexical_results.get("ids", [[]])[0] or []
        lexical_documents = lexical_results.get("documents", [[]])[0] or []
        lexical_metadatas = lexical_results.get("metadatas", [[]])[0] or []
        lexical_distances = lexical_results.get("distances", [[]])[0] or []
        for index, document in enumerate(lexical_documents):
            metadata = lexical_metadatas[index] if index < len(lexical_metadatas) else {}
            try:
                lexical_dist = max(0.0, float(lexical_distances[index]))
            except (TypeError, ValueError, IndexError):
                lexical_dist = 1e9
            result_id = lexical_ids[index]
            transformed = math.exp(-lexical_dist)
            hit = hits_by_id.get(result_id)
            if hit is None:
                hit = RetrievalHit(
                    doc_id=str(metadata.get("document_id", "unknown")),
                    text=str(document),
                    metadata=dict(metadata),
                    score=0.0,
                    vector_score=0.0,
                    lexical_score=transformed,
                )
                hits_by_id[result_id] = hit
            else:
                hit.lexical_score = transformed

        vector_rank = {item_id: rank for rank, item_id in enumerate(vector_ids, start=1)}
        lexical_rank = {item_id: rank for rank, item_id in enumerate(lexical_ids, start=1)}
        max_vector = max((hit.vector_score for hit in hits_by_id.values()), default=1e-9)
        max_lexical = max((hit.lexical_score for hit in hits_by_id.values()), default=1e-9)
        hits: List[RetrievalHit] = []
        for item_id, hit in hits_by_id.items():
            if hit.lexical_score <= 0.0:
                hit.lexical_score = keyword_overlap_score(lexical_query, hit.text)
            v_norm = hit.vector_score / max(1e-9, max_vector)
            l_norm = hit.lexical_score / max(1e-9, max_lexical)
            if mode == "hybrid":
                weighted_sum = vector_weight * v_norm + lexical_weight * l_norm
                rrf_score = 0.0
                if item_id in vector_rank:
                    rrf_score += self._rrf(vector_rank[item_id]) * (0.5 + vector_weight)
                if item_id in lexical_rank:
                    rrf_score += self._rrf(lexical_rank[item_id]) * (0.5 + lexical_weight)
                hit.score = 0.6 * weighted_sum + 0.4 * rrf_score
            elif mode == "vector":
                hit.score = hit.vector_score or keyword_overlap_score(query, hit.text)
            else:
                hit.score = hit.lexical_score
            # Keep the exact child chunk for ranking and context construction.
            # The parent section is still available in metadata for optional UI inspection.
            hits.append(hit)
        return sorted(hits, key=lambda item: item.score, reverse=True)[:top_k]
