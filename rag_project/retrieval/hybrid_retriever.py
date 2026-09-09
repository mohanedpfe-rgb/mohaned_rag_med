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
        try:
            weight = float(vector_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("vector_weight must be numeric") from exc
        if not math.isfinite(weight):
            raise ValueError("vector_weight must be finite")
        self.lexical_mode = lexical_mode
        self.vector_weight = max(0.0, min(1.0, weight))

    def set_embedding_service(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service

    @staticmethod
    def _rrf(rank: int, k: float = 60.0) -> float:
        return 1.0 / (k + float(max(1, int(rank))))

    @staticmethod
    def _unpack_results(result: Any) -> tuple[list[str], list[str], list[dict[str, Any]], list[float]]:
        if not isinstance(result, dict):
            return [], [], [], []

        def first_list(key: str) -> list[Any]:
            value = result.get(key, [])
            if isinstance(value, list) and value and isinstance(value[0], list):
                value = value[0]
            return list(value) if isinstance(value, list) else []

        ids = [str(value) for value in first_list("ids")]
        documents = [str(value) for value in first_list("documents")]
        raw_metadata = first_list("metadatas")
        metadatas = [dict(value) if isinstance(value, dict) else {} for value in raw_metadata]
        raw_distances = first_list("distances")
        distances: list[float] = []
        for value in raw_distances:
            try:
                parsed = float(value)
                distances.append(parsed if math.isfinite(parsed) else 1e9)
            except (TypeError, ValueError):
                distances.append(1e9)
        return ids, documents, metadatas, distances

    def _lexical(self, lexical_query: str, candidate_count: int, where: Dict[str, Any] | None) -> Any:
        return self.vector_store.search_lexical(lexical_query, candidate_count, where)

    def _vector(self, query_embedding: Any, candidate_count: int, where: Dict[str, Any] | None) -> Any:
        return self.vector_store.search(query_embedding, candidate_count, where)

    def retrieve(self, query: str, top_k: int = 6, where: Dict[str, Any] | None = None) -> List[RetrievalHit]:
        original_query = (query or "").strip()
        if not original_query:
            return []
        try:
            top_k = max(1, min(int(top_k), 100))
        except (TypeError, ValueError) as exc:
            raise ValueError("top_k must be an integer") from exc

        lexical_query = " ".join(meaningful_tokens(original_query)).strip()
        candidate_count = max(top_k * 5, 20)
        configured_mode = self.lexical_mode
        vector_weight = self.vector_weight
        lexical_weight = 1.0 - vector_weight

        vector_results: Any = None
        lexical_results: Any = None
        vector_error: Exception | None = None
        lexical_error: Exception | None = None
        query_embedding: Any = None

        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = None
            lexical_future = None

            # Always prepare both branches for hybrid mode. In single-mode operation,
            # the other branch remains available as a real fallback instead of a dead
            # mode switch after the primary branch fails.
            if configured_mode in {"hybrid", "lexical"}:
                lexical_future = executor.submit(self._lexical, lexical_query, candidate_count, where)

            if configured_mode in {"hybrid", "vector"}:
                try:
                    query_embedding = self.embedding_service.embed_query(original_query)
                    vector_future = executor.submit(self._vector, query_embedding, candidate_count, where)
                except (RuntimeError, ValueError, TypeError) as exc:
                    vector_error = exc

            if vector_future is not None:
                try:
                    vector_results = vector_future.result()
                except (RuntimeError, ValueError, TypeError) as exc:
                    vector_error = exc

            if lexical_future is not None:
                try:
                    lexical_results = lexical_future.result()
                except (RuntimeError, ValueError, TypeError) as exc:
                    lexical_error = exc

            # Missing primary branch: execute the missing fallback while we still
            # own the retrieval operation. This fixes the old vector-only and
            # lexical-only fallback paths that could silently return zero hits.
            if configured_mode == "vector" and (vector_error is not None or not self._unpack_results(vector_results)[0]):
                try:
                    lexical_results = self._lexical(lexical_query, candidate_count, where)
                    lexical_error = None
                except (RuntimeError, ValueError, TypeError) as exc:
                    lexical_error = exc
            elif configured_mode == "lexical" and (lexical_error is not None or not self._unpack_results(lexical_results)[0]):
                if query_embedding is None:
                    try:
                        query_embedding = self.embedding_service.embed_query(original_query)
                    except (RuntimeError, ValueError, TypeError) as exc:
                        vector_error = exc
                if query_embedding is not None:
                    try:
                        vector_results = self._vector(query_embedding, candidate_count, where)
                        vector_error = None
                    except (RuntimeError, ValueError, TypeError) as exc:
                        vector_error = exc

        vector_ids, vector_documents, vector_metadatas, vector_distances = self._unpack_results(vector_results)
        lexical_ids, lexical_documents, lexical_metadatas, lexical_distances = self._unpack_results(lexical_results)

        # If hybrid lost one branch, keep the surviving branch. There is no valid
        # reason to discard useful evidence merely because the other backend failed.
        if not vector_ids and not lexical_ids:
            return []

        hits_by_id: dict[str, RetrievalHit] = {}
        vector_document_count = min(len(vector_ids), len(vector_documents))
        for index in range(vector_document_count):
            result_id = vector_ids[index]
            metadata = vector_metadatas[index] if index < len(vector_metadatas) else {}
            distance = vector_distances[index] if index < len(vector_distances) else 1e9
            distance_value = max(0.0, distance)
            hits_by_id[result_id] = RetrievalHit(
                doc_id=str(metadata.get("document_id", "unknown")),
                text=vector_documents[index],
                metadata=metadata,
                score=0.0,
                vector_score=math.exp(-distance_value),
                lexical_score=0.0,
            )

        lexical_document_count = min(len(lexical_ids), len(lexical_documents))
        for index in range(lexical_document_count):
            result_id = lexical_ids[index]
            metadata = lexical_metadatas[index] if index < len(lexical_metadatas) else {}
            distance = lexical_distances[index] if index < len(lexical_distances) else 1e9
            distance_value = max(0.0, distance)
            transformed = math.exp(-distance_value)
            hit = hits_by_id.get(result_id)
            if hit is None:
                hits_by_id[result_id] = RetrievalHit(
                    doc_id=str(metadata.get("document_id", "unknown")),
                    text=lexical_documents[index],
                    metadata=metadata,
                    score=0.0,
                    vector_score=0.0,
                    lexical_score=transformed,
                )
            else:
                hit.lexical_score = transformed

        vector_rank = {item_id: rank for rank, item_id in enumerate(vector_ids, start=1)}
        lexical_rank = {item_id: rank for rank, item_id in enumerate(lexical_ids, start=1)}
        max_vector = max((hit.vector_score for hit in hits_by_id.values()), default=1e-9)
        max_lexical = max((hit.lexical_score for hit in hits_by_id.values()), default=1e-9)

        hits: list[RetrievalHit] = []
        effective_mode = configured_mode
        if configured_mode == "vector" and not vector_ids and lexical_ids:
            effective_mode = "lexical"
        elif configured_mode == "lexical" and not lexical_ids and vector_ids:
            effective_mode = "vector"
        elif configured_mode == "hybrid":
            if not vector_ids and lexical_ids:
                effective_mode = "lexical"
            elif vector_ids and not lexical_ids:
                effective_mode = "vector"

        for item_id, hit in hits_by_id.items():
            if hit.lexical_score <= 0.0:
                hit.lexical_score = keyword_overlap_score(lexical_query, hit.text)
            v_norm = hit.vector_score / max(1e-9, max_vector)
            l_norm = hit.lexical_score / max(1e-9, max_lexical)
            if effective_mode == "hybrid":
                weighted_sum = vector_weight * v_norm + lexical_weight * l_norm
                rrf_score = 0.0
                if item_id in vector_rank:
                    rrf_score += self._rrf(vector_rank[item_id]) * (0.5 + vector_weight)
                if item_id in lexical_rank:
                    rrf_score += self._rrf(lexical_rank[item_id]) * (0.5 + lexical_weight)
                hit.score = 0.6 * weighted_sum + 0.4 * rrf_score
            elif effective_mode == "vector":
                hit.score = hit.vector_score or keyword_overlap_score(original_query, hit.text)
            else:
                hit.score = hit.lexical_score
            hits.append(hit)

        return sorted(hits, key=lambda item: item.score, reverse=True)[:top_k]
