from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Sequence

from rag_project.app.rag_system import (
    EvidenceAlignment,
    RAGSystem,
    QueryQualityClassifier,
    _generate_with_citations,
    sanitize_evidence,
)
from rag_project.configuration.settings import Settings
from rag_project.embeddings.embedding_service import EmbeddingService
from rag_project.generation.llm_client import OllamaLLMClient
from rag_project.retrieval.context_builder import ContextBuilder
from rag_project.retrieval.metadata_filter import MetadataFilter
from rag_project.retrieval.query_rewriter import QueryRewriter
from rag_project.retrieval.hybrid_retriever import RetrievalHit


class ResilientRAGSystem(RAGSystem):
    """Production-facing RAG wrapper that keeps degraded modes usable."""

    def __init__(self, settings: Settings | None = None):
        super().__init__(settings)
        self._last_component_error: str | None = None
        self._rebuild_context_builder()

    def _rebuild_context_builder(self) -> None:
        self.context_builder = ContextBuilder(
            token_budget=self.settings.context_token_budget,
            max_per_document=4 if self.settings.neighbor_expansion else 1,
            neighbor_expansion=self.settings.neighbor_expansion,
            neighbor_resolver=self._fetch_neighbors,
        )

    def _fetch_neighbors(self, document_id: str, chunk_index: int, radius: int = 1) -> list[RetrievalHit]:
        records = self.vector_store.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
        ids = records.get("ids", [])
        documents = records.get("documents", [])
        metadatas = records.get("metadatas", [])
        result: list[RetrievalHit] = []
        wanted = set(range(max(0, int(chunk_index) - radius), int(chunk_index) + radius + 1))
        for index, metadata in enumerate(metadatas):
            try:
                current_index = int((metadata or {}).get("chunk_index"))
            except (TypeError, ValueError):
                continue
            if current_index not in wanted or current_index == int(chunk_index):
                continue
            result.append(
                RetrievalHit(
                    doc_id=str((metadata or {}).get("document_id", document_id)),
                    text=str(documents[index] if index < len(documents) else ""),
                    metadata=dict(metadata or {}),
                    score=0.0,
                    vector_score=0.0,
                    lexical_score=0.0,
                )
            )
        return sorted(result, key=lambda hit: int(hit.metadata.get("chunk_index", 0)))

    def _rebuild_runtime_clients(self) -> None:
        self.embedding_service = EmbeddingService(
            self.settings.ollama_base_url,
            self.settings.embedding_model,
            batch_size=self.settings.embedding_batch_size,
            retries=self.settings.embedding_retries,
            timeout_seconds=self.settings.embedding_timeout_seconds,
            test_mode=self.settings.embedding_test_mode,
            cache_size=self.settings.embedding_cache_size,
            cache_ttl_seconds=self.settings.embedding_cache_ttl_seconds,
            max_concurrency=self.settings.ollama_concurrency,
        )
        self.embedding_startup_error = None
        self.vector_store.set_expected_identity(None)
        try:
            self.embedding_service.discover_dimension()
            self.vector_store.set_expected_identity(self.embedding_service.identity)
        except RuntimeError as exc:
            self.embedding_startup_error = str(exc)
        self.retriever.set_embedding_service(self.embedding_service)
        self.llm = OllamaLLMClient(
            self.settings.ollama_base_url,
            self.settings.generation_model,
            self.settings.generation_timeout_seconds,
            self.settings.generation_max_output_tokens,
        )
        self._rebuild_context_builder()

    def apply_settings_in_place(self, updates: Dict[str, Any]) -> tuple[bool, list[str]]:
        rebuild_needed = any(
            field in updates
            for field in {
                "ollama_base_url",
                "embedding_model",
                "generation_model",
                "embedding_batch_size",
                "embedding_retries",
                "embedding_timeout_seconds",
                "embedding_test_mode",
                "embedding_cache_size",
                "embedding_cache_ttl_seconds",
                "generation_timeout_seconds",
                "generation_max_output_tokens",
            }
        )
        success, warnings = super().apply_settings_in_place(updates)
        if rebuild_needed:
            try:
                self._rebuild_runtime_clients()
                warnings = [
                    warning
                    for warning in warnings
                    if "requires a full RAGSystem restart" not in warning
                ]
            except Exception as exc:
                warnings.append(f"Runtime client rebuild failed: {exc}")
                self._last_component_error = str(exc)
                success = False
        if "neighbor_expansion" in updates or "context_token_budget" in updates:
            self._rebuild_context_builder()
        return success, warnings

    def ingest_file(self, pdf_path: str | Any) -> Dict[str, Any]:
        """Retry failed versions safely and re-probe embeddings before ingestion."""
        from pathlib import Path

        file_path = Path(pdf_path)
        self.embedding_startup_error = None
        try:
            self.embedding_service.discover_dimension()
            self.vector_store.set_expected_identity(self.embedding_service.identity)
        except RuntimeError as exc:
            self.embedding_startup_error = str(exc)

        if file_path.is_file():
            content_hash = self._hash_file(file_path)
            existing = self.state_store.get_by_hash(content_hash)
            if existing and not self.state_store.is_ready_status(existing.get("status")):
                try:
                    self.vector_store.delete_version(existing["document_id"], content_hash)
                except Exception as exc:
                    self._last_component_error = f"Failed to clean retryable index: {exc}"
        return super().ingest_file(file_path)

    def _safe_rewrite(self, question: str) -> str:
        """Use query rewriting only when Ollama responds immediately."""
        try:
            import requests

            response = requests.get(
                f"{self.settings.ollama_base_url.rstrip('/')}/api/tags",
                timeout=(1.5, 2.0),
            )
            response.raise_for_status()
            return QueryRewriter.rewrite(question, self.conversation_memory.history, llm=self.llm)
        except Exception:
            return question.strip()

    @staticmethod
    def _retrieval_mode(hits: Sequence[Any]) -> str:
        has_vector = any(getattr(hit, "vector_score", 0.0) > 0 for hit in hits)
        has_lexical = any(getattr(hit, "lexical_score", 0.0) > 0 for hit in hits)
        if has_vector and has_lexical:
            return "hybrid"
        if has_vector:
            return "vector"
        if has_lexical:
            return "lexical"
        return "unknown"

    def answer(self, question: str, metadata_filter: Dict[str, Any] | None = None) -> Dict[str, Any]:
        answer_started = time.perf_counter()
        original_question = (question or "").strip()
        if not original_question:
            return {
                "status": "invalid_query",
                "answer": "Please enter a question.",
                "citations": [],
                "hits": [],
                "confidence": {"level": "none", "top_score": 0.0, "margin": 0.0},
            }

        self.index_compatibility = self.vector_store.compatibility_report(self.embedding_service.identity)
        rewritten_question = self._safe_rewrite(original_question)
        filter_query = MetadataFilter.build(metadata_filter)

        retrieval_started = time.perf_counter()
        retrieval_error: str | None = None
        try:
            hits = self.retriever.retrieve(rewritten_question, top_k=self.settings.top_k, where=filter_query)
        except Exception as exc:
            retrieval_error = str(exc)
            hits = []
            try:
                lexical = self.vector_store.search_lexical(
                    rewritten_question,
                    n_results=max(self.settings.top_k, 5),
                    where=filter_query,
                )
                ids = lexical.get("ids", [[]])[0]
                docs = lexical.get("documents", [[]])[0]
                metas = lexical.get("metadatas", [[]])[0]
                distances = lexical.get("distances", [[]])[0]
                hits = [
                    RetrievalHit(
                        doc_id=str(meta.get("document_id", item_id)),
                        text=str(doc),
                        metadata=dict(meta),
                        score=1.0 - float(distances[index] if index < len(distances) else 1.0),
                        vector_score=0.0,
                        lexical_score=1.0 - float(distances[index] if index < len(distances) else 1.0),
                    )
                    for index, (item_id, doc, meta) in enumerate(zip(ids, docs, metas, strict=True))
                ]
            except Exception as lexical_exc:
                retrieval_error = f"{retrieval_error}; lexical fallback failed: {lexical_exc}"
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        rerank_started = time.perf_counter()
        reranked = self.reranker.rerank(rewritten_question, hits)
        rerank_ms = (time.perf_counter() - rerank_started) * 1000
        confidence = self.reranker.confidence(reranked)

        sanitized_hits = []
        for hit in reranked:
            sanitized = sanitize_evidence(hit.text)
            if sanitized != hit.text:
                metadata = dict(hit.metadata or {})
                metadata["evidence_sanitized"] = True
                hit = RetrievalHit(
                    doc_id=hit.doc_id,
                    text=sanitized,
                    metadata=metadata,
                    score=hit.score,
                    vector_score=hit.vector_score,
                    lexical_score=hit.lexical_score,
                )
            sanitized_hits.append(hit)

        context, selected_hits = self.context_builder.build(sanitized_hits)
        query_analysis = QueryQualityClassifier.assess(original_question)
        evidence_alignment = EvidenceAlignment.evaluate(original_question, selected_hits)
        if not selected_hits:
            result = {
                "query_id": str(uuid.uuid4()),
                "answer": "I could not find sufficient evidence in the indexed documents.",
                "citations": [],
                "hits": [],
                "confidence": confidence,
                "query_analysis": query_analysis,
                "evidence_alignment": evidence_alignment,
                "retrieval_mode": self._retrieval_mode(hits),
            }
            if retrieval_error:
                result["retrieval_error"] = retrieval_error
            return result

        generation_started = time.perf_counter()
        try:
            answer, _ = _generate_with_citations(
                self.llm,
                question=original_question,
                context=context,
                selected_hits=selected_hits,
                conversation_context=self.conversation_memory.prompt_context(),
                temperature=self.settings.temperature,
            )
        except Exception as exc:
            self._last_component_error = str(exc)
            answer = (
                "The language model is unavailable right now. Here is the grounded evidence "
                "that matched your question:\n\n"
                + "\n\n".join(
                    f"[S{index + 1}] {hit.text}" for index, hit in enumerate(selected_hits)
                )
            )
        generation_ms = (time.perf_counter() - generation_started) * 1000
        answer, answer_grounding, query_coverage, grounding_fallback = self.apply_grounding_guard(
            answer, rewritten_question, selected_hits
        )
        self.conversation_memory.add(original_question, answer)
        citations = self.citation_manager.validate(
            self.citation_manager.build(selected_hits), selected_hits
        )
        query_id = str(uuid.uuid4())
        total_ms = (time.perf_counter() - answer_started) * 1000
        self.state_store.record_query_trace(
            query_id,
            {
                "original_query": original_question,
                "rewritten_query": rewritten_question,
                "retrieval_method": self._retrieval_mode(hits),
                "candidate_count": len(hits),
                "timings_ms": {
                    "retrieval": round(retrieval_ms, 3),
                    "rerank": round(rerank_ms, 3),
                    "generation": round(generation_ms, 3),
                    "total": round(total_ms, 3),
                },
                "confidence": confidence,
                "query_analysis": query_analysis,
                "evidence_alignment": evidence_alignment,
                "grounding": {
                    "answer_evidence_overlap": round(answer_grounding, 3),
                    "query_answer_coverage": round(query_coverage, 3),
                    "fallback_used": grounding_fallback,
                },
                "selected_chunk_ids": [hit.metadata.get("chunk_id", hit.doc_id) for hit in selected_hits],
                "citations": citations,
                "degraded_mode": bool(retrieval_error or self.embedding_startup_error),
            },
        )
        result = {
            "query_id": query_id,
            "answer": answer,
            "citations": citations,
            "hits": selected_hits,
            "confidence": confidence,
            "query_analysis": query_analysis,
            "evidence_alignment": evidence_alignment,
            "retrieval_mode": self._retrieval_mode(hits),
            "degraded_mode": bool(retrieval_error or self.embedding_startup_error),
        }
        if retrieval_error:
            result["retrieval_error"] = retrieval_error
        return result
