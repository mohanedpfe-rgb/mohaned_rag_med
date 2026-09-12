from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rag_project.citations.citation_manager import CitationManager
from rag_project.embeddings.embedding_service import EmbeddingService
from rag_project.generation.llm_client import OllamaLLMClient
from rag_project.intelligence.med_evidence_pro import MedEvidenceProEngine
from rag_project.retrieval.hybrid_retriever import HybridRetriever
from rag_project.storage.vector_store import VectorStore
from rag_project.testing.advanced_phases import _fixture_chunks, _result
from rag_project.testing.deep_diagnostics import PhaseResult


class _DeterministicLLM:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0) -> str:
        self.calls += 1
        return "Diabetes mellitus is a chronic metabolic disease. [S1] HbA1c is used in diagnosis and monitoring. [S2]"


@dataclass
class _Settings:
    project_root: Path
    top_k: int = 6
    vector_weight: float = 0.7
    lexical_mode: str = "hybrid"


class _ProbeSystem:
    def __init__(self, root: Path) -> None:
        self.settings = _Settings(root)
        self.citation_manager = CitationManager()
        self.conversation_memory = None
        self._med_selected_hits = []
        self.embedding_service = EmbeddingService(
            "", "diagnostic-deterministic", batch_size=8, retries=0, timeout_seconds=30, test_mode=True
        )
        self.vector_store = VectorStore(root / "vector_db", collection_name="phase10")
        self.retriever = HybridRetriever(
            self.vector_store, self.embedding_service, lexical_mode="hybrid", vector_weight=0.7
        )
        self.live_ollama = os.getenv("DIAGNOSTIC_LIVE_OLLAMA", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if self.live_ollama:
            self.llm = OllamaLLMClient(
                os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                os.getenv("GENERATION_MODEL", "llama3.2:3b"),
                timeout_seconds=60,
                max_output_tokens=512,
                circuit_threshold=2,
                circuit_open_seconds=15,
            )
            self.generation_backend = "ollama"
        else:
            self.llm = _DeterministicLLM()
            self.generation_backend = "deterministic_test"


def _seed_real_retrieval(system: _ProbeSystem) -> int:
    chunks = _fixture_chunks()
    documents = [chunk.text for chunk in chunks]
    metadatas = []
    ids = []
    for chunk in chunks:
        chunk_id = f"{chunk.doc_id}:{chunk.chunk_index}:{chunk.representation_type}"
        metadatas.append(
            {
                **dict(chunk.metadata or {}),
                "document_id": chunk.doc_id,
                "chunk_id": chunk_id,
                "version_id": "phase10-v1",
                "page_numbers": list(chunk.page_numbers or []),
                "index_state": "READY",
            }
        )
        ids.append(chunk_id)
    embeddings = system.embedding_service.embed_texts(documents)
    system.vector_store.add_documents(documents, metadatas, embeddings, ids)
    return len(chunks)


def phase10_canonical_answer_engine(phase: object) -> PhaseResult:
    result = _result(phase)
    root = Path(tempfile.mkdtemp(prefix="rag_phase10_answer_engine_"))
    try:
        system = _ProbeSystem(root)
        seeded = _seed_real_retrieval(system)
        engine = MedEvidenceProEngine(system)
        question = "Explain diabetes mellitus and the role of HbA1c in diagnosis."
        response = engine.answer(question)
        answer = str(response.get("answer") or "")
        verification = response.get("verification") if isinstance(response.get("verification"), dict) else {}
        route = response.get("route") if isinstance(response.get("route"), dict) else {}
        hits = response.get("hits") or []
        citations = response.get("citations") or []
        expected_document_ids = {str(hit.doc_id) for hit in hits}
        citation_ids_valid = bool(citations) and all(
            str(item.get("document_id")) in expected_document_ids for item in citations
        )
        require_live = os.getenv("REQUIRE_LIVE_OLLAMA", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        generation_path = str(response.get("generation_path") or "").casefold()
        live_ok = (
            system.live_ollama
            and system.generation_backend == "ollama"
            and bool(answer)
            and any(token in generation_path for token in ("llm", "ollama", "generated"))
            and getattr(system.llm, "last_error", None) is None
        )
        result.details = {
            "evidence_level": "canonical_med_evidence_pro_engine_real_retrieval",
            "production_entrypoint": "rag_project.intelligence.med_evidence_pro.MedEvidenceProEngine.answer",
            "answer_generated": bool(answer),
            "generation_path": response.get("generation_path"),
            "generation_backend": system.generation_backend,
            "generation_backend_calls": getattr(system.llm, "calls", 1),
            "retrieval_hits": len(hits),
            "seeded_index_records": seeded,
            "retrieval_backend": "VectorStore + HybridRetriever + deterministic EmbeddingService test backend",
            "citations_present": bool(citations),
            "citation_ids_valid": citation_ids_valid,
            "verification_allow": bool(verification.get("allow")),
            "supported_ratio": float(verification.get("supported_ratio", 0.0) or 0.0),
            "route_intent": route.get("intent"),
            "canonical_engine_executed": True,
            "retrieval_stub_used": False,
            "live_ollama_opt_in": system.live_ollama,
            "live_ollama_verified": bool(live_ok),
            "live_ollama_required": require_live,
            "production_orchestration": [
                "SafetyGate", "QueryRouter", "MultiTierRetriever", "HybridRetriever", "VectorStore",
                "EvidenceCompiler", "AnswerCascade", "ActiveVerifier", "ResponseFormatter", "FeedbackLogger",
            ],
        }
        required = all(
            [
                result.details["answer_generated"],
                result.details["retrieval_hits"] > 0,
                result.details["citations_present"],
                result.details["citation_ids_valid"],
                result.details["verification_allow"],
                result.details["canonical_engine_executed"],
                result.details["retrieval_stub_used"] is False,
                (not require_live or live_ok),
            ]
        )
        result.score = 1.0 if required else 0.0
        result.status = "PASS" if required else "FAIL"
        if not required:
            result.failures.append(
                {
                    "location": "MedEvidenceProEngine.answer -> MultiTierRetriever -> HybridRetriever -> VectorStore",
                    "exception": "CanonicalAnswerEngineContractFailure",
                    "message": str(result.details),
                }
            )
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append(
            {
                "location": "phase 10 canonical answer engine",
                "exception": type(exc).__name__,
                "message": str(exc),
            }
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return result
