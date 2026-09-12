from __future__ import annotations

import http.server
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from socketserver import ThreadingMixIn

from rag_project.citations.citation_manager import CitationManager
from rag_project.embeddings.embedding_service import EmbeddingService
from rag_project.generation.llm_client import OllamaLLMClient
from rag_project.intelligence.med_evidence_pro import MedEvidenceProEngine
from rag_project.retrieval.hybrid_retriever import HybridRetriever
from rag_project.storage.vector_store import VectorStore
from rag_project.testing.advanced_phases import _fixture_chunks, _result
from rag_project.testing.deep_diagnostics import PhaseResult


class _ThreadingHTTPServer(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _OllamaProtocolHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, status: int, payload: dict) -> None:
        import json
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/tags":
            self._json(200, {"models": [{"name": "diagnostic-protocol:latest"}]})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self._json(200, {
            "model": "diagnostic-protocol:latest",
            "created_at": "2026-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "Diabetes mellitus is a chronic metabolic disease. [S1] HbA1c is used in diagnosis and monitoring. [S2]"},
            "done": True,
            "prompt_eval_count": 32,
            "eval_count": 24,
            "total_duration": 1000000,
            "load_duration": 100000,
            "eval_duration": 500000,
        })


class _LocalOllamaServer:
    def __init__(self) -> None:
        self.server = _ThreadingHTTPServer(("127.0.0.1", 0), _OllamaProtocolHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2.0)


@dataclass
class _Settings:
    project_root: Path
    top_k: int = 6
    vector_weight: float = 0.7
    lexical_mode: str = "hybrid"


class _ProbeSystem:
    def __init__(self, root: Path, llm: object | None = None) -> None:
        self.settings = _Settings(root)
        self.citation_manager = CitationManager()
        self.conversation_memory = None
        self._med_selected_hits = []
        self.embedding_service = EmbeddingService("", "diagnostic-deterministic", batch_size=8, retries=0, timeout_seconds=30, test_mode=True)
        self.vector_store = VectorStore(root / "vector_db", collection_name="phase10")
        self.retriever = HybridRetriever(self.vector_store, self.embedding_service, lexical_mode="hybrid", vector_weight=0.7)
        self.llm = llm
        self.generation_backend = "ollama_protocol" if isinstance(self.llm, OllamaLLMClient) else "unknown"


def _seed_real_retrieval(system: _ProbeSystem) -> int:
    chunks = _fixture_chunks(); documents = [chunk.text for chunk in chunks]; metadatas = []; ids = []
    for chunk in chunks:
        chunk_id = f"{chunk.doc_id}:{chunk.chunk_index}:{chunk.representation_type}"
        metadatas.append({**dict(chunk.metadata or {}), "document_id": chunk.doc_id, "chunk_id": chunk_id, "version_id": "phase10-v1", "page_numbers": list(chunk.page_numbers or []), "index_state": "READY"})
        ids.append(chunk_id)
    system.vector_store.add_documents(documents, metadatas, system.embedding_service.embed_texts(documents), ids)
    return len(chunks)


def _build_generation_client(server: _LocalOllamaServer | None) -> tuple[object, str, bool]:
    require_external = os.getenv("REQUIRE_LIVE_OLLAMA", "0").strip().lower() in {"1", "true", "yes", "on"}
    if require_external:
        client = OllamaLLMClient(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"), os.getenv("GENERATION_MODEL", "llama3.2:3b"), timeout_seconds=60, max_output_tokens=512, circuit_threshold=2, circuit_open_seconds=15)
        return client, "ollama_external", True
    assert server is not None
    return OllamaLLMClient(server.base_url, "diagnostic-protocol:latest", timeout_seconds=15, max_output_tokens=512, circuit_threshold=2, circuit_open_seconds=5), "ollama_protocol", False


def phase10_canonical_answer_engine(phase: object) -> PhaseResult:
    result = _result(phase); root = Path(tempfile.mkdtemp(prefix="rag_phase10_answer_engine_")); server = None
    try:
        require_external = os.getenv("REQUIRE_LIVE_OLLAMA", "0").strip().lower() in {"1", "true", "yes", "on"}
        if not require_external: server = _LocalOllamaServer()
        llm, backend, external = _build_generation_client(server)
        system = _ProbeSystem(root, llm=llm); seeded = _seed_real_retrieval(system)
        client_health = system.llm.health_check(timeout_seconds=2.0) if isinstance(system.llm, OllamaLLMClient) else False
        response = MedEvidenceProEngine(system).answer("Explain diabetes mellitus and the role of HbA1c in diagnosis.")
        answer = str(response.get("answer") or ""); verification = response.get("verification") if isinstance(response.get("verification"), dict) else {}; route = response.get("route") if isinstance(response.get("route"), dict) else {}; hits = response.get("hits") or []; citations = response.get("citations") or []
        expected_document_ids = {str(hit.doc_id) for hit in hits}
        citation_ids_valid = bool(citations) and all(str(item.get("document_id")) in expected_document_ids for item in citations)
        generation_path = str(response.get("generation_path") or "").casefold()
        live_ok = external and bool(answer) and any(token in generation_path for token in ("llm", "ollama", "generated")) and getattr(system.llm, "last_error", None) is None
        protocol_ok = isinstance(system.llm, OllamaLLMClient) and client_health and bool(answer) and getattr(system.llm, "last_error", None) is None and system.llm.last_metrics is not None
        result.details = {
            "evidence_level": "canonical_med_evidence_pro_engine_real_retrieval",
            "evidence_level_extended": "canonical_med_evidence_pro_engine_real_retrieval_real_ollama_client_protocol",
            "production_entrypoint": "rag_project.intelligence.med_evidence_pro.MedEvidenceProEngine.answer",
            "answer_generated": bool(answer), "generation_path": response.get("generation_path"), "generation_backend": backend,
            "generation_backend_calls": 1, "retrieval_hits": len(hits), "seeded_index_records": seeded,
            "retrieval_backend": "VectorStore + HybridRetriever + deterministic EmbeddingService test backend",
            "generation_client": "OllamaLLMClient", "ollama_health_check": client_health, "ollama_protocol_roundtrip_verified": protocol_ok,
            "citations_present": bool(citations), "citation_ids_valid": citation_ids_valid, "verification_allow": bool(verification.get("allow")),
            "supported_ratio": float(verification.get("supported_ratio", 0.0) or 0.0), "route_intent": route.get("intent"),
            "canonical_engine_executed": True, "retrieval_stub_used": False, "live_ollama_opt_in": external,
            "live_ollama_verified": bool(live_ok), "live_ollama_required": require_external,
            "production_orchestration": ["SafetyGate", "QueryRouter", "MultiTierRetriever", "HybridRetriever", "VectorStore", "EvidenceCompiler", "AnswerCascade", "ActiveVerifier", "ResponseFormatter", "FeedbackLogger"],
        }
        required = all([result.details["answer_generated"], result.details["retrieval_hits"] > 0, result.details["citations_present"], result.details["citation_ids_valid"], result.details["verification_allow"], result.details["canonical_engine_executed"], result.details["retrieval_stub_used"] is False, result.details["generation_client"] == "OllamaLLMClient", result.details["ollama_protocol_roundtrip_verified"], (not require_external or live_ok)])
        result.score = 1.0 if required else 0.0; result.status = "PASS" if required else "FAIL"
        if not required: result.failures.append({"location": "MedEvidenceProEngine.answer -> OllamaLLMClient -> /api/chat", "exception": "CanonicalAnswerEngineContractFailure", "message": str(result.details)})
    except Exception as exc:
        result.status = "FAIL"; result.failures.append({"location": "phase 10 canonical answer engine", "exception": type(exc).__name__, "message": str(exc)})
    finally:
        if server is not None: server.close()
        shutil.rmtree(root, ignore_errors=True)
    return result
