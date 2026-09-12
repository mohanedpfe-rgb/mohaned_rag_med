from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_project.intelligence.med_evidence_pro import MedEvidenceProEngine
from rag_project.retrieval.hybrid_retriever import RetrievalHit
from rag_project.testing.advanced_phases import _fixture_chunks, _result
from rag_project.testing.deep_diagnostics import PhaseResult


class _DeterministicLLM:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0) -> str:
        self.calls += 1
        return "Diabetes mellitus is a chronic metabolic disease. [S1] HbA1c is used in diagnosis and monitoring. [S2]"


class _CitationManager:
    def build(self, hits: list[RetrievalHit]) -> list[dict[str, Any]]:
        return [{"source": f"S{index + 1}", "document_id": hit.doc_id} for index, hit in enumerate(hits)]

    def validate(self, citations: list[dict[str, Any]], hits: list[RetrievalHit]) -> list[dict[str, Any]]:
        valid = {hit.doc_id for hit in hits}
        return [item for item in citations if item.get("document_id") in valid]


@dataclass
class _Settings:
    project_root: Path


class _ProbeSystem:
    def __init__(self, root: Path) -> None:
        self.settings = _Settings(root)
        self.llm = _DeterministicLLM()
        self.citation_manager = _CitationManager()
        self._med_selected_hits: list[RetrievalHit] = []
        self.conversation_memory = None


def phase10_canonical_answer_engine(phase: Any) -> PhaseResult:
    result = _result(phase)
    root = Path(tempfile.mkdtemp(prefix="rag_phase10_answer_engine_"))
    try:
        system = _ProbeSystem(root)
        chunks = _fixture_chunks()
        hits = [
            RetrievalHit(
                doc_id=chunk.doc_id,
                text=chunk.text,
                metadata={**dict(chunk.metadata or {}), "chunk_id": f"{chunk.doc_id}:{chunk.chunk_index}"},
                score=1.0 / (index + 1),
            )
            for index, chunk in enumerate(chunks[:6])
        ]
        engine = MedEvidenceProEngine(system)

        class _RetrievalStub:
            def retrieve(self, question: str, route: Any, metadata_filter: dict[str, Any] | None = None) -> tuple[list[RetrievalHit], dict[str, Any]]:
                return hits, {
                    "tier": "diagnostic-production-stub",
                    "candidate_count": len(hits),
                    "final_hits": len(hits),
                    "cache_hit": False,
                    "early_exit": False,
                }

        engine.retrieval = _RetrievalStub()
        response = engine.answer("Explain the mechanism and management of diabetes mellitus and the role of HbA1c.")
        answer = str(response.get("answer") or "")
        verification = response.get("verification") if isinstance(response.get("verification"), dict) else {}
        route = response.get("route") if isinstance(response.get("route"), dict) else {}
        result.details = {
            "evidence_level": "canonical_med_evidence_pro_engine",
            "production_entrypoint": "rag_project.intelligence.med_evidence_pro.MedEvidenceProEngine.answer",
            "answer_generated": bool(answer),
            "generation_path": response.get("generation_path"),
            "generation_backend_calls": system.llm.calls,
            "retrieval_hits": len(response.get("hits") or []),
            "citations_present": bool(response.get("citations")),
            "citation_ids_valid": all(item.get("document_id") in {hit.doc_id for hit in hits} for item in response.get("citations") or []),
            "verification_allow": bool(verification.get("allow")),
            "supported_ratio": float(verification.get("supported_ratio", 0.0) or 0.0),
            "route_intent": route.get("intent"),
            "canonical_engine_executed": True,
            "production_orchestration": [
                "SafetyGate",
                "QueryRouter",
                "MultiTierRetriever",
                "EvidenceCompiler",
                "AnswerCascade",
                "ActiveVerifier",
                "ResponseFormatter",
                "FeedbackLogger",
            ],
        }
        required = all(
            [
                result.details["answer_generated"],
                result.details["citations_present"],
                result.details["citation_ids_valid"],
                result.details["verification_allow"],
                result.details["canonical_engine_executed"],
            ]
        )
        result.score = 1.0 if required else 0.0
        result.status = "PASS" if required else "FAIL"
        if not required:
            result.failures.append({
                "location": "MedEvidenceProEngine.answer",
                "exception": "CanonicalAnswerEngineContractFailure",
                "message": str(result.details),
            })
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({
            "location": "phase 10 canonical answer engine",
            "exception": type(exc).__name__,
            "message": str(exc),
        })
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
    return result
