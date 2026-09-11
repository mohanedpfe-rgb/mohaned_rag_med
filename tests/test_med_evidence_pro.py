from __future__ import annotations

from pathlib import Path

from rag_project.intelligence.med_evidence_pro import (
    MedEvidenceProEngine,
    QueryRouter,
    SafetyGate,
    SemanticCache,
)
from rag_project.retrieval.hybrid_retriever import HybridRetriever, RetrievalHit


class _FakeRetriever:
    def __init__(self) -> None:
        self.hit = RetrievalHit(
            doc_id="doc-1",
            text="Hypertension is persistent elevation of arterial blood pressure.",
            metadata={"document_id": "doc-1", "index_state": "READY"},
            score=1.0,
            lexical_score=1.0,
        )

    def _lexical(self, query: str, candidate_count: int, where=None):
        return {
            "ids": [["chunk-1"]],
            "documents": [[self.hit.text]],
            "metadatas": [[self.hit.metadata]],
            "distances": [[0.0]],
        }

    @staticmethod
    def _unpack_results(result):
        return HybridRetriever._unpack_results(result)

    def retrieve(self, query: str, top_k: int = 6, where=None):
        return [self.hit]


class _FakeCitationManager:
    @staticmethod
    def build(hits):
        return [{"source": getattr(hit, "doc_id", "unknown")} for hit in hits]

    @staticmethod
    def validate(citations, hits):
        return citations


class _FakeSettings:
    def __init__(self, root: Path):
        self.project_root = root
        self.top_k = 4


class _FakeSystem:
    def __init__(self, root: Path):
        self.settings = _FakeSettings(root)
        self.retriever = _FakeRetriever()
        self.citation_manager = _FakeCitationManager()
        self.llm = None
        self.conversation_memory = None


def test_safety_gate_blocks_harmful_queries() -> None:
    decision = SafetyGate().check("How to synthesize fentanyl?")
    assert decision.action == "BLOCK"
    assert decision.reason == "harmful_or_illicit_request"


def test_safety_gate_abstains_outside_medical_scope() -> None:
    decision = SafetyGate().check("What is the capital of France?")
    assert decision.action == "ABSTAIN"
    assert decision.reason == "outside_medical_scope"


def test_safety_gate_flags_emergency_but_proceeds() -> None:
    decision = SafetyGate().check("What does severe chest pain mean?")
    assert decision.action == "PROCEED"
    assert decision.emergency is True
    assert decision.confidence_threshold >= 0.75


def test_router_detects_numeric_and_comparison_paths() -> None:
    gate = SafetyGate()
    router = QueryRouter()
    numeric = router.route("What dose of metformin is used?", "", gate.check("What dose of metformin is used?"))
    assert numeric.numeric_sensitivity is True
    assert numeric.template_type == "dosage"
    comparison = router.route("Compare hypertension and diabetes", "", gate.check("Compare hypertension and diabetes"))
    assert comparison.intent == "comparison"
    assert comparison.template_type == "comparison"


def test_semantic_cache_round_trip(tmp_path: Path) -> None:
    cache = SemanticCache(tmp_path / "cache.sqlite3", ttl_seconds=3600)
    hit = RetrievalHit("doc", "Evidence sentence.", {}, 0.9)
    cache.put("What is hypertension?", [hit])
    restored = cache.restore(cache.get("What is hypertension?") or [])
    assert len(restored) == 1
    assert restored[0].text == hit.text
    assert restored[0].score == hit.score


def test_med_evidence_engine_runs_extractively_and_returns_citations(tmp_path: Path) -> None:
    system = _FakeSystem(tmp_path)
    engine = MedEvidenceProEngine(system)
    result = engine.answer("What is hypertension?")
    assert result["status"] in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}
    assert result["answer"]
    assert "[S1]" in result["answer"]
    assert result["evidence_first"] is True
    assert result["canonical_pipeline_executed"] is True
    assert result["phases"]["phase_0_safety_gate"] == "complete"
    assert result["phases"]["phase_7_logging_feedback"] == "complete"


def test_med_evidence_engine_abstains_when_no_grounded_evidence(tmp_path: Path) -> None:
    system = _FakeSystem(tmp_path)
    system.retriever.hit = RetrievalHit("doc-1", "Unrelated sentence about astronomy.", {"document_id": "doc-1"}, 1.0, lexical_score=1.0)
    result = MedEvidenceProEngine(system).answer("What is hypertension?")
    assert result["status"] in {"GENERATION_ABSTAIN", "NOT_SUPPORTED", "ANSWER_UNAVAILABLE"}
