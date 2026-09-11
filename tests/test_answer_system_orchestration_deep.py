from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_project.app.rag_system import RetrievalHit
from rag_project.app.production_rag import ProductionRAGSystem
from rag_project.intelligence import god_mode
from rag_project.intelligence import god_mode_100
from rag_project.intelligence.advanced_clinical_reasoner import assess_clinical_reasoning
from rag_project.intelligence.evidence_guard import ClaimCheck, grounding_decision, verify_claims
from rag_project.intelligence.final_answer_contract import verify_final_answer
from rag_project.intelligence.query_intelligence import plan_query
from rag_project.intelligence.semantic_reasoning import QueryUnderstanding, extract_clinical_entities
from rag_project.intelligence.top_level_pipeline import deterministic_phase1


def hit(text: str, score: float = 0.8, *, doc: str = "d1", chunk: str = "c1", page: int = 1):
    return RetrievalHit(
        doc_id=doc,
        text=text,
        metadata={"document_id": doc, "chunk_id": chunk, "page_numbers": [page], "file_name": f"{doc}.pdf", "language": "fr"},
        score=score,
        vector_score=score,
        lexical_score=score / 2,
    )


def settings(top_k=4, max_query_variants=8, retrieval_candidate_multiplier=5):
    return SimpleNamespace(top_k=top_k, max_query_variants=max_query_variants, retrieval_candidate_multiplier=retrieval_candidate_multiplier)


def understanding(question: str, *, entities=(), intent="factual", relations=(), constraints=(), confidence=0.8):
    return QueryUnderstanding(
        normalized=question.casefold(), intents=(intent,), primary_intent=intent,
        entities=tuple(entities), relations=tuple(relations), constraints=tuple(constraints),
        answer_shape="explanation", semantic_terms=tuple(question.casefold().split()), confidence=confidence,
    )


class FakeRetriever:
    def __init__(self, batches=None, error_on=None):
        self.batches = batches or {}
        self.error_on = set(error_on or ())
        self.calls = []

    def retrieve(self, query, top_k=8, where=None):
        self.calls.append((query, top_k, where))
        if query in self.error_on:
            raise RuntimeError(f"boom:{query}")
        return list(self.batches.get(query, []))[:top_k]


class FakeReranker:
    def __init__(self, reverse=False, explode=False):
        self.reverse = reverse
        self.explode = explode

    def rerank(self, query, candidates):
        if self.explode:
            raise RuntimeError("reranker down")
        return list(reversed(candidates)) if self.reverse else sorted(candidates, key=lambda x: x.score, reverse=True)


# ---------------------------------------------------------------------------
# _safe_hits orchestration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("top_k", [1, 2, 4, 8, 12])
def test_safe_hits_respects_query_variant_limit(top_k):
    retriever = FakeRetriever({"q": [hit("A", 0.9)]})
    system = SimpleNamespace(settings=settings(top_k=top_k, max_query_variants=2), retriever=retriever, reranker=FakeReranker())
    plan = plan_query("What is diabetes?")
    plan = plan.__class__(plan.original, "q", plan.intent, ("q", "q2", "q3"), ("q", "q2", "q3"), plan.entities, plan.needs_numeric, plan.needs_table, plan.needs_figure, plan.needs_multi_hop, plan.needs_neighborhood, plan.score_boosts)
    result = god_mode._safe_hits(system, plan, None)
    assert len(retriever.calls) <= 2
    assert result


@pytest.mark.parametrize("scores", [
    [0.95, 0.1, 0.2], [0.1, 0.95, 0.2], [0.2, 0.1, 0.95], [0.5, 0.5, 0.4],
])
def test_safe_hits_does_not_drop_late_high_score_candidate(scores):
    batches = {"What is diabetes?": [hit(f"candidate-{i}", score=s, chunk=f"c{i}") for i, s in enumerate(scores)]}
    system = SimpleNamespace(settings=settings(top_k=4, max_query_variants=1), retriever=FakeRetriever(batches), reranker=FakeReranker())
    result = god_mode._safe_hits(system, plan_query("What is diabetes?"), None)
    assert result
    assert max(h.score for h in result) >= max(scores)


@pytest.mark.parametrize("failed_query", ["What is diabetes?", "diabetes", "What is diabetes? diabetes"])
def test_safe_hits_survives_one_retrieval_branch_failure(failed_query):
    retriever = FakeRetriever({"What is diabetes?": [hit("Diabetes is chronic.", 0.8)], "diabetes": [hit("Diabetes is a disease.", 0.7)]}, error_on={failed_query})
    system = SimpleNamespace(settings=settings(top_k=4), retriever=retriever, reranker=FakeReranker())
    result = god_mode._safe_hits(system, plan_query("What is diabetes?"), None)
    assert isinstance(result, list)


@pytest.mark.parametrize("reverse", [False, True])
def test_safe_hits_rerank_failure_preserves_candidates(reverse):
    retriever = FakeRetriever({"What is diabetes?": [hit("Diabetes is chronic.", 0.8), hit("Diabetes is metabolic.", 0.6)]})
    system = SimpleNamespace(settings=settings(top_k=4), retriever=retriever, reranker=FakeReranker(reverse=reverse, explode=True))
    result = god_mode._safe_hits(system, plan_query("What is diabetes?"), None)
    assert result
    assert all(0.0 <= h.score <= 1.0 for h in result)


@pytest.mark.parametrize("score", [-1.0, 0.0, 0.2, 0.8, 1.0, 2.0, float("nan")])
def test_safe_hits_score_sanitization(score):
    value = 0.0 if score != score else score
    retriever = FakeRetriever({"What is diabetes?": [hit("Diabetes is chronic.", value)]})
    system = SimpleNamespace(settings=settings(top_k=4), retriever=retriever, reranker=FakeReranker())
    result = god_mode._safe_hits(system, plan_query("What is diabetes?"), None)
    assert all(0.0 <= float(h.score) <= 1.0 for h in result)


@pytest.mark.parametrize("duplicates", [2, 4, 8, 16])
def test_safe_hits_deduplicates_same_chunk(duplicates):
    repeated = [hit("Diabetes is chronic.", 0.8 - i * 0.01, chunk="same") for i in range(duplicates)]
    retriever = FakeRetriever({"What is diabetes?": repeated})
    system = SimpleNamespace(settings=settings(top_k=8), retriever=retriever, reranker=FakeReranker())
    result = god_mode._safe_hits(system, plan_query("What is diabetes?"), None)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Production composition: enhance_result / exact final-answer gate
# ---------------------------------------------------------------------------

@pytest.fixture
def base_result():
    return {
        "status": "SUCCESS",
        "answer": "Diabetes is chronic.",
        "citations": ["S1"],
        "hits": [hit("Diabetes is chronic.", 0.9)],
        "confidence": {"level": "high", "evidence_confidence": 0.9},
        "query_analysis": {"intent": "factual", "entities": [], "needs_numeric": False},
        "semantic_understanding": {"confidence": 0.9},
        "semantic_alignment": {"score": 0.9, "entity_coverage": 0.0},
        "advanced_reasoning": {"entity_coverage": 1.0, "source_agreement": 0.8, "safety_conflict": 0.0},
        "contradiction_report": {"has_contradiction": False},
    }


def fake_completed(result):
    return {
        **result,
        "answer": result.get("answer", ""),
        "hits": result.get("hits", []),
        "phase_plan": {"intent": "factual", "entities": [], "needs_numeric": False},
        "rewritten_question": "What is diabetes?",
        "phases": {
            "phase_1_query_understanding": "complete",
            "phase_2_retrieval_precision": "complete",
            "phase_4_verification": "complete",
            "phase_5_intelligence_visibility": "complete",
        },
        "adaptive_retrieval": {"stage": 1, "queries": 1, "final_hits": len(result.get("hits", [])), "escalated": False},
        "two_stage_synthesis": {"required": False, "attempted": False, "used": False, "fallback": False, "verification": {}},
    }


@pytest.mark.parametrize("answer", [
    "Diabetes is chronic.",
    "Diabetes is chronic. Sources: [S1] book.pdf",
])
def test_enhance_result_accepts_grounded_final_answer(monkeypatch, base_result, answer):
    base_result["answer"] = answer
    monkeypatch.setattr(god_mode_100, "complete_phases", lambda system, question, result, metadata_filter: fake_completed(result))
    enhanced = god_mode_100.enhance_result(SimpleNamespace(), "What is diabetes?", base_result, None)
    assert enhanced["final_verification"]["allow"] is True
    assert enhanced["status"] == "SUCCESS"
    assert enhanced["answer"]


@pytest.mark.parametrize("answer", [
    "Diabetes is chronic. The moon is blue.",
    "Diabetes is not chronic.",
    "The dose is 600 mg.",
])
def test_enhance_result_blocks_bad_final_answer(monkeypatch, base_result, answer):
    base_result["answer"] = answer
    if "dose" in answer:
        base_result["query_analysis"] = {"intent": "numeric", "entities": ["dose"], "needs_numeric": True}
    monkeypatch.setattr(god_mode_100, "complete_phases", lambda system, question, result, metadata_filter: fake_completed(result))
    enhanced = god_mode_100.enhance_result(SimpleNamespace(), "What is diabetes?", base_result, None)
    assert enhanced["abstained"] is True
    assert enhanced["status"] == "REASONING_ABSTAIN"
    assert enhanced["citations"] == []


@pytest.mark.parametrize("entities", [[], ["diabetes"], ["diabetes", "hypertension"]])
def test_enhance_result_entity_free_confidence_is_not_forced_to_zero(monkeypatch, base_result, entities):
    base_result["query_analysis"] = {"intent": "factual", "entities": entities, "needs_numeric": False}
    monkeypatch.setattr(god_mode_100, "complete_phases", lambda system, question, result, metadata_filter: fake_completed(result))
    enhanced = god_mode_100.enhance_result(SimpleNamespace(), "What is diabetes?", base_result, None)
    assert 0.0 <= enhanced["confidence"]["evidence_confidence"] <= 1.0


@pytest.mark.parametrize("final_matrix", [[], [{"status": "NOT_ENTAILED"}], [{"status": "ENTAILED"}]])
def test_runtime_phase_implementation_reflects_matrix_and_verification(final_matrix):
    completed = fake_completed({"hits": []})
    completed["confidence_calibration"] = {"calibrated": 0.8}
    implementation = god_mode_100._runtime_phase_implementation(completed, final_matrix, {"checked": True, "blocked_claims": 0})
    assert implementation["phase_4_verification"]["claim_count"] == len(final_matrix)
    assert implementation["phase_4_verification"]["final_answer_checked"] is True


# ---------------------------------------------------------------------------
# Public final verifier: exact output is the security boundary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("answer,evidence", [
    ("Diabetes is chronic.", "Diabetes is chronic."),
    ("Hypertension is common.", "Hypertension is common."),
    ("Metformin lowers glucose.", "Metformin lowers glucose."),
    ("The dose is 500 mg.", "The dose is 0.5 g."),
])
def test_final_verifier_accepts_supported_answers(answer, evidence):
    result = verify_final_answer(answer, [hit(evidence, 0.95)])
    assert result["allow"] is True
    assert result["blocked_claims"] == 0


@pytest.mark.parametrize("answer,evidence", [
    ("Diabetes is not chronic.", "Diabetes is chronic."),
    ("The dose is 600 mg.", "The dose is 500 mg."),
    ("The moon is blue.", "Diabetes is chronic."),
])
def test_final_verifier_rejects_unsupported_answers(answer, evidence):
    result = verify_final_answer(answer, [hit(evidence, 0.95)])
    assert result["allow"] is False
    assert result["blocked_claims"] >= 1


@pytest.mark.parametrize("answer", ["", "   ", "Sources: [S1] book.pdf", "[S1]"])
def test_final_verifier_rejects_no_verifiable_claims(answer):
    result = verify_final_answer(answer, [hit("Diabetes is chronic.", 0.9)])
    assert result["allow"] is False


# ---------------------------------------------------------------------------
# Production conversation isolation / exception restoration
# ---------------------------------------------------------------------------


def bare_production(fake_answer, history=None):
    obj = object.__new__(ProductionRAGSystem)
    obj._production_feature_contract = {"all_resolved": True}
    obj.conversation_memory = SimpleNamespace(history=list(history or []))
    obj.settings = SimpleNamespace(medical_high_risk_evidence_threshold=0.8)
    obj._certified_god_answer = fake_answer
    return obj


@pytest.mark.parametrize("history", [[], [("What is diabetes?", "Diabetes is chronic.")], [("A", "B"), ("C", "D")]])
def test_production_isolated_question_restores_previous_history(monkeypatch, history):
    def fake_answer(question, metadata_filter=None):
        return {"status": "SUCCESS", "answer": "Hypertension is elevated blood pressure.", "citations": [], "confidence": {"evidence_confidence": 0.5}}
    obj = bare_production(fake_answer, history)
    monkeypatch.setattr("rag_project.app.production_rag.sanitize_trace", lambda x: x)
    before = list(history)
    ProductionRAGSystem.answer(obj, "What is hypertension?")
    assert obj.conversation_memory.history[:len(before)] == before
    assert obj.conversation_memory.history[-1][0] == "What is hypertension?"


@pytest.mark.parametrize("question", ["What is diabetes?", "What is hypertension?", "What are the main findings?"])
def test_production_answer_always_sets_pipeline_authority(monkeypatch, question):
    obj = bare_production(lambda q, metadata_filter=None: {"status": "SUCCESS", "answer": "evidence", "citations": [], "confidence": {"evidence_confidence": 0.5}})
    monkeypatch.setattr("rag_project.app.production_rag.sanitize_trace", lambda x: x)
    result = ProductionRAGSystem.answer(obj, question)
    assert result["pipeline_authority"].endswith("complete_phases")


@pytest.mark.parametrize("question", ["What is diabetes?", "What are the main findings?", "Define hypertension."])
def test_production_answer_handles_certified_answer_exception_without_corrupting_history(monkeypatch, question):
    history = [("previous", "answer")]
    def broken_answer(q, metadata_filter=None):
        raise RuntimeError("generation failure")
    obj = bare_production(broken_answer, history)
    monkeypatch.setattr("rag_project.app.production_rag.sanitize_trace", lambda x: x)
    result = ProductionRAGSystem.answer(obj, question)
    assert result["recovery"]["attempted"] is True
    assert result["recovery"]["pipeline_error"] == "RuntimeError"
    assert obj.conversation_memory.history == history


# ---------------------------------------------------------------------------
# Reasoning + verification interaction invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", ["What is diabetes?", "What is hypertension?", "What is albuminuria?"])
def test_simple_entity_reasoning_never_requires_multi_hop(question):
    u = understanding(question, entities=extract_clinical_entities(question), intent="factual")
    result = assess_clinical_reasoning(u, [hit(question.replace("What is ", "") + " is a medical concept.", 0.9)])
    assert result.mode in {"DIRECT", "DIRECT_SUMMARY", "INSUFFICIENT"}
    assert result.depth in {0, 1}


@pytest.mark.parametrize("answer", ["Diabetes is chronic.", "Hypertension is common.", "Albuminuria is albumin in urine."])
def test_verified_claims_have_valid_source_ids(answer):
    checks = verify_claims(answer, [answer], ["S1"])
    assert checks
    assert all(src.startswith("S") for check in checks for src in check.sources)


@pytest.mark.parametrize("support", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
def test_grounding_ratio_is_monotonic_with_supported_claim_count(support):
    good = ClaimCheck("good", support, "SUPPORTED", ("S1",))
    bad = ClaimCheck("bad", 0.0, "UNSUPPORTED", ())
    one = grounding_decision([good])
    two = grounding_decision([good, bad])
    assert one["supported_ratio"] >= two["supported_ratio"]


# ---------------------------------------------------------------------------
# Query-plan / production interaction: never expand arrays unboundedly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [0, 1, 2, 4, 8, 16, 32])
def test_deterministic_phase1_entity_limit(n):
    question = "What are " + " ".join(f"condition{i}" for i in range(n)) + "?"
    plan = deterministic_phase1(question)
    assert len(plan.entities) <= 16
    assert len(plan.rewritten_queries) <= 8
    assert len(plan.sub_questions) <= 6


@pytest.mark.parametrize("intent", ["factual", "comparison", "diagnosis", "management", "etiology", "mechanism", "prognosis", "numeric", "table_lookup", "figure_lookup", "relationship"])
def test_plan_query_output_schema_for_all_supported_intents(intent):
    phrases = {
        "factual": "What is diabetes?", "comparison": "Compare diabetes and hypertension.", "diagnosis": "What are diagnostic criteria?",
        "management": "What is the treatment?", "etiology": "What causes DKA?", "mechanism": "How does insulin work?",
        "prognosis": "What is the prognosis?", "numeric": "What is the dose in mg?", "table_lookup": "Which table has the values?",
        "figure_lookup": "What does figure 2 show?", "relationship": "What is diabetes related to?",
    }
    plan = plan_query(phrases[intent])
    assert plan.intent == intent
    assert isinstance(plan.variants, tuple)
    assert len(plan.variants) <= 10
