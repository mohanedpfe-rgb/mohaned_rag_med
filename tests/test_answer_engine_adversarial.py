from __future__ import annotations

import re
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from rag_project.app.rag_system import RetrievalHit
from rag_project.app.production_rag import _is_explicit_followup, ProductionRAGSystem
from rag_project.generation.latency_budget import budget_scope, exhausted, remaining
from rag_project.intelligence.evidence_guard import (
    ClaimCheck,
    contradiction_report,
    detect_contradiction,
    extract_measurements,
    grounding_decision,
    numeric_consistency,
    semantic_support,
    split_claims,
    verify_claims,
)
from rag_project.intelligence.god_mode import _answer_with_ladder, _safe_hits, _simple_extractive_answer
from rag_project.intelligence.query_intelligence import decompose_query, extract_query_entities, normalize_query, plan_query
from rag_project.intelligence.semantic_reasoning import QueryUnderstanding, extract_clinical_entities
from rag_project.intelligence.small_model_reasoner import should_use_small_model
from rag_project.intelligence.top_level_pipeline import (
    _hard_query,
    compress_context,
    deterministic_phase1,
    extractive_draft,
    medical_term_layer,
    rewrite_follow_up,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def hit(
    text: str,
    score: float = 0.9,
    *,
    doc: str = "doc-1",
    chunk: str | None = None,
    page: int = 1,
    language: str = "fr",
    evidence_types: tuple[str, ...] = ("text",),
):
    return RetrievalHit(
        doc_id=doc,
        text=text,
        metadata={
            "document_id": doc,
            "chunk_id": chunk or f"{doc}-chunk-{page}",
            "page_numbers": [page],
            "file_name": f"{doc}.pdf",
            "language": language,
            "evidence_types": list(evidence_types),
        },
        score=score,
        vector_score=score,
        lexical_score=score * 0.5,
    )


class FakeRetriever:
    def __init__(self, mapping=None, errors=None):
        self.mapping = mapping or {}
        self.errors = set(errors or ())
        self.calls: list[tuple[str, int, object]] = []

    def retrieve(self, query, top_k=8, where=None):
        self.calls.append((query, top_k, where))
        if query in self.errors:
            raise RuntimeError(f"retrieval:{query}")
        return list(self.mapping.get(query, ()))[:top_k]


class FakeReranker:
    def __init__(self, error=False):
        self.error = error

    def rerank(self, query, candidates):
        if self.error:
            raise RuntimeError("reranker failure")
        return sorted(candidates, key=lambda x: float(x.score), reverse=True)


class FakeLLM:
    def __init__(self, outputs=None, error=False, delay=0.0):
        self.outputs = list(outputs or [])
        self.error = error
        self.delay = delay
        self.calls = []

    def generate(self, prompt, system_prompt=None, temperature=0.2):
        self.calls.append((prompt, system_prompt, temperature))
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise RuntimeError("llm failure")
        return self.outputs.pop(0) if self.outputs else ""

    def generate_json(self, prompt, system_prompt=None, temperature=0.0, max_tokens=180):
        self.calls.append((prompt, system_prompt, temperature, max_tokens))
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise RuntimeError("llm json failure")
        return self.outputs.pop(0) if self.outputs else "{}"


def understanding(
    question: str,
    *,
    entities=(),
    intent="factual",
    relations=(),
    confidence=0.9,
):
    return QueryUnderstanding(
        normalized=question.casefold(),
        intents=(intent,),
        primary_intent=intent,
        entities=tuple(entities),
        relations=tuple(relations),
        constraints=(),
        answer_shape="explanation",
        semantic_terms=tuple(question.casefold().split()),
        confidence=confidence,
    )


def fake_settings(**overrides):
    base = dict(
        top_k=4,
        max_query_variants=4,
        retrieval_candidate_multiplier=5,
        context_token_budget=1800,
        temperature=0.2,
        generation_latency_budget_seconds=45.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# A. Query normalization / language matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("What is diabetes?", "What is diabetes?"),
        ("  What   is   diabetes?  ", "What is diabetes?"),
        ("What's diabetes?", "What is diabetes?"),
        ("what's hypertension?", "what is hypertension?"),
        ("diabetes – hypertension", "diabetes - hypertension"),
        ("diabetes — hypertension", "diabetes - hypertension"),
        ("A\nB\tC", "A B C"),
        ("العربية", "العربية"),
        ("ما هو السكري؟", "ما هو السكري؟"),
        ("Qu'est-ce que le diabète ?", "Qu'est-ce que le diabète ?"),
        ("  ", ""),
        ("", ""),
    ],
)
def test_query_normalization_matrix(raw, expected):
    assert normalize_query(raw) == expected


@pytest.mark.parametrize(
    "question",
    [
        "What is diabetes?",
        "Define diabetes.",
        "What is HbA1c?",
        "What is DKA?",
        "Qu'est-ce que le diabète ?",
        "ما هو السكري؟",
        "Quel est le site principal de régulation du phosphate ?",
        "ما هي مضاعفات الحماض الكيتوني السكري؟",
    ],
)
def test_simple_queries_have_no_fake_question_entities(question):
    entities = {str(x).casefold() for x in extract_query_entities(question)}
    fake = {"what", "is", "the", "main", "principal", "biais", "follow-up", "relevant", "entities"}
    assert not entities.intersection(fake)


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What is diabetes?", "factual"),
        ("Define hypertension.", "factual"),
        ("What is the dose of metformin?", "numeric"),
        ("How many mg is the dose?", "numeric"),
        ("Which table contains potassium values?", "table_lookup"),
        ("What does Figure 3 show?", "figure_lookup"),
        ("Compare diabetes and hypertension.", "comparison"),
        ("What causes DKA?", "etiology"),
        ("How does insulin work?", "mechanism"),
        ("What is the treatment of DKA?", "management"),
        ("What are the diagnostic criteria?", "diagnosis"),
        ("What is the prognosis?", "prognosis"),
    ],
)
def test_query_intent_matrix(question, expected):
    assert plan_query(question).intent == expected


@pytest.mark.parametrize(
    "question",
    [
        "What is diabetes?",
        "What is HbA1c?",
        "What are the main findings?",
        "Define hypertension.",
        "What is albuminuria?",
        "What is DKA?",
        "What does the document report?",
        "Summarize this section.",
        "List the complications.",
    ],
)
def test_easy_question_does_not_trigger_strict_small_model(question):
    plan = plan_query(question)
    q = understanding(question, intent=plan.intent, entities=extract_clinical_entities(question))
    expected = bool(
        plan.needs_multi_hop
        or plan.needs_numeric
        or plan.needs_table
        or plan.needs_figure
        or plan.intent in {"diagnosis", "management", "etiology", "mechanism", "prognosis", "comparison"}
        or len(question.split()) >= 28
    )
    assert should_use_small_model(question, q) is expected


# ---------------------------------------------------------------------------
# B. Follow-up isolation and contamination attacks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question,expected",
    [
        ("What about complications?", True),
        ("And the treatment?", True),
        ("What about this?", True),
        ("And this?", True),
        ("et le traitement ?", True),
        ("والمضاعفات؟", True),
        ("وما هي المضاعفات؟", True),
        ("What is diabetes?", False),
        ("What is hypertension?", False),
        ("Define insulin resistance.", False),
        ("Quel est le site principal de régulation du phosphate ?", False),
        ("ما هو السكري؟", False),
    ],
)
def test_explicit_followup_classifier(question, expected):
    assert _is_explicit_followup(question) is expected


@pytest.mark.parametrize(
    "question",
    [
        "What is diabetes?",
        "What is hypertension?",
        "Define HbA1c.",
        "What is DKA?",
        "What are the complications of diabetes?",
        "What is the main site of phosphate regulation?",
        "Quel est le site principal de régulation du phosphate ?",
        "ما هو الموقع الرئيسي لتنظيم الفوسفات؟",
    ],
)
def test_standalone_questions_are_not_rewritten_even_with_history(question):
    history = [("What is diabetes?", "Diabetes is a chronic disease.")]
    rewritten = rewrite_follow_up(question, history)
    assert rewritten == question
    assert "Follow-up:" not in rewritten
    assert "Relevant entities:" not in rewritten


@pytest.mark.parametrize(
    "question",
    [
        "What about complications?",
        "And the treatment?",
        "What about this?",
        "والمضاعفات؟",
        "وما هي المضاعفات؟",
    ],
)
def test_real_followups_keep_context(question):
    history = [("What is diabetes?", "Diabetes is chronic. It may have complications.")]
    rewritten = rewrite_follow_up(question, history)
    assert rewritten != question
    assert "Follow-up:" in rewritten
    assert "diabetes" in rewritten.casefold() or "السكري" in rewritten


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_short_complete_question_is_not_automatically_followup(tokens):
    question = "What is " + "acute " * max(1, tokens - 2) + "diabetes?"
    history = [("What is hypertension?", "Hypertension is common.")]
    rewritten = rewrite_follow_up(question, history)
    assert "Follow-up:" not in rewritten


# ---------------------------------------------------------------------------
# C. Decomposition / query planning preservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question,required",
    [
        ("Compare diabetes and hypertension.", ["diabetes", "hypertension"]),
        ("What are the causes and complications of DKA?", ["causes", "complications", "DKA"]),
        ("What is the dose of metformin?", ["metformin"]),
        ("What is the relationship between obesity and diabetes?", ["obesity", "diabetes"]),
        ("What is HbA1c and how is it used?", ["HbA1c"]),
        ("ما هي أسباب ومضاعفات الحماض الكيتوني السكري؟", ["الحماض", "مضاعفات"]),
        ("Quelles sont les causes et complications du diabète ?", ["causes", "complications"]),
    ],
)
def test_decomposition_preserves_user_concepts(question, required):
    joined = " ".join(decompose_query(question)).casefold()
    for term in required:
        assert term.casefold() in joined


@pytest.mark.parametrize(
    "question",
    [
        "What is the exact dose?",
        "How many mg?",
        "Which table contains potassium values?",
        "What does Figure 2 show?",
        "How does insulin affect glucose?",
    ],
)
def test_hard_query_detection_is_true_for_expensive_questions(question):
    plan = plan_query(question)
    q = understanding(question, intent=plan.intent, entities=extract_clinical_entities(question))
    assert _hard_query(plan, q, question) is True


@pytest.mark.parametrize(
    "question",
    [
        "What is diabetes?",
        "Define hypertension.",
        "What is HbA1c?",
        "What is DKA?",
    ],
)
def test_simple_factual_question_not_marked_hard(question):
    plan = plan_query(question)
    q = understanding(question, intent=plan.intent, entities=extract_clinical_entities(question))
    assert _hard_query(plan, q, question) is False


# ---------------------------------------------------------------------------
# D. Evidence splitting and metadata pollution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "[Section: 3. Régulation :] Le rein est le principal site de régulation de la concentration plasmatique du Pi.",
        "[Section: Introduction] Diabetes is chronic.",
        "[Page 12] HbA1c reflects long-term glycemic control.",
        "Source: book.pdf\nDiabetes is chronic.",
        "References:\n[1] Example\nHypertension is common.",
    ],
)
def test_metadata_does_not_become_a_meaningful_claim(text):
    claims = split_claims(text)
    assert all("[Section:" not in c for c in claims)
    assert all(not c.casefold().startswith("source:") for c in claims)
    assert all(not c.casefold().startswith("references:") for c in claims)


@pytest.mark.parametrize(
    "text,needle",
    [
        ("[Section: 3. Régulation :] Le rein est le principal site de régulation de la concentration plasmatique du Pi.", "Le rein est le principal site"),
        ("[Page 12] HbA1c reflects long-term glycemic control.", "HbA1c reflects"),
        ("Source: x.pdf\nDiabetes is chronic.", "Diabetes is chronic"),
    ],
)
def test_metadata_cleanup_preserves_actual_evidence(text, needle):
    claims = split_claims(text)
    assert any(needle.casefold() in c.casefold() for c in claims)


# ---------------------------------------------------------------------------
# E. Extractive fast path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question,evidence",
    [
        ("What is the main site of phosphate regulation?", "Le rein est le principal site de régulation de la concentration plasmatique du Pi."),
        ("What is diabetes?", "Diabetes mellitus is a chronic metabolic disorder."),
        ("What is HbA1c?", "HbA1c reflects average glycemic exposure over time."),
        ("What is DKA?", "Diabetic ketoacidosis is a metabolic emergency characterized by hyperglycemia and ketosis."),
        ("Quel est le site principal de régulation du phosphate ?", "Le rein est le principal site de régulation du phosphate plasmatique."),
        ("ما هو الموقع الرئيسي لتنظيم الفوسفات؟", "الكلية هي الموقع الرئيسي لتنظيم الفوسفات في البلازما."),
    ],
)
def test_simple_extractive_answer_returns_relevant_sentence(question, evidence):
    answer = _simple_extractive_answer(question, [hit(evidence)])
    assert answer
    assert any(term.casefold() in answer.casefold() for term in evidence.split() if len(term) > 4)


@pytest.mark.parametrize(
    "question",
    [
        "What is diabetes?",
        "Define HbA1c.",
        "What is the main site of phosphate regulation?",
        "What is DKA?",
    ],
)
def test_simple_answer_path_does_not_call_llm(question):
    llm = FakeLLM(outputs=["THIS MUST NOT BE USED"])
    selected = [hit("The requested fact is directly stated here.")]
    answer, path = _answer_with_ladder(
        SimpleNamespace(llm=llm, settings=fake_settings()),
        question,
        "The requested fact is directly stated here.",
        selected,
        "",
        "",
    )
    assert path == "fast_extractive"
    assert not llm.calls
    assert "THIS MUST NOT BE USED" not in answer


@pytest.mark.parametrize(
    "question",
    [
        "Why does diabetic ketoacidosis happen?",
        "How does insulin affect glucose?",
        "Compare diabetes and hypertension.",
        "What is the treatment of DKA?",
        "What is the exact dose of metformin?",
    ],
)
def test_hard_question_does_call_llm_when_evidence_is_available(question):
    llm = FakeLLM(outputs=["[S1] Supported answer."])
    answer, path = _answer_with_ladder(
        SimpleNamespace(llm=llm, settings=fake_settings()),
        question,
        "Supported evidence.",
        [hit("Supported evidence.")],
        "",
        "",
    )
    assert path in {"primary", "extractive_fallback", "abstained"}
    assert llm.calls
    assert answer


# ---------------------------------------------------------------------------
# F. Retrieval resilience and ordering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("top_k", [1, 2, 4, 6, 8, 12])
def test_safe_hits_respects_variant_budget(top_k):
    retriever = FakeRetriever({"What is diabetes?": [hit("Diabetes is chronic.")]})
    system = SimpleNamespace(settings=fake_settings(top_k=top_k, max_query_variants=2), retriever=retriever, reranker=FakeReranker())
    assert _safe_hits(system, plan_query("What is diabetes?"), None)
    assert len(retriever.calls) <= 2


@pytest.mark.parametrize("failed", ["What is diabetes?", "diabetes", "What is diabetes? diabetes"])
def test_safe_hits_survives_retrieval_branch_error(failed):
    retriever = FakeRetriever(
        {
            "What is diabetes?": [hit("Diabetes is chronic.")],
            "diabetes": [hit("Diabetes is a metabolic disorder.", 0.8)],
        },
        errors={failed},
    )
    system = SimpleNamespace(settings=fake_settings(), retriever=retriever, reranker=FakeReranker())
    result = _safe_hits(system, plan_query("What is diabetes?"), None)
    assert isinstance(result, list)


@pytest.mark.parametrize("value", [-1.0, 0.0, 0.1, 0.5, 1.0, 2.0, float("nan")])
def test_safe_hits_score_bounds(value):
    actual = 0.0 if value != value else value
    retriever = FakeRetriever({"What is diabetes?": [hit("Diabetes is chronic.", actual)]})
    system = SimpleNamespace(settings=fake_settings(), retriever=retriever, reranker=FakeReranker())
    result = _safe_hits(system, plan_query("What is diabetes?"), None)
    assert all(0.0 <= float(x.score) <= 1.0 for x in result)


@pytest.mark.parametrize("count", [1, 2, 4, 8, 16])
def test_safe_hits_deduplicates_identical_chunks(count):
    values = [hit("Same evidence.", 0.9 - i * 0.01, chunk="same") for i in range(count)]
    retriever = FakeRetriever({"What is diabetes?": values})
    system = SimpleNamespace(settings=fake_settings(top_k=8), retriever=retriever, reranker=FakeReranker())
    result = _safe_hits(system, plan_query("What is diabetes?"), None)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# G. Context compression and evidence preservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question,evidence",
    [
        ("What is diabetes?", "Diabetes is chronic.\n" * 20),
        ("What is HbA1c?", "HbA1c reflects glycemic exposure.\n" * 20),
        ("What is the dose?", "The dose is 500 mg.\n" * 20),
    ],
)
def test_context_compression_returns_nonempty_relevant_context(question, evidence):
    compact, meta = compress_context(question, [hit(evidence)])
    assert compact
    assert meta["selected_sentences"] >= 1
    assert meta["input_sentences"] >= meta["selected_sentences"]


@pytest.mark.parametrize("max_chars", [50, 100, 200, 500, 1000, 2000])
def test_context_compression_respects_hard_character_limit(max_chars):
    compact, _ = compress_context("What is diabetes?", [hit("Diabetes is chronic. " * 100)], max_chars=max_chars)
    assert len(compact) <= max_chars


@pytest.mark.parametrize("question", ["What is diabetes?", "What is HbA1c?", "What is DKA?"])
def test_extractive_draft_is_stable_with_duplicate_evidence(question):
    phase = deterministic_phase1(question)
    evidence = [hit("The answer is stated here clearly.", 0.9, chunk="c1"), hit("The answer is stated here clearly.", 0.8, chunk="c2")]
    draft, meta = extractive_draft(question, evidence, phase)
    assert draft
    assert meta["supported"] is True


# ---------------------------------------------------------------------------
# H. Medical term layer coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question,expected",
    [
        ("What is HbA1c?", "hba1c"),
        ("What is the dose of metformin?", "metformin"),
        ("What are the effects of dapagliflozin?", "dapagliflozin"),
        ("Is 500 mg the correct dose?", "500 mg"),
        ("Is the level 10%?", "10%"),
        ("What is hypokalemia?", "hypokalemia"),
    ],
)
def test_medical_term_layer_detects_important_terms(question, expected):
    result = medical_term_layer(question)
    joined = " ".join(str(x) for x in result["terms"] + result["units"])
    assert expected.casefold() in joined.casefold()


# ---------------------------------------------------------------------------
# I. Evidence support / contradiction / numeric consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "claim,evidence,expected_supported",
    [
        ("Diabetes is chronic.", "Diabetes is chronic.", True),
        ("HbA1c reflects glycemic exposure.", "HbA1c reflects glycemic exposure.", True),
        ("Diabetes causes pneumonia.", "Diabetes is chronic.", False),
        ("The moon is blue.", "Diabetes is chronic.", False),
    ],
)
def test_semantic_support_matrix(claim, evidence, expected_supported):
    result = semantic_support(claim, evidence)
    assert bool(result) is expected_supported


@pytest.mark.parametrize(
    "a,b",
    [
        ("Diabetes is present.", "Diabetes is absent."),
        ("Hyperglycemia is present.", "There is no hyperglycemia."),
        ("The patient has DKA.", "The patient does not have DKA."),
    ],
)
def test_contradiction_detector_flags_presence_absence(a, b):
    assert detect_contradiction(a, b) is True


@pytest.mark.parametrize(
    "a,b",
    [
        ("Diabetes is chronic.", "Hypertension is common."),
        ("HbA1c reflects glycemic exposure.", "Albuminuria is kidney damage."),
    ],
)
def test_contradiction_detector_does_not_overflag_unrelated_claims(a, b):
    assert detect_contradiction(a, b) is False


@pytest.mark.parametrize(
    "claim,evidence",
    [
        ("The dose is 500 mg.", "The dose is 500 mg."),
        ("The dose is 0.5 g.", "The dose is 500 mg."),
        ("The dose is 600 mg.", "The dose is 500 mg."),
    ],
)
def test_numeric_consistency_matrix(claim, evidence):
    result = numeric_consistency(claim, evidence)
    assert isinstance(result, bool)


@pytest.mark.parametrize(
    "text",
    [
        "10%",
        "500 mg",
        "0.5 g",
        "12.5 mL",
        "120 mmHg",
        "5 mmol/L",
        "70 kg",
        "37°C",
        "100 IU",
    ],
)
def test_measurement_extraction_recognizes_common_units(text):
    values = extract_measurements(text)
    assert values


# ---------------------------------------------------------------------------
# J. Claim verification invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "answer,evidence",
    [
        ("Diabetes is chronic.", ["Diabetes is chronic."]),
        ("HbA1c reflects glycemic exposure.", ["HbA1c reflects glycemic exposure."]),
        ("The dose is 500 mg.", ["The dose is 500 mg."]),
    ],
)
def test_verify_claims_accepts_directly_supported_answer(answer, evidence):
    checks = verify_claims(answer, evidence, ["S1"])
    assert checks
    assert all(c.status not in {"UNSUPPORTED", "CONTRADICTED", "NUMERIC_MISMATCH"} for c in checks)


@pytest.mark.parametrize(
    "answer,evidence",
    [
        ("Diabetes is chronic. The moon is blue.", ["Diabetes is chronic."]),
        ("Diabetes is not chronic.", ["Diabetes is chronic."]),
        ("The dose is 600 mg.", ["The dose is 500 mg."]),
        ("Hypertension causes DKA.", ["Hypertension is common."]),
    ],
)
def test_verify_claims_rejects_unsupported_or_conflicting_answer(answer, evidence):
    checks = verify_claims(answer, evidence, ["S1"])
    assert checks
    assert any(c.status in {"UNSUPPORTED", "WEAK", "CONTRADICTED", "NUMERIC_MISMATCH"} or c.contradiction for c in checks)


@pytest.mark.parametrize(
    "checks,allow",
    [
        ([ClaimCheck("supported", "S1", 0.9, "ENTAILED", False, False)], True),
        ([ClaimCheck("bad", None, 0.0, "UNSUPPORTED", False, False)], False),
        ([ClaimCheck("bad", "S1", 0.0, "CONTRADICTED", True, False)], False),
    ],
)
def test_grounding_decision_matrix(checks, allow):
    result = grounding_decision(checks, min_supported_ratio=0.60)
    assert result["allow"] is allow


@pytest.mark.parametrize(
    "checks",
    [
        [],
        [ClaimCheck("supported", "S1", 0.9, "ENTAILED", False, False)],
        [ClaimCheck("bad", None, 0.0, "UNSUPPORTED", False, False)],
        [ClaimCheck("bad", "S1", 0.0, "CONTRADICTED", True, False)],
    ],
)
def test_contradiction_report_is_structurally_stable(checks):
    result = contradiction_report(checks)
    assert "has_contradiction" in result


# ---------------------------------------------------------------------------
# K. Latency budget semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seconds", [0.05, 0.1, 0.25, 0.5])
def test_latency_budget_expires_when_deadline_passes(seconds):
    with budget_scope(seconds):
        assert remaining() is not None
        time.sleep(seconds + 0.02)
        assert exhausted() is True


def test_latency_budget_is_shared_not_reset_for_nested_scopes():
    with budget_scope(0.30):
        first = remaining()
        time.sleep(0.10)
        with budget_scope(0.50):
            nested = remaining()
            assert nested <= first + 0.01
            time.sleep(0.10)
        assert remaining() < first


def test_latency_budget_allows_fast_operations():
    with budget_scope(1.0):
        time.sleep(0.01)
        assert not exhausted()


# ---------------------------------------------------------------------------
# L. Production answer isolation / error recovery
# ---------------------------------------------------------------------------


def bare_production(fake_answer, history=None):
    obj = object.__new__(ProductionRAGSystem)
    obj._production_feature_contract = {"all_resolved": True}
    obj.conversation_memory = SimpleNamespace(history=list(history or []))
    obj.settings = SimpleNamespace(medical_high_risk_evidence_threshold=0.8)
    obj._certified_god_answer = fake_answer
    return obj


@pytest.mark.parametrize(
    "question",
    [
        "What is diabetes?",
        "What is hypertension?",
        "What is HbA1c?",
        "What is DKA?",
        "What is the main site of phosphate regulation?",
        "Quel est le site principal de régulation du phosphate ?",
        "ما هو الموقع الرئيسي لتنظيم الفوسفات؟",
    ],
)
def test_production_independent_question_preserves_previous_history(monkeypatch, question):
    history = [("previous", "previous answer")]
    obj = bare_production(lambda q, metadata_filter=None: {"status": "SUCCESS", "answer": "evidence", "citations": [], "confidence": {"evidence_confidence": 0.8}}, history)
    monkeypatch.setattr("rag_project.app.production_rag.sanitize_trace", lambda x: x)
    ProductionRAGSystem.answer(obj, question)
    assert obj.conversation_memory.history[0] == history[0]
    assert obj.conversation_memory.history[-1][0] == question


def test_production_failure_restores_history(monkeypatch):
    history = [("previous", "previous answer")]

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    obj = bare_production(broken, history)
    monkeypatch.setattr("rag_project.app.production_rag.sanitize_trace", lambda x: x)
    result = ProductionRAGSystem.answer(obj, "What is diabetes?")
    assert obj.conversation_memory.history == history
    assert result["recovery"]["attempted"] is True
    assert result["recovery"]["pipeline_error"] == "RuntimeError"


# ---------------------------------------------------------------------------
# M. High-value adversarial question corpus
# ---------------------------------------------------------------------------

ADVERSARIAL_QUESTIONS = [
    "What is diabetes?",
    "Define diabetes mellitus.",
    "What is HbA1c?",
    "What is DKA?",
    "What is hypoglycemia?",
    "What is hyperglycemia?",
    "What is hypokalemia?",
    "What is albuminuria?",
    "What is the main site of phosphate regulation?",
    "What are the main complications of diabetes?",
    "What are the causes of diabetic ketoacidosis?",
    "Why does diabetic ketoacidosis happen?",
    "How does insulin affect glucose?",
    "Compare type 1 and type 2 diabetes.",
    "What is the treatment of DKA?",
    "What are the diagnostic criteria for DKA?",
    "What is the exact dose of metformin?",
    "Which table contains potassium values?",
    "What does Figure 3 show?",
    "What does the document say about renal phosphate handling?",
    "What is the relationship between obesity and diabetes?",
    "What complications are associated with nephropathy?",
    "What about complications?",
    "And the treatment?",
    "والمضاعفات؟",
    "وما هو العلاج؟",
    "Quelles sont les complications du diabète ?",
    "Quel est le site principal de régulation du phosphate ?",
    "ما هو الموقع الرئيسي لتنظيم الفوسفات؟",
    "Quel est le traitement de l'acidocétose diabétique ?",
]


@pytest.mark.parametrize("question", ADVERSARIAL_QUESTIONS)
def test_adversarial_question_corpus_has_deterministic_plan(question):
    plan = plan_query(question)
    assert plan.normalized
    assert plan.intent
    assert isinstance(plan.entities, tuple)
    assert isinstance(plan.variants, tuple)


@pytest.mark.parametrize("question", ADVERSARIAL_QUESTIONS)
def test_adversarial_question_corpus_does_not_crash_hard_query_classifier(question):
    plan = plan_query(question)
    q = understanding(question, intent=plan.intent, entities=extract_clinical_entities(question))
    result = _hard_query(plan, q, question)
    assert isinstance(result, bool)


@pytest.mark.parametrize("question", ADVERSARIAL_QUESTIONS)
def test_adversarial_question_corpus_extractive_path_is_safe(question):
    evidence = [hit("The indexed evidence contains a directly relevant sentence about the topic.")]
    answer = _simple_extractive_answer(question, evidence)
    assert isinstance(answer, str)


# ---------------------------------------------------------------------------
# N. Regression tests for the previously observed failure trace
# ---------------------------------------------------------------------------


def test_phosphate_trace_never_creates_followup_metadata_from_standalone_query():
    question = "What is the main site of phosphate regulation?"
    history = [("What is the principal bias follow-up?", "irrelevant previous answer")]
    rewritten = rewrite_follow_up(question, history)
    assert rewritten == question
    assert "Follow-up" not in rewritten
    assert "Relevant entities" not in rewritten


def test_phosphate_trace_extractive_answer_uses_medical_sentence_not_section_fragment():
    evidence = [
        hit(
            "[Section: 3. Régulation :] Le rein est le principal site de régulation de la concentration plasmatique du Pi."
        )
    ]
    answer = _simple_extractive_answer("What is the main site of phosphate regulation?", evidence)
    assert "Le rein est le principal site" in answer
    assert "[Section:" not in answer
    assert "85mg" not in answer
    assert "phospholipides" not in answer


def test_phosphate_trace_fast_path_does_not_call_llm():
    llm = FakeLLM(outputs=["slow answer"])
    system = SimpleNamespace(llm=llm, settings=fake_settings())
    answer, path = _answer_with_ladder(
        system,
        "What is the main site of phosphate regulation?",
        "Le rein est le principal site de régulation de la concentration plasmatique du Pi.",
        [hit("Le rein est le principal site de régulation de la concentration plasmatique du Pi.")],
        "",
        "",
    )
    assert path == "fast_extractive"
    assert not llm.calls
    assert "Le rein est le principal site" in answer


# ---------------------------------------------------------------------------
# O. Determinism / repeated-call invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "What is diabetes?",
    "What is HbA1c?",
    "What is DKA?",
    "What is the main site of phosphate regulation?",
    "What about complications?",
])
def test_query_plan_is_repeatably_deterministic(question):
    first = plan_query(question).to_dict()
    second = plan_query(question).to_dict()
    assert first == second


@pytest.mark.parametrize("question", [
    "What is diabetes?",
    "What is HbA1c?",
    "What is DKA?",
    "What is the main site of phosphate regulation?",
])
def test_extractive_answer_is_repeatably_deterministic(question):
    evidence = [hit("This is the directly relevant evidence sentence for the question.")]
    first = _simple_extractive_answer(question, evidence)
    second = _simple_extractive_answer(question, evidence)
    assert first == second


# ---------------------------------------------------------------------------
# P. Boundary / malformed-input resilience
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question",
    [
        "",
        " ",
        "?",
        "!!!",
        "...",
        "a",
        "###",
        "null",
        "None",
        "What is?",
        "Why?",
        "How?",
        "؟؟؟",
        "ما؟",
    ],
)
def test_malformed_or_low_signal_query_does_not_crash_planner(question):
    plan = plan_query(question)
    assert plan is not None


@pytest.mark.parametrize(
    "evidence",
    [
        "",
        " ",
        "123",
        "[Section: x]",
        "Source: file.pdf",
        "\\x00",
        "```",
        "<script>alert(1)</script>",
    ],
)
def test_malformed_evidence_does_not_crash_extractive_answer(evidence):
    answer = _simple_extractive_answer("What is diabetes?", [hit(evidence)])
    assert isinstance(answer, str)


# ---------------------------------------------------------------------------
# Q. Explicit invariants for the answer engine contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", ADVERSARIAL_QUESTIONS)
def test_answer_engine_question_contract_is_json_like(question):
    plan = plan_query(question)
    payload = plan.to_dict()
    required = {
        "original",
        "normalized",
        "intent",
        "subqueries",
        "variants",
        "entities",
        "needs_numeric",
        "needs_table",
        "needs_figure",
        "needs_multi_hop",
    }
    assert required.issubset(payload)


@pytest.mark.parametrize(
    "question,should_have_table",
    [
        ("What is the dose of metformin?", False),
        ("How many mg?", False),
        ("Which table contains potassium values?", True),
        ("Show the table of potassium values.", True),
        ("What is 500 mg?", False),
    ],
)
def test_numeric_and_table_queries_are_separated(question, should_have_table):
    plan = plan_query(question)
    assert plan.needs_table is should_have_table


@pytest.mark.parametrize(
    "question,should_be_followup",
    [
        ("What is diabetes?", False),
        ("What about complications?", True),
        ("And the treatment?", True),
        ("What is the treatment of diabetes?", False),
        ("والمضاعفات؟", True),
        ("ما هي مضاعفات السكري؟", False),
    ],
)
def test_followup_boundary_matrix(question, should_be_followup):
    assert _is_explicit_followup(question) is should_be_followup


# ---------------------------------------------------------------------------
# R. Answer quality / no-empty-success invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question,evidence",
    [
        ("What is diabetes?", "Diabetes is a chronic metabolic disorder."),
        ("What is HbA1c?", "HbA1c reflects average glycemic exposure."),
        ("What is DKA?", "DKA is characterized by hyperglycemia and ketosis."),
        ("What is the main site of phosphate regulation?", "The kidney is the main site of phosphate regulation."),
    ],
)
def test_grounded_factual_question_produces_nonempty_extractive_answer(question, evidence):
    answer = _simple_extractive_answer(question, [hit(evidence)])
    assert answer.strip()
    assert "- " in answer


@pytest.mark.parametrize(
    "question",
    [
        "What is diabetes?",
        "What is HbA1c?",
        "What is DKA?",
    ],
)
def test_no_llm_fast_path_output_contains_citation_marker(question):
    answer = _simple_extractive_answer(question, [hit("The requested fact is stated directly in this sentence.")])
    assert re.search(r"\[S\d+\]", answer)
