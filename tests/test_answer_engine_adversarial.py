from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
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
    def __init__(self, reverse: bool = False):
        self.reverse = reverse
        self.calls = []

    def rerank(self, query, hits):
        self.calls.append((query, list(hits)))
        rows = list(hits)
        return list(reversed(rows)) if self.reverse else rows


class FakeLLM:
    def __init__(self, outputs=None):
        self.outputs = list(outputs or [])
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0) if self.outputs else ""

    def generate_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0) if self.outputs else None



def fake_settings(**overrides):
    base = dict(
        top_k=4,
        retrieval_candidate_multiplier=5,
        max_query_variants=8,
        temperature=0.0,
        generation_latency_budget_seconds=5.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# B. Follow-up and conversation preservation
# ---------------------------------------------------------------------------

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
    assert "Follow-up:" not in rewritten
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
def test_query_planning_preserves_core_terms(question, required):
    plan = plan_query(question)
    joined = " ".join(plan.subqueries + plan.variants + plan.entities).casefold()
    for term in required:
        assert term.casefold() in joined or term.casefold().replace("é", "e") in joined


# ---------------------------------------------------------------------------
# D. Query normalization / hard-query invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question,expected",
    [
        ("  What   is   diabetes?  ", "What is diabetes?"),
        ("What's diabetes?", "What is diabetes?"),
        ("Type 1–Type 2", "Type 1-Type 2"),
        ("A vs. B", "A versus B"),
    ],
)
def test_query_normalization_matrix(question, expected):
    assert normalize_query(question) == expected


# ---------------------------------------------------------------------------
# E. Small-model / planner boundary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", ["Hi", "Thanks", "ok", "What is diabetes?"])
def test_small_model_usage_is_bounded(question):
    result = should_use_small_model(plan_query(question), SimpleNamespace(confidence=0.2))
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# F. Retrieval safety / score boundaries
# ---------------------------------------------------------------------------

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


@pytest.mark.parametrize("claim,evidence", [("The dose is 500 mg.", "The dose is 500 mg."), ("The dose is 0.5 g.", "The dose is 500 mg."), ("The dose is 600 mg.", "The dose is 500 mg.")])
def test_numeric_consistency_matrix(claim,evidence):
    result=numeric_consistency(claim,evidence)
    assert isinstance(result,dict)
    assert "mismatch" in result and "checked" in result


@pytest.mark.parametrize("text", ["10%","500 mg","0.5 g","12.5 mL","120 mmHg","5 mmol/L","70 kg","37°C","100 IU"])
def test_measurement_extraction_recognizes_common_units(text):
    assert extract_measurements(text)


# ---------------------------------------------------------------------------
# J. Claim verification invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("answer,evidence", [("Diabetes is chronic.",["Diabetes is chronic."]), ("HbA1c reflects glycemic exposure.",["HbA1c reflects glycemic exposure."]), ("The dose is 500 mg.",["The dose is 500 mg."])])
def test_verify_claims_accepts_directly_supported_answer(answer,evidence):
    checks=verify_claims(answer,evidence,["S1"]);assert checks;assert all(c.status not in {"UNSUPPORTED","CONTRADICTED","NUMERIC_MISMATCH"} for c in checks)

@pytest.mark.parametrize("answer,evidence", [("Diabetes is chronic. The moon is blue.",["Diabetes is chronic."]), ("Diabetes is not chronic.",["Diabetes is chronic."]), ("The dose is 600 mg.",["The dose is 500 mg."]), ("Hypertension causes DKA.",["Hypertension is common."])])
def test_verify_claims_rejects_unsupported_or_conflicting_answer(answer,evidence):
    checks=verify_claims(answer,evidence,["S1"]);assert checks;assert any(c.status in {"UNSUPPORTED","WEAK","CONTRADICTED","NUMERIC_MISMATCH"} or c.contradiction for c in checks)

@pytest.mark.parametrize("checks,allow", [([ClaimCheck("supported","S1",0.9,"ENTAILED",False,False)],True), ([ClaimCheck("bad",None,0.0,"UNSUPPORTED",False,False)],False), ([ClaimCheck("bad","S1",0.0,"CONTRADICTED",True,False)],False)])
def test_grounding_decision_matrix(checks,allow):
    assert grounding_decision(checks,min_supported_ratio=0.60)["allow"] is allow

@pytest.mark.parametrize("checks", [[],[ClaimCheck("supported","S1",0.9,"ENTAILED",False,False)],[ClaimCheck("bad",None,0.0,"UNSUPPORTED",False,False)],[ClaimCheck("bad","S1",0.0,"CONTRADICTED",True,False)]])
def test_contradiction_report_is_structurally_stable(checks):
    assert "has_contradiction" in contradiction_report(checks)


# ---------------------------------------------------------------------------
# K. Latency budget semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seconds",[0.05,0.1,0.25,0.5])
def test_latency_budget_expires_when_deadline_passes(seconds):
    with budget_scope(seconds):
        assert remaining() is not None
        import time;time.sleep(seconds+0.02)
        assert exhausted() is True

def test_latency_budget_is_shared_not_reset_for_nested_scopes():
    import time
    with budget_scope(0.30):
        first=remaining();time.sleep(0.10)
        with budget_scope(0.50):
            nested=remaining();assert nested<=first+0.01;time.sleep(0.10)
        assert remaining()<first

def test_latency_budget_allows_fast_operations():
    import time
    with budget_scope(1.0):
        time.sleep(0.01);assert not exhausted()


# ---------------------------------------------------------------------------
# L. Adversarial corpus and helper invariants
# ---------------------------------------------------------------------------

ADVERSARIAL_QUESTIONS=["What is diabetes?","What is HbA1c?","What is DKA?","Why does diabetic ketoacidosis happen?","How does insulin affect glucose?","Compare type 1 and type 2 diabetes.","What is the treatment of DKA?","What are the diagnostic criteria for DKA?","What is the exact dose of metformin?","Which table contains potassium values?","What does Figure 3 show?","What does the document say about renal phosphate handling?","What is the relationship between obesity and diabetes?","What complications are associated with nephropathy?","What about complications?","And the treatment?","والمضاعفات؟","وما هو العلاج؟","Quelles sont les complications du diabète ?","Quel est le site principal de régulation du phosphate ?","ما هو الموقع الرئيسي لتنظيم الفوسفات؟","Quel est le traitement de l'acidocétose diabétique ?"]

@pytest.mark.parametrize("question",ADVERSARIAL_QUESTIONS)
def test_adversarial_question_corpus_has_deterministic_plan(question):
    plan=plan_query(question);assert plan.normalized;assert plan.intent;assert isinstance(plan.entities,tuple);assert isinstance(plan.variants,tuple)

@pytest.mark.parametrize("question",ADVERSARIAL_QUESTIONS)
def test_adversarial_question_corpus_does_not_crash_hard_query_classifier(question):
    plan=plan_query(question);q=SimpleNamespace(primary_intent=plan.intent);assert isinstance(_hard_query(plan,q,question),bool)

@pytest.mark.parametrize("question",ADVERSARIAL_QUESTIONS)
def test_adversarial_question_corpus_extractive_path_is_safe(question):
    assert isinstance(_simple_extractive_answer(question,[hit("The indexed evidence contains a directly relevant sentence about the topic.")]),str)


def test_phosphate_trace_never_creates_followup_metadata_from_standalone_query():
    question="What is the main site of phosphate regulation?";history=[("What is the principal bias follow-up?","irrelevant previous answer")];rewritten=rewrite_follow_up(question,history)
    assert rewritten==question;assert "Follow-up" not in rewritten;assert "Relevant entities" not in rewritten

def test_phosphate_trace_extractive_answer_uses_medical_sentence_not_section_fragment():
    answer=_simple_extractive_answer("What is the main site of phosphate regulation?",[hit("[Section: 3. Régulation :] Le rein est le principal site de régulation de la concentration plasmatique du Pi.")])
    assert "Le rein est le principal site" in answer;assert "[Section:" not in answer;assert "85mg" not in answer;assert "phospholipides" not in answer

def test_phosphate_trace_fast_path_does_not_call_llm():
    llm=FakeLLM(outputs=["slow answer"]);system=SimpleNamespace(llm=llm,settings=fake_settings());answer,path=_answer_with_ladder(system,"What is the main site of phosphate regulation?","Le rein est le principal site de régulation de la concentration plasmatique du Pi.",[hit("Le rein est le principal site de régulation de la concentration plasmatique du Pi.")],"","")
    assert path=="fast_extractive";assert not llm.calls;assert "Le rein est le principal site" in answer

@pytest.mark.parametrize("question",["What is diabetes?","What is HbA1c?","What is DKA?","What is the main site of phosphate regulation?","What about complications?"])
def test_query_plan_is_repeatably_deterministic(question):
    first=plan_query(question).to_dict();second=plan_query(question).to_dict();assert first==second
