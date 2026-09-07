from __future__ import annotations

from pathlib import Path

from rag_project.storage.vector_store import VectorStore
from rag_project.utils.text_utils import (
    detect_language,
    keyword_proximity_score,
    meaningful_tokens,
    normalize_arabic,
)
from rag_project.retrieval.hybrid_retriever import RetrievalHit
from rag_project.reranking.reranker import Reranker
from rag_project.app.rag_system import RAGSystem
from rag_project.evaluation.metrics import (
    compute_keyphrase_precision,
    is_negative_refusal,
)
from rag_project.evaluation.dataset import load_dataset


def test_query_normalization_removes_question_stopwords_and_keeps_unicode():
    assert meaningful_tokens("What is DIABÉTIQUE nutrition?") == [
        "diabétique",
        "nutrition",
    ]


def test_proximity_prefers_terms_that_are_close():
    close = keyword_proximity_score("diabetes nutrition", "diabetes nutrition guidance")
    distant = keyword_proximity_score(
        "diabetes nutrition",
        "diabetes " + "unrelated " * 40 + "nutrition guidance",
    )
    assert close > distant


def test_multilingual_normalization_and_language_hints():
    assert normalize_arabic("إِنَّ السُّكَّر") == "ان السكر"
    assert "السكر" in meaningful_tokens("إِنَّ السُّكَّر")
    assert detect_language("ما هي التغذية؟") == "ar"
    assert detect_language("Quelle est la méthode ?") == "fr"
    assert detect_language("What is the method?") == "en"


def test_multilingual_keyphrase_matching_across_supported_scripts():
    cases = [
        (
            "Insulin therapy requires glucose monitoring.",
            "insulin glucose",
            ["insulin", "glucose"],
        ),
        (
            "La nutrition équilibrée réduit le risque.",
            "nutrition risque",
            ["nutrition", "risque"],
        ),
        (
            "تتطلب معالجة السكري مراقبة السكر.",
            "معالجة السكري والسكر",
            ["السكري", "السكر"],
        ),
    ]
    for answer, evidence, keyphrases in cases:
        assert compute_keyphrase_precision(answer, evidence, keyphrases) == 1.0


def test_multilingual_negative_refusal_detection():
    assert is_negative_refusal("I cannot find sufficient evidence.")
    assert is_negative_refusal("Je ne sais pas d'information disponible.")
    assert is_negative_refusal("لا توجد معلومات كافية في الوثيقة.")


def test_review_only_dataset_filter_excludes_drafts():
    reviewed = load_dataset("v4", only_reviewed=True)
    assert reviewed == []


def test_retrieval_confidence_reports_low_and_high_evidence():
    low = RetrievalHit("doc", "text", {}, 0.1, 0.0, 0.0)
    high = RetrievalHit("doc", "text", {}, 0.8, 0.0, 0.0)
    second = RetrievalHit("doc", "text", {}, 0.4, 0.0, 0.0)
    assert Reranker.confidence([low])["level"] == "low"
    assert Reranker.confidence([high, second])["level"] == "high"


def test_grounding_guard_falls_back_for_unsupported_answer():
    hit = RetrievalHit(
        "doc",
        "Insulin therapy requires monitoring blood glucose.",
        {"file_name": "guide.pdf", "page_numbers": [4]},
        0.8,
        0.8,
        0.8,
    )
    answer, evidence_overlap, query_coverage, used_fallback = (
        RAGSystem.apply_grounding_guard("The weather is sunny.", "How is insulin monitored?", [hit])
    )
    assert used_fallback is True
    assert "[S1]" in answer
    assert evidence_overlap > 0.0
    assert query_coverage > 0.0


def test_low_quality_queries_trigger_abstention_and_clarification():
    analysis = RAGSystem.assess_query_quality("How are sont and plus related in the text?")
    assert analysis["query_quality"] == "LOW_QUALITY_QUERY"
    assert analysis["should_abstain"] is True
    assert "rephrase" in analysis["clarification"].lower()


def test_evidence_alignment_flags_related_but_non_answering_material():
    hits = [
        RetrievalHit(
            "doc",
            "This section discusses patient monitoring and treatment planning in general.",
            {"file_name": "guide.pdf", "page_numbers": [5]},
            0.8,
            0.8,
            0.8,
        ),
        RetrievalHit(
            "doc",
            "The chapter reviews insulin dosing and follow-up scheduling.",
            {"file_name": "guide.pdf", "page_numbers": [6]},
            0.7,
            0.7,
            0.7,
        ),
    ]
    alignment = RAGSystem.evaluate_evidence_alignment("Does this directly answer the exact dosage question?", hits)
    assert alignment["decision"] in {"RELATED_BUT_NOT_ANSWERING", "PARTIALLY_SUPPORTED"}


def test_building_chunks_are_not_searchable_until_activated(tmp_path: Path):
    store = VectorStore(tmp_path / "vectors")
    metadata = {
        "document_id": "doc",
        "chunk_id": "doc-0",
        "page_numbers": [1],
        "version_id": "v1",
        "index_state": "BUILDING",
    }
    store.add_lexical_documents(["insulin nutrition guidance"], [metadata], ["doc-0"])
    assert store.search_lexical("insulin nutrition")["ids"] == [[]]
    store.set_document_index_state("doc", "READY")
    assert store.search_lexical("insulin nutrition")["ids"] == [["doc-0"]]
