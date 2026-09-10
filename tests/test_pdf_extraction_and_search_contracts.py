from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz

from rag_project.parsing.pdf_extractor import PDFExtractor
from rag_project.ingestion.document_classifier import DocumentClassifier
from rag_project.retrieval.hybrid_retriever import HybridRetriever
from rag_project.intelligence.query_intelligence import plan_query
from rag_project.retrieval.query_rewriter import QueryRewriter


@dataclass
class _Hit:
    doc_id: str
    text: str
    score: float
    metadata: dict
    vector_score: float = 0.0
    lexical_score: float = 0.0


class _EmbeddingStub:
    def __init__(self, vector=None):
        self.vector = vector or [0.1, 0.2, 0.3]
        self.calls: list[str] = []

    def embed_query(self, query: str):
        self.calls.append(query)
        return list(self.vector)


class _VectorStoreStub:
    def __init__(self, vector=None, lexical=None):
        self.vector = vector or {
            "ids": ["v1", "v2"],
            "documents": ["Amoxicillin dose is 500 mg twice daily.", "Paracetamol dose is 1 g daily."],
            "metadatas": [
                {"document_id": "doc-amox", "chunk_id": "c1", "page_numbers": [4]},
                {"document_id": "doc-para", "chunk_id": "c2", "page_numbers": [7]},
            ],
            "distances": [0.05, 0.8],
        }
        self.lexical = lexical or {
            "ids": ["v1", "v2"],
            "documents": ["Amoxicillin dose is 500 mg twice daily.", "Paracetamol dose is 1 g daily."],
            "metadatas": [
                {"document_id": "doc-amox", "chunk_id": "c1", "page_numbers": [4]},
                {"document_id": "doc-para", "chunk_id": "c2", "page_numbers": [7]},
            ],
            "distances": [0.1, 1.0],
        }
        self.calls: list[tuple[str, int, dict | None]] = []

    def search(self, query_embedding, candidate_count, where):
        return self.vector

    def search_lexical(self, query, candidate_count, where):
        self.calls.append((query, candidate_count, where))
        return self.lexical


def _make_pdf(path: Path, pages: list[str]) -> None:
    document = fitz.open()
    for text in pages:
        page = document.new_page()
        rect = fitz.Rect(50, 50, page.rect.width - 50, page.rect.height - 50)
        page.insert_textbox(rect, text, fontsize=11)
    document.save(str(path))
    document.close()


def test_pdf_extraction_preserves_page_order_text_and_identity(tmp_path: Path):
    pdf = tmp_path / "medical.pdf"
    _make_pdf(pdf, [
        "Chapter 1\nAmoxicillin is an antibiotic. Dose: 500 mg twice daily.",
        "Chapter 2\nAdverse effects include nausea and diarrhea.",
        "Chapitre 3\nLa dose doit être adaptée à la fonction rénale.",
    ])

    pages = PDFExtractor(ocr_enabled=False).extract(pdf, "doc-123")

    assert len(pages) == 3
    assert [page.page_number for page in pages] == [1, 2, 3]
    assert all(page.document_id == "doc-123" for page in pages)
    assert "Amoxicillin" in pages[0].text
    assert "500 mg" in pages[0].text
    assert "fonction rénale" in pages[2].text
    assert all(page.source_path == str(pdf) for page in pages)


def test_pdf_extraction_cleans_repeated_whitespace(tmp_path: Path):
    pdf = tmp_path / "whitespace.pdf"
    _make_pdf(pdf, ["Drug    dose\n\n\nAmoxicillin     500 mg"])

    page = PDFExtractor(ocr_enabled=False).extract(pdf, "doc-space")[0]

    assert "     " not in page.text
    assert "\n\n\n" not in page.text
    assert "500 mg" in page.text


def test_pdf_extraction_flags_image_only_page_for_ocr_when_disabled(tmp_path: Path):
    image = tmp_path / "page.png"
    from PIL import Image

    Image.new("RGB", (1000, 700), "white").save(image)
    pdf = tmp_path / "scan.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_image(page.rect, filename=str(image))
    document.save(str(pdf))
    document.close()

    extracted = PDFExtractor(ocr_enabled=False).extract(pdf, "scan-doc")[0]

    assert extracted.has_images is True
    assert extracted.ocr_required is True
    assert extracted.ocr_status == "skipped_disabled"
    assert extracted.metadata["ocr_disabled"] is True


def test_pdf_extraction_merges_native_and_ocr_without_duplicate_paragraphs():
    merged = PDFExtractor._merge_native_and_ocr_text(
        "Drug dose 500 mg.\n\nTake twice daily.",
        "Drug dose 500 mg.\n\nTake twice daily.\n\nWith water.",
    )

    assert merged.count("Drug dose 500 mg") == 1
    assert merged.count("Take twice daily") == 1
    assert "With water" in merged


def test_pdf_classifier_distinguishes_text_based_document(tmp_path: Path):
    pdf = tmp_path / "text-book.pdf"
    text = "This is a sufficiently long medical paragraph used to classify this document as text based. " * 3
    _make_pdf(pdf, [text, text])

    result = DocumentClassifier.classify(pdf)

    assert result["page_count"] == 2
    assert result["document_type"] == "text_based"
    assert result["text_pages"] == 2
    assert result["ocr_required"] is False
    assert result["classification_scope"] == "all_pages"


def test_pdf_classifier_marks_image_only_document_for_ocr(tmp_path: Path):
    image = tmp_path / "scan.png"
    from PIL import Image

    Image.new("RGB", (900, 900), "white").save(image)
    pdf = tmp_path / "image-book.pdf"
    document = fitz.open()
    for _ in range(2):
        page = document.new_page()
        page.insert_image(page.rect, filename=str(image))
    document.save(str(pdf))
    document.close()

    result = DocumentClassifier.classify(pdf)

    assert result["page_count"] == 2
    assert result["document_type"] == "scanned_or_ocr_required"
    assert result["ocr_required"] is True


def test_query_planner_handles_english_french_and_arabic_intent():
    english = plan_query("What is the dose of amoxicillin in mg?")
    french = plan_query("Quelle est la dose en mg de l'amoxicilline ?")
    arabic = plan_query("ما هي جرعة الأموكسيسيلين؟")

    assert english.needs_numeric is True
    assert french.needs_numeric is True
    assert arabic.needs_numeric is True
    assert all(plan.normalized for plan in (english, french, arabic))
    assert all(plan.variants for plan in (english, french, arabic))


def test_query_planner_detects_table_figure_and_comparison_signals():
    table = plan_query("What is in the table for dose values?")
    figure = plan_query("What does Figure 3 show?")
    comparison = plan_query("Compare amoxicillin versus cefalexin doses")

    assert table.needs_table is True
    assert figure.needs_figure is True
    assert comparison.intent == "comparison"
    assert comparison.needs_multi_hop is True


def test_query_rewriter_falls_back_safely_without_llm():
    assert QueryRewriter.rewrite("  What is the dose?  ") == "What is the dose?"
    assert QueryRewriter.rewrite("What about it?", [("What is amoxicillin?", "..." )]) == "What is amoxicillin? Follow-up question: What about it?"


def test_hybrid_retriever_returns_ranked_hits_and_calls_embedding():
    embedding = _EmbeddingStub()
    store = _VectorStoreStub()
    retriever = HybridRetriever(store, embedding, lexical_mode="hybrid", vector_weight=0.7)

    hits = retriever.retrieve("amoxicillin dose 500 mg", top_k=2)

    assert len(hits) == 2
    assert hits[0].metadata["chunk_id"] == "c1"
    assert hits[0].score >= hits[1].score
    assert embedding.calls == ["amoxicillin dose 500 mg"]
    assert store.calls


def test_hybrid_retriever_empty_query_is_a_noop():
    embedding = _EmbeddingStub()
    store = _VectorStoreStub()
    retriever = HybridRetriever(store, embedding)

    assert retriever.retrieve("   ") == []
    assert embedding.calls == []
    assert store.calls == []


def test_hybrid_retriever_clamps_top_k_to_safe_bounds():
    embedding = _EmbeddingStub()
    store = _VectorStoreStub()
    retriever = HybridRetriever(store, embedding)

    one = retriever.retrieve("dose", top_k=0)
    many = retriever.retrieve("dose", top_k=1000)

    assert len(one) == 1
    assert len(many) <= 100


def test_hybrid_retriever_lexical_mode_prefers_token_overlap():
    store = _VectorStoreStub(
        vector={"ids": [], "documents": [], "metadatas": [], "distances": []},
        lexical={
            "ids": ["a", "b"],
            "documents": ["warfarin dose adjustment monitoring INR", "unrelated respiratory symptoms"],
            "metadatas": [{"document_id": "d1", "chunk_id": "a"}, {"document_id": "d2", "chunk_id": "b"}],
            "distances": [0.05, 0.9],
        },
    )
    retriever = HybridRetriever(store, _EmbeddingStub(), lexical_mode="lexical", vector_weight=0.0)

    hits = retriever.retrieve("warfarin dose INR", top_k=2)

    assert [hit.metadata["chunk_id"] for hit in hits] == ["a", "b"]
    assert hits[0].lexical_score > hits[1].lexical_score


def test_hybrid_retriever_vector_failure_falls_back_to_lexical():
    class BrokenVectorStore(_VectorStoreStub):
        def search(self, query_embedding, candidate_count, where):
            raise RuntimeError("vector backend unavailable")

    store = BrokenVectorStore(
        vector={"ids": [], "documents": [], "metadatas": [], "distances": []},
        lexical={
            "ids": ["lex1"],
            "documents": ["Metformin is used for type 2 diabetes."],
            "metadatas": [{"document_id": "d1", "chunk_id": "lex1"}],
            "distances": [0.1],
        },
    )
    retriever = HybridRetriever(store, _EmbeddingStub(), lexical_mode="vector", vector_weight=1.0)

    hits = retriever.retrieve("metformin diabetes", top_k=1)

    assert len(hits) == 1
    assert hits[0].metadata["chunk_id"] == "lex1"


def test_hybrid_retriever_handles_missing_metadata_and_distances():
    store = _VectorStoreStub(
        vector={
            "ids": ["x", "y"],
            "documents": ["one", "two"],
            "metadatas": [None, {"document_id": "d2"}],
            "distances": [float("nan"), "not-a-number"],
        },
        lexical={"ids": [], "documents": [], "metadatas": [], "distances": []},
    )
    retriever = HybridRetriever(store, _EmbeddingStub(), lexical_mode="vector", vector_weight=1.0)

    hits = retriever.retrieve("one", top_k=2)

    assert len(hits) == 2
    assert all(isinstance(hit.metadata, dict) for hit in hits)
    assert all(hit.score >= 0.0 for hit in hits)


def test_hybrid_retriever_forwards_metadata_filter_to_lexical_backend():
    embedding = _EmbeddingStub()
    store = _VectorStoreStub()
    retriever = HybridRetriever(store, embedding)
    where = {"document_id": "doc-amox"}

    retriever.retrieve("amoxicillin", top_k=2, where=where)

    assert store.calls[-1][2] == where


def test_search_scores_are_finite_even_with_extreme_distance_values():
    store = _VectorStoreStub(
        vector={
            "ids": ["x"],
            "documents": ["amoxicillin 500 mg"],
            "metadatas": [{"document_id": "d1", "chunk_id": "x"}],
            "distances": [-999999999.0],
        },
        lexical={"ids": [], "documents": [], "metadatas": [], "distances": []},
    )
    retriever = HybridRetriever(store, _EmbeddingStub(), lexical_mode="vector", vector_weight=1.0)

    hits = retriever.retrieve("amoxicillin", top_k=1)

    assert len(hits) == 1
    assert hits[0].score >= 0.0
    assert hits[0].score < float("inf")
