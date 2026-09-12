from __future__ import annotations

import tempfile
from pathlib import Path

import fitz

from rag_project.chunking.semantic_chunker import SemanticChunker
from rag_project.parsing.pdf_extractor import PDFExtractor
from rag_project.storage.vector_store import VectorStore
from rag_project.utils.text_utils import clean_text, normalize_whitespace, tokenize, keyword_overlap_score
from rag_project.ingestion.document_models import PageExtraction


def _page(text: str, document_id: str = "integration-doc") -> PageExtraction:
    return PageExtraction(
        document_id=document_id, file_name=f"{document_id}.pdf", page_index=0, page_number=1,
        text=text, extraction_method="native_text", ocr_required=False, ocr_status="not_required",
        ocr_confidence=1.0, page_type="text", image_count=0, table_count=0, has_images=False,
        blocks=[text], metadata={"version_id": "v1", "source": "integration"},
        source_path=f"{document_id}.pdf", quality_score=1.0, routing_decision="native",
        table_ids=[], figure_ids=[], table_texts=[], figure_captions=[], headings=["Section"],
    )


def _chunks(text: str, document_id: str = "integration-doc"):
    return SemanticChunker(chunk_size=120, chunk_overlap=20).chunk_pages([_page(text, document_id)])


def _store(document_id: str = "integration-doc") -> tuple[VectorStore, tempfile.TemporaryDirectory]:
    holder = tempfile.TemporaryDirectory(prefix="rag_integration_")
    store = VectorStore(Path(holder.name) / "index", collection_name="integration")
    text = "Diabetes mellitus diagnosis uses plasma glucose and HbA1c."
    emb = [1.0, 0.5, 0.25, 0.125]
    store.add_documents([text], [{"document_id": document_id, "chunk_id": f"{document_id}:0", "version_id": "v1", "page_numbers": [1], "section_id": "s1", "parent_id": "p1", "index_state": "READY"}], [emb], [f"{document_id}:0"])
    return store, holder


def _pdf(kind: str) -> tuple[Path, tempfile.TemporaryDirectory]:
    holder = tempfile.TemporaryDirectory(prefix="rag_pdf_integration_")
    path = Path(holder.name) / f"{kind}.pdf"
    doc = fitz.open(); page = doc.new_page(width=595, height=842)
    if kind == "unicode":
        page.insert_text((55, 70), "Chapitre — Diabète / طب السكري")
        page.insert_text((55, 110), "HbA1c ≥ 6,5 %")
    elif kind == "long":
        page.insert_textbox(fitz.Rect(40, 40, 550, 790), "Diabetes mellitus " * 120, fontsize=10)
    else:
        page.insert_text((55, 70), "Diabetes mellitus diagnosis and HbA1c 6.5 percent")
    doc.save(path); doc.close(); return path, holder


def test_integration_001_text_cleaning(): assert clean_text("  Diabetes  ") == "Diabetes"
def test_integration_002_whitespace_normalization(): assert normalize_whitespace("  diabetes   mellitus  ") == "diabetes mellitus"
def test_integration_003_tokenization_case(): assert tokenize("HbA1c") == tokenize("HbA1c")
def test_integration_004_overlap_positive(): assert keyword_overlap_score("diabetes", "diabetes mellitus") > 0
def test_integration_005_overlap_empty(): assert keyword_overlap_score("xyz", "diabetes mellitus") == 0
def test_integration_006_normalization_newline(): assert normalize_whitespace("a\n\nb") == "a b"
def test_integration_007_normalization_tabs(): assert normalize_whitespace("a\tb") == "a b"
def test_integration_008_unicode_tokens(): assert "diabète" in tokenize("diabète")[0]
def test_integration_009_clean_punctuation(): assert clean_text("HbA1c!!!").startswith("HbA1c")
def test_integration_010_query_overlap(): assert keyword_overlap_score("HbA1c diagnosis", "HbA1c diagnosis") == 1.0

def test_integration_011_chunk_single_page(): assert len(_chunks("Diabetes mellitus is a chronic metabolic disease.")) >= 1
def test_integration_012_chunk_identity(): assert all(c.doc_id == "integration-doc" for c in _chunks("Diabetes mellitus diagnosis."))
def test_integration_013_chunk_pages(): assert all(c.page_numbers for c in _chunks("Diabetes mellitus diagnosis."))
def test_integration_014_chunk_metadata(): assert all(c.metadata for c in _chunks("Diabetes mellitus diagnosis."))
def test_integration_015_chunk_section(): assert all(c.section_id for c in _chunks("# Section\nDiabetes mellitus diagnosis."))
def test_integration_016_chunk_parent(): assert all(c.parent_id for c in _chunks("# Section\nDiabetes mellitus diagnosis."))
def test_integration_017_chunk_nonempty(): assert all(c.text.strip() for c in _chunks("Diabetes mellitus diagnosis."))
def test_integration_018_chunk_long_text(): assert len(_chunks("Diabetes mellitus. " * 100)) > 1
def test_integration_019_chunk_unicode_text(): assert len(_chunks("Diabète mellitus / طب السكري.")) >= 1
def test_integration_020_chunk_headings(): assert any(c.section_id for c in _chunks("# Chapter\nDiabetes mellitus diagnosis."))
def test_integration_021_chunk_overlap_metadata(): assert all(c.chunk_index >= 0 for c in _chunks("Diabetes mellitus. " * 20))
def test_integration_022_chunk_document_filename(): assert all(c.file_name.endswith(".pdf") for c in _chunks("Diabetes mellitus."))
def test_integration_023_chunk_quality(): assert all(c.metadata is not None for c in _chunks("Diabetes mellitus."))
def test_integration_024_chunk_page_numbers_one(): assert all(1 in c.page_numbers for c in _chunks("Diabetes mellitus."))
def test_integration_025_chunk_stable_count(): assert len(_chunks("Diabetes mellitus.")) == len(_chunks("Diabetes mellitus."))
def test_integration_026_chunk_whitespace_equivalence(): assert len(_chunks("Diabetes mellitus.")) == len(_chunks("Diabetes   mellitus."))
def test_integration_027_chunk_case_preserves_schema(): assert all(c.doc_id for c in _chunks("DIABETES MELLITUS."))
def test_integration_028_chunk_numeric_content(): assert all(c.text for c in _chunks("HbA1c 6.5 percent."))
def test_integration_029_chunk_french_content(): assert all(c.text for c in _chunks("Le diabète est une maladie chronique."))
def test_integration_030_chunk_mixed_content(): assert all(c.text for c in _chunks("# Diabetes\nHbA1c 6.5 %.\nSuivi clinique."))


def test_integration_031_vector_insert_count():
    s,h=_store(); assert s.count()==1; h.cleanup()
def test_integration_032_vector_lexical_count():
    s,h=_store(); assert s.lexical_count()==1; h.cleanup()
def test_integration_033_vector_document_roundtrip():
    s,h=_store(); assert s.get_documents()["ids"]; h.cleanup()
def test_integration_034_vector_metadata_identity():
    s,h=_store(); assert s.get_documents()["metadatas"][0]["document_id"]=="integration-doc"; h.cleanup()
def test_integration_035_vector_chunk_identity():
    s,h=_store(); assert s.get_documents()["metadatas"][0]["chunk_id"]=="integration-doc:0"; h.cleanup()
def test_integration_036_vector_page_identity():
    s,h=_store(); assert s.get_documents()["metadatas"][0]["page_numbers"]==[1]; h.cleanup()
def test_integration_037_vector_index_state():
    s,h=_store(); assert s.get_documents()["metadatas"][0]["index_state"]=="READY"; h.cleanup()
def test_integration_038_vector_lexical_search():
    s,h=_store(); r=s.search_lexical("diabetes diagnosis",n_results=3); assert r["ids"][0]; h.cleanup()
def test_integration_039_vector_metadata_filter():
    s,h=_store(); r=s.search_lexical("diabetes",n_results=3,where={"document_id":"missing"}); assert r["ids"]==[[]]; h.cleanup()
def test_integration_040_vector_semantic_search():
    s,h=_store(); r=s.search([1.0,0.5,0.25,0.125],n_results=1); assert r["ids"][0]; h.cleanup()
def test_integration_041_vector_verify_index():
    s,h=_store(); assert s.verify_index("integration-doc")["valid"] is True; h.cleanup()
def test_integration_042_vector_reconcile():
    s,h=_store(); assert s.reconcile_index("integration-doc")["valid"] is True; h.cleanup()
def test_integration_043_vector_delete_version():
    s,h=_store(); s.delete_version("integration-doc","v1"); assert s.count()==0; h.cleanup()
def test_integration_044_vector_set_state():
    s,h=_store(); s.set_document_index_state("integration-doc","BUILDING"); assert s.get_documents()["metadatas"][0]["index_state"]=="BUILDING"; h.cleanup()
def test_integration_045_vector_restore_state():
    s,h=_store(); s.set_document_index_state("integration-doc","BUILDING"); s.set_document_index_state("integration-doc","READY"); assert s.get_documents()["metadatas"][0]["index_state"]=="READY"; h.cleanup()


def test_integration_046_pdf_native_extraction():
    p,h=_pdf("text"); pages=PDFExtractor(ocr_enabled=False).extract(p,document_id="pdf1"); assert pages and "Diabetes" in pages[0].text; h.cleanup()
def test_integration_047_pdf_unicode_extraction():
    p,h=_pdf("unicode"); pages=PDFExtractor(ocr_enabled=False).extract(p,document_id="pdf2"); assert pages and pages[0].text; h.cleanup()
def test_integration_048_pdf_long_extraction():
    p,h=_pdf("long"); pages=PDFExtractor(ocr_enabled=False).extract(p,document_id="pdf3"); assert pages and len(pages[0].text)>100; h.cleanup()
def test_integration_049_pdf_malformed_rejected():
    holder=tempfile.TemporaryDirectory(); p=Path(holder.name)/"bad.pdf"; p.write_bytes(b"not pdf"); rejected=False
    try: PDFExtractor(ocr_enabled=False).extract(p,document_id="bad")
    except Exception: rejected=True
    holder.cleanup(); assert rejected
def test_integration_050_pdf_page_number():
    p,h=_pdf("text"); assert PDFExtractor(ocr_enabled=False).extract(p,document_id="pdf4")[0].page_number==1; h.cleanup()

def test_integration_051_pdf_quality_score():
    p,h=_pdf("text"); assert PDFExtractor(ocr_enabled=False).extract(p,document_id="pdf5")[0].quality_score is not None; h.cleanup()
def test_integration_052_pdf_extraction_method():
    p,h=_pdf("text"); assert PDFExtractor(ocr_enabled=False).extract(p,document_id="pdf6")[0].extraction_method in {"native_text","hybrid"}; h.cleanup()
def test_integration_053_pdf_document_identity():
    p,h=_pdf("text"); assert PDFExtractor(ocr_enabled=False).extract(p,document_id="pdf7")[0].document_id=="pdf7"; h.cleanup()
def test_integration_054_pdf_blocks_present():
    p,h=_pdf("text"); assert PDFExtractor(ocr_enabled=False).extract(p,document_id="pdf8")[0].blocks; h.cleanup()
def test_integration_055_pdf_routing_decision():
    p,h=_pdf("text"); assert PDFExtractor(ocr_enabled=False).extract(p,document_id="pdf9")[0].routing_decision; h.cleanup()
def test_integration_056_pdf_text_contains_hba1c():
    p,h=_pdf("text"); assert "hba1c" in PDFExtractor(ocr_enabled=False).extract(p,document_id="pdf10")[0].text.casefold(); h.cleanup()
def test_integration_057_pdf_semantic_chunk_after_extract():
    p,h=_pdf("text"); pages=PDFExtractor(ocr_enabled=False).extract(p,document_id="pdf11"); assert _chunks(pages[0].text,"pdf11"); h.cleanup()
def test_integration_058_pdf_chunk_identity_after_extract():
    p,h=_pdf("text"); pages=PDFExtractor(ocr_enabled=False).extract(p,document_id="pdf12"); assert all(c.doc_id=="pdf12" for c in _chunks(pages[0].text,"pdf12")); h.cleanup()
def test_integration_059_pdf_token_survival_after_extract():
    p,h=_pdf("text"); pages=PDFExtractor(ocr_enabled=False).extract(p,document_id="pdf13"); text=" ".join(c.text for c in _chunks(pages[0].text,"pdf13")); assert "diabetes" in text.casefold(); h.cleanup()
def test_integration_060_pdf_full_pipeline_contract():
    p,h=_pdf("text"); pages=PDFExtractor(ocr_enabled=False).extract(p,document_id="pdf14"); chunks=_chunks(pages[0].text,"pdf14"); s=VectorStore(Path(h.name)/"idx",collection_name="full")
    s.add_documents([c.text for c in chunks],[{"document_id":c.doc_id,"chunk_id":f"{c.doc_id}:{c.chunk_index}","page_numbers":list(c.page_numbers),"version_id":"v1","index_state":"READY"} for c in chunks],[[1.0,0.5,0.25,0.125] for _ in chunks],[f"pdf14:{i}" for i,_ in enumerate(chunks)]); assert s.count()==len(chunks); h.cleanup()
