"""Final overrides for strict benchmark execution."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from rag_project.testing.advanced_phases import _cleanup_store, _embedding, _result

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"
CORPUS = TESTS / "support" / "gold_sets" / "diagnostic_independent_corpus.jsonl"
GOLD = TESTS / "support" / "gold_sets" / "diagnostic_independent_gold.jsonl"


def phase16_independent_gold(phase: Any):
    result = _result(phase)
    tmp = None
    try:
        from rag_project.ingestion.document_models import PageExtraction
        from rag_project.chunking.semantic_chunker import SemanticChunker
        from rag_project.storage.vector_store import VectorStore
        if not CORPUS.exists() or not GOLD.exists():
            raise FileNotFoundError("independent corpus and separate gold labels are required")
        corpus = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
        gold = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
        corpus_ids = {row["doc_id"] for row in corpus}
        if not corpus_ids or any(not row.get("text") for row in corpus):
            raise ValueError("independent corpus must contain non-empty document text")
        if any(not set(row.get("expected_doc_ids", [])) <= corpus_ids for row in gold):
            raise ValueError("gold labels reference missing corpus documents")
        tmp = Path(tempfile.mkdtemp(prefix="rag_independent_gold_v2_"))
        store = VectorStore(tmp / "index", collection_name="independent_gold_v2")
        pages = [PageExtraction(document_id=row["doc_id"], file_name=f"{row['doc_id']}.pdf", page_index=0, page_number=1, text=row["text"], extraction_method="native_text", ocr_required=False, ocr_status="not_required", ocr_confidence=1.0, page_type="text", image_count=0, table_count=0, has_images=False, blocks=[row["text"]], metadata={"version_id":"v1"}, source_path=f"{row['doc_id']}.pdf", quality_score=1.0, routing_decision="native", table_ids=[], figure_ids=[], table_texts=[], figure_captions=[], headings=[]) for row in corpus]
        chunks = SemanticChunker(chunk_size=220, chunk_overlap=30).chunk_pages(pages)
        docs, metas, embeddings, ids = [], [], [], []
        for chunk in chunks:
            cid = f"{chunk.doc_id}:{chunk.chunk_index}"
            docs.append(chunk.text)
            metas.append({**dict(chunk.metadata or {}), "document_id": chunk.doc_id, "chunk_id": cid, "version_id": "v1"})
            embeddings.append(_embedding(chunk.text))
            ids.append(cid)
        store.add_documents(docs, metas, embeddings, ids)
        rows = []
        for case in gold:
            question = str(case["question"])
            expected = set(case["expected_doc_ids"])
            lexical = store.search_lexical(question, n_results=min(8, len(docs)))
            semantic = store.search(_embedding(question), n_results=min(8, len(docs)))
            lexical_ids = [str(x) for x in (lexical.get("ids") or [[]])[0]]
            semantic_ids = [str(x) for x in (semantic.get("ids") or [[]])[0]]
            retrieved_docs = {value.split(":", 1)[0] for value in lexical_ids + semantic_ids}
            hit = bool(expected & retrieved_docs)
            rows.append({"id": case["id"], "hit": hit, "expected_doc_ids": sorted(expected), "lexical_top": lexical_ids[:3], "semantic_top": semantic_ids[:3]})
        recall = sum(int(row["hit"]) for row in rows) / max(1, len(rows))
        result.details = {"mode": "independent_corpus_local", "corpus_path": str(CORPUS.relative_to(ROOT)), "gold_path": str(GOLD.relative_to(ROOT)), "case_count": len(rows), "corpus_document_count": len(corpus), "indexed_chunk_count": len(chunks), "retrieval_recall": recall, "gold_labels_independent_of_corpus_text": True, "clinical_correctness_claimed": False, "results": rows}
        result.score = round(recall, 3)
        result.status = "PASS" if recall >= 0.80 else "FAIL"
        if result.status == "FAIL":
            result.failures.append({"location": str(GOLD.relative_to(ROOT)), "exception": "IndependentGoldenRecallFailure", "message": f"recall={recall:.3f}"})
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "phase 16 independent benchmark", "exception": type(exc).__name__, "message": str(exc)})
    finally:
        if tmp is not None:
            _cleanup_store(tmp)
    result.duration_s = 0.0 if not result.started_at else __import__('time').time() - result.started_at
    return result
