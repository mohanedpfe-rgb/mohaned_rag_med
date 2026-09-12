from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from rag_project.testing.advanced_phases import _embedding, _store_fixture, _cleanup_store
from rag_project.testing.deep_diagnostics import PhaseResult

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "tests" / "support" / "gold_sets" / "diagnostic_independent_corpus.jsonl"
GOLD = ROOT / "tests" / "support" / "gold_sets" / "diagnostic_independent_gold.jsonl"


def _result(phase: Any) -> PhaseResult:
    return PhaseResult(phase.number, phase.key, phase.name, started_at=time.time())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _build_independent_chunks() -> list[Any]:
    from rag_project.chunking.semantic_chunker import SemanticChunker
    from rag_project.ingestion.document_models import PageExtraction

    chunks: list[Any] = []
    corpus = _load_jsonl(CORPUS)
    for index, row in enumerate(corpus):
        doc_id = str(row["doc_id"])
        page = PageExtraction(
            document_id=doc_id,
            file_name=f"{doc_id}.pdf",
            page_index=0,
            page_number=1,
            text=str(row["text"]),
            extraction_method="native_text",
            ocr_required=False,
            ocr_status="not_required",
            ocr_confidence=1.0,
            page_type="text",
            image_count=0,
            table_count=0,
            has_images=False,
            blocks=[str(row["text"])],
            metadata={"version_id": "independent-v1", "source": "independent-corpus", "corpus_row": index},
            source_path=f"{doc_id}.pdf",
            quality_score=1.0,
            routing_decision="native",
            table_ids=[],
            figure_ids=[],
            table_texts=[],
            figure_captions=[],
            headings=[],
        )
        chunks.extend(SemanticChunker(chunk_size=320, chunk_overlap=40).chunk_pages([page]))
    if not chunks:
        raise AssertionError("independent corpus produced no production chunks")
    return chunks


def phase9_independent_retrieval(phase: Any) -> PhaseResult:
    result = _result(phase)
    tmp = None
    try:
        corpus = _load_jsonl(CORPUS)
        gold = _load_jsonl(GOLD)
        corpus_ids = {str(row["doc_id"]) for row in corpus}
        gold_ids = {str(doc_id) for row in gold for doc_id in row.get("expected_doc_ids", [])}
        if not corpus or not gold:
            raise RuntimeError("independent corpus/gold set is empty")
        if not gold_ids.issubset(corpus_ids):
            raise RuntimeError(f"gold references missing corpus ids: {sorted(gold_ids - corpus_ids)}")

        chunks = _build_independent_chunks()
        store, tmp = _store_fixture(chunks)
        query_rows: list[dict[str, Any]] = []
        lexical_hits = 0
        semantic_hits = 0
        lexical_rr: list[float] = []
        semantic_rr: list[float] = []

        for case in gold:
            question = str(case["question"])
            expected = {str(doc_id) for doc_id in case.get("expected_doc_ids", [])}
            lexical = store.search_lexical(question, n_results=min(5, max(1, store.lexical_count())))
            semantic = store.search(_embedding(question), n_results=min(5, max(1, store.count())))
            lexical_ids = [str(value) for value in (lexical.get("ids") or [[]])[0]]
            semantic_ids = [str(value) for value in (semantic.get("ids") or [[]])[0]]
            def doc_id(item_id: str) -> str:
                return item_id.split(":", 1)[0]
            lexical_docs = [doc_id(value) for value in lexical_ids]
            semantic_docs = [doc_id(value) for value in semantic_ids]
            lex_rank = next((index + 1 for index, value in enumerate(lexical_docs) if value in expected), None)
            sem_rank = next((index + 1 for index, value in enumerate(semantic_docs) if value in expected), None)
            lex_hit = bool(lex_rank is not None and lex_rank <= 3)
            sem_hit = bool(sem_rank is not None and sem_rank <= 3)
            lexical_hits += int(lex_hit)
            semantic_hits += int(sem_hit)
            if lex_rank:
                lexical_rr.append(1 / lex_rank)
            if sem_rank:
                semantic_rr.append(1 / sem_rank)
            query_rows.append({
                "id": case["id"],
                "question": question,
                "expected_doc_ids": sorted(expected),
                "lexical_rank": lex_rank,
                "semantic_rank": sem_rank,
                "lexical_recall_at_3": lex_hit,
                "semantic_recall_at_3": sem_hit,
            })

        filtered = store.search_lexical("diabetes", n_results=5, where={"document_id": "__missing__"})
        filter_empty = not bool((filtered.get("ids") or [[]])[0])
        case_count = len(query_rows)
        lexical_recall = lexical_hits / case_count
        semantic_recall = semantic_hits / case_count
        result.details = {
            "evidence_level": "independent_corpus_and_gold_retrieval",
            "corpus_path": str(CORPUS.relative_to(ROOT)),
            "gold_path": str(GOLD.relative_to(ROOT)),
            "gold_labels_independent_of_corpus_text": True,
            "production_chunker": "SemanticChunker.chunk_pages",
            "production_store": "VectorStore",
            "query_count": case_count,
            "queries": query_rows,
            "lexical_recall_at_3": lexical_recall,
            "semantic_recall_at_3": semantic_recall,
            "lexical_mrr": statistics.fmean(lexical_rr) if lexical_rr else 0.0,
            "semantic_mrr": statistics.fmean(semantic_rr) if semantic_rr else 0.0,
            "metadata_filter_correct": filter_empty,
            "vector_count": store.count(),
            "lexical_count": store.lexical_count(),
        }
        result.score = round((lexical_recall + semantic_recall + int(filter_empty)) / 3, 3)
        result.status = "PASS" if lexical_recall >= 0.80 and semantic_recall >= 0.80 and filter_empty else "FAIL"
        if result.status == "FAIL":
            result.failures.append({
                "location": "VectorStore lexical/semantic retrieval against independent labels",
                "exception": "IndependentRetrievalFailure",
                "message": json.dumps(result.details, sort_keys=True),
            })
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "phase 9 independent retrieval probe", "exception": type(exc).__name__, "message": str(exc)})
    finally:
        if tmp is not None:
            _cleanup_store(tmp)
    result.duration_s = round(time.time() - result.started_at, 3)
    return result
