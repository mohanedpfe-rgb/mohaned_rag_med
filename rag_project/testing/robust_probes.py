"""Production-oriented representation and retrieval probes."""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from .deep_diagnostics import PhaseResult
from .advanced_phases import _cleanup_store, _embedding, _fixture_chunks, _fixture_pages, _store_fixture, _survival, _token_set, _result

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "tests" / "support" / "gold_sets" / "diagnostic_independent_gold.jsonl"


def information_loss(phase: Any) -> PhaseResult:
    result = _result(phase)
    try:
        pages = _fixture_pages()
        chunks = _fixture_chunks(pages)
        store, tmp = _store_fixture(chunks)
        try:
            stored = store.get_documents()
            vector_docs = [str(v or "") for v in stored.get("documents", [])]
            vector_meta = [dict(v or {}) for v in stored.get("metadatas", [])]
            import sqlite3
            with sqlite3.connect(store.lexical_database) as connection:
                lexical_rows = connection.execute("SELECT document, metadata FROM lexical_documents ORDER BY id").fetchall()
            source_text = "\n".join(page.text for page in pages)
            chunk_text = "\n".join(chunk.text for chunk in chunks)
            lexical_text = "\n".join(str(row[0]) for row in lexical_rows)
            vector_fields = {field for meta in vector_meta for field in meta}
            lexical_fields: set[str] = set()
            for _, raw in lexical_rows:
                try:
                    lexical_fields |= set(json.loads(raw).keys())
                except Exception:
                    pass
            field_survival = {
                "document_id": {"source": True, "chunk": all(bool(c.doc_id) for c in chunks), "vector": "document_id" in vector_fields, "lexical": "document_id" in lexical_fields},
                "page_numbers": {"source": all(page.page_number is not None for page in pages), "chunk": all(bool(c.page_numbers) for c in chunks), "vector": "page_numbers" in vector_fields, "lexical": "page_numbers" in lexical_fields},
                "chapter_section_parent": {"source": True, "chunk": all(bool(c.section_id and c.parent_id) for c in chunks), "vector": all(m.get("section_id") and m.get("parent_id") for m in vector_meta), "lexical": all("section_id" in json.loads(raw) and "parent_id" in json.loads(raw) for _, raw in lexical_rows)},
                "table_figure_identity": {"source": bool(any(page.table_ids for page in pages) and any(page.figure_ids for page in pages)), "chunk": bool(any(c.table_id for c in chunks) and any(c.figure_id for c in chunks)), "vector": bool(any(m.get("table_id") for m in vector_meta) and any(m.get("figure_id") for m in vector_meta)), "lexical": bool(any("table_id" in json.loads(raw) for _, raw in lexical_rows) and any("figure_id" in json.loads(raw) for _, raw in lexical_rows))},
                "quality_ocr": {"source": all(page.quality_score is not None and page.ocr_status is not None for page in pages), "chunk": all(c.metadata.get("quality_score") is not None and c.metadata.get("ocr_status") is not None for c in chunks), "vector": all(m.get("quality_score") is not None and m.get("ocr_status") is not None for m in vector_meta), "lexical": all("quality_score" in json.loads(raw) and "ocr_status" in json.loads(raw) for _, raw in lexical_rows)},
                "index_state": {"vector": all(m.get("index_state") == "READY" for m in vector_meta), "lexical": all(json.loads(raw).get("index_state") == "READY" for _, raw in lexical_rows)},
            }
            failed_fields = [name for name, surfaces in field_survival.items() if not all(bool(value) for value in surfaces.values())]
            measurements = {
                "source_to_chunk_token_survival": _survival(_token_set(source_text), _token_set(chunk_text)),
                "chunk_to_vector_document_token_survival": _survival(_token_set(chunk_text), _token_set(" ".join(vector_docs))),
                "vector_to_lexical_document_survival": _survival(_token_set(" ".join(vector_docs)), _token_set(lexical_text)),
            }
            failed_text = [name for name, value in measurements.items() if value < 0.90]
            failures = [*failed_fields, *failed_text]
            result.details = {"representation_chain": ["PageExtraction", "Chunk", "VectorStore", "SQLite lexical"], "field_survival": field_survival, "token_survival": measurements, "failed_invariants": failures, "expected_field_semantics": "fields are checked only at layers where the production schema owns them; inherited/vector-only fields are not incorrectly required on PageExtraction or Chunk objects"}
            result.score = round(1.0 - min(1.0, len(failures) / max(1, len(field_survival) + len(measurements))), 3)
            result.status = "PASS" if not failures else "FAIL"
            if failures:
                result.failures.append({"location": "production representation chain", "exception": "InformationLossFailure", "message": f"failed invariants: {failures}"})
        finally:
            _cleanup_store(tmp)
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "rag_project/testing/robust_probes.py", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


def _load_independent_gold() -> list[dict[str, Any]]:
    if not GOLD.exists():
        raise FileNotFoundError(GOLD)
    rows = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 3:
        raise RuntimeError("independent gold set is too small")
    return rows


def _metadata_by_id(store: Any) -> dict[str, dict[str, Any]]:
    payload = store.get_documents()
    ids = payload.get("ids") or []
    metadatas = payload.get("metadatas") or []
    return {str(item_id): dict(meta or {}) for item_id, meta in zip(ids, metadatas, strict=False)}


def retrieval_microscope(phase: Any) -> PhaseResult:
    result = _result(phase)
    try:
        chunks = _fixture_chunks()
        gold = _load_independent_gold()
        store, tmp = _store_fixture(chunks)
        try:
            metadata = _metadata_by_id(store)
            metrics: list[dict[str, Any]] = []
            for case in gold:
                expected_doc_ids = {str(value) for value in case.get("expected_doc_ids", [])}
                if not expected_doc_ids:
                    raise RuntimeError(f"gold case {case.get('id')} has no expected_doc_ids")
                query = str(case["question"])
                lexical = store.search_lexical(query, n_results=5)
                semantic = store.search(_embedding(query), n_results=5)
                lexical_ids = [str(v) for v in (lexical.get("ids") or [[]])[0]]
                semantic_ids = [str(v) for v in (semantic.get("ids") or [[]])[0]]
                lexical_docs = {str((metadata.get(item) or {}).get("document_id")) for item in lexical_ids if item in metadata}
                semantic_docs = {str((metadata.get(item) or {}).get("document_id")) for item in semantic_ids if item in metadata}
                lex_rank = next((i + 1 for i, item in enumerate(lexical_ids) if str((metadata.get(item) or {}).get("document_id")) in expected_doc_ids), None)
                sem_rank = next((i + 1 for i, item in enumerate(semantic_ids) if str((metadata.get(item) or {}).get("document_id")) in expected_doc_ids), None)
                metrics.append({"id": case["id"], "question": query, "expected_doc_ids": sorted(expected_doc_ids), "lexical_hit": bool(expected_doc_ids & lexical_docs), "semantic_hit": bool(expected_doc_ids & semantic_docs), "lexical_rank": lex_rank, "semantic_rank": sem_rank, "lexical_recall_at_3": bool(lex_rank and lex_rank <= 3), "semantic_recall_at_3": bool(sem_rank and sem_rank <= 3)})
            filtered = store.search_lexical("diabetes", n_results=5, where={"document_id": "does-not-exist"})
            filter_ok = not bool((filtered.get("ids") or [[]])[0])
            lexical_recall = statistics.fmean([int(row["lexical_recall_at_3"]) for row in metrics]) if metrics else 0.0
            semantic_recall = statistics.fmean([int(row["semantic_recall_at_3"]) for row in metrics]) if metrics else 0.0
            reciprocal = [1.0 / row["lexical_rank"] for row in metrics if row["lexical_rank"]]
            mrr = statistics.fmean(reciprocal) if reciprocal else 0.0
            result.details = {"evidence_level": "independent_gold_retrieval_microscope", "gold_path": str(GOLD.relative_to(ROOT)), "gold_cases": len(metrics), "gold_independent_of_corpus_text": True, "queries": metrics, "lexical_recall_at_3": lexical_recall, "semantic_recall_at_3": semantic_recall, "lexical_mrr": mrr, "metadata_filter_correct": filter_ok, "vector_count": store.count(), "lexical_count": store.lexical_count(), "relevance_derived_from_fixture_text": False}
            result.score = round((lexical_recall + semantic_recall + int(filter_ok)) / 3.0, 3)
            result.status = "PASS" if lexical_recall >= 0.67 and semantic_recall >= 0.67 and filter_ok else "FAIL"
            if result.status == "FAIL":
                result.failures.append({"location": "VectorStore.search/search_lexical", "exception": "RetrievalMicroscopeFailure", "message": json.dumps(result.details, sort_keys=True)})
        finally:
            _cleanup_store(tmp)
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "rag_project/testing/robust_probes.py", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3)
    return result
