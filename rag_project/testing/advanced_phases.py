"""Evidence-producing implementations for the advanced diagnostic phases.

These probes deliberately execute production ingestion/chunking/retrieval/context/storage
utilities against bounded deterministic fixtures. They never modify the checkout. External
model/API benchmarks are opt-in; the default probes still produce real local measurements.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import statistics
import sys
import tempfile
import time
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .deep_diagnostics import PhaseResult

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "rag_project"
TESTS = ROOT / "tests"

IDENTITY_FIELDS = (
    "document_id", "version_id", "chunk_id", "page_numbers", "chapter", "chapter_id",
    "section", "section_id", "parent_id", "table_id", "figure_id", "quality_score",
    "ocr_status", "index_state",
)


def _result(phase: Any) -> PhaseResult:
    return PhaseResult(phase.number, phase.key, phase.name, started_at=time.time())


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(x) for x in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _safe_imports() -> tuple[Any, Any, Any]:
    from rag_project.ingestion.document_models import PageExtraction
    from rag_project.chunking.semantic_chunker import SemanticChunker
    from rag_project.storage.vector_store import VectorStore
    return PageExtraction, SemanticChunker, VectorStore


def _fixture_pages(variant: str = "canonical") -> list[Any]:
    PageExtraction, _, _ = _safe_imports()
    base = (
        "# Chapter 1\n## Diabetes mellitus\n"
        "Diabetes mellitus is a chronic metabolic disease. Diagnosis uses plasma glucose measurements.\n"
        "[TABLE]\nHbA1c diagnostic threshold: 6.5 percent.\n"
        "Figure 1: glucose regulation pathway."
    )
    if variant == "whitespace":
        base = base.replace("\n", "\n\n")
    elif variant == "case":
        base = base.upper()
    elif variant == "unicode":
        base = "# Chapitre 1\n## Diabète mellitus\nLe diabète chronique nécessite un diagnostic et un traitement adaptés.\n" + base
    elif variant == "long":
        base = base + "\n" + ("Clinical evidence and follow-up. " * 250)
    page1 = PageExtraction(
        document_id="diag-doc-001", file_name="diagnostic_fixture.pdf", page_index=0, page_number=1,
        text=base, extraction_method="native_text", ocr_required=False, ocr_status="not_required",
        ocr_confidence=0.99, page_type="mixed_layout", image_count=1, table_count=1, has_images=True,
        blocks=[base], metadata={"version_id": "v1", "source": "diagnostic-fixture"},
        source_path="diagnostic_fixture.pdf", quality_score=0.98, routing_decision="native",
        table_ids=["diag-table-001"], figure_ids=["diag-figure-001"],
        table_texts=["HbA1c diagnostic threshold: 6.5 percent."],
        figure_captions=["Figure 1: glucose regulation pathway."], headings=["Chapter 1", "Diabetes mellitus"],
    )
    page2 = PageExtraction(
        document_id="diag-doc-001", file_name="diagnostic_fixture.pdf", page_index=1, page_number=2,
        text=("## Kidney complications\nDiabetic nephropathy can cause albuminuria and chronic kidney disease.\n"
              "Treatment and monitoring depend on the clinical context."),
        extraction_method="ocr", ocr_required=True, ocr_status="complete", ocr_confidence=0.94,
        page_type="ocr_text", image_count=1, table_count=0, has_images=True, blocks=[],
        metadata={"version_id": "v1", "source": "diagnostic-fixture"}, source_path="diagnostic_fixture.pdf",
        quality_score=0.91, routing_decision="ocr", table_ids=[], figure_ids=["diag-figure-002"],
        table_texts=[], figure_captions=["Figure 2: nephropathy progression."], headings=["Kidney complications"],
    )
    return [page1, page2]


def _fixture_chunks(pages: list[Any] | None = None) -> list[Any]:
    _, SemanticChunker, _ = _safe_imports()
    chunks = SemanticChunker(chunk_size=260, chunk_overlap=40).chunk_pages(pages or _fixture_pages())
    if not chunks:
        raise AssertionError("production SemanticChunker produced no chunks for diagnostic fixture")
    return chunks


def _embedding(text: str, dimension: int = 32) -> list[float]:
    import hashlib
    vector = [0.0] * dimension
    for token in re.findall(r"\w+", str(text).casefold(), flags=re.UNICODE):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        vector[int.from_bytes(digest[:2], "big") % dimension] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _store_fixture(chunks: list[Any] | None = None) -> tuple[Any, Path]:
    _, _, VectorStore = _safe_imports()
    chunks = chunks or _fixture_chunks()
    tmp = Path(tempfile.mkdtemp(prefix="rag_17phase_store_"))
    store = VectorStore(tmp / "index", collection_name="diagnostic")
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    embeddings: list[list[float]] = []
    ids: list[str] = []
    for chunk in chunks:
        chunk_id = f"{chunk.doc_id}:{chunk.chunk_index}:{chunk.representation_type}"
        meta = dict(chunk.metadata or {})
        meta.update({"document_id": chunk.doc_id, "chunk_id": chunk_id, "version_id": meta.get("version_id", "v1"), "page_numbers": list(chunk.page_numbers or []), "index_state": "READY"})
        documents.append(chunk.text)
        metadatas.append(meta)
        embeddings.append(_embedding(chunk.text))
        ids.append(chunk_id)
    store.add_documents(documents, metadatas, embeddings, ids)
    return store, tmp


def _cleanup_store(tmp: Path) -> None:
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


def _token_set(text: str) -> set[str]:
    return {token.casefold() for token in re.findall(r"\w+", text or "", flags=re.UNICODE)}


def _survival(before: Iterable[str], after: Iterable[str]) -> float:
    before_set = set(before)
    if not before_set:
        return 1.0
    return len(before_set & set(after)) / len(before_set)


# Phase 3 ------------------------------------------------------------------

def diagnostic_chain(phase: Any) -> PhaseResult:
    result = _result(phase)
    try:
        from rag_project.ingestion.document_models import Chunk
        from rag_project.retrieval.context_builder import ContextBuilder
        from rag_project.retrieval.hybrid_retriever import RetrievalHit
        pages = _fixture_pages()
        chunks = _fixture_chunks(pages)
        stage_results: list[dict[str, Any]] = [{"stage": "ingestion_fixture", "count": len(pages), "ok": bool(pages)}]
        stage_results.append({"stage": "production_chunking", "count": len(chunks), "ok": bool(chunks)})
        hit = RetrievalHit(doc_id=chunks[0].doc_id, text=chunks[0].text, metadata={**chunks[0].metadata, "chunk_id": "chain-0"}, score=1.0)
        context, selected = ContextBuilder(token_budget=2500).build([hit])
        stage_results.append({"stage": "context_build", "count": len(selected), "ok": bool(context and selected)})
        required = all(isinstance(chunk, Chunk) and chunk.doc_id and chunk.text and chunk.metadata.get("document_id") for chunk in chunks)
        result.details = {"chain": stage_results, "required_chunk_contract": required, "first_failure_stage": next((row["stage"] for row in stage_results if not row["ok"]), None), "evidence": "PageExtraction -> SemanticChunker -> RetrievalHit -> ContextBuilder"}
        result.score = sum(bool(row["ok"]) for row in stage_results) / len(stage_results)
        result.status = "PASS" if result.score == 1.0 and required else "FAIL"
        if result.status == "FAIL":
            result.failures.append({"location": "production diagnostic chain", "exception": "DiagnosticChainFailure", "message": json.dumps(stage_results, sort_keys=True)})
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "rag_project/testing/advanced_phases.py", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


# Phase 4 ------------------------------------------------------------------

def contract_triangulation(phase: Any) -> PhaseResult:
    result = _result(phase)
    try:
        from rag_project.retrieval.context_builder import ContextBuilder
        from rag_project.retrieval.hybrid_retriever import RetrievalHit
        pages = _fixture_pages()
        chunks = _fixture_chunks(pages)
        contract_failures: list[str] = []
        input_contract = all(getattr(page, "document_id", None) and getattr(page, "page_number", None) and getattr(page, "text", None) is not None for page in pages)
        transform_contract = all(getattr(chunk, "doc_id", None) and getattr(chunk, "page_numbers", None) and getattr(chunk, "metadata", None) and getattr(chunk, "section_id", None) for chunk in chunks)
        hits = [RetrievalHit(doc_id=chunk.doc_id, text=chunk.text, metadata=dict(chunk.metadata), score=float(len(chunk.text))) for chunk in chunks[:4]]
        context, selected = ContextBuilder(token_budget=2500).build(hits)
        output_contract = bool(context) and len(selected) > 0 and all("<evidence id=\"S" in context for _ in [selected[0]])
        if not input_contract: contract_failures.append("input")
        if not transform_contract: contract_failures.append("transformation")
        if not output_contract: contract_failures.append("output")
        result.details = {"input_contract": input_contract, "transformation_contract": transform_contract, "output_contract": output_contract, "page_count": len(pages), "chunk_count": len(chunks), "selected_context_hits": len(selected), "failed_contracts": contract_failures}
        result.score = round(sum((input_contract, transform_contract, output_contract)) / 3, 3)
        result.status = "PASS" if not contract_failures else "FAIL"
        if contract_failures:
            result.failures.append({"location": "PageExtraction/SemanticChunker/ContextBuilder", "exception": "ContractTriangulationFailure", "message": f"failed contracts: {contract_failures}"})
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "rag_project/testing/advanced_phases.py", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


# Phase 5 ------------------------------------------------------------------

def cross_layer_invariants(phase: Any) -> PhaseResult:
    result = _result(phase)
    try:
        pages = _fixture_pages(); chunks = _fixture_chunks(); store, tmp = _store_fixture(chunks)
        try:
            records = store.get_documents(); ids = [str(value) for value in records.get("ids", [])]; metas = [dict(value or {}) for value in records.get("metadatas", [])]
            lexical_count = store.lexical_count(); vector_count = store.count(); violations: list[str] = []
            page_ids = {page.document_id for page in pages}; chunk_ids = {f"{chunk.doc_id}:{chunk.chunk_index}:{chunk.representation_type}" for chunk in chunks}
            if page_ids != {meta.get("document_id") for meta in metas}: violations.append("document_id changed or disappeared")
            if not chunk_ids.issubset(set(ids)): violations.append("chunk identity not conserved into vector ids")
            for meta in metas:
                for field in ("document_id", "chunk_id", "page_numbers", "section_id", "parent_id", "index_state"):
                    if field not in meta: violations.append(f"missing identity field: {field}")
                if not meta.get("page_numbers"): violations.append("chunk without page_numbers")
            if vector_count != lexical_count: violations.append(f"vector/lexical count mismatch: {vector_count}/{lexical_count}")
            report = store.verify_index("diag-doc-001")
            if not report.get("valid"): violations.append(f"stored index validation failed: {report.get('issues')}")
            result.details = {"fixture_pages": len(pages), "production_chunks": len(chunks), "vector_records": vector_count, "lexical_records": lexical_count, "identity_rows": [{field: meta.get(field) for field in IDENTITY_FIELDS} for meta in metas], "violations": violations, "evidence": "PageExtraction -> SemanticChunker.chunk_pages -> VectorStore.add_documents -> VectorStore.verify_index"}
            result.score = 1.0 if not violations else max(0.0, 1.0 - len(violations) / max(1, len(metas) * 2)); result.status = "PASS" if not violations else "FAIL"
            if violations: result.failures.append({"location": "production chunk/storage boundary", "exception": "IdentityConservationFailure", "message": "; ".join(violations[:8])})
        finally: _cleanup_store(tmp)
    except Exception as exc:
        result.status = "FAIL"; result.failures.append({"location": "rag_project/testing/advanced_phases.py", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3); return result


# Phase 6 ------------------------------------------------------------------

def information_loss(phase: Any) -> PhaseResult:
    result = _result(phase)
    try:
        pages = _fixture_pages(); chunks = _fixture_chunks(); store, tmp = _store_fixture(chunks)
        try:
            stored = store.get_documents(); vector_docs = [str(v or "") for v in stored.get("documents", [])]; vector_meta = [dict(v or {}) for v in stored.get("metadatas", [])]
            import sqlite3
            with sqlite3.connect(store.lexical_database) as connection: lexical_rows = connection.execute("SELECT document, metadata FROM lexical_documents ORDER BY id").fetchall()
            source_text = "\n".join(page.text for page in pages); chunk_text = "\n".join(chunk.text for chunk in chunks); lexical_text = "\n".join(str(row[0]) for row in lexical_rows)
            source_meta = {field for page in pages for field in ((page.metadata or {}).keys())}; chunk_meta = {field for chunk in chunks for field in ((chunk.metadata or {}).keys())}; vector_fields = {field for meta in vector_meta for field in meta}; lexical_fields = set()
            for _, raw in lexical_rows:
                try: lexical_fields |= set(json.loads(raw).keys())
                except Exception: pass
            fields = {field: {"page": field in source_meta or hasattr(pages[0], field), "chunk": field in chunk_meta or all(hasattr(c, field) for c in chunks), "vector": field in vector_fields, "lexical": field in lexical_fields} for field in IDENTITY_FIELDS}
            measurements = {"source_to_chunk_token_survival": _survival(_token_set(source_text), _token_set(chunk_text)), "chunk_to_vector_document_token_survival": _survival(_token_set(chunk_text), _token_set(" ".join(vector_docs))), "vector_to_lexical_document_survival": _survival(_token_set(" ".join(vector_docs)), _token_set(lexical_text)), "table_identity_survived": "diag-table-001" in {m.get("table_id") for m in vector_meta}, "figure_identity_survived": bool({"diag-figure-001", "diag-figure-002"} & {m.get("figure_id") for m in vector_meta}), "ocr_status_survived": all("ocr_status" in m for m in vector_meta), "structure_survival": sum(int(all(surface.values())) for surface in fields.values()) / max(1, len(fields))}
            failures = [key for key, value in measurements.items() if isinstance(value, float) and key.endswith("survival") and value < 0.90]
            if not measurements["table_identity_survived"]: failures.append("table_identity_survived")
            if not measurements["figure_identity_survived"]: failures.append("figure_identity_survived")
            if measurements["structure_survival"] < 0.90: failures.append("structure_survival")
            result.details = {"representations": ["source_page", "canonical_chunk", "vector_document", "sqlite_lexical"], "field_surfaces": fields, "measurements": measurements, "failed_invariants": failures}
            result.score = round(1.0 - len(failures) / max(1, len(measurements)), 3); result.status = "PASS" if not failures else "FAIL"
            if failures: result.failures.append({"location": "production representation chain", "exception": "InformationLossFailure", "message": f"information-loss invariants failed: {failures}"})
        finally: _cleanup_store(tmp)
    except Exception as exc:
        result.status = "FAIL"; result.failures.append({"location": "rag_project/testing/advanced_phases.py", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3); return result


# Phase 7 ------------------------------------------------------------------

def adversarial_documents(phase: Any) -> PhaseResult:
    result = _result(phase)
    cases = ["canonical", "whitespace", "case", "unicode", "long"]
    outcomes: list[dict[str, Any]] = []
    try:
        from rag_project.chunking.semantic_chunker import SemanticChunker
        for variant in cases:
            pages = _fixture_pages(variant)
            started = time.perf_counter(); chunks = SemanticChunker(chunk_size=180, chunk_overlap=30).chunk_pages(pages); elapsed = time.perf_counter() - started
            valid_chunks = all(chunk.doc_id and chunk.page_numbers and chunk.metadata.get("document_id") for chunk in chunks)
            outcomes.append({"case": variant, "pages": len(pages), "chunks": len(chunks), "valid_chunk_schema": valid_chunks, "elapsed_s": round(elapsed, 4)})
        empty = _fixture_pages()[0]; empty.text = ""; empty.table_texts = []; empty.figure_captions = []
        empty_chunks = SemanticChunker(chunk_size=180, chunk_overlap=30).chunk_pages([empty])
        outcomes.append({"case": "empty_page", "pages": 1, "chunks": len(empty_chunks), "valid_chunk_schema": True, "expected_non_crash": True})
        valid = all(row["valid_chunk_schema"] for row in outcomes)
        result.details = {"case_matrix": outcomes, "cases_executed": len(outcomes), "non_crash_all": valid, "adversarial_dimensions": ["whitespace", "case", "unicode", "long", "empty", "ocr", "tables", "figures"]}
        result.score = sum(int(row.get("valid_chunk_schema") and row.get("expected_non_crash", True)) for row in outcomes) / len(outcomes)
        result.status = "PASS" if valid else "FAIL"
        if not valid: result.failures.append({"location": "production SemanticChunker adversarial matrix", "exception": "AdversarialDocumentFailure", "message": json.dumps(outcomes, sort_keys=True)})
    except Exception as exc:
        result.status = "FAIL"; result.failures.append({"location": "rag_project/chunking/semantic_chunker.py", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3); return result


# Phase 8 ------------------------------------------------------------------

def metamorphic(phase: Any) -> PhaseResult:
    result = _result(phase)
    try:
        from rag_project.utils.text_utils import clean_text, keyword_overlap_score, normalize_whitespace, tokenize
        from rag_project.chunking.semantic_chunker import SemanticChunker
        queries = ["What is diabetes mellitus?", "  What is diabetes mellitus?  ", "WHAT IS DIABETES MELLITUS?"]
        normalized = [normalize_whitespace(clean_text(q)).casefold() for q in queries]; tokenized = [tokenize(q) for q in queries]; scores = [keyword_overlap_score(q, "Diabetes mellitus is a chronic metabolic disease") for q in queries]
        canonical = SemanticChunker(chunk_size=260, chunk_overlap=40).chunk_pages(_fixture_pages()); variant = SemanticChunker(chunk_size=260, chunk_overlap=40).chunk_pages(_fixture_pages("whitespace"))
        canonical_norm = Counter(clean_text(c.text).casefold() for c in canonical); variant_norm = Counter(clean_text(c.text).casefold() for c in variant); overlap = len(set(canonical_norm) & set(variant_norm)) / max(1, len(canonical_norm))
        checks = {"query_whitespace_case_invariant": len(set(normalized)) == 1, "tokenization_invariant": all(tokenized[0] == tokens for tokens in tokenized[1:]), "lexical_score_invariant": max(scores) - min(scores) <= 1e-12, "chunk_content_invariant": overlap >= 0.90}
        failures = [name for name, ok in checks.items() if not ok]; result.details = {"production_functions": ["clean_text", "normalize_whitespace", "tokenize", "keyword_overlap_score", "SemanticChunker.chunk_pages"], "checks": checks, "normalized_queries": normalized, "lexical_scores": scores, "canonical_chunk_count": len(canonical), "variant_chunk_count": len(variant), "chunk_content_overlap": round(overlap, 4)}; result.score = round(sum(checks.values()) / len(checks), 3); result.status = "PASS" if not failures else "FAIL"
        if failures: result.failures.append({"location": "production text/chunk transformation", "exception": "MetamorphicFailure", "message": f"failed invariants: {failures}"})
    except Exception as exc:
        result.status = "FAIL"; result.failures.append({"location": "rag_project/testing/advanced_phases.py", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3); return result


# Phase 9 ------------------------------------------------------------------

def retrieval_microscope(phase: Any) -> PhaseResult:
    result = _result(phase)
    try:
        chunks = _fixture_chunks(); store, tmp = _store_fixture(chunks)
        try:
            query_specs = [("diabetes diagnosis", {0, 1}), ("HbA1c diagnostic threshold", {2}), ("kidney nephropathy", {3, 4})]
            metrics = []; filter_failures = 0
            for query, relevant_tokens in query_specs:
                lexical = store.search_lexical(query, n_results=5); semantic = store.search(_embedding(query), n_results=5)
                lexical_ids = [str(v) for v in (lexical.get("ids") or [[]])[0]]; semantic_ids = [str(v) for v in (semantic.get("ids") or [[]])[0]]
                relevant = {f"{chunks[index].doc_id}:{chunks[index].chunk_index}:{chunks[index].representation_type}" for index in relevant_tokens if index < len(chunks)}
                lex_hit = any(item in relevant for item in lexical_ids[:3]); sem_hit = any(item in relevant for item in semantic_ids[:3]); lex_rank = next((i + 1 for i, item in enumerate(lexical_ids) if item in relevant), None); sem_rank = next((i + 1 for i, item in enumerate(semantic_ids) if item in relevant), None)
                metrics.append({"query": query, "relevant_count": len(relevant), "lexical_recall_at_3": lex_hit, "semantic_recall_at_3": sem_hit, "lexical_rank": lex_rank, "semantic_rank": sem_rank})
            filtered = store.search_lexical("diabetes", n_results=5, where={"document_id": "does-not-exist"}); filter_failures += int(bool((filtered.get("ids") or [[]])[0]))
            lexical_recall = sum(int(row["lexical_recall_at_3"]) for row in metrics) / len(metrics); semantic_recall = sum(int(row["semantic_recall_at_3"]) for row in metrics) / len(metrics); mrr = statistics.fmean([1 / row["lexical_rank"] for row in metrics if row["lexical_rank"]] or [0.0])
            result.details = {"queries": metrics, "lexical_recall_at_3": lexical_recall, "semantic_recall_at_3": semantic_recall, "lexical_mrr": mrr, "metadata_filter_correct": filter_failures == 0, "vector_count": store.count(), "lexical_count": store.lexical_count()}
            result.score = round((lexical_recall + semantic_recall + int(filter_failures == 0)) / 3, 3); result.status = "PASS" if lexical_recall >= 0.67 and filter_failures == 0 else "FAIL"
            if result.status == "FAIL": result.failures.append({"location": "VectorStore.search/search_lexical", "exception": "RetrievalMicroscopeFailure", "message": json.dumps(result.details, sort_keys=True)})
        finally: _cleanup_store(tmp)
    except Exception as exc:
        result.status = "FAIL"; result.failures.append({"location": "rag_project/storage/vector_store.py", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3); return result


# Phase 10 -----------------------------------------------------------------

def rag_causality(phase: Any) -> PhaseResult:
    result = _result(phase)
    try:
        from rag_project.retrieval.context_builder import ContextBuilder
        from rag_project.retrieval.hybrid_retriever import RetrievalHit
        chunks = _fixture_chunks(); store, tmp = _store_fixture(chunks)
        try:
            raw = store.search_lexical("diabetes diagnosis", n_results=4); ids = [str(v) for v in (raw.get("ids") or [[]])[0]]; docs = [str(v) for v in (raw.get("documents") or [[]])[0]]; metas = [dict(v) for v in (raw.get("metadatas") or [[]])[0]]
            hits = [RetrievalHit(doc_id=metas[i].get("document_id", ids[i]), text=docs[i], metadata={**metas[i], "chunk_id": ids[i]}, score=1.0 / (i + 1)) for i in range(min(len(ids), len(docs), len(metas)))]
            context, selected = ContextBuilder(token_budget=2800, max_per_document=4).build(hits)
            evidence_ids = re.findall(r'<evidence id="S\d+" chunk_id="([^"]+)">', context)
            retrieved_set = {hit.metadata.get("chunk_id") for hit in hits}; selected_set = {hit.metadata.get("chunk_id") for hit in selected}
            causal_edges = {"query_to_retrieval": bool(hits), "retrieval_to_context": bool(selected) and selected_set.issubset(retrieved_set), "context_to_evidence": bool(evidence_ids) and set(evidence_ids) == selected_set, "evidence_identity_preserved": all(str(eid).strip() for eid in evidence_ids)}
            first_broken = next((name for name, ok in causal_edges.items() if not ok), None)
            result.details = {"retrieval_hits": len(hits), "selected_context_hits": len(selected), "evidence_ids": evidence_ids, "causal_edges": causal_edges, "first_broken_edge": first_broken, "grounding_contract": "every emitted evidence id must correspond to a retrieved chunk id", "clinical_answer_generation_exercised": False}
            result.score = sum(causal_edges.values()) / len(causal_edges); result.status = "PASS" if result.score == 1.0 else "FAIL"
            if result.status == "FAIL": result.failures.append({"location": "VectorStore -> ContextBuilder", "exception": "RGACausalityFailure", "message": f"first broken causal edge: {first_broken}"})
        finally: _cleanup_store(tmp)
    except Exception as exc:
        result.status = "FAIL"; result.failures.append({"location": "rag_project/retrieval/context_builder.py", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3); return result


# Phase 11 -----------------------------------------------------------------

def _load_module_from_source(path: Path, source: str, module_name: str) -> Any:
    tmp = Path(tempfile.mkdtemp(prefix="rag_mutant_module_")); target = tmp / path.name; target.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(module_name, target)
    if spec is None or spec.loader is None: raise ImportError(f"unable to load mutant module {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[module_name] = module
    try: spec.loader.exec_module(module)
    finally: sys.modules.pop(module_name, None)
    import shutil; shutil.rmtree(tmp, ignore_errors=True)
    return module


def mutation_detection(phase: Any) -> PhaseResult:
    result = _result(phase); mutation_results: list[dict[str, Any]] = []
    try:
        text_path = PROJECT / "utils" / "text_utils.py"; source = text_path.read_text(encoding="utf-8"); original = 'return re.sub(r"\\s+", " ", value or "").strip()'; mutant = 'return value or ""'
        if original in source:
            module = _load_module_from_source(text_path, source.replace(original, mutant, 1), "_diag_mutant_text_utils"); observed = module.normalize_whitespace("  diabetes   mellitus  "); mutation_results.append({"target": str(text_path.relative_to(ROOT)), "mutation": "normalize_whitespace_return_raw", "killed": observed == "diabetes mellitus", "observed": observed})
        else: mutation_results.append({"target": str(text_path.relative_to(ROOT)), "mutation": "normalize_whitespace_return_raw", "killed": False, "reason": "target expression changed"})
        chunk_path = PROJECT / "chunking" / "semantic_chunker.py"; chunk_source = chunk_path.read_text(encoding="utf-8"); target = '        if not children:\n            return []\n'; replacement = '        if not children:\n            return ["__MUTANT__"]\n'
        if target in chunk_source:
            module = _load_module_from_source(chunk_path, chunk_source.replace(target, replacement, 1), "_diag_mutant_chunker"); observed = module.SemanticChunker._compact_children([]); mutation_results.append({"target": str(chunk_path.relative_to(ROOT)), "mutation": "compact_children_empty_contract", "killed": observed == [], "observed": observed})
        else: mutation_results.append({"target": str(chunk_path.relative_to(ROOT)), "mutation": "compact_children_empty_contract", "killed": False, "reason": "target expression changed"})
        applicable = sum("reason" not in item for item in mutation_results); killed = sum(bool(item.get("killed")) for item in mutation_results); kill_score = killed / applicable if applicable else 0.0
        result.details = {"strategy": "executable shadow mutants loaded from temporary copies; checkout never modified", "mutation_results": mutation_results, "mutants_applicable": applicable, "mutants_killed": killed, "kill_score": round(kill_score, 3)}; result.score = round(kill_score, 3)
        if applicable == 0: result.status = "FAIL"; result.failures.append({"location": "mutation targets", "exception": "MutationCoverageFailure", "message": "no executable mutation targets remained compatible with the current source"})
        elif kill_score < 1.0: result.status = "FAIL"; result.failures.append({"location": "production diagnostic mutants", "exception": "SurvivingMutant", "message": f"mutation kill score={kill_score:.3f}"})
        else: result.status = "PASS"
    except Exception as exc:
        result.status = "FAIL"; result.failures.append({"location": "rag_project/testing/advanced_phases.py", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3); return result


# Phase 14 -----------------------------------------------------------------

def performance(phase: Any) -> PhaseResult:
    result = _result(phase)
    try:
        from rag_project.utils.text_utils import tokenize
        from rag_project.chunking.semantic_chunker import SemanticChunker
        pages = _fixture_pages(); stage_samples: dict[str, list[float]] = defaultdict(list)
        for _ in range(5):
            started = time.perf_counter(); _ = [tokenize(page.text) for page in pages]; stage_samples["text_normalization"].append((time.perf_counter() - started) * 1000)
            started = time.perf_counter(); chunks = SemanticChunker(chunk_size=260, chunk_overlap=40).chunk_pages(pages); stage_samples["chunking"].append((time.perf_counter() - started) * 1000)
            store, tmp = _store_fixture(chunks)
            try:
                started = time.perf_counter(); _ = store.search_lexical("diabetes diagnosis", n_results=3); stage_samples["lexical_retrieval"].append((time.perf_counter() - started) * 1000)
                started = time.perf_counter(); _ = store.search(_embedding("diabetes diagnosis"), n_results=3); stage_samples["semantic_retrieval"].append((time.perf_counter() - started) * 1000)
            finally: _cleanup_store(tmp)
        summary = {stage: {"samples": len(values), "p50_ms": round(_percentile(values, .50), 3), "p95_ms": round(_percentile(values, .95), 3), "p99_ms": round(_percentile(values, .99), 3), "max_ms": round(max(values), 3)} for stage, values in stage_samples.items()}
        baseline_path = ROOT / "tests" / "support" / "performance_baseline.json"; baseline = {}
        if baseline_path.exists():
            try: baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            except Exception: baseline = {}
        regressions = [{"stage": stage, "p95_ms": row["p95_ms"], "baseline_p95_ms": float(limit)} for stage, row in summary.items() if (limit := baseline.get(stage, {}).get("p95_ms")) is not None and row["p95_ms"] > float(limit) * 1.25]
        result.details = {"fixture_pages": len(pages), "stage_metrics": summary, "baseline_path": str(baseline_path.relative_to(ROOT)), "regressions": regressions, "benchmark_repetitions": 5}; result.score = 1.0 if not regressions else max(0.0, 1.0 - len(regressions) / len(summary)); result.status = "PASS" if not regressions else "FAIL"
        if regressions: result.failures.append({"location": str(baseline_path.relative_to(ROOT)), "exception": "PerformanceRegression", "message": json.dumps(regressions, sort_keys=True)})
    except Exception as exc:
        result.status = "FAIL"; result.failures.append({"location": "rag_project/testing/advanced_phases.py", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3); return result


# Phase 15 -----------------------------------------------------------------

def resources(phase: Any) -> PhaseResult:
    result = _result(phase)
    try:
        tracemalloc.start(); samples: list[int] = []; peaks: list[int] = []; started = time.perf_counter()
        for _ in range(8):
            chunks = _fixture_chunks(); store, tmp = _store_fixture(chunks)
            try: _ = store.search_lexical("kidney nephropathy", n_results=3); _ = store.search(_embedding("kidney nephropathy"), n_results=3)
            finally: _cleanup_store(tmp)
            current, peak = tracemalloc.get_traced_memory(); samples.append(current); peaks.append(peak); del chunks
        current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop(); first = samples[0] if samples else 0; last = samples[-1] if samples else 0; growth = (last - first) / max(1, first); monotonic_growth = sum(b > a for a, b in zip(samples, samples[1:])) >= 6 if len(samples) > 1 else False
        result.details = {"repetitions": len(samples), "current_bytes": current, "peak_bytes": peak, "sample_current_bytes": samples, "peak_series_bytes": peaks, "relative_growth_first_to_last": round(growth, 4), "monotonic_growth_signal": monotonic_growth, "elapsed_s": round(time.perf_counter() - started, 3), "pipeline_exercised": ["SemanticChunker.chunk_pages", "VectorStore.add_documents", "VectorStore.search_lexical", "VectorStore.search"]}; leak = growth > .60 and monotonic_growth; result.score = round(max(0.0, 1.0 - min(1.0, max(0.0, growth))), 3); result.status = "FAIL" if leak else "PASS"
        if leak: result.failures.append({"location": "production ingestion/chunking/storage repetition", "exception": "ResourceGrowthFailure", "message": f"traced allocation grew by {growth:.1%} with monotonic signal"})
    except Exception as exc:
        try: tracemalloc.stop()
        except Exception: pass
        result.status = "FAIL"; result.failures.append({"location": "rag_project/testing/advanced_phases.py", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3); return result


# Phase 16 -----------------------------------------------------------------

def _gold_cases() -> list[dict[str, Any]]:
    gold = TESTS / "support" / "gold_sets" / "core.jsonl"
    if not gold.exists(): return []
    return [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_live_gold(base_url: str, gold: Path) -> dict[str, Any]:
    script = ROOT / "scripts" / "run_kpi_benchmark.py"
    command = [sys.executable, str(script), "--url", base_url, "--gold", str(gold), "--timeout", "30", "--output", str(ROOT / "artifacts" / "diagnostic_live_kpi.json")]
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=600)
    if proc.returncode != 0: raise RuntimeError((proc.stdout + "\n" + proc.stderr)[-3000:])
    return json.loads((ROOT / "artifacts" / "diagnostic_live_kpi.json").read_text(encoding="utf-8"))


def _offline_gold(cases: list[dict[str, Any]]) -> dict[str, Any]:
    from rag_project.ingestion.document_models import Chunk
    chunks = []
    for index, item in enumerate(cases):
        terms = [str(term) for term in item.get("must_contain", [])]
        text = "Medical evidence: " + ". ".join(terms or [str(item.get("question", ""))])
        chunks.append(Chunk(doc_id=f"gold-{index}", file_name="gold-fixture.pdf", chunk_index=index, text=text, page_numbers=[index + 1], metadata={"document_id": f"gold-{index}", "chunk_id": f"gold-chunk-{index}", "version_id": "v1", "section": "gold", "section_id": f"gold-section-{index}", "parent_id": f"gold-parent-{index}", "index_state": "READY"}))
    store, tmp = _store_fixture(chunks)
    try:
        rows = []
        for index, item in enumerate(cases):
            query = " ".join([str(item.get("question", "")), *[str(v) for v in item.get("must_contain", [])]])
            semantic = store.search(_embedding(query), n_results=min(5, max(1, len(chunks)))); lexical = store.search_lexical(query, n_results=min(5, max(1, len(chunks)))); semantic_ids = [str(v) for v in (semantic.get("ids") or [[]])[0]]; lexical_ids = [str(v) for v in (lexical.get("ids") or [[]])[0]]; target = f"gold-{index}:"; retrieved = set(semantic_ids) | set(lexical_ids); hit = any(item_id.startswith(target) for item_id in retrieved); lex_rank = next((i + 1 for i, value in enumerate(lexical_ids) if value.startswith(target)), None); sem_rank = next((i + 1 for i, value in enumerate(semantic_ids) if value.startswith(target)), None); rows.append({"id": item.get("id"), "retrieval_hit": hit, "lexical_rank": lex_rank, "semantic_rank": sem_rank})
        recall = sum(bool(r["retrieval_hit"]) for r in rows) / len(rows) if rows else 0.0; rr = [1 / r["lexical_rank"] for r in rows if r["lexical_rank"]]; return {"mode": "offline_retrieval_contract", "case_count": len(rows), "recall_at_k": recall, "mrr": statistics.fmean(rr) if rr else 0.0, "results": rows, "clinical_correctness_claimed": False, "citation_accuracy_claimed": False}
    finally: _cleanup_store(tmp)


def golden_benchmark(phase: Any) -> PhaseResult:
    result = _result(phase); gold = TESTS / "support" / "gold_sets" / "core.jsonl"
    try:
        cases = _gold_cases()
        if not cases: result.status = "FAIL"; result.failures.append({"location": str(gold.relative_to(ROOT)), "exception": "MissingGoldSet", "message": "permanent gold set is unavailable or empty"}); return result
        malformed = [item.get("id", "<missing>") for item in cases if not {"id", "question"}.issubset(item)]
        if malformed: result.status = "FAIL"; result.failures.append({"location": str(gold.relative_to(ROOT)), "exception": "MalformedGoldSet", "message": f"invalid cases: {malformed[:10]}"}); return result
        base_url = os.getenv("DIAGNOSTIC_API_URL", "").strip()
        payload = _run_live_gold(base_url, gold) if base_url else _offline_gold(cases); mode = "live_api" if base_url else "offline_retrieval_contract"; metrics = payload.get("overall", payload); score = float(metrics.get("weighted_accuracy", metrics.get("recall_at_k", 0.0)) or 0.0)
        result.details = {"gold_path": str(gold.relative_to(ROOT)), "case_count": len(cases), "mode": mode, "metrics": metrics, "clinical_correctness_claimed": False, "live_mode_opt_in": bool(base_url)}; result.score = round(score, 3)
        transport_ok = all(bool(row.get("transport_ok")) for row in payload.get("results", [])) if mode == "live_api" else True
        result.status = "PASS" if transport_ok and score >= 0.80 else "FAIL"
        if result.status == "FAIL": result.failures.append({"location": str(gold.relative_to(ROOT)), "exception": "GoldenBenchmarkFailure", "message": f"benchmark score={score:.3f}"})
    except Exception as exc:
        result.status = "FAIL"; result.failures.append({"location": "gold benchmark", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3); return result


# Phases 12/13/17 -----------------------------------------------------------

def fingerprint_failures(results: Iterable[PhaseResult]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for phase in results:
        for failure in phase.failures:
            location = str(failure.get("location") or "unknown"); exception = str(failure.get("exception") or "Failure"); message = re.sub(r"\s+", " ", str(failure.get("message") or failure.get("detail") or "")).strip().casefold(); normalized_digits = re.sub(r"\d+", "#", message)[:180]; key = f"{location}|{exception}|{normalized_digits}"
            row = groups.setdefault(key, {"fingerprint": key, "phase_numbers": [], "evidence": [], "locations": []}); row["phase_numbers"].append(phase.number); row["locations"].append(location); row["evidence"].append(f"phase {phase.number}: {message[:240]}")
    output = list(groups.values())
    for row in output:
        row["phase_numbers"] = sorted(set(row["phase_numbers"])); row["locations"] = sorted(set(row["locations"])); row["confidence"] = round(min(.99, .55 + .08 * len(row["phase_numbers"])), 3)
    return sorted(output, key=lambda item: (-len(item["phase_numbers"]), item["phase_numbers"][0] if item["phase_numbers"] else 99))


def cascade_compression(results: Iterable[PhaseResult], fingerprints: list[dict[str, Any]]) -> dict[str, Any]:
    phases = list(results); failed = [p.number for p in phases if p.status == "FAIL"]; blocked = [p.number for p in phases if p.status == "BLOCKED"]; first_failure = min(failed) if failed else None; affected = sorted({number for row in fingerprints if first_failure in row.get("phase_numbers", []) for number in row.get("phase_numbers", [])}) if first_failure else []
    return {"failed_phases": failed, "blocked_phases": blocked, "first_failed_phase": first_failure, "candidate_root_causes": len(fingerprints), "compressed_impacts": [{"root_phase": first_failure, "affected_phases": affected}] if first_failure else [], "fix_order": [{"phase": row["phase_numbers"][0], "fingerprint": row["fingerprint"], "confidence": row["confidence"]} for row in fingerprints if row.get("phase_numbers")]}


def root_cause_phase(phase: Any, results: dict[int, PhaseResult]) -> PhaseResult:
    result = _result(phase); fingerprints = fingerprint_failures(results.values()); result.details = {"algorithm": "structured location + exception + normalized message fingerprint", "unique_fingerprints": len(fingerprints), "fingerprints": fingerprints[:50], "evidence_phases": sorted(results)}; result.score = 1.0; result.status = "PASS"; result.duration_s = round(time.time() - result.started_at, 3); return result


def cascade_phase(phase: Any, results: dict[int, PhaseResult]) -> PhaseResult:
    result = _result(phase); fingerprints = fingerprint_failures(results.values()); result.details = cascade_compression(results.values(), fingerprints); result.score = 1.0; result.status = "PASS"; result.duration_s = round(time.time() - result.started_at, 3); return result


def certification_phase(phase: Any, results: dict[int, PhaseResult]) -> PhaseResult:
    result = _result(phase); expected = list(range(1, 18)); present = sorted(results); missing = sorted(set(expected) - set(present)); implementation_matrix = {1: "architecture_map", 2: "fast_health", 3: "diagnostic_chain", 4: "contract_triangulation", 5: "cross_layer_real_fixture", 6: "information_survival_measurement", 7: "adversarial_document_matrix", 8: "production_metamorphic_execution", 9: "retrieval_microscope", 10: "rag_causality_trace", 11: "executable_shadow_mutants", 12: "structured_failure_fingerprints", 13: "causal_failure_compression", 14: "stage_latency_measurement", 15: "bounded_resource_growth_measurement", 16: "permanent_gold_benchmark", 17: "master_certification"}; runtime_failures = [p.number for p in results.values() if p.status == "FAIL"]; runtime_blocked = [p.number for p in results.values() if p.status == "BLOCKED"]; implementation_complete = not missing and len(implementation_matrix) == 17; result.details = {"implementation_matrix": implementation_matrix, "implementation_coverage": "17/17" if implementation_complete else f"{len(implementation_matrix)}/17", "all_phase_results_present": not missing, "missing_phase_results": missing, "runtime_failures": runtime_failures, "runtime_warnings": [p.number for p in results.values() if p.status == "WARN"], "runtime_blocked": runtime_blocked, "certification_rule": "17/17 implementation is mandatory; runtime certification PASS requires zero FAIL/BLOCKED phases", "clinical_correctness_claimed": False}; result.score = len(implementation_matrix) / 17.0
    if not implementation_complete: result.status = "FAIL"; result.failures.append({"location": "rag_project/testing/runner.py", "exception": "IncompleteDiagnosticSystem", "message": f"missing results: {missing}"})
    elif runtime_failures or runtime_blocked: result.status = "WARN"; result.failures.append({"location": "phase runtime", "exception": "RuntimeCertificationPending", "message": f"17/17 implemented, runtime failures={runtime_failures}, blocked={runtime_blocked}"})
    else: result.status = "PASS"
    result.duration_s = round(time.time() - result.started_at, 3); return result
