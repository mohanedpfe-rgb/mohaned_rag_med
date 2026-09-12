from __future__ import annotations

from typing import Any

from rag_project.testing.deep_diagnostics import PhaseResult
from rag_project.testing.advanced_phases import _cleanup_store, _embedding, _fixture_chunks, _fixture_pages, _store_fixture


def run_full_metamorphic_suite(phase: Any) -> PhaseResult:
    import time
    result = PhaseResult(phase.number, phase.key, phase.name, started_at=time.time())
    tmp = None
    try:
        from rag_project.chunking.semantic_chunker import SemanticChunker
        from rag_project.utils.text_utils import clean_text, normalize_whitespace, tokenize
        chunks = _fixture_chunks()
        store, tmp = _store_fixture(chunks)
        variants = [
            "diabetes diagnosis",
            "  diabetes diagnosis  ",
            "DIABETES DIAGNOSIS",
            "diabetes   diagnosis",
        ]
        top_docs = []
        top_ids = []
        for query in variants:
            lexical = store.search_lexical(query, n_results=5)
            semantic = store.search(_embedding(query), n_results=5)
            lex_ids = [str(v) for v in (lexical.get("ids") or [[]])[0]]
            sem_ids = [str(v) for v in (semantic.get("ids") or [[]])[0]]
            top_ids.append({"lexical": lex_ids[:3], "semantic": sem_ids[:3]})
            top_docs.append({"lexical": [value.split(":", 1)[0] for value in lex_ids[:3]], "semantic": [value.split(":", 1)[0] for value in sem_ids[:3]]})
        canonical_docs = top_docs[0]
        retrieval_invariant = all(row == canonical_docs for row in top_docs[1:])

        canonical_pages = _fixture_pages("canonical")
        whitespace_pages = _fixture_pages("whitespace")
        canonical_chunks = SemanticChunker(chunk_size=260, chunk_overlap=40).chunk_pages(canonical_pages)
        whitespace_chunks = SemanticChunker(chunk_size=260, chunk_overlap=40).chunk_pages(whitespace_pages)
        normalize = lambda value: normalize_whitespace(clean_text(value)).casefold()
        canonical_text = {normalize(chunk.text) for chunk in canonical_chunks}
        whitespace_text = {normalize(chunk.text) for chunk in whitespace_chunks}
        content_invariant = canonical_text == whitespace_text

        normalized_queries = [normalize_whitespace(clean_text(value)).casefold() for value in variants]
        token_variants = [tokenize(value) for value in variants]
        query_invariant = len(set(normalized_queries)) == 1 and all(tokens == token_variants[0] for tokens in token_variants)

        checks = {
            "query_normalization_invariant": query_invariant,
            "lexical_semantic_top_documents_stable": retrieval_invariant,
            "canonical_vs_whitespace_chunk_content_stable": content_invariant,
            "production_chunker_executed_for_both_variants": bool(canonical_chunks and whitespace_chunks),
        }
        failures = [name for name, value in checks.items() if not value]
        result.details = {
            "production_functions": ["VectorStore.search_lexical", "VectorStore.search", "SemanticChunker.chunk_pages", "clean_text", "normalize_whitespace", "tokenize"],
            "checks": checks,
            "query_variants": variants,
            "top_retrieval_ids": top_ids,
            "top_retrieved_documents": top_docs,
            "normalized_queries": normalized_queries,
            "canonical_chunk_count": len(canonical_chunks),
            "whitespace_chunk_count": len(whitespace_chunks),
            "mutation_kind": "semantics-preserving whitespace/case transformations",
        }
        result.score = sum(checks.values()) / len(checks)
        result.status = "PASS" if not failures else "FAIL"
        if failures:
            result.failures.append({"location": "phase 8 full retrieval metamorphic suite", "exception": "MetamorphicInvariantFailure", "message": str(failures)})
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "phase 8 full retrieval metamorphic suite", "exception": type(exc).__name__, "message": str(exc)})
    finally:
        if tmp is not None:
            _cleanup_store(tmp)
    result.duration_s = round(time.time() - result.started_at, 3)
    return result
