from __future__ import annotations

from typing import Any

from rag_project.testing.deep_diagnostics import PhaseResult
from rag_project.testing.advanced_phases import _cleanup_store, _embedding, _fixture_chunks, _fixture_pages, _store_fixture


def run_full_metamorphic_suite(phase: Any) -> PhaseResult:
    import shutil
    import tempfile
    import time
    from pathlib import Path

    result = PhaseResult(phase.number, phase.key, phase.name, started_at=time.time())
    tmp = None
    server = None
    answer_roots: list[Path] = []
    try:
        from rag_project.chunking.semantic_chunker import SemanticChunker
        from rag_project.generation.llm_client import OllamaLLMClient
        from rag_project.intelligence.med_evidence_pro import MedEvidenceProEngine
        from rag_project.testing.production_answer_probes import _LocalOllamaServer, _ProbeSystem, _seed_real_retrieval
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

        server = _LocalOllamaServer()
        answer_outputs = []
        answer_citations = []
        answer_verification = []
        protocol_health = []
        generation_paths = []
        for index, query in enumerate([
            "Explain diabetes mellitus and the role of HbA1c in diagnosis.",
            "  Explain diabetes mellitus and the role of HbA1c in diagnosis.  ",
            "EXPLAIN DIABETES MELLITUS AND THE ROLE OF HBA1C IN DIAGNOSIS.",
            "Explain   diabetes mellitus and the role of HbA1c in diagnosis.",
        ]):
            root = Path(tempfile.mkdtemp(prefix=f"rag_phase8_answer_{index}_")); answer_roots.append(root)
            system = _ProbeSystem(root, llm=OllamaLLMClient(server.base_url, "diagnostic-protocol:latest", timeout_seconds=15, max_output_tokens=512))
            system.generation_backend = "ollama_protocol"
            _seed_real_retrieval(system)
            protocol_health.append(bool(system.llm.health_check(timeout_seconds=2.0)))
            response = MedEvidenceProEngine(system).answer(query)
            answer_outputs.append(normalize(str(response.get("answer") or "")))
            answer_citations.append(tuple(sorted(str(item.get("document_id")) for item in (response.get("citations") or []))))
            answer_verification.append(bool((response.get("verification") or {}).get("allow")))
            generation_paths.append(str(response.get("generation_path") or ""))

        answer_invariant = bool(answer_outputs[0]) and len(set(answer_outputs)) == 1
        citation_invariant = bool(answer_citations[0]) and len(set(answer_citations)) == 1
        verification_invariant = bool(answer_verification) and all(answer_verification)
        client_path_invariant = bool(protocol_health) and all(protocol_health) and all("ollama" in path.casefold() or "generated" in path.casefold() for path in generation_paths)

        checks = {
            "query_normalization_invariant": query_invariant,
            "lexical_semantic_top_documents_stable": retrieval_invariant,
            "canonical_vs_whitespace_chunk_content_stable": content_invariant,
            "production_chunker_executed_for_both_variants": bool(canonical_chunks and whitespace_chunks),
            "answer_semantics_stable_under_query_formatting": answer_invariant,
            "citation_identity_stable_under_query_formatting": citation_invariant,
            "answer_verification_stable_under_query_formatting": verification_invariant,
            "ollama_client_protocol_stable_under_query_formatting": client_path_invariant,
        }
        failures = [name for name, value in checks.items() if not value]
        result.details = {
            "evidence_level": "end_to_end_rag_metamorphic_execution",
            "production_functions": [
                "VectorStore.search_lexical", "VectorStore.search", "SemanticChunker.chunk_pages",
                "clean_text", "normalize_whitespace", "tokenize", "MedEvidenceProEngine.answer",
                "CitationManager", "OllamaLLMClient",
            ],
            "checks": checks,
            "query_variants": variants,
            "top_retrieval_ids": top_ids,
            "top_retrieved_documents": top_docs,
            "normalized_queries": normalized_queries,
            "answer_invariance_outputs": answer_outputs,
            "citation_invariance_identities": [list(value) for value in answer_citations],
            "verification_results": answer_verification,
            "generation_paths": generation_paths,
            "ollama_protocol_health_checks": protocol_health,
            "canonical_chunk_count": len(canonical_chunks),
            "whitespace_chunk_count": len(whitespace_chunks),
            "mutation_kind": "semantics-preserving whitespace/case transformations across retrieval, chunking, generation, verification and citation identity",
            "end_to_end_answer_path_executed": True,
            "ollama_protocol_path_executed": True,
            "transformation_count": len(checks),
        }
        result.score = sum(checks.values()) / len(checks)
        result.status = "PASS" if not failures else "FAIL"
        if failures:
            result.failures.append({"location": "phase 8 full end-to-end RAG metamorphic suite", "exception": "MetamorphicInvariantFailure", "message": str(failures)})
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "phase 8 full end-to-end RAG metamorphic suite", "exception": type(exc).__name__, "message": str(exc)})
    finally:
        if server is not None:
            server.close()
        for root in answer_roots:
            shutil.rmtree(root, ignore_errors=True)
        if tmp is not None:
            _cleanup_store(tmp)
    result.duration_s = round(time.time() - result.started_at, 3)
    return result
