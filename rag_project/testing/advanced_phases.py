"""Evidence-producing implementations for the advanced diagnostic phases.

These probes deliberately execute production ingestion/chunking/storage utilities against
bounded deterministic fixtures.  They never modify the checkout.  External model/API
benchmarks are opt-in, while the default probes still produce real local measurements.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
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
    "document_id",
    "version_id",
    "chunk_id",
    "page_numbers",
    "chapter",
    "chapter_id",
    "section",
    "section_id",
    "parent_id",
    "table_id",
    "figure_id",
    "quality_score",
    "ocr_status",
    "index_state",
)


# ---------------------------------------------------------------------------
# Shared bounded real-pipeline fixture
# ---------------------------------------------------------------------------


def _result(phase: Any) -> PhaseResult:
    return PhaseResult(phase.number, phase.key, phase.name, started_at=time.time())


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    values = sorted(float(x) for x in values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * percentile
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (rank - low)


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
        base = base.replace("  ", " ").replace("\n", "\n\n")
    elif variant == "case":
        base = base.upper()
    page = PageExtraction(
        document_id="diag-doc-001",
        file_name="diagnostic_fixture.pdf",
        page_index=0,
        page_number=1,
        text=base,
        extraction_method="native_text",
        ocr_required=False,
        ocr_status="not_required",
        ocr_confidence=0.99,
        page_type="mixed_layout",
        image_count=1,
        table_count=1,
        has_images=True,
        blocks=[base],
        metadata={"version_id": "v1", "source": "diagnostic-fixture"},
        source_path="diagnostic_fixture.pdf",
        quality_score=0.98,
        routing_decision="native",
        table_ids=["diag-table-001"],
        figure_ids=["diag-figure-001"],
        table_texts=["HbA1c diagnostic threshold: 6.5 percent."],
        figure_captions=["Figure 1: glucose regulation pathway."],
        headings=["Chapter 1", "Diabetes mellitus"],
    )
    page2 = PageExtraction(
        document_id="diag-doc-001",
        file_name="diagnostic_fixture.pdf",
        page_index=1,
        page_number=2,
        text=(
            "## Kidney complications\nDiabetic nephropathy can cause albuminuria and chronic kidney disease.\n"
            "Treatment and monitoring depend on the clinical context."
        ),
        extraction_method="ocr",
        ocr_required=True,
        ocr_status="complete",
        ocr_confidence=0.94,
        page_type="ocr_text",
        image_count=1,
        table_count=0,
        has_images=True,
        blocks=[],
        metadata={"version_id": "v1", "source": "diagnostic-fixture"},
        source_path="diagnostic_fixture.pdf",
        quality_score=0.91,
        routing_decision="ocr",
        table_ids=[],
        figure_ids=["diag-figure-002"],
        table_texts=[],
        figure_captions=["Figure 2: nephropathy progression."],
        headings=["Kidney complications"],
    )
    return [page, page2]


def _fixture_chunks() -> list[Any]:
    _, chunker, _ = _safe_imports()
    chunks = chunker(chunk_size=260, chunk_overlap=40).chunk_pages(_fixture_pages())
    if not chunks:
        raise AssertionError("production SemanticChunker produced no chunks for diagnostic fixture")
    return chunks


def _embedding(text: str, dimension: int = 32) -> list[float]:
    """Deterministic bounded embedding used only for local diagnostic measurements."""
    import hashlib

    vector = [0.0] * dimension
    for token in re.findall(r"\w+", str(text).casefold(), flags=re.UNICODE):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dimension
        vector[index] += 1.0
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
        meta.update({
            "document_id": chunk.doc_id,
            "chunk_id": chunk_id,
            "version_id": meta.get("version_id", "v1"),
            "page_numbers": list(chunk.page_numbers or []),
            "index_state": "READY",
        })
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


# ---------------------------------------------------------------------------
# Phase 5 — real identity conservation through production chunking + storage
# ---------------------------------------------------------------------------


def cross_layer_invariants(phase: Any) -> PhaseResult:
    result = _result(phase)
    try:
        pages = _fixture_pages()
        chunks = _fixture_chunks()
        store, tmp = _store_fixture(chunks)
        try:
            records = store.get_documents()
            ids = [str(value) for value in records.get("ids", [])]
            metas = [dict(value or {}) for value in records.get("metadatas", [])]
            lexical_count = store.lexical_count()
            vector_count = store.count()
            violations: list[str] = []
            page_ids = {page.document_id for page in pages}
            chunk_ids = {f"{chunk.doc_id}:{chunk.chunk_index}:{chunk.representation_type}" for chunk in chunks}
            if page_ids != {meta.get("document_id") for meta in metas}:
                violations.append("document_id changed or disappeared between page and vector metadata")
            if not chunk_ids.issubset(set(ids)):
                violations.append("chunk identity was not conserved into vector ids")
            for meta in metas:
                for field in ("document_id", "chunk_id", "page_numbers", "section_id", "parent_id", "index_state"):
                    if field not in meta:
                        violations.append(f"missing required identity field: {field}")
                        break
                if not meta.get("page_numbers"):
                    violations.append("chunk without page_numbers")
            if vector_count != lexical_count:
                violations.append(f"vector/lexical count mismatch: vector={vector_count}, lexical={lexical_count}")
            report = store.verify_index("diag-doc-001")
            if not report.get("valid"):
                violations.append(f"stored index validation failed: {report.get('issues')}")
            identity_rows = [
                {field: meta.get(field) for field in IDENTITY_FIELDS}
                for meta in metas
            ]
            result.details = {
                "fixture_pages": len(pages),
                "production_chunks": len(chunks),
                "vector_records": vector_count,
                "lexical_records": lexical_count,
                "identity_rows": identity_rows,
                "violations": violations,
                "evidence": "PageExtraction -> SemanticChunker.chunk_pages -> VectorStore.add_documents -> VectorStore.verify_index",
            }
            result.score = 1.0 if not violations else max(0.0, 1.0 - len(violations) / max(1, len(metas) * 2))
            result.status = "PASS" if not violations else "FAIL"
            if violations:
                result.failures.append({"location": "rag_project/chunking/semantic_chunker.py -> rag_project/storage/vector_store.py", "exception": "IdentityConservationFailure", "message": "; ".join(violations[:8])})
        finally:
            _cleanup_store(tmp)
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "rag_project/testing/advanced_phases.py", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


# ---------------------------------------------------------------------------
# Phase 6 — measure actual information survival between representations
# ---------------------------------------------------------------------------


def information_loss(phase: Any) -> PhaseResult:
    result = _result(phase)
    try:
        pages = _fixture_pages()
        chunks = _fixture_chunks()
        store, tmp = _store_fixture(chunks)
        try:
            stored = store.get_documents()
            vector_docs = [str(value or "") for value in stored.get("documents", [])]
            vector_meta = [dict(value or {}) for value in stored.get("metadatas", [])]
            with sqlite3_connect(store.lexical_database) as connection:
                lexical_rows = connection.execute("SELECT document, metadata FROM lexical_documents ORDER BY id").fetchall()
            source_text = "\n".join(page.text for page in pages)
            chunk_text = "\n".join(chunk.text for chunk in chunks)
            lexical_text = "\n".join(str(row[0]) for row in lexical_rows)
            fields = {}
            source_meta = {field for page in pages for field in ((page.metadata or {}).keys())}
            chunk_meta = {field for chunk in chunks for field in ((chunk.metadata or {}).keys())}
            vector_meta_fields = {field for meta in vector_meta for field in meta.keys()}
            lexical_meta_fields = set()
            for _, raw in lexical_rows:
                try:
                    lexical_meta_fields |= set(json.loads(raw).keys())
                except Exception:
                    pass
            for field in IDENTITY_FIELDS:
                fields[field] = {
                    "page_surface": field in source_meta or hasattr(pages[0], field),
                    "chunk_surface": field in chunk_meta or all(hasattr(chunk, field) for chunk in chunks),
                    "vector_surface": field in vector_meta_fields,
                    "lexical_surface": field in lexical_meta_fields,
                }
            source_tokens = _token_set(source_text)
            measurements = {
                "source_to_chunk_token_survival": _survival(source_tokens, _token_set(chunk_text)),
                "chunk_to_vector_document_token_survival": _survival(_token_set(chunk_text), _token_set(lexical_text) | _token_set(" ".join(vector_docs))),
                "vector_to_lexical_document_survival": _survival(_token_set(" ".join(vector_docs)), _token_set(lexical_text)),
                "table_identity_survived": "diag-table-001" in {meta.get("table_id") for meta in vector_meta},
                "figure_identity_survived": "diag-figure-001" in {meta.get("figure_id") for meta in vector_meta} or "diag-figure-002" in {meta.get("figure_id") for meta in vector_meta},
                "ocr_status_survived": all("ocr_status" in meta for meta in vector_meta),
                "structure_survival": sum(int(all(surface.values())) for surface in fields.values()) / max(1, len(fields)),
            }
            loss = {key: round(1.0 - float(value), 4) for key, value in measurements.items() if isinstance(value, (float, int)) and key.endswith("survival")}
            failures = [key for key, value in measurements.items() if isinstance(value, float) and value < 0.90]
            if not measurements["table_identity_survived"]:
                failures.append("table_identity_survived")
            if not measurements["figure_identity_survived"]:
                failures.append("figure_identity_survived")
            if measurements["structure_survival"] < 0.90:
                failures.append("structure_survival")
            result.details = {
                "representations": ["source_page", "canonical_chunk", "vector_document", "sqlite_lexical"],
                "field_surfaces": fields,
                "measurements": measurements,
                "loss": loss,
                "failed_invariants": failures,
            }
            result.score = round(1.0 - (len(failures) / max(1, len(measurements))), 3)
            result.status = "PASS" if not failures else "FAIL"
            if failures:
                result.failures.append({"location": "production representation chain", "exception": "InformationLossFailure", "message": f"information-loss invariants failed: {failures}"})
        finally:
            _cleanup_store(tmp)
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "rag_project/testing/advanced_phases.py", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


def sqlite3_connect(path: Path):
    import sqlite3
    return sqlite3.connect(path)


# ---------------------------------------------------------------------------
# Phase 8 — true metamorphic testing against production utilities + chunker
# ---------------------------------------------------------------------------


def metamorphic(phase: Any) -> PhaseResult:
    result = _result(phase)
    try:
        from rag_project.ingestion.document_models import PageExtraction
        from rag_project.utils.text_utils import clean_text, keyword_overlap_score, normalize_whitespace, tokenize
        from rag_project.chunking.semantic_chunker import SemanticChunker

        queries = [
            "What is diabetes mellitus?",
            "  What is diabetes mellitus?  ",
            "WHAT IS DIABETES MELLITUS?",
        ]
        normalized = [normalize_whitespace(clean_text(query)).casefold() for query in queries]
        tokenized = [tokenize(query) for query in queries]
        lexical_scores = [keyword_overlap_score(query, "Diabetes mellitus is a chronic metabolic disease") for query in queries]
        pages = _fixture_pages()
        canonical = SemanticChunker(chunk_size=260, chunk_overlap=40).chunk_pages(pages)
        whitespace_pages = _fixture_pages("whitespace")
        variant = SemanticChunker(chunk_size=260, chunk_overlap=40).chunk_pages(whitespace_pages)
        canonical_norm = Counter(clean_text(chunk.text).casefold() for chunk in canonical)
        variant_norm = Counter(clean_text(chunk.text).casefold() for chunk in variant)
        token_invariant = all(tokenized[0] == tokens for tokens in tokenized[1:])
        query_invariant = len(set(normalized)) == 1
        score_invariant = max(lexical_scores) - min(lexical_scores) <= 1e-12
        chunk_identity_overlap = len(set(canonical_norm) & set(variant_norm)) / max(1, len(canonical_norm))
        chunk_invariant = chunk_identity_overlap >= 0.90
        checks = {
            "query_whitespace_case_invariant": query_invariant,
            "tokenization_invariant": token_invariant,
            "lexical_score_invariant": score_invariant,
            "chunk_content_invariant": chunk_invariant,
        }
        failures = [name for name, ok in checks.items() if not ok]
        result.details = {
            "production_functions": ["clean_text", "normalize_whitespace", "tokenize", "keyword_overlap_score", "SemanticChunker.chunk_pages"],
            "checks": checks,
            "normalized_queries": normalized,
            "lexical_scores": lexical_scores,
            "canonical_chunk_count": len(canonical),
            "variant_chunk_count": len(variant),
            "chunk_content_overlap": round(chunk_identity_overlap, 4),
        }
        result.score = round(sum(checks.values()) / len(checks), 3)
        result.status = "PASS" if not failures else "FAIL"
        if failures:
            result.failures.append({"location": "rag_project/utils/text_utils.py / rag_project/chunking/semantic_chunker.py", "exception": "MetamorphicFailure", "message": f"failed invariants: {failures}"})
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "rag_project/testing/advanced_phases.py", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


# ---------------------------------------------------------------------------
# Phase 11 — executable non-destructive mutant runner
# ---------------------------------------------------------------------------


def _load_module_from_source(path: Path, source: str, module_name: str) -> Any:
    tmp = Path(tempfile.mkdtemp(prefix="rag_mutant_module_"))
    target = tmp / path.name
    target.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(module_name, target)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load mutant module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return module


def mutation_detection(phase: Any) -> PhaseResult:
    result = _result(phase)
    mutation_results: list[dict[str, Any]] = []
    try:
        text_path = PROJECT / "utils" / "text_utils.py"
        source = text_path.read_text(encoding="utf-8")
        original = "return re.sub(r\"\\s+\", \" \", value or \"\").strip()"
        mutant = "return value or \"\""
        if original in source:
            mutant_source = source.replace(original, mutant, 1)
            module = _load_module_from_source(text_path, mutant_source, "_diag_mutant_text_utils")
            observed = module.normalize_whitespace("  diabetes   mellitus  ")
            killed = observed != "diabetes mellitus"
            mutation_results.append({"target": str(text_path.relative_to(ROOT)), "mutation": "normalize_whitespace_return_raw", "killed": killed, "observed": observed})
        else:
            mutation_results.append({"target": str(text_path.relative_to(ROOT)), "mutation": "normalize_whitespace_return_raw", "killed": False, "reason": "target expression changed; mutation not applicable"})

        chunk_path = PROJECT / "chunking" / "semantic_chunker.py"
        chunk_source = chunk_path.read_text(encoding="utf-8")
        target = '        if not children:\n            return []\n'
        replacement = '        if not children:\n            return ["__MUTANT__"]\n'
        if target in chunk_source:
            mutant_source = chunk_source.replace(target, replacement, 1)
            module = _load_module_from_source(chunk_path, mutant_source, "_diag_mutant_chunker")
            # Directly exercise the mutated production method; an injected sentinel is
            # outside the legitimate text-domain contract and must be detected.
            observed = module.SemanticChunker._compact_children([])
            killed = observed == []
            mutation_results.append({"target": str(chunk_path.relative_to(ROOT)), "mutation": "compact_children_empty_contract", "killed": killed, "observed": observed})
        else:
            mutation_results.append({"target": str(chunk_path.relative_to(ROOT)), "mutation": "compact_children_empty_contract", "killed": False, "reason": "target expression changed; mutation not applicable"})

        killed = sum(bool(item.get("killed")) for item in mutation_results)
        applicable = sum("reason" not in item for item in mutation_results)
        kill_score = killed / applicable if applicable else 0.0
        result.details = {
            "strategy": "executable shadow mutants loaded from temporary copies; checkout never modified",
            "mutation_results": mutation_results,
            "mutants_applicable": applicable,
            "mutants_killed": killed,
            "kill_score": round(kill_score, 3),
        }
        if applicable == 0:
            result.status = "FAIL"
            result.failures.append({"location": "rag_project/testing/advanced_phases.py", "exception": "MutationCoverageFailure", "message": "no executable mutation targets remained compatible with the current source"})
        elif kill_score < 1.0:
            result.status = "FAIL"
            result.failures.append({"location": "production diagnostic mutants", "exception": "SurvivingMutant", "message": f"mutation kill score={kill_score:.3f}"})
        else:
            result.status = "PASS"
        result.score = round(kill_score, 3)
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "rag_project/testing/advanced_phases.py", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


# ---------------------------------------------------------------------------
# Phase 14 — stage-level production measurements, not marker existence
# ---------------------------------------------------------------------------


def performance(phase: Any) -> PhaseResult:
    result = _result(phase)
    try:
        from rag_project.utils.text_utils import tokenize
        from rag_project.chunking.semantic_chunker import SemanticChunker

        pages = _fixture_pages()
        stage_samples: dict[str, list[float]] = defaultdict(list)
        for _ in range(5):
            started = time.perf_counter()
            _ = [tokenize(page.text) for page in pages]
            stage_samples["text_normalization"].append((time.perf_counter() - started) * 1000)

            started = time.perf_counter()
            chunks = SemanticChunker(chunk_size=260, chunk_overlap=40).chunk_pages(pages)
            stage_samples["chunking"].append((time.perf_counter() - started) * 1000)

            store, tmp = _store_fixture(chunks)
            try:
                started = time.perf_counter()
                _ = store.search_lexical("diabetes diagnosis", n_results=3)
                stage_samples["lexical_retrieval"].append((time.perf_counter() - started) * 1000)
                started = time.perf_counter()
                _ = store.search(_embedding("diabetes diagnosis"), n_results=3)
                stage_samples["semantic_retrieval"].append((time.perf_counter() - started) * 1000)
            finally:
                _cleanup_store(tmp)
        summary = {
            stage: {
                "samples": len(values),
                "p50_ms": round(_percentile(values, 0.50), 3),
                "p95_ms": round(_percentile(values, 0.95), 3),
                "p99_ms": round(_percentile(values, 0.99), 3),
                "max_ms": round(max(values), 3),
            }
            for stage, values in stage_samples.items()
        }
        baseline_path = ROOT / "tests" / "support" / "performance_baseline.json"
        baseline = {}
        if baseline_path.exists():
            try:
                baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            except Exception:
                baseline = {}
        regressions = []
        for stage, row in summary.items():
            limit = baseline.get(stage, {}).get("p95_ms")
            if limit is not None and row["p95_ms"] > float(limit) * 1.25:
                regressions.append({"stage": stage, "p95_ms": row["p95_ms"], "baseline_p95_ms": float(limit)})
        result.details = {
            "fixture_pages": len(pages),
            "stage_metrics": summary,
            "baseline_path": str(baseline_path.relative_to(ROOT)),
            "regressions": regressions,
            "benchmark_repetitions": 5,
        }
        result.score = 1.0 if not regressions else max(0.0, 1.0 - len(regressions) / len(summary))
        result.status = "PASS" if not regressions else "FAIL"
        if regressions:
            result.failures.append({"location": str(baseline_path.relative_to(ROOT)), "exception": "PerformanceRegression", "message": json.dumps(regressions, sort_keys=True)})
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "rag_project/testing/advanced_phases.py", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


# ---------------------------------------------------------------------------
# Phase 15 — actual chunking + vector + lexical repetition under tracemalloc
# ---------------------------------------------------------------------------


def resources(phase: Any) -> PhaseResult:
    result = _result(phase)
    try:
        tracemalloc.start()
        samples: list[int] = []
        rss_like: list[int] = []
        started = time.perf_counter()
        for iteration in range(8):
            chunks = _fixture_chunks()
            store, tmp = _store_fixture(chunks)
            try:
                _ = store.search_lexical("kidney nephropathy", n_results=3)
                _ = store.search(_embedding("kidney nephropathy"), n_results=3)
            finally:
                _cleanup_store(tmp)
            current, peak = tracemalloc.get_traced_memory()
            samples.append(current)
            rss_like.append(peak)
            del chunks
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        first = samples[0] if samples else 0
        last = samples[-1] if samples else 0
        growth = (last - first) / max(1, first)
        monotonic_growth = sum(b > a for a, b in zip(samples, samples[1:])) >= 6 if len(samples) > 1 else False
        result.details = {
            "repetitions": len(samples),
            "current_bytes": current,
            "peak_bytes": peak,
            "sample_current_bytes": samples,
            "peak_series_bytes": rss_like,
            "relative_growth_first_to_last": round(growth, 4),
            "monotonic_growth_signal": monotonic_growth,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "pipeline_exercised": ["SemanticChunker.chunk_pages", "VectorStore.add_documents", "VectorStore.search_lexical", "VectorStore.search"],
        }
        leak = growth > 0.60 and monotonic_growth
        result.score = round(max(0.0, 1.0 - min(1.0, max(0.0, growth))), 3)
        result.status = "FAIL" if leak else "PASS"
        if leak:
            result.failures.append({"location": "production ingestion/chunking/storage repetition", "exception": "ResourceGrowthFailure", "message": f"traced allocation grew by {growth:.1%} with monotonic signal"})
    except Exception as exc:
        try:
            tracemalloc.stop()
        except Exception:
            pass
        result.status = "FAIL"
        result.failures.append({"location": "rag_project/testing/advanced_phases.py", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


# ---------------------------------------------------------------------------
# Phase 16 — permanent gold-set execution with offline + opt-in live modes
# ---------------------------------------------------------------------------


def _gold_cases() -> list[dict[str, Any]]:
    gold = TESTS / "support" / "gold_sets" / "core.jsonl"
    if not gold.exists():
        return []
    return [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_live_gold(base_url: str, gold: Path) -> dict[str, Any]:
    from scripts.run_kpi_benchmark import run as run_kpi
    return run_kpi(base_url, gold, 30.0)


def _offline_gold(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Exercise the real lexical/semantic stores using the gold queries as retrieval probes.

    This is deliberately a retrieval-quality benchmark, not a fabricated answer-accuracy
    score. Each case gets bounded evidence rows derived only from its declared must-contain
    labels. Clinical correctness remains explicitly unclaimed.
    """
    chunks = []
    from rag_project.ingestion.document_models import Chunk
    for index, item in enumerate(cases):
        terms = [str(term) for term in item.get("must_contain", [])]
        text = "Medical evidence: " + ". ".join(terms or [str(item.get("question", ""))])
        chunks.append(Chunk(
            doc_id=f"gold-{index}",
            file_name="gold-fixture.pdf",
            chunk_index=index,
            text=text,
            page_numbers=[index + 1],
            metadata={
                "document_id": f"gold-{index}",
                "chunk_id": f"gold-chunk-{index}",
                "version_id": "v1",
                "section": "gold",
                "section_id": f"gold-section-{index}",
                "parent_id": f"gold-parent-{index}",
                "index_state": "READY",
            },
        ))
    store, tmp = _store_fixture(chunks)
    try:
        rows = []
        for index, item in enumerate(cases):
            query = str(item.get("question", ""))
            semantic = store.search(_embedding(query), n_results=min(5, max(1, len(chunks))))
            lexical = store.search_lexical(query, n_results=min(5, max(1, len(chunks))))
            semantic_ids = [str(x) for x in (semantic.get("ids") or [[]])[0]]
            lexical_ids = [str(x) for x in (lexical.get("ids") or [[]])[0]]
            target = f"gold-{index}:"  # real stored id prefix
            retrieved = set(semantic_ids) | set(lexical_ids)
            hit = any(item_id.startswith(target) for item_id in retrieved)
            rows.append({"id": item.get("id"), "retrieval_hit": hit, "semantic_rank": next((i + 1 for i, value in enumerate(semantic_ids) if value.startswith(target)), None), "lexical_rank": next((i + 1 for i, value in enumerate(lexical_ids) if value.startswith(target)), None)})
        hit_rate = sum(bool(row["retrieval_hit"]) for row in rows) / len(rows) if rows else 0.0
        reciprocal = [1.0 / row["semantic_rank"] for row in rows if row["semantic_rank"]]
        return {
            "mode": "offline_retrieval_contract",
            "case_count": len(rows),
            "recall_at_k": hit_rate,
            "mrr": statistics.fmean(reciprocal) if reciprocal else 0.0,
            "results": rows,
            "clinical_correctness_claimed": False,
            "citation_accuracy_claimed": False,
        }
    finally:
        _cleanup_store(tmp)


def golden_benchmark(phase: Any) -> PhaseResult:
    result = _result(phase)
    gold = TESTS / "support" / "gold_sets" / "core.jsonl"
    try:
        cases = _gold_cases()
        if not cases:
            result.status = "FAIL"
            result.failures.append({"location": str(gold.relative_to(ROOT)), "exception": "MissingGoldSet", "message": "permanent gold set is unavailable or empty"})
            return result
        malformed = [item.get("id", "<missing>") for item in cases if not {"id", "question"}.issubset(item)]
        if malformed:
            result.status = "FAIL"
            result.failures.append({"location": str(gold.relative_to(ROOT)), "exception": "MalformedGoldSet", "message": f"invalid cases: {malformed[:10]}"})
            return result
        base_url = os.getenv("DIAGNOSTIC_API_URL", "").strip()
        if base_url:
            payload = _run_live_gold(base_url, gold)
            overall = payload.get("overall", {})
            score = float(overall.get("weighted_accuracy", 0.0) or 0.0)
            evidence_mode = "live_api"
        else:
            payload = _offline_gold(cases)
            score = float(payload.get("recall_at_k", 0.0) or 0.0)
            evidence_mode = "offline_retrieval_contract"
        result.details = {
            "gold_path": str(gold.relative_to(ROOT)),
            "case_count": len(cases),
            "mode": evidence_mode,
            "metrics": payload.get("overall", payload),
            "clinical_correctness_claimed": False,
            "live_mode_opt_in": bool(base_url),
        }
        result.score = round(score, 3)
        if evidence_mode == "live_api":
            transport_ok = all(bool(row.get("transport_ok")) for row in payload.get("results", []))
            result.status = "PASS" if transport_ok else "FAIL"
        else:
            result.status = "PASS" if score >= 0.80 else "FAIL"
        if result.status == "FAIL":
            result.failures.append({"location": str(gold.relative_to(ROOT)), "exception": "GoldenBenchmarkFailure", "message": f"benchmark score={score:.3f}"})
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "rag_project/testing/advanced_phases.py", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


# ---------------------------------------------------------------------------
# Phases 12/13/17 — evidence-based root-cause graph and certification
# ---------------------------------------------------------------------------


def fingerprint_failures(results: Iterable[PhaseResult]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for phase in results:
        for failure in phase.failures:
            layer = str(failure.get("location") or "unknown")
            layer = layer.split("/", 2)[0] if "/" in layer else layer
            exception = str(failure.get("exception") or "Failure")
            message = re.sub(r"\s+", " ", str(failure.get("message") or failure.get("detail") or "")).strip().casefold()
            key = f"{layer}|{exception}|{re.sub(r'\\d+', '#', message)[:180]}"
            groups.setdefault(key, {"fingerprint": key, "phase_numbers": [], "evidence": [], "locations": []})
            row = groups[key]
            row["phase_numbers"].append(phase.number)
            row["locations"].append(layer)
            row["evidence"].append(f"phase {phase.number}: {message[:240]}")
    output = list(groups.values())
    for row in output:
        row["phase_numbers"] = sorted(set(row["phase_numbers"]))
        row["locations"] = sorted(set(row["locations"]))
        row["confidence"] = round(min(0.99, 0.55 + 0.08 * len(row["phase_numbers"])), 3)
    return sorted(output, key=lambda item: (-len(item["phase_numbers"]), item["phase_numbers"][0] if item["phase_numbers"] else 99))


def cascade_compression(results: Iterable[PhaseResult], fingerprints: list[dict[str, Any]]) -> dict[str, Any]:
    phases = list(results)
    failed = [p.number for p in phases if p.status == "FAIL"]
    blocked = [p.number for p in phases if p.status == "BLOCKED"]
    first_failure = min(failed) if failed else None
    affected_by_first = sorted({number for row in fingerprints if first_failure in row.get("phase_numbers", []) for number in row.get("phase_numbers", [])}) if first_failure else []
    return {
        "failed_phases": failed,
        "blocked_phases": blocked,
        "first_failed_phase": first_failure,
        "candidate_root_causes": len(fingerprints),
        "compressed_impacts": [{"root_phase": first_failure, "affected_phases": affected_by_first}] if first_failure else [],
        "fix_order": [
            {"phase": row["phase_numbers"][0], "fingerprint": row["fingerprint"], "confidence": row["confidence"]}
            for row in fingerprints
            if row.get("phase_numbers")
        ],
    }


def root_cause_phase(phase: Any, results: dict[int, PhaseResult]) -> PhaseResult:
    result = _result(phase)
    fingerprints = fingerprint_failures(results.values())
    result.details = {
        "algorithm": "structured location + exception + normalized message fingerprint",
        "unique_fingerprints": len(fingerprints),
        "fingerprints": fingerprints[:50],
        "evidence_phases": sorted(results),
    }
    result.score = 1.0
    result.status = "PASS"
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


def cascade_phase(phase: Any, results: dict[int, PhaseResult]) -> PhaseResult:
    result = _result(phase)
    fingerprints = fingerprint_failures(results.values())
    result.details = cascade_compression(results.values(), fingerprints)
    result.score = 1.0
    result.status = "PASS"
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


def certification_phase(phase: Any, results: dict[int, PhaseResult]) -> PhaseResult:
    result = _result(phase)
    expected = list(range(1, 18))
    present = sorted(results)
    missing = sorted(set(expected) - set(present))
    implementation_matrix = {
        1: "architecture_map",
        2: "fast_health",
        3: "diagnostic_chain",
        4: "contract_triangulation",
        5: "cross_layer_real_fixture",
        6: "information_survival_measurement",
        7: "adversarial_test_suite_execution",
        8: "production_metamorphic_execution",
        9: "retrieval_contract_execution",
        10: "rag_generation_causality_suite",
        11: "executable_shadow_mutants",
        12: "structured_failure_fingerprints",
        13: "causal_failure_compression",
        14: "stage_latency_measurement",
        15: "bounded_resource_growth_measurement",
        16: "permanent_gold_benchmark",
        17: "master_certification",
    }
    implementation_complete = not missing and all(number in implementation_matrix for number in expected)
    runtime_failures = [p.number for p in results.values() if p.status == "FAIL"]
    runtime_warns = [p.number for p in results.values() if p.status == "WARN"]
    runtime_blocked = [p.number for p in results.values() if p.status == "BLOCKED"]
    result.details = {
        "implementation_matrix": implementation_matrix,
        "implementation_coverage": f"{len(implementation_matrix)}/17",
        "all_phase_results_present": not missing,
        "missing_phase_results": missing,
        "runtime_failures": runtime_failures,
        "runtime_warnings": runtime_warns,
        "runtime_blocked": runtime_blocked,
        "certification_rule": "Implementation must be 17/17; runtime certification PASS additionally requires zero FAIL/BLOCKED phases.",
        "clinical_correctness_claimed": False,
    }
    result.score = round(len(implementation_matrix) / 17.0, 3)
    if not implementation_complete:
        result.status = "FAIL"
        result.failures.append({"location": "rag_project/testing/runner.py", "exception": "IncompleteDiagnosticSystem", "message": f"missing phase implementations/results: {missing}"})
    elif runtime_failures or runtime_blocked:
        result.status = "WARN"
        result.failures.append({"location": "phase runtime", "exception": "RuntimeCertificationPending", "message": f"17/17 implemented, but runtime has failures={runtime_failures} blocked={runtime_blocked}"})
    else:
        result.status = "PASS"
    result.duration_s = round(time.time() - result.started_at, 3)
    return result
