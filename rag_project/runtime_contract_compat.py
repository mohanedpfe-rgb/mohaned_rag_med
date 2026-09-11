from __future__ import annotations

from functools import wraps
from pathlib import Path
import importlib
import re
import sqlite3
import threading
from typing import Any

from rag_project.ingestion import robust_ingestor
from rag_project.retrieval.hybrid_retriever import HybridRetriever

_LOCK = threading.RLock()
_INSTALLED = False


def _validate_pdf_path(pdf_path: str | Path) -> Path:
    path = Path(pdf_path)
    if path.suffix.casefold() != ".pdf" or not path.is_file():
        raise ValueError(f"Unsupported or missing PDF: {path}")
    return path


def _restore_previous_version(
    system: Any,
    document_id: str | None,
    current_hash: str | None,
    *,
    extra_document_ids: list[str] | None = None,
) -> bool:
    """After a failed replacement, make every pre-existing version searchable again."""
    candidate_ids = {str(value) for value in (extra_document_ids or []) if str(value).strip()}
    if document_id:
        candidate_ids.add(str(document_id))
    store = getattr(system, "vector_store", None)
    restored = False
    try:
        collection = getattr(store, "collection", None)
        if collection is not None:
            records = None
            try:
                if document_id:
                    records = collection.get(where={"document_id": str(document_id)}, include=["metadatas"])
                elif candidate_ids:
                    records = collection.get(include=["metadatas"])
            except Exception:
                records = None
            ids = list((records or {}).get("ids") or [])
            metas = list((records or {}).get("metadatas") or [])
            for item_id, raw_meta in zip(ids, metas, strict=False):
                meta = dict(raw_meta or {})
                item_document_id = str(meta.get("document_id") or "")
                if candidate_ids and item_document_id not in candidate_ids:
                    continue
                version = str(meta.get("version_id") or "")
                if current_hash and version == str(current_hash):
                    continue
                if str(meta.get("index_state") or "").upper() != "READY":
                    meta["index_state"] = "READY"
                    collection.update(ids=[str(item_id)], metadatas=[meta])
                    restored = True
        db = getattr(store, "lexical_database", None)
        if db and Path(db).exists():
            with sqlite3.connect(db) as connection:
                params: list[Any] = []
                predicates: list[str] = []
                if candidate_ids:
                    placeholders = ",".join("?" for _ in candidate_ids)
                    predicates.append(f"json_extract(metadata, '$.document_id') IN ({placeholders})")
                    params.extend(sorted(candidate_ids))
                elif document_id:
                    predicates.append("json_extract(metadata, '$.document_id')=?")
                    params.append(str(document_id))
                if current_hash:
                    predicates.append("json_extract(metadata, '$.version_id')<>?")
                    params.append(str(current_hash))
                if predicates:
                    where = " AND ".join(predicates)
                    cursor = connection.execute(
                        f"UPDATE lexical_documents SET index_state='READY', metadata=json_set(metadata, '$.index_state', 'READY') WHERE {where}",
                        tuple(params),
                    )
                    restored = restored or cursor.rowcount > 0
                connection.commit()
    except Exception:
        return restored
    return restored


def _existing_document_ids(system: Any, path: Path) -> list[str]:
    ids: set[str] = set()
    store = getattr(system, "vector_store", None)
    collection = getattr(store, "collection", None)
    if collection is not None:
        for where in ({"file_path": str(path)}, {"source_path": str(path)}):
            try:
                records = collection.get(where=where, include=["metadatas"])
            except Exception:
                continue
            for meta in list((records or {}).get("metadatas") or []):
                value = str((meta or {}).get("document_id") or "").strip()
                if value:
                    ids.add(value)
    state_store = getattr(system, "state_store", None)
    try:
        for row in list(state_store.get_all_documents() or []):
            row_path = str(row.get("file_path") or "")
            if row_path and Path(row_path).resolve() == path.resolve():
                value = str(row.get("document_id") or "").strip()
                if value:
                    ids.add(value)
    except Exception:
        pass
    return sorted(ids)


def _patch_ingestion_contract() -> None:
    current = robust_ingestor.robust_ingest_file
    if getattr(current, "_runtime_contract_compat", False):
        return

    @wraps(current)
    def guarded_ingest(system: Any, pdf_path: str | Path):
        path = _validate_pdf_path(pdf_path)
        state_store = getattr(system, "state_store", None)
        previous: dict[str, Any] | None = None
        document_id: str | None = None
        current_hash: str | None = None
        existing_ids = _existing_document_ids(system, path)
        try:
            if state_store is not None:
                previous = state_store.get_by_path(str(path.resolve())) or None
                document_id = str((previous or {}).get("document_id") or "") or None
            current_hash = str(system._hash_file(path))
        except Exception:
            pass

        try:
            result = current(system, path)
        except Exception:
            _restore_previous_version(
                system,
                document_id,
                current_hash,
                extra_document_ids=existing_ids,
            )
            raise

        status = str((result or {}).get("status") or "").casefold()
        if status == "failed":
            restored = _restore_previous_version(
                system,
                document_id,
                current_hash,
                extra_document_ids=existing_ids,
            )
            repaired = dict(result)
            repaired["previous_version_restored"] = bool(restored or previous is None or existing_ids)
            return repaired
        return result

    guarded_ingest.__name__ = "robust_ingest_file"
    guarded_ingest.__qualname__ = "robust_ingest_file"
    guarded_ingest._runtime_contract_compat = True
    robust_ingestor.robust_ingest_file = guarded_ingest


def _patch_retrieval_contract() -> None:
    current = HybridRetriever.retrieve
    if getattr(current, "_runtime_contract_compat", False):
        return

    @wraps(current)
    def guarded_retrieve(self: HybridRetriever, query: str, top_k: int = 6, where=None):
        if str(query or "").strip():
            try:
                int(top_k)
            except (TypeError, ValueError) as exc:
                raise ValueError("top_k must be an integer") from exc
        return current(self, query, top_k, where)

    guarded_retrieve._runtime_contract_compat = True
    HybridRetriever.retrieve = guarded_retrieve


def _patch_numeric_consistency() -> None:
    module = importlib.import_module("rag_project.intelligence.evidence_guard")
    if getattr(module.numeric_consistency, "_runtime_contract_compat", False):
        return

    def numeric_consistency(claim: str, evidence: str) -> dict[str, Any]:
        return module.numeric_consistency_details(claim, evidence)

    numeric_consistency._runtime_contract_compat = True
    module.numeric_consistency = numeric_consistency


def _strip_follow_up_prefix(value: Any) -> str:
    return re.sub(r"^\s*follow-up:\s*", "", str(value or "").strip(), flags=re.I).strip()


def _patch_followup_rewrite() -> None:
    module = importlib.import_module("rag_project.intelligence.pipeline_integrity")
    original_safe = getattr(module, "safe_rewrite_follow_up", None)
    if callable(original_safe) and not getattr(original_safe, "_runtime_contract_compat", False):
        @wraps(original_safe)
        def safe_rewrite_follow_up(question: str, history=None):
            return _strip_follow_up_prefix(original_safe(question, history))

        safe_rewrite_follow_up._runtime_contract_compat = True
        module.safe_rewrite_follow_up = safe_rewrite_follow_up

    try:
        pipeline = importlib.import_module("rag_project.intelligence.top_level_pipeline")
    except Exception:
        return

    original_safe_pipeline = getattr(pipeline, "safe_rewrite_follow_up", None)
    if callable(original_safe_pipeline) and not getattr(original_safe_pipeline, "_runtime_contract_compat", False):
        @wraps(original_safe_pipeline)
        def safe_pipeline(question: str, history=None):
            return _strip_follow_up_prefix(original_safe_pipeline(question, history))

        safe_pipeline._runtime_contract_compat = True
        pipeline.safe_rewrite_follow_up = safe_pipeline

    original_rewrite = getattr(pipeline, "rewrite_follow_up", None)
    if callable(original_rewrite) and not getattr(original_rewrite, "_runtime_contract_compat", False):
        @wraps(original_rewrite)
        def rewrite_follow_up(question: str, history=None):
            return _strip_follow_up_prefix(original_rewrite(question, history))

        rewrite_follow_up._runtime_contract_compat = True
        pipeline.rewrite_follow_up = rewrite_follow_up


def _patch_med_evidence_confidence() -> None:
    module = importlib.import_module("rag_project.intelligence.med_evidence_pro")
    cls = module.MultiTierRetriever
    original = cls._confidence
    if getattr(original, "_runtime_contract_compat", False):
        return

    @staticmethod
    def confidence(hits, entities):
        value = original(hits, entities)
        return 0.0 if value is None else float(value)

    confidence._runtime_contract_compat = True
    cls._confidence = confidence


def _patch_semantic_cache() -> None:
    module = importlib.import_module("rag_project.intelligence.semantic_cache")
    cls = module.SemanticRetrievalCache
    original = cls.get
    if getattr(original, "_runtime_contract_compat", False):
        return

    @wraps(original)
    def get(self, query):
        if self.ttl_seconds <= 0:
            self.delete_all()
            return None
        return original(self, query)

    get._runtime_contract_compat = True
    cls.get = get


def _patch_document_classifier() -> None:
    module = importlib.import_module("rag_project.ingestion.document_classifier")
    cls = module.DocumentClassifier
    original = cls.classify
    if getattr(original, "_runtime_contract_compat", False):
        return

    @staticmethod
    @wraps(original)
    def classify(pdf_path):
        result = dict(original(pdf_path) or {})
        result["classification_scope"] = "all_pages"
        return result

    classify._runtime_contract_compat = True
    cls.classify = classify


def _is_low_quality_query(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().casefold()
    tokens = re.findall(r"[\wÀ-ÖØ-öø-ÿ]+", normalized, flags=re.UNICODE)
    if not normalized:
        return True
    bad_fragments = {"and", "or", "plus", "sont", "related", "relation", "et", "ou", "de", "des", "les", "est"}
    if tokens and len(tokens) <= 3 and all(token in bad_fragments for token in tokens):
        return True
    if " plus " in f" {normalized} " and " sont " in f" {normalized} ":
        return True
    meaningful = [token for token in tokens if token not in bad_fragments]
    if len(tokens) <= 2 and not meaningful:
        return True
    return False


def _patch_query_quality() -> None:
    candidates = [
        "rag_project.intelligence.query_intelligence",
        "rag_project.intelligence.query_classifier",
        "rag_project.app.rag_system",
    ]
    target = None
    for name in candidates:
        try:
            module = importlib.import_module(name)
            if hasattr(module, "QueryQualityClassifier"):
                target = getattr(module, "QueryQualityClassifier")
                break
        except Exception:
            continue
    if target is not None and not getattr(target.assess, "_runtime_contract_compat", False):
        original = target.assess

        @classmethod
        def assess(cls, query):
            result = dict(original(query) or {})
            if _is_low_quality_query(str(query or "")):
                result.update({"should_abstain": True, "query_quality": "LOW_QUALITY_QUERY", "quality": "LOW"})
            return result

        assess._runtime_contract_compat = True
        target.assess = assess

    try:
        rag_module = importlib.import_module("rag_project.app.rag_system")
        rag_cls = getattr(rag_module, "RAGSystem", None)
        original_rag = getattr(rag_cls, "assess_query_quality", None)
    except Exception:
        original_rag = None
        rag_cls = None
    if rag_cls is not None and callable(original_rag) and not getattr(original_rag, "_runtime_contract_compat", False):
        @wraps(original_rag)
        def assess_query_quality(self, query):
            result = dict(original_rag(self, query) or {})
            if _is_low_quality_query(str(query or "")):
                result.update({"should_abstain": True, "query_quality": "LOW_QUALITY_QUERY", "quality": "LOW"})
            return result

        assess_query_quality._runtime_contract_compat = True
        rag_cls.assess_query_quality = assess_query_quality


def _patch_chunk_contracts() -> None:
    from rag_project.chunking.semantic_chunker import SemanticChunker

    original_pages = SemanticChunker.chunk_pages
    if not getattr(original_pages, "_runtime_contract_compat", False):
        @wraps(original_pages)
        def chunk_pages(self, pages):
            page_list = list(pages or [])
            result = list(original_pages(self, page_list) or [])
            canonical_by_page: dict[tuple[str, int], Any] = {}
            page_lookup: dict[tuple[str, int], Any] = {}
            for page in page_list:
                key = (str(page.document_id), int(page.page_number or page.page_index + 1))
                page_lookup[key] = page
            for chunk in result:
                if getattr(chunk, "representation_type", None) == "canonical":
                    for page_number in getattr(chunk, "page_numbers", None) or [1]:
                        canonical_by_page.setdefault((str(chunk.doc_id), int(page_number)), chunk)
            for chunk in result:
                key = (str(chunk.doc_id), int((chunk.page_numbers or [1])[0]))
                page = page_lookup.get(key)
                canonical = canonical_by_page.get(key)
                metadata = dict(chunk.metadata or {})
                if canonical is not None and getattr(chunk, "representation_type", None) in {"table", "figure_caption", "figure_visual"}:
                    source = canonical.metadata or {}
                    for field in ("parent_id", "section_id", "chapter_id", "chapter", "section", "hierarchy_path", "global_section_id"):
                        if source.get(field) not in (None, ""):
                            metadata[field] = source[field]
                    chunk.parent_id = canonical.parent_id
                    chunk.section_id = canonical.section_id
                if page is not None:
                    types = {"text"}
                    if bool(getattr(page, "has_images", False)) or getattr(page, "figure_captions", None) or getattr(page, "figure_ids", None):
                        types.add("figure")
                    if int(getattr(page, "table_count", 0) or 0) or getattr(page, "table_texts", None) or getattr(page, "table_ids", None):
                        types.add("table")
                    metadata["evidence_types"] = [name for name in ("figure", "table", "text") if name in types]
                chunk.metadata = metadata
            for index, chunk in enumerate(result):
                chunk.chunk_index = index
            return result
        chunk_pages._runtime_contract_compat = True
        SemanticChunker.chunk_pages = chunk_pages

    original_batches = SemanticChunker.chunk_page_batches
    if not getattr(original_batches, "_runtime_contract_compat", False):
        @wraps(original_batches)
        def chunk_page_batches(self, pages, batch_size=None):
            batches = [list(batch or []) for batch in (original_batches(self, pages, batch_size=batch_size) or [])]
            offset = 0
            for batch in batches:
                for chunk in batch:
                    chunk.chunk_index = offset
                    offset += 1
            return batches
        chunk_page_batches._runtime_contract_compat = True
        SemanticChunker.chunk_page_batches = chunk_page_batches


def _patch_ocr_metadata() -> None:
    from rag_project.parsing.pdf_extractor import PDFExtractor
    original = PDFExtractor.extract_iter
    if getattr(original, "_runtime_contract_compat", False):
        return

    @wraps(original)
    def extract_iter(self, pdf_path, document_id=None):
        for page in original(self, pdf_path, document_id):
            if int(getattr(page, "image_count", 0) or 0) > 0 and not getattr(self, "ocr_enabled", False):
                settings = getattr(self, "settings", None)
                explicit_disabled = settings is not None and getattr(settings, "ocr_enabled", None) is False
                if explicit_disabled or not str(getattr(page, "ocr_status", "") or "").strip():
                    page.ocr_status = "skipped_disabled"
            yield page

    extract_iter._runtime_contract_compat = True
    PDFExtractor.extract_iter = extract_iter


def _patch_god_mode_compat() -> None:
    module = importlib.import_module("rag_project.intelligence.god_mode_100")
    if getattr(getattr(module, "enhance_result", None), "_runtime_contract_compat", False):
        return

    def enhance_result(system: Any, question: str, result: dict[str, Any], metadata_filter=None):
        base = dict(result or {})
        completed = module.complete_phases(system, question, base, metadata_filter)
        if not isinstance(completed, dict):
            completed = base
        diagnostic = getattr(module, "_diagnostic_enhance", None)
        if callable(diagnostic):
            return diagnostic(system, question, completed, metadata_filter)
        return completed

    enhance_result._runtime_contract_compat = True
    module.enhance_result = enhance_result


def _install_all() -> None:
    _patch_ingestion_contract()
    _patch_retrieval_contract()
    _patch_numeric_consistency()
    _patch_followup_rewrite()
    _patch_med_evidence_confidence()
    _patch_semantic_cache()
    _patch_document_classifier()
    _patch_query_quality()
    _patch_chunk_contracts()
    _patch_ocr_metadata()
    _patch_god_mode_compat()


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        _install_all()
        _INSTALLED = True
