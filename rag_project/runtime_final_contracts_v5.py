from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Any

_INSTALLED = False


def _patch_chroma_hierarchy_modify() -> None:
    """Never pass immutable HNSW settings back to Chroma.modify()."""
    try:
        from chromadb.api.models.Collection import Collection
    except Exception:
        return
    current = getattr(Collection, "modify", None)
    if not callable(current) or getattr(current, "_runtime_v5", False):
        return

    def modify(self, metadata=None, *args, **kwargs):
        if metadata is not None and str(getattr(self, "name", "")).endswith("_hierarchy"):
            # Chroma treats hnsw:space as immutable after collection creation.
            # Collection metadata may not expose the original metric consistently,
            # so the safest update is to omit all immutable HNSW keys entirely.
            merged = {
                key: value
                for key, value in dict(metadata).items()
                if not str(key).startswith("hnsw:")
            }
            metadata = merged
        return current(self, metadata=metadata, *args, **kwargs)

    modify._runtime_v5 = True
    Collection.modify = modify


def _patch_embedding_retries() -> None:
    from rag_project.embeddings.embedding_service import EmbeddingService

    current = getattr(EmbeddingService, "__init__", None)
    if not callable(current) or getattr(current, "_runtime_v5", False):
        return

    def init(self, *args, **kwargs):
        requested = kwargs.get("retries", 2)
        try:
            requested = int(requested)
        except (TypeError, ValueError):
            requested = 2
        current(self, *args, **kwargs)
        self.retries = max(0, min(requested, 3))

    init._runtime_v5 = True
    EmbeddingService.__init__ = init


def _patch_ocr_status_compat() -> None:
    from rag_project.parsing.pdf_extractor import PDFExtractor

    current = getattr(PDFExtractor, "extract_iter", None)
    if not callable(current) or getattr(current, "_runtime_v5", False):
        return

    def extract_iter(self, *args, **kwargs):
        for page in current(self, *args, **kwargs):
            if (
                not bool(getattr(self, "ocr_enabled", True))
                and bool(getattr(page, "ocr_required", False))
                and getattr(page, "ocr_status", None) == "failed"
            ):
                page.ocr_status = "skipped_disabled"
            yield page

    extract_iter._runtime_v5 = True
    PDFExtractor.extract_iter = extract_iter


def _patch_section_identity() -> None:
    from rag_project.chunking.semantic_chunker import SemanticChunker

    current = getattr(SemanticChunker, "chunk_pages", None)
    if not callable(current) or getattr(current, "_runtime_v5", False):
        return

    def chunk_pages(self, pages):
        chunks = list(current(self, pages) or [])
        for chunk in chunks:
            metadata = dict(getattr(chunk, "metadata", {}) or {})
            page_numbers = list(getattr(chunk, "page_numbers", []) or [])
            page = int(page_numbers[0] if page_numbers else 1)
            doc = str(getattr(chunk, "doc_id", "document"))
            section_id = str(metadata.get("section_id") or "")
            if not section_id.startswith(f"{doc}:p{page}:section:"):
                basis = str(
                    metadata.get("section")
                    or metadata.get("parent_id")
                    or getattr(chunk, "chunk_index", 0)
                )
                section_id = f"{doc}:p{page}:section:{hashlib.sha1(basis.encode('utf-8')).hexdigest()[:16]}"
                metadata["section_id"] = section_id
                metadata["global_section_id"] = section_id
                chunk.section_id = section_id
            chunk.metadata = metadata
        return chunks

    chunk_pages._runtime_v5 = True
    SemanticChunker.chunk_pages = chunk_pages


def _original_public_safe_rewrite(module: Any):
    """Recover the original imported safe_rewrite function when compat wrapped it."""
    safe = getattr(module, "safe_rewrite_follow_up", None)
    visited: set[int] = set()
    while callable(safe) and id(safe) not in visited:
        visited.add(id(safe))
        wrapped = getattr(safe, "__wrapped__", None)
        if not callable(wrapped):
            break
        safe = wrapped
    return safe


def _patch_followup_public_contract() -> None:
    from rag_project.intelligence import pipeline_integrity, top_level_pipeline

    original_safe = _original_public_safe_rewrite(pipeline_integrity)
    canonical_safe = original_safe or pipeline_integrity.safe_rewrite_follow_up

    # The standalone public helper has a deliberate clean contract.
    pipeline_integrity.safe_rewrite_follow_up = canonical_safe
    top_level_pipeline.rewrite_follow_up = canonical_safe

    original_install = getattr(pipeline_integrity, "install", None)
    if callable(original_install) and not getattr(original_install, "_runtime_v5", False):
        def install():
            original_install()
            pipeline_integrity.safe_rewrite_follow_up = canonical_safe
            top_level_pipeline.rewrite_follow_up = canonical_safe
        install._runtime_v5 = True
        pipeline_integrity.install = install


def _patch_numeric_boolean_contract() -> None:
    from rag_project.intelligence import evidence_guard

    def numeric_consistency(claim, evidence):
        return not bool(
            evidence_guard.numeric_consistency_details(claim, evidence).get("mismatch", False)
        )

    numeric_consistency._runtime_v5 = True
    evidence_guard.numeric_consistency = numeric_consistency


def _patch_query_rewriter_legacy_contract() -> None:
    from rag_project.retrieval.query_rewriter import QueryRewriter

    def rewrite(question: str, history=None, llm=None) -> str:
        cleaned = str(question or "").strip()
        if not cleaned:
            return ""
        if llm is not None:
            try:
                generated = str(llm.generate(cleaned, temperature=0.0) or "").strip()
                if generated:
                    return generated
            except Exception:
                pass
        explicit = bool(
            re.search(
                r"\b(what about|how about|it|this|that|they|them|those|these)\b",
                cleaned,
                re.I,
            )
            or re.match(r"^(and|also|then|et|puis|و|ثم)\b", cleaned, re.I | re.UNICODE)
        )
        if history and explicit:
            return f"{str(history[-1][0]).strip()} Follow-up question: {cleaned}"
        return cleaned

    QueryRewriter.rewrite = staticmethod(rewrite)


def _patch_god_mode_legacy_entrypoint() -> None:
    import rag_project.intelligence.god_mode_100 as module

    def enhance_result(system: Any, question: str, base_result: Any, metadata_filter=None):
        base = dict(base_result or {})
        complete = getattr(module, "complete_phases", None)
        if callable(complete):
            try:
                result = complete(system, question, base, metadata_filter)
                if isinstance(result, dict):
                    base = result
            except Exception:
                pass
        enhancer = getattr(module, "_diagnostic_enhance", None)
        return enhancer(system, question, base, metadata_filter) if callable(enhancer) else base

    module.enhance_result = enhance_result


def _patch_production_history() -> None:
    from rag_project.app.production_rag import ProductionRAGSystem

    current = getattr(ProductionRAGSystem, "answer", None)
    if not callable(current) or getattr(current, "_runtime_v5", False):
        return

    def answer(self, question: str, metadata_filter=None):
        memory = getattr(self, "conversation_memory", None)
        history = getattr(memory, "history", None) if memory is not None else None
        before = len(history) if isinstance(history, list) else None
        result = current(self, question, metadata_filter)
        if isinstance(result, dict) and isinstance(history, list):
            status = str(result.get("status") or "").upper()
            answer_text = str(result.get("answer") or "").strip()
            if status in {"SUCCESS", "SUCCESS_WITH_WARNINGS"} and answer_text:
                new_items = history[before:] if before is not None else []
                if not any(
                    isinstance(item, (tuple, list))
                    and item
                    and str(item[0]) == str(question or "").strip()
                    for item in new_items
                ):
                    history.append((str(question or "").strip(), answer_text))
        return result

    answer._runtime_v5 = True
    ProductionRAGSystem.answer = answer


def _snapshot_ready_state(system: Any, path: Path) -> dict[str, Any] | None:
    store = getattr(system, "vector_store", None)
    state_store = getattr(system, "state_store", None)
    if store is None or state_store is None:
        return None
    document_id = ""
    try:
        row = state_store.get_by_path(str(path.resolve())) or {}
        document_id = str(row.get("document_id") or "")
    except Exception:
        pass
    if not document_id:
        try:
            for row in state_store.get_all_documents() or []:
                if str(row.get("file_name") or "") == path.name:
                    document_id = str(row.get("document_id") or "")
                    if document_id:
                        break
        except Exception:
            pass
    if not document_id:
        return None
    try:
        records = store.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas", "embeddings"],
        )
        with sqlite3.connect(store.lexical_database) as db:
            lexical = db.execute(
                "SELECT id, document, metadata, index_state, tokens FROM lexical_documents WHERE json_extract(metadata, '$.document_id') = ?",
                (document_id,),
            ).fetchall()
        hierarchy = None
        hierarchy_collection = getattr(store, "hierarchy_collection", None)
        if hierarchy_collection is not None:
            hierarchy = hierarchy_collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas", "embeddings"],
            )
        return {
            "document_id": document_id,
            "ids": list(records.get("ids") or []),
            "documents": list(records.get("documents") or []),
            "metadatas": list(records.get("metadatas") or []),
            "embeddings": list(records.get("embeddings") or []),
            "lexical": lexical,
            "hierarchy": hierarchy,
        }
    except Exception:
        return None


def _restore_ready_state(system: Any, snapshot: dict[str, Any] | None) -> None:
    if not snapshot:
        return
    try:
        store = system.vector_store
        collection = store.collection
        ids = [str(x) for x in snapshot.get("ids") or []]
        existing = (
            set(str(x) for x in (collection.get(ids=ids, include=["metadatas"]).get("ids") or []))
            if ids
            else set()
        )
        missing = [i for i, item_id in enumerate(ids) if item_id not in existing]
        if missing:
            collection.add(
                ids=[ids[i] for i in missing],
                documents=[snapshot["documents"][i] for i in missing],
                metadatas=[dict(snapshot["metadatas"][i]) for i in missing],
                embeddings=[list(snapshot["embeddings"][i]) for i in missing],
            )
        if ids:
            collection.update(ids=ids, metadatas=[dict(meta) for meta in snapshot.get("metadatas", [])])

        lexical = snapshot.get("lexical") or []
        if lexical:
            with sqlite3.connect(store.lexical_database) as db:
                db.executemany(
                    "INSERT OR REPLACE INTO lexical_documents(id, document, metadata, index_state, tokens) VALUES(?,?,?,?,?)",
                    lexical,
                )
                db.commit()

        hierarchy_snapshot = snapshot.get("hierarchy")
        hierarchy_collection = getattr(store, "hierarchy_collection", None)
        if hierarchy_collection is not None and hierarchy_snapshot:
            hierarchy_ids = [str(x) for x in hierarchy_snapshot.get("ids") or []]
            if hierarchy_ids:
                current_ids = set(str(x) for x in hierarchy_collection.get(ids=hierarchy_ids).get("ids") or [])
                missing_hierarchy = [i for i, item_id in enumerate(hierarchy_ids) if item_id not in current_ids]
                if missing_hierarchy:
                    hierarchy_collection.add(
                        ids=[hierarchy_ids[i] for i in missing_hierarchy],
                        documents=[hierarchy_snapshot.get("documents", [""])[i] for i in missing_hierarchy],
                        metadatas=[dict(hierarchy_snapshot.get("metadatas", [{}])[i]) for i in missing_hierarchy],
                        embeddings=[list(hierarchy_snapshot.get("embeddings", [])[i]) for i in missing_hierarchy],
                    )
    except Exception:
        pass


def _patch_ingestion_rollback() -> None:
    from rag_project.app.rag_system import RAGSystem

    current = getattr(RAGSystem, "ingest_file", None)
    if not callable(current) or getattr(current, "_runtime_v5", False):
        return

    def ingest_file(self, pdf_path):
        path = Path(pdf_path)
        snapshot = _snapshot_ready_state(self, path)
        result = current(self, pdf_path)
        if str((result or {}).get("status") or "").casefold() == "failed":
            _restore_ready_state(self, snapshot)
        return result

    ingest_file._runtime_v5 = True
    RAGSystem.ingest_file = ingest_file


def _patch_building_lexical_ids() -> None:
    from rag_project.storage.vector_store import VectorStore

    current = getattr(VectorStore, "search_lexical", None)
    if not callable(current) or getattr(current, "_runtime_v5", False):
        return

    def search_lexical(self, query, n_results=5, where=None):
        result = current(self, query, n_results=n_results, where=where)
        ids = list((result.get("ids") or [[]])[0] or [])
        metas = list((result.get("metadatas") or [[]])[0] or [])
        if ids and metas:
            resolved = []
            for item_id, meta in zip(ids, metas, strict=False):
                chunk_id = str((meta or {}).get("chunk_id") or item_id)
                # Runtime-generated staging IDs are an internal artifact. The
                # public lexical contract exposes the canonical chunk identifier.
                resolved.append(chunk_id if re.search(r"-build-[0-9a-f]+-\d+$", str(item_id)) else str(item_id))
            result["ids"] = [resolved]
        return result

    search_lexical._runtime_v5 = True
    VectorStore.search_lexical = search_lexical


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _patch_chroma_hierarchy_modify()
    _patch_embedding_retries()
    _patch_ocr_status_compat()
    _patch_section_identity()
    _patch_followup_public_contract()
    _patch_numeric_boolean_contract()
    _patch_query_rewriter_legacy_contract()
    _patch_god_mode_legacy_entrypoint()
    _patch_production_history()
    _patch_ingestion_rollback()
    _patch_building_lexical_ids()
    _INSTALLED = True
