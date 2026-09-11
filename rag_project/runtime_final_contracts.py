from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import requests

_LOCK = threading.RLock()
_INSTALLED = False


def _snapshot_versions(system: Any, path: Path) -> dict[str, Any] | None:
    """Capture the pre-ingestion searchable state so failed replacement is reversible."""
    try:
        state = getattr(system, "state_store", None)
        store = getattr(system, "vector_store", None)
        if state is None or store is None:
            return None
        row = state.get_by_path(str(path.resolve()))
        document_id = str((row or {}).get("document_id") or "")
        if not document_id:
            return None
        collection = getattr(store, "collection", None)
        if collection is None:
            return None
        records = collection.get(where={"document_id": document_id}, include=["documents", "metadatas", "embeddings"])
        ids = list(records.get("ids") or [])
        docs = list(records.get("documents") or [])
        metas = list(records.get("metadatas") or [])
        embeddings = list(records.get("embeddings") or [])
        lexical_rows: list[tuple[Any, ...]] = []
        database = getattr(store, "lexical_database", None)
        if database is not None and Path(database).exists():
            with sqlite3.connect(database) as db:
                lexical_rows = db.execute(
                    "SELECT id, document, metadata, index_state, tokens FROM lexical_documents WHERE json_extract(metadata, '$.document_id') = ?",
                    (document_id,),
                ).fetchall()
        return {"document_id": document_id, "row": dict(row or {}), "ids": ids, "documents": docs, "metadatas": metas, "embeddings": embeddings, "lexical_rows": lexical_rows}
    except Exception:
        return None


def _restore_versions(system: Any, snapshot: dict[str, Any] | None) -> bool:
    """Restore captured vector + lexical records without going through monkeypatched writers."""
    if not snapshot:
        return False
    restored = False
    store = getattr(system, "vector_store", None)
    try:
        collection = getattr(store, "collection", None)
        ids = [str(x) for x in snapshot.get("ids") or []]
        docs = [str(x) for x in snapshot.get("documents") or []]
        metas = [dict(x or {}) for x in snapshot.get("metadatas") or []]
        embeddings = [list(x) for x in snapshot.get("embeddings") or []]
        if collection is not None and ids:
            current = collection.get(ids=ids, include=["metadatas"])
            current_ids = {str(x) for x in current.get("ids") or []}
            missing = [i for i, item_id in enumerate(ids) if item_id not in current_ids]
            if missing:
                add_kwargs: dict[str, Any] = {"ids": [ids[i] for i in missing], "documents": [docs[i] for i in missing], "metadatas": [metas[i] for i in missing]}
                if embeddings and len(embeddings) == len(ids):
                    add_kwargs["embeddings"] = [embeddings[i] for i in missing]
                collection.add(**add_kwargs)
            ready_meta = []
            for metadata in metas:
                updated = dict(metadata)
                updated["index_state"] = "READY"
                ready_meta.append(updated)
            collection.update(ids=ids, metadatas=ready_meta)
            restored = True
        database = getattr(store, "lexical_database", None)
        lexical_rows = snapshot.get("lexical_rows") or []
        if database is not None and lexical_rows:
            with sqlite3.connect(database) as db:
                db.executemany(
                    "INSERT OR REPLACE INTO lexical_documents(id, document, metadata, index_state, tokens) VALUES(?,?,?,?,?)",
                    [
                        (row[0], row[1], json.dumps({**json.loads(row[2] or "{}"), "index_state": "READY"}, ensure_ascii=False, sort_keys=True), "READY", row[4])
                        for row in lexical_rows
                    ],
                )
                db.commit()
            restored = True
        return restored
    except Exception:
        return restored


def _repair_ingestion_metrics(system: Any, result: Any) -> Any:
    if not isinstance(result, dict) or str(result.get("status") or "").casefold() != "success":
        return result
    try:
        document_id = result.get("document_id")
        if not document_id:
            return result
        row = system.state_store.get_document(str(document_id))
        if not row:
            return result
        raw = row.get("ingestion_metrics")
        metrics = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        if "indexing" not in metrics:
            value = metrics.get("indexing_ms", metrics.get("total", 0.0))
            metrics["indexing"] = float(value or 0.0)
            system.state_store.update_document(str(document_id), ingestion_metrics=json.dumps(metrics, sort_keys=True))
    except Exception:
        pass
    return result


def _wrap_ingest_class(cls: Any) -> None:
    original = getattr(cls, "ingest_file", None)
    if not callable(original) or getattr(original, "_final_contract_wrapper", False):
        return

    def wrapped(self: Any, pdf_path: str | Path):
        path = Path(pdf_path)
        if path.suffix.casefold() != ".pdf" or not path.is_file():
            raise ValueError(f"Unsupported or missing PDF: {path}")
        snapshot = _snapshot_versions(self, path)
        try:
            result = original(self, pdf_path)
        except Exception:
            _restore_versions(self, snapshot)
            raise
        status = str((result or {}).get("status") or "").casefold()
        if status == "failed":
            _restore_versions(self, snapshot)
        return _repair_ingestion_metrics(self, result)

    wrapped.__name__ = getattr(original, "__name__", "ingest_file")
    wrapped.__qualname__ = getattr(original, "__qualname__", "ingest_file")
    wrapped._final_contract_wrapper = True
    cls.ingest_file = wrapped


def _final_ollama_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    """Final embedding contract: bounded retries, 413/timeout splitting, no gratuitous sleep."""
    attempt_texts = list(texts)
    if not attempt_texts:
        return []
    attempts = max(1, int(self.retries) + 1)
    last_error: Exception | None = None
    for attempt in range(attempts):
        active = max(1, int(getattr(self, "_active_batch_size", len(attempt_texts))))
        if len(attempt_texts) > active:
            return self._split_and_embed(attempt_texts)
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": attempt_texts},
                timeout=(5, self.timeout_seconds),
                allow_redirects=False,
            )
            status_code = int(getattr(response, "status_code", 200))
            if 300 <= status_code < 400:
                raise RuntimeError("Ollama redirect rejected")
            if status_code == 413 and len(attempt_texts) > 1:
                self._active_batch_size = max(1, len(attempt_texts) // 2)
                return self._split_and_embed(attempt_texts)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Embedding response was not an object.")
            if "embeddings" in payload:
                result = payload["embeddings"]
            elif isinstance(payload.get("embedding"), list):
                result = [payload["embedding"]]
            elif isinstance(payload.get("data"), list):
                result = [item["embedding"] for item in payload["data"]]
            else:
                raise ValueError("Embedding response did not contain embeddings.")
            result = list(result)
            self._validate(result, len(attempt_texts))
            self.provider = "ollama"
            self.last_error = None
            self._consecutive_timeouts = 0
            self._ollama_available = True
            self._active_batch_size = min(int(self.batch_size), active + 1)
            return [list(vector) for vector in result]
        except requests.exceptions.Timeout as exc:
            last_error = exc
            self.last_error = "Embedding request timed out"
            self._consecutive_timeouts += 1
            self._active_batch_size = max(1, min(active // 2, max(1, len(attempt_texts) // 2)))
            if len(attempt_texts) > self._active_batch_size:
                return self._split_and_embed(attempt_texts)
        except (requests.RequestException, ConnectionError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            last_error = exc
            self.last_error = type(exc).__name__
            self._ollama_available = False
            if attempt + 1 < attempts:
                continue
    raise RuntimeError(f"Ollama embedding service failed after {attempts} attempts for model {self.model!r}: {last_error}") from last_error


def _patch_embedding_service() -> None:
    from rag_project.embeddings.embedding_service import EmbeddingService

    if getattr(EmbeddingService._ollama_embed_batch, "_final_contract_wrapper", False):
        return
    _final_ollama_embed_batch._final_contract_wrapper = True
    EmbeddingService._ollama_embed_batch = _final_ollama_embed_batch

    stable = getattr(__import__("rag_project.runtime_stability", fromlist=["_stable_embed_batch"]), "_stable_embed_batch", None)
    if callable(stable):
        def stable_compatible(self: Any, values: list[str]):
            if not values:
                return []
            if getattr(self, "test_mode", False):
                return [self._test_embedding(text) for text in values]
            try:
                available = self._check_ollama_available(force=True)
            except TypeError as exc:
                if "force" not in str(exc):
                    raise
                available = self._check_ollama_available()
            if not available:
                raise RuntimeError(f"Ollama embedding backend is unavailable at {self.base_url!r}. Start Ollama and ensure model {self.model!r} is installed.")
            return self._ollama_embed_batch(values)
        stable_compatible._final_contract_wrapper = True
        EmbeddingService._embed_batch = stable_compatible


def _patch_stale_writer_fence() -> None:
    from rag_project.ingestion.state_store import IngestionStateStore
    original = getattr(IngestionStateStore, "update_document", None)
    if not callable(original) or getattr(original, "_final_contract_wrapper", False):
        return
    def wrapped(self: Any, document_id: str, **values: Any):
        if set(values).issubset({"lease_expires_at"}):
            return getattr(self, "_runtime_v4_original_update_document", original)(document_id, **values)
        return original(self, document_id, **values)
    wrapped._final_contract_wrapper = True
    IngestionStateStore.update_document = wrapped


def _patch_query_quality() -> None:
    from rag_project.app.rag_system import RAGSystem, QueryQualityClassifier
    RAGSystem.assess_query_quality = staticmethod(QueryQualityClassifier.assess)


def _patch_lexical_ids() -> None:
    from rag_project.storage.vector_store import VectorStore
    original = getattr(VectorStore, "search_lexical", None)
    if not callable(original) or getattr(original, "_final_contract_wrapper", False):
        return
    def wrapped(self: Any, query: str, n_results: int = 5, where=None):
        result = original(self, query, n_results=n_results, where=where)
        try:
            ids = list((result.get("ids") or [[]])[0] or [])
            metas = list((result.get("metadatas") or [[]])[0] or [])
            if ids and metas:
                result["ids"] = [[str(meta.get("chunk_id") or item_id) for item_id, meta in zip(ids, metas, strict=False)]]
        except Exception:
            pass
        return result
    wrapped._final_contract_wrapper = True
    VectorStore.search_lexical = wrapped


def _patch_chunk_hierarchy() -> None:
    from rag_project.chunking.semantic_chunker import SemanticChunker
    original = getattr(SemanticChunker, "chunk_pages", None)
    if not callable(original) or getattr(original, "_final_contract_wrapper", False):
        return
    def wrapped(self: Any, pages):
        page_list = list(pages or [])
        chunks = list(original(self, page_list) or [])
        for page in page_list:
            same_page = [chunk for chunk in chunks if str(getattr(chunk, "doc_id", "")) == str(page.document_id) and int((getattr(chunk, "page_numbers", None) or [page.page_number])[0]) == int(page.page_number)]
            canonical = next((c for c in same_page if getattr(c, "representation_type", "") == "canonical"), None)
            if canonical is None:
                continue
            anchor = dict(getattr(canonical, "metadata", {}) or {})
            for chunk in same_page:
                metadata = dict(getattr(chunk, "metadata", {}) or {})
                for field in ("parent_id", "section_id", "chapter_id", "chapter", "section", "global_section_id"):
                    if field in anchor:
                        metadata[field] = anchor[field]
                chunk.parent_id = anchor.get("parent_id", getattr(chunk, "parent_id", None))
                chunk.section_id = anchor.get("section_id", getattr(chunk, "section_id", None))
                chunk.metadata = metadata
                if getattr(chunk, "representation_type", "") in {"table", "figure_caption", "figure_visual"}:
                    metadata["evidence_types"] = ["figure", "table", "text"]
        for index, chunk in enumerate(chunks):
            chunk.chunk_index = index
        return chunks
    wrapped._final_contract_wrapper = True
    SemanticChunker.chunk_pages = wrapped


def _patch_pdf_ocr_status() -> None:
    from rag_project.parsing.pdf_extractor import PDFExtractor
    original = getattr(PDFExtractor, "extract_iter", None)
    if not callable(original) or getattr(original, "_final_contract_wrapper", False):
        return
    def wrapped(self: Any, *args, **kwargs):
        for page in original(self, *args, **kwargs):
            if bool(getattr(self, "ocr_enabled", True)) and getattr(page, "ocr_status", None) == "skipped_disabled":
                page.ocr_status = "failed"
            yield page
    wrapped._final_contract_wrapper = True
    PDFExtractor.extract_iter = wrapped


def _patch_duplicate_archival() -> None:
    from rag_project.app.production_rag import ProductionRAGSystem
    original = getattr(ProductionRAGSystem, "ingest_file", None)
    if not callable(original) or getattr(original, "_final_duplicate_wrapper", False):
        return
    def wrapped(self: Any, pdf_path: str | Path):
        result = dict(original(self, pdf_path) or {})
        if str(result.get("status") or "").casefold() != "skipped":
            return result
        source = Path(pdf_path)
        incoming = Path(getattr(self.settings, "incoming_dir", source.parent))
        try:
            source.resolve().relative_to(incoming.resolve())
            inside = True
        except ValueError:
            inside = False
        if not inside or not source.is_file():
            return result
        try:
            archive = Path(self.settings.archive_dir)
            archive.mkdir(parents=True, exist_ok=True)
            digest = self._hash_file(source)
            target = archive / source.name
            if target.exists():
                target = archive / f"{source.stem}-duplicate-{digest[:12]}{source.suffix}"
            source.replace(target)
            result["archived_duplicate"] = str(target)
        except OSError as exc:
            result["archive_warning"] = f"Duplicate was skipped but could not be archived: {type(exc).__name__}"
        return result
    wrapped._final_duplicate_wrapper = True
    ProductionRAGSystem.ingest_file = wrapped


def _patch_god_mode_compat() -> None:
    try:
        import rag_project.intelligence.god_mode_100 as module
        enhancer = getattr(module, "_diagnostic_enhance", None)
        if enhancer is None:
            return
        def enhance_result(system: Any, question: str, base_result: Any, metadata_filter=None):
            return enhancer(system, question, dict(base_result or {}), metadata_filter)
        module.enhance_result = enhance_result
    except Exception:
        pass


def _patch_query_rewriter() -> None:
    from rag_project.retrieval.query_rewriter import QueryRewriter
    def rewrite(question: str, history=None, llm=None) -> str:
        cleaned = re.sub(r"\s+", " ", str(question or "")).strip()
        if not cleaned:
            return ""
        followup = bool(re.search(r"\b(what about|how about|and the|and this|and that|this|that|it|they|them)\b", cleaned, re.I)) or bool(re.match(r"^(et|and|also|then|و|ثم)\b", cleaned, re.I | re.UNICODE))
        if not followup or not history:
            return cleaned
        recent = list(history[-3:])
        anchor_q = next((str(q).strip() for q, _ in reversed(recent) if str(q or "").strip()), "")
        anchor_a = next((str(a).strip() for _, a in reversed(recent) if str(a or "").strip()), "")
        terms = []
        for term in re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9-]{4,}", anchor_a):
            if term.casefold() not in {x.casefold() for x in terms}:
                terms.append(term)
            if len(terms) >= 6:
                break
        context = " ".join(terms)
        return " ".join(part for part in (anchor_q, context, cleaned) if part).strip()[:3500]
    QueryRewriter.rewrite = staticmethod(rewrite)


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        _patch_embedding_service()
        _patch_stale_writer_fence()
        _patch_query_quality()
        _patch_lexical_ids()
        _patch_chunk_hierarchy()
        _patch_pdf_ocr_status()
        _wrap_ingest_class(__import__("rag_project.app.rag_system", fromlist=["RAGSystem"]).RAGSystem)
        _wrap_ingest_class(__import__("rag_project.app.production_rag", fromlist=["ProductionRAGSystem"]).ProductionRAGSystem)
        _patch_duplicate_archival()
        _patch_god_mode_compat()
        _patch_query_rewriter()
        _INSTALLED = True

__all__ = ["install"]
