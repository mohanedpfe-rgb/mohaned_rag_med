from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


def _as_list(value: Any) -> list[Any]:
    """Normalize Python/NumPy/connector containers without implicit truth tests."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except (TypeError, ValueError):
        return []


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _heartbeat_loop(system: Any, document_id: str, worker_id: str, stop: threading.Event) -> None:
    interval = max(5.0, min(30.0, float(system.settings.ingestion_lease_seconds) / 4.0))
    while not stop.wait(interval):
        try:
            if not system.state_store.heartbeat_document(document_id, worker_id, lease_seconds=system.settings.ingestion_lease_seconds):
                system.logger.warning("Ingestion lease lost for %s", document_id)
                return
        except Exception:
            system.logger.exception("Ingestion heartbeat failed for %s", document_id)
            return


def _stable_extract_iter(self: Any, pdf_path: str | Path, document_id: str | None = None) -> Iterator[Any]:
    """Use the native extractor path; never spawn a child process per PDF page."""
    original = getattr(type(self), "_original_extract_iter", None)
    if original is None:
        raise RuntimeError("Native PDF extractor implementation is unavailable.")
    yield from original(self, pdf_path, document_id)


def _stable_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    """Ollama-only production embeddings with bounded failure behavior."""
    if not texts:
        return []
    if self.test_mode:
        return [self._test_embedding(text) for text in texts]
    if not self._check_ollama_available(force=True):
        raise RuntimeError(f"Ollama embedding backend is unavailable at {self.base_url!r}. Start Ollama and ensure model {self.model!r} is installed.")
    return self._ollama_embed_batch(texts)


def _stable_embedding_init(self: Any, *args: Any, **kwargs: Any) -> None:
    """Cap pathological embedding waits while retaining user configuration as the upper intent."""
    cls = type(self)
    original = getattr(cls, "_runtime_stability_original_init", None)
    if original is None:
        original = cls.__dict__.get("__init__")
    if original is None or original is _stable_embedding_init:
        raise RuntimeError("EmbeddingService initializer is unavailable.")
    original(self, *args, **kwargs)
    self.timeout_seconds = min(float(self.timeout_seconds), 300.0)
    self.retries = min(max(1, int(self.retries)), 3)


def _stable_set_version_state(self: Any, document_id: str, version_id: str, state: str) -> None:
    """Promote or demote a document version in one bounded Chroma update + one SQL update."""
    normalized = str(state).upper()
    records = self.collection.get(where={"document_id": document_id}, include=["metadatas"])
    ids: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for item_id, metadata in zip(_as_list(records.get("ids")), _as_list(records.get("metadatas")), strict=False):
        meta = self._coerce_metadata(metadata)
        if str(meta.get("version_id") or "") != str(version_id):
            continue
        meta["index_state"] = normalized
        ids.append(str(item_id))
        metadatas.append(meta)
    if ids:
        self.collection.update(ids=ids, metadatas=metadatas)
    with sqlite3.connect(self.lexical_database) as connection:
        connection.execute("UPDATE lexical_documents SET index_state = ?, metadata = json_set(metadata, '$.index_state', ?) WHERE json_extract(metadata, '$.document_id') = ? AND json_extract(metadata, '$.version_id') = ?", (normalized, normalized, str(document_id), str(version_id)))
        connection.commit()


def _stable_delete_version(self: Any, document_id: str, version_id: str) -> None:
    """Delete a version using set-based operations instead of per-record updates."""
    records = self.collection.get(where={"document_id": document_id}, include=["metadatas"])
    removable = [str(item_id) for item_id, metadata in zip(_as_list(records.get("ids")), _as_list(records.get("metadatas")), strict=False) if str(self._coerce_metadata(metadata).get("version_id") or "") == str(version_id)]
    if removable:
        self.collection.delete(ids=removable)
    with sqlite3.connect(self.lexical_database) as connection:
        connection.execute("DELETE FROM lexical_documents WHERE json_extract(metadata, '$.document_id') = ? AND json_extract(metadata, '$.version_id') = ?", (str(document_id), str(version_id)))
        connection.commit()


def _stable_validate_document_index(self: Any, document_id: str, version_id: str | None = None, *, sample_size: int = 8) -> dict[str, Any]:
    """Validate metadata for all chunks but inspect only a bounded embedding sample."""
    records = self.collection.get(where={"document_id": document_id}, include=["metadatas"])
    ids = [str(item_id) for item_id in _as_list(records.get("ids"))]
    metadatas = [self._coerce_metadata(item) for item in _as_list(records.get("metadatas"))]
    if version_id is not None:
        pairs = [(item_id, metadata) for item_id, metadata in zip(ids, metadatas, strict=False) if str(metadata.get("version_id") or "") == str(version_id)]
    else:
        pairs = list(zip(ids, metadatas, strict=False))
    issues: list[str] = []
    seen_chunk_ids: set[str] = set()
    for item_id, metadata in pairs:
        chunk_id = str(metadata.get("chunk_id") or item_id or "")
        if not chunk_id:
            issues.append("missing chunk_id")
        elif chunk_id in seen_chunk_ids:
            issues.append(f"duplicate chunk_id: {chunk_id}")
        seen_chunk_ids.add(chunk_id)
        if str(metadata.get("index_state") or "").upper() not in {"READY", "BUILDING"}:
            issues.append(f"unexpected index_state: {metadata.get('index_state')}" )

    chosen_ids = [item_id for item_id, _ in pairs[: max(1, int(sample_size))]]
    if len(pairs) > 1 and len(chosen_ids) < sample_size:
        chosen_ids.append(pairs[-1][0])
    if len(pairs) > 2:
        chosen_ids.append(pairs[len(pairs) // 2][0])
    chosen_ids = list(dict.fromkeys(chosen_ids))[: max(1, int(sample_size))]
    if chosen_ids:
        try:
            sampled = self.collection.get(ids=chosen_ids, include=["embeddings"])
            embeddings = _as_list(sampled.get("embeddings"))
            expected_dim = self._collection_dim()
            for vector in embeddings:
                if not self._valid_vector(vector, expected_dim):
                    issues.append("invalid semantic embedding in validation sample")
        except Exception as exc:
            issues.append(f"embedding validation sample failed: {type(exc).__name__}")

    return {"document_id": document_id, "count": len(pairs), "valid": bool(pairs) and not issues, "issues": issues, "sampled_embeddings": min(len(chosen_ids), len(pairs))}


def _stable_ingest_directory(self: Any, directory: str | Path | None = None) -> list[dict[str, Any]]:
    """Serialize local-disk ingestion; Chroma and SQLite are local stores and should not be contended."""
    source_dir = Path(directory) if directory else self.settings.incoming_dir
    source_dir.mkdir(parents=True, exist_ok=True)
    return [self.ingest_file(path) for path in sorted(source_dir.glob("*.pdf"))]


def _stable_ingest_file(self: Any, pdf_path: str | Path) -> dict[str, Any]:
    """Stable compatibility ingestion state machine with one authoritative version id."""
    from rag_project.chunking.semantic_chunker import SemanticChunker
    from rag_project.ingestion.document_classifier import DocumentClassifier
    from rag_project.parsing.pdf_extractor import PDFExtractor
    from rag_project.utils.text_utils import detect_language

    started = time.perf_counter()
    source = Path(pdf_path)
    if source.suffix.lower() != ".pdf" or not source.is_file():
        raise ValueError(f"Unsupported or missing PDF: {source}")
    content_hash = self._hash_file(source)
    existing = self.state_store.get_by_hash(content_hash)
    previous = self.state_store.get_by_path(str(source.resolve()))
    document_id = existing["document_id"] if existing else (previous["document_id"] if previous else content_hash)
    previous_version = (str(previous.get("version_id") or previous.get("content_hash") or "") if previous and previous.get("content_hash") != content_hash else None)
    previous_content_hash = str(previous.get("content_hash") or "") if previous and previous.get("content_hash") != content_hash else None
    parser_version = "pdf-extractor-v4-stable"
    chunk_config = {"size": self.settings.chunk_size, "overlap": self.settings.chunk_overlap}
    ocr_config = {"engine": "rapidocr", "enabled": bool(getattr(self.settings, "ocr_enabled", False)), "confidence_threshold": float(getattr(self.settings, "ocr_confidence_threshold", 0.55))}
    version_id = self._ingestion_version_id(content_hash=content_hash, parser_version=parser_version, ocr_config=ocr_config, chunking_config=chunk_config, embedding_model=self.settings.embedding_model, embedding_profile=None, embedding_dimension=None)
    if existing and self.state_store.is_ready_status(existing.get("status")) and existing.get("version_id") == version_id:
        try:
            validation = self.vector_store.validate_document_index(existing["document_id"], version_id)
        except Exception:
            validation = {"valid": False, "count": 0}
        if validation.get("valid") and validation.get("count", 0) > 0:
            return {"status": "skipped", "file_name": source.name, "document_id": existing["document_id"], "reason": "identical indexed version already validated"}
    try:
        self.state_store.recover_stale_documents()
    except Exception:
        self.logger.exception("Unable to recover stale document leases before %s", source.name)
    for stale_version in dict.fromkeys(v for v in (previous_version, previous_content_hash) if v and v != version_id):
        try:
            self.vector_store.delete_version(document_id, stale_version)
        except Exception:
            self.logger.exception("Unable to retire stale version %s for %s", stale_version, source.name)
    try:
        self.vector_store.delete_version(document_id, version_id)
    except Exception:
        pass
    if previous and previous["content_hash"] != content_hash:
        self.state_store.delete_pages(document_id)
    stat = source.stat()
    self.state_store.upsert_document({"document_id": document_id, "content_hash": content_hash, "file_path": str(source.resolve()), "file_name": source.name, "file_size": stat.st_size, "created_at": _now(), "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(), "ingestion_started_at": _now(), "current_stage": "DISCOVERED", "current_page": 0, "total_pages": 0, "status": "RUNNING", "parser_version": parser_version, "ocr_config": json.dumps(ocr_config, sort_keys=True), "chunking_config": json.dumps(chunk_config, sort_keys=True), "embedding_model": self.settings.embedding_model, "embedding_dimension": None, "index_state": "BUILDING", "version_id": version_id, "error": None})
    worker_id = str(uuid.uuid4())
    if not self.state_store.claim_document(document_id, worker_id, lease_seconds=self.settings.ingestion_lease_seconds):
        return {"status": "busy", "file_name": source.name, "document_id": document_id, "reason": "document is currently leased by another worker"}
    stop_heartbeat = threading.Event()
    heartbeat = threading.Thread(target=_heartbeat_loop, args=(self, document_id, worker_id, stop_heartbeat), name=f"ingest-heartbeat-{document_id[:8]}", daemon=True)
    heartbeat.start()
    semantic_enabled = True
    chunk_count = 0
    embedding_count = 0
    page_count = 0
    current_page = 0
    document_language: str | None = None
    extraction_ms = 0.0
    try:
        classification = DocumentClassifier.classify(source)
        page_count = int(classification.get("page_count") or 0)
        self.state_store.transition_document_state(document_id, "VALIDATING", current_page=0, total_pages=page_count)
        self.state_store.record_event(document_id, stage="VALIDATING", status="RUNNING", event_type="classification", message=f"Validated {source.name}: {page_count} pages", details=classification, current_page=0, total_pages=page_count, file_name=source.name)
        if classification.get("document_type") == "encrypted":
            raise RuntimeError("FAILED_EXTRACTION: PDF is password-protected.")
        if classification.get("document_type") == "invalid_pdf":
            raise RuntimeError(f"FAILED_EXTRACTION: {classification.get('error', 'invalid PDF')}")
        if page_count <= 0:
            raise RuntimeError("FAILED_EXTRACTION: PDF contains no pages.")
        self.state_store.transition_document_state(document_id, "EXTRACTING", current_page=0, total_pages=page_count)
        self.state_store.record_event(document_id, stage="EXTRACTING", status="RUNNING", event_type="extract", message=f"Extracting text page-by-page: 0/{page_count}", current_page=0, total_pages=page_count, file_name=source.name)
        extractor = PDFExtractor(self.state_store, ocr_enabled=bool(getattr(self.settings, "ocr_enabled", True)), ocr_confidence_threshold=float(getattr(self.settings, "ocr_confidence_threshold", 0.55)), ocr_min_char_density=float(getattr(self.settings, "ocr_min_char_density", 0.001)), ocr_image_coverage_threshold=float(getattr(self.settings, "ocr_image_coverage_threshold", 0.55)))
        extraction_started = time.perf_counter()
        pages = extractor.extract_iter(source, document_id)
        extraction_ms = (time.perf_counter() - extraction_started) * 1000.0
        chunker = SemanticChunker(self.settings.chunk_size, self.settings.chunk_overlap)
        batches = chunker.chunk_page_batches(pages, batch_size=max(1, int(self.settings.page_batch_size)))
        first_batch = True
        for batch_number, batch in enumerate(batches, start=1):
            if not batch:
                continue
            if first_batch:
                self.state_store.transition_document_state(document_id, "CHUNKING", current_page=0, total_pages=page_count)
                self.state_store.record_event(document_id, stage="CHUNKING", status="RUNNING", event_type="chunk", message="Splitting extracted pages into searchable sections", details={"chunk_size": self.settings.chunk_size, "chunk_overlap": self.settings.chunk_overlap, "page_batch_size": self.settings.page_batch_size}, current_page=0, total_pages=page_count, file_name=source.name)
                first_batch = False
            page_numbers = [int(page) for chunk in batch for page in _as_list(getattr(chunk, "page_numbers", None))]
            current_page = min(page_count, max(page_numbers, default=current_page))
            documents: list[str] = []
            chunks: list[Any] = []
            for chunk in batch:
                text = str(chunk.text or "").strip()
                if text:
                    documents.append(text)
                    chunks.append(chunk)
            if not documents:
                self.state_store.update_document(document_id, current_stage="CHUNKING", current_page=current_page, total_pages=page_count)
                continue
            if document_language is None:
                document_language = detect_language(" ".join(documents))
            ids: list[str] = []
            metadatas: list[dict[str, Any]] = []
            for offset, chunk in enumerate(chunks):
                global_index = chunk_count + offset
                chunk_id = f"{document_id}-{content_hash[:12]}-{global_index}"
                chunk.chunk_index = global_index
                ids.append(chunk_id)
                metadatas.append({"document_id": document_id, "chunk_id": chunk_id, "file_name": chunk.file_name, "page_numbers": _as_list(getattr(chunk, "page_numbers", None)), "chunk_index": global_index, "document_type": classification.get("document_type", "unknown"), "language": document_language, "evidence_types": _as_list(chunk.metadata.get("evidence_types")) or ["text"], "index_state": "BUILDING", "version_id": version_id})
            if semantic_enabled:
                self.state_store.transition_document_state(document_id, "EMBEDDING", current_page=current_page, total_pages=page_count)
                try:
                    vectors = _as_list(self.embedding_service.embed_texts(documents))
                    if len(vectors) != len(documents) or len(vectors) == 0:
                        raise RuntimeError(f"Embedding backend returned {len(vectors)} vectors for {len(documents)} chunks.")
                    self.embedding_service.validate_batch(vectors)
                    self.vector_store.set_expected_identity(self.embedding_service.identity)
                    compatibility = self.vector_store.compatibility_report(self.embedding_service.identity)
                    if not compatibility.get("valid"):
                        raise RuntimeError(compatibility.get("message", "embedding index is incompatible"))
                except Exception:
                    semantic_enabled = False
                    embedding_count = 0
                    try:
                        self.vector_store.delete_version(document_id, version_id)
                    except Exception:
                        pass
            if semantic_enabled:
                self.state_store.transition_document_state(document_id, "INDEXING", current_page=current_page, total_pages=page_count)
                self.vector_store.add_documents(documents, metadatas, vectors, ids)
                embedding_count += len(vectors)
            else:
                self.state_store.transition_document_state(document_id, "INDEXING", current_page=current_page, total_pages=page_count)
                self.vector_store.add_lexical_documents(documents, metadatas, ids)
            chunk_count += len(documents)
            self.state_store.update_document(document_id, current_stage="INDEXING", current_page=current_page, total_pages=page_count, embedding_dimension=self.embedding_service.dimension if semantic_enabled else None, ingestion_metrics=json.dumps({"batch": batch_number, "page": current_page, "pages": page_count, "chunks": chunk_count, "embeddings": embedding_count, "retrieval_mode": "hybrid" if semantic_enabled else "lexical", "elapsed_ms": round((time.perf_counter() - started) * 1000, 3), "extraction": round(extraction_ms, 3)}, sort_keys=True))
        if chunk_count == 0:
            raise RuntimeError(f"FAILED_EXTRACTION: No extractable text was produced for {source.name}.")
        self.state_store.transition_document_state(document_id, "INDEXING", current_page=page_count, total_pages=page_count)
        if semantic_enabled:
            if embedding_count != chunk_count:
                raise RuntimeError(f"FAILED_EMBEDDING: expected {chunk_count} embeddings, produced {embedding_count}.")
            validation = self.vector_store.validate_document_index(document_id, version_id)
            if not validation.get("valid") or int(validation.get("count") or 0) != embedding_count:
                raise RuntimeError("FAILED_INDEXING: semantic index validation failed: " + "; ".join(_as_list(validation.get("issues"))))
        self.vector_store.set_version_index_state(document_id, version_id, "READY")
        target = self.settings.processed_dir / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            if target.exists():
                archived = _unique_path(self.settings.archive_dir, target, content_hash[:12])
                archived.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, archived)
            os.replace(source, target)
        warning = "Semantic embeddings were unavailable; lexical search is active for this document." if not semantic_enabled else None
        self.state_store.transition_document_state(document_id, "READY", current_page=page_count, total_pages=page_count, ingestion_completed_at=_now(), file_path=str(target.resolve()), embedding_dimension=self.embedding_service.dimension if semantic_enabled else None, version_id=version_id, index_state="READY", error=warning, ingestion_metrics=json.dumps({"page_count": page_count, "chunk_count": chunk_count, "embedding_count": embedding_count, "retrieval_mode": "hybrid" if semantic_enabled else "lexical", "total_ms": round((time.perf_counter() - started) * 1000, 3), "extraction": round(extraction_ms, 3)}, sort_keys=True))
        return {"status": "success", "document_id": document_id, "file_name": source.name, "document_type": classification.get("document_type", "unknown"), "page_count": page_count, "chunk_count": chunk_count, "embedding_count": embedding_count, "retrieval_mode": "hybrid" if semantic_enabled else "lexical", "degraded": not semantic_enabled, "warning": warning}
    except Exception as exc:
        self.logger.exception("Stable ingestion failed for %s", source.name)
        for stale_version in dict.fromkeys(v for v in (version_id, previous_version, previous_content_hash) if v):
            try:
                self.vector_store.delete_version(document_id, stale_version)
            except Exception:
                self.logger.exception("Stable rollback failed for %s", source.name)
        failure_stage = "FAILED_EXTRACTION" if str(exc).startswith("FAILED_EXTRACTION:") else ("FAILED_EMBEDDING" if str(exc).startswith("FAILED_EMBEDDING:") else "FAILED_INDEXING")
        try:
            self.state_store.transition_document_state(document_id, failure_stage, current_page=current_page, total_pages=page_count, error=str(exc), index_state="FAILED")
        except Exception:
            try:
                self.state_store.update_document(document_id, current_stage=failure_stage, status=failure_stage, error=str(exc), current_page=current_page, total_pages=page_count, index_state="FAILED")
            except Exception:
                self.logger.exception("Could not persist failure state for %s", source.name)
        failed = _unique_path(self.settings.failed_dir, source, content_hash[:12])
        quarantine_error = None
        try:
            if source.exists() and source.resolve() != failed.resolve():
                failed.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, failed)
                source.unlink()
        except OSError as quarantine_exc:
            quarantine_error = str(quarantine_exc)
            self.logger.exception("Could not quarantine failed PDF %s", source.name)
        return {"status": "failed", "document_id": document_id, "file_name": source.name, "error": str(exc), "quarantine_error": quarantine_error}
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=2.0)
        self.state_store.release_document(document_id, worker_id)


def _unique_path(directory: Path, source: Path, suffix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / source.name
    if not candidate.exists():
        return candidate
    candidate = directory / f"{source.stem}-{suffix}{source.suffix}"
    if not candidate.exists():
        return candidate
    return directory / f"{source.stem}-{suffix}-{uuid.uuid4().hex[:10]}{source.suffix}"


def install() -> None:
    from rag_project.app.rag_system import RAGSystem
    from rag_project.embeddings.embedding_service import EmbeddingService
    from rag_project.parsing.pdf_extractor import PDFExtractor
    from rag_project.storage.vector_store import VectorStore

    if hasattr(PDFExtractor, "_original_extract_iter"):
        PDFExtractor.extract_iter = _stable_extract_iter
    if not hasattr(EmbeddingService, "_runtime_stability_original_embed_batch"):
        EmbeddingService._runtime_stability_original_embed_batch = EmbeddingService._embed_batch
        EmbeddingService._embed_batch = _stable_embed_batch
    if not hasattr(VectorStore, "_runtime_stability_original_set_version_state"):
        VectorStore._runtime_stability_original_set_version_state = VectorStore.set_version_index_state
        VectorStore.set_version_index_state = _stable_set_version_state
    if not hasattr(VectorStore, "_runtime_stability_original_delete_version"):
        VectorStore._runtime_stability_original_delete_version = VectorStore.delete_version
        VectorStore.delete_version = _stable_delete_version
    if not hasattr(VectorStore, "_runtime_stability_original_validate_document_index"):
        VectorStore._runtime_stability_original_validate_document_index = VectorStore.validate_document_index
        VectorStore.validate_document_index = _stable_validate_document_index
    if not hasattr(RAGSystem, "_runtime_stability_original_ingest_directory"):
        RAGSystem._runtime_stability_original_ingest_directory = RAGSystem.ingest_directory
        RAGSystem.ingest_directory = _stable_ingest_directory
    RAGSystem.ingest_file = _stable_ingest_file
    if not hasattr(EmbeddingService, "_runtime_stability_original_init"):
        EmbeddingService._runtime_stability_original_init = EmbeddingService.__init__
        EmbeddingService.__init__ = _stable_embedding_init
