from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Sequence


def _safe_add_documents(
    self: Any,
    documents: Sequence[str],
    metadatas: Sequence[dict[str, Any]],
    embeddings: Sequence[Sequence[float]],
    ids: Sequence[str],
) -> None:
    """Best-effort transactional protocol across Chroma and SQLite."""
    if not documents:
        return
    if not (len(documents) == len(metadatas) == len(embeddings) == len(ids)):
        raise ValueError("documents, metadatas, embeddings, and ids must have the same length")
    lock = getattr(self, "_transaction_lock", None)
    if lock is None:
        lock = threading.RLock()
        self._transaction_lock = lock
    with lock:
        dim = self._resolve_dimension(embeddings)
        self._apply_collection_metadata(dim)
        normalized: list[dict[str, Any]] = []
        safe_ids = [str(item) for item in ids]
        for index, item in enumerate(documents):
            metadata = self._coerce_metadata(metadatas[index])
            metadata.setdefault("document_id", "unknown")
            metadata.setdefault("chunk_id", safe_ids[index])
            metadata["index_state"] = "BUILDING"
            metadata.setdefault("version_id", metadata.get("document_id", "legacy"))
            metadata.setdefault("page_numbers", [])
            if not self._valid_vector(embeddings[index], dim):
                raise ValueError(f"Invalid semantic embedding at index {index}.")
            normalized.append(metadata)

        try:
            self.collection.upsert(
                ids=safe_ids,
                documents=[str(item) for item in documents],
                metadatas=normalized,
                embeddings=[list(map(float, vector)) for vector in embeddings],
            )
            self._update_collection_identity(self.expected_identity)
            try:
                self._upsert_lexical_records(documents, normalized, safe_ids)
            except Exception:
                try:
                    self.collection.delete(ids=safe_ids)
                except Exception:
                    pass
                raise
            # Make records visible only after both stores contain the batch.
            ready_metadata = []
            for metadata in normalized:
                ready = dict(metadata)
                ready["index_state"] = "READY"
                ready_metadata.append(ready)
            self.collection.update(ids=safe_ids, metadatas=ready_metadata)
        except Exception:
            raise


def _safe_ingestion_version_id(
    *,
    content_hash: str,
    parser_version: str,
    ocr_config: str | dict[str, Any] | None,
    chunking_config: str | dict[str, Any] | None,
    embedding_model: str,
    embedding_profile: str | None,
    embedding_dimension: int | None,
) -> str:
    """Stable content/index fingerprint including the actual embedding profile inputs."""
    def as_dict(value: str | dict[str, Any] | None) -> dict[str, Any]:
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else {"value": parsed}
            except json.JSONDecodeError:
                return {"value": value}
        return value or {}

    dimension = embedding_dimension
    if not dimension:
        try:
            env_dim = int(os.getenv("EMBEDDING_DIMENSION", "0"))
            dimension = env_dim or None
        except ValueError:
            dimension = None
    endpoint = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    profile = embedding_profile or hashlib.sha256(
        f"ollama|{endpoint}|{embedding_model}|{dimension or 'unknown'}".encode("utf-8")
    ).hexdigest()
    payload = {
        "content_hash": content_hash,
        "parser_version": parser_version,
        "ocr_config": as_dict(ocr_config),
        "chunking_config": as_dict(chunking_config),
        "embedding_model": embedding_model,
        "embedding_profile": profile,
        "embedding_dimension": dimension,
        "endpoint": endpoint,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _safe_ingest_directory(self: Any, directory: str | Path | None = None) -> list[dict[str, Any]]:
    """Track directory-ingestion futures so clear/cancel can drain them safely."""
    source_dir = Path(directory) if directory else self.settings.incoming_dir
    source_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(source_dir.glob("*.pdf"))
    results: list[dict[str, Any]] = []
    max_workers = max(1, min(int(self.settings.max_workers), 4))
    if max_workers <= 1 or len(pdf_paths) <= 1:
        for pdf_path in pdf_paths:
            results.append(self.ingest_file(pdf_path))
        return results

    from concurrent.futures import ThreadPoolExecutor, as_completed

    workers = min(max_workers, len(pdf_paths))
    future_lock = getattr(self, "_ingest_future_lock", None)
    if future_lock is None:
        future_lock = threading.RLock()
        self._ingest_future_lock = future_lock
    self._ingest_futures = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rag-ingest") as executor:
        future_to_path = {}
        for path in pdf_paths:
            future = executor.submit(self.ingest_file, path)
            future_to_path[future] = path
            with future_lock:
                self._ingest_futures[id(future)] = future
        try:
            for future in as_completed(future_to_path):
                result = future.result()
                results.append(result)
                with future_lock:
                    self._ingest_futures.pop(id(future), None)
        finally:
            with future_lock:
                remaining = list(self._ingest_futures.values())
            for future in remaining:
                try:
                    future.cancel()
                except Exception:
                    pass
            with future_lock:
                self._ingest_futures.clear()
    results.sort(key=lambda result: result.get("file_name", ""))
    return results


def _safe_extract_markdown(self: Any, pdf_path: str | Path) -> str:
    """Use the same killable worker for the legacy whole-document Markdown API."""
    from rag_project.runtime_hardening import _run_markdown_worker, _timed_process

    path = Path(pdf_path)
    import fitz

    pdf = fitz.open(str(path))
    try:
        page_count = pdf.page_count
    finally:
        pdf.close()
    # Whole-document conversion remains optional and is given a bounded global budget.
    # Concatenate per-page results so one pathological page cannot freeze the process.
    chunks: list[str] = []
    timeout_per_page = max(3.0, float(os.getenv("RAG_PAGE_TIMEOUT", "30")))
    for index in range(page_count):
        ok, payload, error = _timed_process(_run_markdown_worker, (str(path), index), timeout_per_page)
        if ok and payload:
            value = str(payload[0] or "").strip()
            if value:
                chunks.append(value)
        elif error:
            logger = getattr(self, "logger", None)
            if logger:
                logger.warning("Whole-document Markdown page %d skipped: %s", index + 1, error)
    if not chunks:
        raise RuntimeError(f"No Markdown could be extracted from {path.name}.")
    return "\n\n".join(chunks)


def _safe_classify_with_password(pdf_path: str | Path) -> dict[str, Any]:
    import fitz
    from rag_project.runtime_hardening import _safe_classify

    path = Path(pdf_path)
    try:
        pdf = fitz.open(str(path))
    except Exception as exc:
        return {
            "file_name": path.name,
            "page_count": 0,
            "document_type": "invalid_pdf",
            "ocr_required": False,
            "classification_scope": "preflight",
            "error": str(exc),
        }
    try:
        if getattr(pdf, "needs_pass", False):
            return {
                "file_name": path.name,
                "page_count": pdf.page_count,
                "document_type": "encrypted",
                "ocr_required": False,
                "password_required": True,
                "classification_scope": "preflight",
            }
    finally:
        pdf.close()
    return _safe_classify(path)


def _lexical_fallback_after_embedding_failure(self: Any, pdf_path: str | Path) -> dict[str, Any]:
    """Degrade to a searchable lexical index when Ollama embeddings are unavailable."""
    from rag_project.chunking.semantic_chunker import SemanticChunker
    from rag_project.ingestion.document_classifier import DocumentClassifier
    from rag_project.parsing.pdf_extractor import PDFExtractor

    file_path = Path(pdf_path)
    content_hash = self._hash_file(file_path)
    document = self.state_store.get_by_hash(content_hash) or self.state_store.get_by_path(str(file_path.resolve()))
    if not document:
        return {"status": "failed", "file_name": file_path.name, "error": "No state record for lexical fallback."}
    document_id = document["document_id"]
    worker_id = f"lexical-{threading.get_ident()}-{time.time_ns()}"
    if not self.state_store.claim_document(
        document_id, worker_id, lease_seconds=self.settings.ingestion_lease_seconds
    ):
        return {"status": "busy", "file_name": file_path.name, "document_id": document_id}

    indexed = 0
    try:
        classification = DocumentClassifier.classify(file_path)
        self.state_store.transition_document_state(
            document_id, "EXTRACTING", total_pages=int(classification.get("page_count") or 0)
        )
        extractor = PDFExtractor(
            self.state_store,
            ocr_enabled=getattr(self.settings, "ocr_enabled", False),
            ocr_confidence_threshold=getattr(self.settings, "ocr_confidence_threshold", 0.55),
            ocr_min_char_density=getattr(self.settings, "ocr_min_char_density", 0.001),
            ocr_image_coverage_threshold=getattr(self.settings, "ocr_image_coverage_threshold", 0.55),
        )
        pages = extractor.extract_iter(file_path, document_id)
        chunker = SemanticChunker(self.settings.chunk_size, self.settings.chunk_overlap)
        self.state_store.transition_document_state(document_id, "CHUNKING")
        batches = chunker.chunk_page_batches(pages, batch_size=self.settings.page_batch_size)

        self.vector_store.delete_version(document_id, content_hash)
        for batch in batches:
            if not batch:
                continue
            documents = [chunk.text for chunk in batch]
            metadatas: list[dict[str, Any]] = []
            ids: list[str] = []
            for offset, chunk in enumerate(batch):
                chunk_index = indexed + offset
                chunk_id = f"{document_id}-{content_hash[:12]}-{chunk_index}"
                ids.append(chunk_id)
                metadatas.append({
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "file_name": chunk.file_name,
                    "page_numbers": chunk.page_numbers,
                    "chunk_index": chunk_index,
                    "document_type": classification.get("document_type", "unknown"),
                    "language": None,
                    "evidence_types": chunk.metadata.get("evidence_types", ["text"]),
                    "index_state": "BUILDING",
                    "version_id": content_hash,
                    "retrieval_mode": "lexical",
                })
            self.vector_store.add_lexical_documents(documents, metadatas, ids)
            indexed += len(batch)
            self.state_store.update_document(document_id, current_page=indexed)

        if indexed == 0:
            raise ValueError(f"No extractable text was produced for lexical fallback: {file_path.name}")
        self.vector_store.set_version_index_state(document_id, content_hash, "READY")
        self.state_store.transition_document_state(
            document_id,
            "DEGRADED_LEXICAL",
            embedding_dimension=None,
            error="Semantic embeddings unavailable; lexical retrieval index is active.",
        )
        return {
            "status": "success",
            "document_id": document_id,
            "file_name": file_path.name,
            "document_type": classification.get("document_type", "unknown"),
            "page_count": classification.get("page_count", 0),
            "chunk_count": indexed,
            "embedding_count": 0,
            "retrieval_mode": "lexical",
            "degraded": True,
        }
    except Exception as exc:
        try:
            self.vector_store.delete_version(document_id, content_hash)
            self.state_store.transition_document_state(document_id, "FAILED_INDEXING", error=str(exc))
        except Exception:
            pass
        return {"status": "failed", "file_name": file_path.name, "document_id": document_id, "error": str(exc)}
    finally:
        self.state_store.release_document(document_id, worker_id)


def _safe_ingest_file(self: Any, pdf_path: str | Path) -> dict[str, Any]:
    original = self._original_safe_ingest_file
    result = original(pdf_path)
    if result.get("status") == "failed" and "FAILED_EMBEDDING" in str(result.get("error", "")):
        fallback = _lexical_fallback_after_embedding_failure(self, pdf_path)
        if fallback.get("status") == "success":
            return fallback
    return result


def install() -> None:
    from rag_project.storage.vector_store import VectorStore
    from rag_project.app.rag_system import RAGSystem
    from rag_project.ingestion.document_classifier import DocumentClassifier
    from rag_project.parsing.pdf_extractor import PDFExtractor

    if not hasattr(VectorStore, "_original_safe_add_documents"):
        VectorStore._original_safe_add_documents = VectorStore.add_documents
        VectorStore.add_documents = _safe_add_documents

    if not hasattr(RAGSystem, "_original_safe_ingestion_version_id"):
        RAGSystem._original_safe_ingestion_version_id = RAGSystem._ingestion_version_id
        RAGSystem._ingestion_version_id = staticmethod(_safe_ingestion_version_id)

    if not hasattr(RAGSystem, "_original_safe_ingest_directory"):
        RAGSystem._original_safe_ingest_directory = RAGSystem.ingest_directory
        RAGSystem.ingest_directory = _safe_ingest_directory

    if hasattr(PDFExtractor, "extract_markdown") and not hasattr(PDFExtractor, "_original_safe_extract_markdown"):
        PDFExtractor._original_safe_extract_markdown = PDFExtractor.extract_markdown
        PDFExtractor.extract_markdown = _safe_extract_markdown

    if not hasattr(DocumentClassifier, "_original_safe_password_classify"):
        DocumentClassifier._original_safe_password_classify = DocumentClassifier.classify
        DocumentClassifier.classify = staticmethod(_safe_classify_with_password)

    if not hasattr(RAGSystem, "_original_safe_ingest_file"):
        RAGSystem._original_safe_ingest_file = RAGSystem.ingest_file
        RAGSystem.ingest_file = _safe_ingest_file
