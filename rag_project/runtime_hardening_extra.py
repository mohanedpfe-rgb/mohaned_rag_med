from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Sequence


def _safe_add_documents(
    self: Any,
    documents: Sequence[str],
    metadatas: Sequence[dict[str, Any]],
    embeddings: Sequence[Sequence[float]],
    ids: Sequence[str],
) -> None:
    """Best-effort atomic protocol: BUILDING records stay hidden until finalization."""
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
        promote_ready = True
        for index, item in enumerate(documents):
            metadata = self._coerce_metadata(metadatas[index])
            metadata.setdefault("document_id", "unknown")
            metadata.setdefault("chunk_id", safe_ids[index])
            requested_state = str(metadata.get("index_state", "READY")).upper()
            metadata["index_state"] = requested_state
            promote_ready = promote_ready and requested_state == "READY"
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
            if promote_ready:
                self.collection.update(
                    ids=safe_ids,
                    metadatas=[{**meta, "index_state": "READY"} for meta in normalized],
                )
                self._set_lexical_ready_safe(safe_ids)
        except Exception:
            raise


def _set_lexical_ready_safe(self: Any, ids: Sequence[str]) -> None:
    if not ids:
        return
    import sqlite3

    with sqlite3.connect(self.lexical_database) as connection:
        connection.executemany(
            "UPDATE lexical_documents SET index_state='READY', metadata=json_set(metadata, '$.index_state', 'READY') WHERE id = ?",
            [(str(item),) for item in ids],
        )
        connection.commit()


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
    """Stable content/index fingerprint including embedding endpoint/profile/dimension."""
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
    """Track directory-ingestion futures so cancellation/clear can drain them safely."""
    source_dir = Path(directory) if directory else self.settings.incoming_dir
    source_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(source_dir.glob("*.pdf"))
    if not pdf_paths:
        return []
    max_workers = max(1, min(int(self.settings.max_workers), 2))
    if max_workers == 1 or len(pdf_paths) == 1:
        return [self.ingest_file(path) for path in pdf_paths]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    future_lock = getattr(self, "_ingest_future_lock", None)
    if future_lock is None:
        future_lock = threading.RLock()
        self._ingest_future_lock = future_lock
    results: list[dict[str, Any]] = []
    self._ingest_futures = {}
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rag-ingest") as executor:
        futures = {}
        for path in pdf_paths:
            future = executor.submit(self.ingest_file, path)
            futures[future] = path
            with future_lock:
                self._ingest_futures[id(future)] = future
        try:
            for future in as_completed(futures):
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
    return sorted(results, key=lambda result: result.get("file_name", ""))


def _safe_extract_markdown(self: Any, pdf_path: str | Path) -> str:
    """Use the same killable worker for the legacy whole-document Markdown API."""
    from rag_project.runtime_hardening import _run_markdown_worker, _timed_process
    import fitz

    path = Path(pdf_path)
    pdf = fitz.open(str(path))
    try:
        page_count = pdf.page_count
    finally:
        pdf.close()
    timeout_per_page = max(3.0, float(os.getenv("RAG_PAGE_TIMEOUT", "30")))
    chunks: list[str] = []
    for index in range(page_count):
        ok, payload, error = _timed_process(
            _run_markdown_worker, (str(path), index), timeout_per_page
        )
        if ok and payload:
            value = str(payload[0] or "").strip()
            if value:
                chunks.append(value)
        elif error and getattr(self, "logger", None):
            self.logger.warning("Whole-document Markdown page %d skipped: %s", index + 1, error)
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


def _start_lease_heartbeat(self: Any, document_id: str, worker_id: str):
    interval = max(5.0, min(60.0, float(self.settings.ingestion_lease_seconds) / 3.0))
    stop = threading.Event()

    def run() -> None:
        while not stop.wait(interval):
            try:
                if not self.state_store.heartbeat_document(
                    document_id, worker_id, lease_seconds=self.settings.ingestion_lease_seconds
                ):
                    self.logger.warning("Lease ownership lost for %s", document_id)
                    break
            except Exception:
                self.logger.exception("Lease heartbeat failed for %s", document_id)
                break

    thread = threading.Thread(target=run, name=f"lease-{document_id[:8]}", daemon=True)
    thread.start()
    return stop, thread


def _safe_move_into_processed(self: Any, source: Path, content_hash: str) -> Path:
    target = Path(self.settings.processed_dir) / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return target
    if target.exists():
        archived = Path(self.settings.archive_dir) / source.name
        if archived.exists():
            archived = archived.with_name(f"{source.stem}-{content_hash[:12]}{source.suffix}")
        archived.parent.mkdir(parents=True, exist_ok=True)
        if os.stat(target.parent).st_dev == os.stat(archived.parent).st_dev:
            os.replace(target, archived)
        else:
            tmp = archived.with_name(archived.name + f".{uuid.uuid4().hex}.part")
            shutil.copy2(target, tmp)
            os.replace(tmp, archived)
            target.unlink()
    if os.stat(source.parent).st_dev == os.stat(target.parent).st_dev:
        os.replace(source, target)
    else:
        tmp = target.with_name(target.name + f".{uuid.uuid4().hex}.part")
        shutil.copy2(source, tmp)
        os.replace(tmp, target)
        source.unlink()
    return target


def _safe_clear_pdf_data(self: Any) -> list[str]:
    """Cancel workers, wait for their cooperative shutdown, then clear both stores."""
    if callable(getattr(self, "cancel_all_ingests", None)):
        self.cancel_all_ingests()
    deadline = time.monotonic() + max(10.0, float(os.getenv("RAG_CLEAR_DRAIN_TIMEOUT", "120")))
    while time.monotonic() < deadline:
        flags = globals().get("_last_flags")
        try:
            from rag_project.app.rag_system import _INGEST_CANCEL_FLAGS
            active_flags = list(_INGEST_CANCEL_FLAGS.values())
        except Exception:
            active_flags = []
        futures_obj = getattr(self, "_ingest_futures", None)
        futures = list(futures_obj.values()) if hasattr(futures_obj, "values") else list(futures_obj or [])
        if not active_flags and not [f for f in futures if not getattr(f, "done", lambda: True)()]:
            break
        for future in futures:
            try:
                future.cancel()
            except Exception:
                pass
        time.sleep(0.15)

    lock = getattr(self, "_index_write_lock", None)
    if lock is None:
        lock = threading.RLock()
        self._index_write_lock = lock
    removed: list[str] = []
    with lock:
        self.vector_store.clear_all()
        self.state_store.clear_all()
        for raw_dir in (
            getattr(self.settings, "incoming_dir", None),
            getattr(self.settings, "processed_dir", None),
            getattr(self.settings, "failed_dir", None),
            getattr(self.settings, "archive_dir", None),
        ):
            if raw_dir is None:
                continue
            directory = Path(raw_dir)
            directory.mkdir(parents=True, exist_ok=True)
            for child in directory.iterdir():
                try:
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                    removed.append(str(child))
                except OSError:
                    self.logger.warning("Unable to remove %s", child, exc_info=True)
    try:
        self.conversation_memory.history.clear()
    except Exception:
        pass
    return removed


def _multilingual_query_assess(question: str) -> dict[str, Any]:
    from rag_project.utils.text_utils import meaningful_tokens

    raw = (question or "").strip()
    if not raw:
        return {
            "query_quality": "LOW_QUALITY_QUERY",
            "query_intent": "UNKNOWN",
            "should_abstain": True,
            "decision": "NOT_SUPPORTED",
            "reason": "The question is empty.",
            "clarification": "Please provide a more precise question.",
        }
    tokens = [token for token in meaningful_tokens(raw) if token]
    lowered = raw.casefold()
    generic = {
        "related", "relation", "relationship", "compare", "difference", "versus", "vs",
        "between", "and", "or", "et", "ou", "entre", "lien", "rapport", "relation",
        "ما", "ماذا", "كيف", "هل", "بين", "علاقة", "مرتبط", "مرتبطان", "و", "أو",
    }
    meaningful = [token for token in tokens if token not in generic and len(token) > 1]
    only_generic = bool(tokens) and not meaningful
    short_ambiguous = len(tokens) <= 2 and only_generic
    if short_ambiguous:
        return {
            "query_quality": "LOW_QUALITY_QUERY",
            "query_intent": "AMBIGUOUS_OR_MALFORMED",
            "should_abstain": True,
            "decision": "NOT_SUPPORTED",
            "reason": "The query contains insufficient domain-specific information.",
            "clarification": "Please include the exact medical concept, condition, drug, finding, or comparison you want answered.",
        }
    if any(term in lowered for term in ("related", "relation", "relationship", "علاقة", "مرتبط", "lien")):
        intent = "RELATIONSHIP"
    elif any(term in lowered for term in ("compare", "difference", "versus", " vs ", "between", "comparez", "différence", "فرق", "مقارنة", "بين")):
        intent = "COMPARISON"
    else:
        intent = "DIRECT_QUESTION"
    return {
        "query_quality": "HIGH_QUALITY_QUERY",
        "query_intent": intent,
        "should_abstain": False,
        "decision": "DIRECTLY_SUPPORTED",
        "reason": "Question contains enough lexical signal to evaluate against evidence.",
        "clarification": "",
    }


def _safe_ingest_file_streaming(self: Any, pdf_path: str | Path) -> dict[str, Any]:
    """Bounded-memory ingestion: extract -> chunk -> embed/index batch-by-batch."""
    from datetime import datetime, timezone
    from rag_project.app.rag_system import _INGEST_CANCEL_FLAGS, _INGEST_LOCK, _IngestCancelFlag
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
    previous_version = previous.get("content_hash") if previous and previous.get("content_hash") != content_hash else None

    try:
        self.embedding_service.discover_dimension()
    except Exception:
        # Lexical degradation is handled below; do not make model discovery a hard dependency.
        pass
    identity = self.embedding_service.identity
    dimension = getattr(identity, "dimension", None)
    profile = getattr(identity, "fingerprint", getattr(identity, "configuration_fingerprint", None))
    ocr_config = {
        "engine": "rapidocr",
        "enabled": bool(getattr(self.settings, "ocr_enabled", False)),
        "scale": 2,
        "confidence_threshold": float(getattr(self.settings, "ocr_confidence_threshold", 0.55)),
        "max_pixels": int(os.getenv("RAG_OCR_MAX_PIXELS", "12000000")),
    }
    chunk_config = {"size": self.settings.chunk_size, "overlap": self.settings.chunk_overlap}
    version_id = self._ingestion_version_id(
        content_hash=content_hash,
        parser_version="pdf-extractor-v3",
        ocr_config=ocr_config,
        chunking_config=chunk_config,
        embedding_model=self.settings.embedding_model,
        embedding_profile=profile,
        embedding_dimension=dimension,
    )
    if existing and self.state_store.is_ready_status(existing.get("status")) and existing.get("version_id") == version_id:
        return {
            "status": "skipped",
            "file_name": source.name,
            "document_id": existing["document_id"],
            "reason": "identical content is already indexed for the active parser/embedding profile",
        }

    stat = source.stat()
    self.state_store.upsert_document({
        "document_id": document_id,
        "content_hash": content_hash,
        "file_path": str(source.resolve()),
        "file_name": source.name,
        "file_size": stat.st_size,
        "created_at": utc_now(),
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "ingestion_started_at": utc_now(),
        "current_stage": "DISCOVERED",
        "current_page": 0,
        "total_pages": 0,
        "status": "RUNNING",
        "parser_version": "pdf-extractor-v3",
        "ocr_config": json.dumps(ocr_config, sort_keys=True),
        "chunking_config": json.dumps(chunk_config, sort_keys=True),
        "embedding_model": self.settings.embedding_model,
        "embedding_dimension": dimension,
        "index_state": "BUILDING",
        "version_id": version_id,
    })

    worker_id = str(uuid.uuid4())
    if not self.state_store.claim_document(document_id, worker_id, lease_seconds=self.settings.ingestion_lease_seconds):
        return {"status": "busy", "file_name": source.name, "document_id": document_id}
    cancel_flag = _IngestCancelFlag()
    with _INGEST_LOCK:
        _INGEST_CANCEL_FLAGS[document_id] = cancel_flag

    heartbeat_stop, heartbeat_thread = _start_lease_heartbeat(self, document_id, worker_id)
    semantic_enabled = True
    indexed = 0
    embeddings_count = 0
    document_language: str | None = None
    page_count = 0
    target: Path | None = None
    try:
        # Remove stale BUILDING/failed records for this exact version before starting.
        try:
            self.vector_store.delete_version(document_id, content_hash)
        except Exception:
            pass

        classification = DocumentClassifier.classify(source)
        page_count = int(classification.get("page_count") or 0)
        self.state_store.transition_document_state(document_id, "VALIDATING", total_pages=page_count)
        if classification.get("document_type") == "encrypted":
            raise RuntimeError("FAILED_EXTRACTION: PDF is password-protected and requires a password before ingestion.")
        if classification.get("document_type") == "invalid_pdf":
            raise RuntimeError(f"FAILED_EXTRACTION: {classification.get('error', 'invalid PDF')}")

        self.state_store.transition_document_state(document_id, "EXTRACTING", total_pages=page_count)
        extractor = PDFExtractor(
            self.state_store,
            ocr_enabled=getattr(self.settings, "ocr_enabled", False),
            ocr_confidence_threshold=getattr(self.settings, "ocr_confidence_threshold", 0.55),
            ocr_min_char_density=getattr(self.settings, "ocr_min_char_density", 0.001),
            ocr_image_coverage_threshold=getattr(self.settings, "ocr_image_coverage_threshold", 0.55),
        )
        pages = extractor.extract_iter(source, document_id)
        chunker = SemanticChunker(self.settings.chunk_size, self.settings.chunk_overlap)
        batches = chunker.chunk_page_batches(pages, batch_size=self.settings.page_batch_size)

        self.state_store.transition_document_state(document_id, "CHUNKING", total_pages=page_count)
        for batch in batches:
            if cancel_flag.cancelled:
                raise RuntimeError("Ingestion cancelled by user.")
            if not batch:
                continue
            documents = [str(chunk.text or "").strip() for chunk in batch if str(chunk.text or "").strip()]
            if not documents:
                continue
            if document_language is None:
                document_language = detect_language(" ".join(documents))
            metadatas: list[dict[str, Any]] = []
            ids: list[str] = []
            for offset, chunk in enumerate(batch):
                if not str(chunk.text or "").strip():
                    continue
                chunk_index = indexed + offset
                chunk.chunk_index = chunk_index
                chunk_id = f"{document_id}-{content_hash[:12]}-{chunk_index}"
                ids.append(chunk_id)
                metadatas.append({
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "file_name": chunk.file_name,
                    "page_numbers": list(chunk.page_numbers or []),
                    "chunk_index": chunk_index,
                    "document_type": classification.get("document_type", "unknown"),
                    "language": document_language,
                    "evidence_types": chunk.metadata.get("evidence_types", ["text"]),
                    "index_state": "BUILDING",
                    "version_id": content_hash,
                })
            if not ids:
                continue

            if semantic_enabled:
                self.state_store.transition_document_state(document_id, "EMBEDDING", current_page=indexed)
                try:
                    vectors = self.embedding_service.embed_texts(documents)
                    if len(vectors) != len(documents):
                        raise RuntimeError(f"Embedding backend returned {len(vectors)} vectors for {len(documents)} chunks.")
                    if not vectors:
                        raise RuntimeError("Embedding backend returned no vectors.")
                    self.vector_store.set_expected_identity(self.embedding_service.identity)
                    compatibility = self.vector_store.compatibility_report(self.embedding_service.identity)
                    if not compatibility["valid"]:
                        raise RuntimeError(compatibility["message"])
                    self.vector_store.add_documents(documents, metadatas, vectors, ids)
                    embeddings_count += len(vectors)
                except Exception as exc:
                    self.logger.warning("Semantic embedding unavailable; switching document to lexical-only mode: %s", exc)
                    semantic_enabled = False
            if not semantic_enabled:
                self.vector_store.add_lexical_documents(documents, metadatas, ids)
            indexed += len(ids)
            self.state_store.update_document(document_id, current_stage="INDEXING", current_page=indexed, total_pages=page_count)

        if indexed == 0:
            raise ValueError(f"No extractable text was produced for {source.name}.")

        self.state_store.transition_document_state(document_id, "INDEXING", current_page=page_count, total_pages=page_count)
        if semantic_enabled and embeddings_count == indexed:
            validation = self.vector_store.validate_document_index(document_id, content_hash)
            if not validation.get("valid") or int(validation.get("count", 0)) != embeddings_count:
                raise RuntimeError("FAILED_INDEXING: semantic index validation failed: " + "; ".join(validation.get("issues", [])))
        self._safe_move_into_processed(source, content_hash)
        if semantic_enabled:
            self.vector_store.set_version_index_state(document_id, content_hash, "READY")
        else:
            self.vector_store.set_version_index_state(document_id, content_hash, "READY")
        if previous_version:
            self.vector_store.delete_version(document_id, previous_version)

        if semantic_enabled:
            self.state_store.transition_document_state(
                document_id,
                "READY",
                embedding_dimension=getattr(self.embedding_service, "dimension", dimension),
                current_page=page_count,
                total_pages=page_count,
                version_id=version_id,
                index_state="READY",
                ingestion_completed_at=utc_now(),
                file_path=str((Path(self.settings.processed_dir) / source.name).resolve()),
                ingestion_metrics=json.dumps({
                    "page_count": page_count,
                    "chunk_count": indexed,
                    "embedding_count": embeddings_count,
                    "retrieval_mode": "hybrid",
                    "total_ms": round((time.perf_counter() - started) * 1000, 3),
                }, sort_keys=True),
            )
        else:
            self.state_store.transition_document_state(
                document_id,
                "DEGRADED_LEXICAL",
                embedding_dimension=None,
                current_page=page_count,
                total_pages=page_count,
                version_id=version_id,
                index_state="READY",
                ingestion_completed_at=utc_now(),
                file_path=str((Path(self.settings.processed_dir) / source.name).resolve()),
                error="Semantic embeddings unavailable; lexical retrieval index is active.",
                ingestion_metrics=json.dumps({
                    "page_count": page_count,
                    "chunk_count": indexed,
                    "embedding_count": 0,
                    "retrieval_mode": "lexical",
                    "total_ms": round((time.perf_counter() - started) * 1000, 3),
                }, sort_keys=True),
            )
        return {
            "status": "success",
            "document_id": document_id,
            "file_name": source.name,
            "document_type": classification.get("document_type", "unknown"),
            "page_count": page_count,
            "chunk_count": indexed,
            "embedding_count": embeddings_count,
            "retrieval_mode": "hybrid" if semantic_enabled else "lexical",
            "degraded": not semantic_enabled,
        }
    except Exception as exc:
        self.logger.exception("Failed to process %s", source.name)
        try:
            self.vector_store.delete_version(document_id, content_hash)
        except Exception:
            self.logger.exception("Failed to roll back index for %s", source.name)
        failed_stage = "FAILED_EXTRACTION" if str(exc).startswith("FAILED_EXTRACTION:") else (
            "FAILED_EMBEDDING" if str(exc).startswith("FAILED_EMBEDDING:") else "FAILED_INDEXING"
        )
        try:
            self.state_store.transition_document_state(document_id, failed_stage, error=str(exc))
        except Exception:
            try:
                self.state_store.update_document(document_id, current_stage=failed_stage, status=failed_stage, index_state="FAILED", error=str(exc))
            except Exception:
                pass
        failed_path = Path(self.settings.failed_dir) / source.name
        try:
            if source.exists() and source.resolve() != failed_path.resolve():
                failed_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = failed_path.with_name(failed_path.name + f".{uuid.uuid4().hex}.part")
                shutil.copy2(source, tmp)
                os.replace(tmp, failed_path)
        except OSError as quarantine_exc:
            self.logger.warning("Could not quarantine %s: %s", source.name, quarantine_exc)
        return {"status": "failed", "file_name": source.name, "document_id": document_id, "error": str(exc)}
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=2.0)
        self.state_store.release_document(document_id, worker_id)
        with _INGEST_LOCK:
            _INGEST_CANCEL_FLAGS.pop(document_id, None)


def _safe_llm_generate_raw(self: Any, prompt: str, system_prompt: str | None, temperature: float) -> dict[str, Any]:
    """Explicit bounded retry loop; unlike decorator retries it has a real wall-clock budget."""
    import requests

    retry_budget = max(5.0, float(os.getenv("OLLAMA_RETRY_BUDGET_SECONDS", "60")))
    max_attempts = max(1, int(os.getenv("OLLAMA_MAX_ATTEMPTS", "3")))
    deadline = time.monotonic() + retry_budget
    payload = {
        "model": self.model,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": self.max_output_tokens},
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        payload["messages"].insert(0, {"role": "system", "content": system_prompt})
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        read_timeout = max(1.0, min(float(self.timeout_seconds), remaining))
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=(min(5.0, read_timeout), read_timeout),
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError("Ollama returned a non-object JSON response.")
            return data
        except (requests.RequestException, ValueError, TypeError, RuntimeError) as exc:
            last_error = exc
            self.retries_encountered += 1
            if attempt >= max_attempts:
                break
            sleep_for = min(2.0 ** (attempt - 1), max(0.0, deadline - time.monotonic()))
            if sleep_for > 0:
                time.sleep(sleep_for)
    raise RuntimeError(f"Generation service failed for model {self.model!r}: {last_error}") from last_error


def install() -> None:
    from rag_project.storage.vector_store import VectorStore
    from rag_project.app.rag_system import RAGSystem, QueryQualityClassifier
    from rag_project.ingestion.document_classifier import DocumentClassifier
    from rag_project.parsing.pdf_extractor import PDFExtractor
    from rag_project.generation.llm_client import OllamaLLMClient

    if not hasattr(VectorStore, "_original_safe_add_documents"):
        VectorStore._original_safe_add_documents = VectorStore.add_documents
        VectorStore._set_lexical_ready_safe = _set_lexical_ready_safe.__get__(VectorStore, VectorStore)
        VectorStore.add_documents = _safe_add_documents

    if not hasattr(RAGSystem, "_original_safe_ingestion_version_id"):
        RAGSystem._original_safe_ingestion_version_id = RAGSystem._ingestion_version_id
        RAGSystem._ingestion_version_id = staticmethod(_safe_ingestion_version_id)

    if not hasattr(RAGSystem, "_original_safe_ingest_directory"):
        RAGSystem._original_safe_ingest_directory = RAGSystem.ingest_directory
        RAGSystem.ingest_directory = _safe_ingest_directory

    if not hasattr(RAGSystem, "_original_safe_ingest_file_streaming"):
        RAGSystem._original_safe_ingest_file_streaming = RAGSystem.ingest_file
        RAGSystem.ingest_file = _safe_ingest_file_streaming

    if hasattr(PDFExtractor, "extract_markdown") and not hasattr(PDFExtractor, "_original_safe_extract_markdown"):
        PDFExtractor._original_safe_extract_markdown = PDFExtractor.extract_markdown
        PDFExtractor.extract_markdown = _safe_extract_markdown

    if not hasattr(DocumentClassifier, "_original_safe_password_classify"):
        DocumentClassifier._original_safe_password_classify = DocumentClassifier.classify
        DocumentClassifier.classify = staticmethod(_safe_classify_with_password)

    if not hasattr(RAGSystem, "_original_safe_clear_pdf_data"):
        RAGSystem._original_safe_clear_pdf_data = RAGSystem.clear_pdf_data
        RAGSystem.clear_pdf_data = _safe_clear_pdf_data

    if not hasattr(QueryQualityClassifier, "_original_safe_multilingual_assess"):
        QueryQualityClassifier._original_safe_multilingual_assess = QueryQualityClassifier.assess
        QueryQualityClassifier.assess = staticmethod(_multilingual_query_assess)

    if not hasattr(OllamaLLMClient, "_original_safe_generate_raw"):
        OllamaLLMClient._original_safe_generate_raw = OllamaLLMClient._generate_raw
        OllamaLLMClient._generate_raw = _safe_llm_generate_raw
