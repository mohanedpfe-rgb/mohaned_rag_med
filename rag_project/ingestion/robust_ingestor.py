from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from rag_project.chunking.semantic_chunker import SemanticChunker
from rag_project.ingestion.atomic_claim import ensure_and_claim
from rag_project.ingestion.document_classifier import DocumentClassifier
from rag_project.ingestion.state_store import utc_now
from rag_project.parsing.pdf_extractor import PDFExtractor
from rag_project.utils.text_utils import detect_language


def _unique_archive_path(directory: Path, source: Path, suffix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    extension = source.suffix
    candidates = [directory / source.name, directory / f"{stem}-{suffix}{extension}"]
    for candidate in candidates:
        if not candidate.exists():
            return candidate
    for _ in range(100):
        candidate = directory / f"{stem}-{suffix}-{uuid.uuid4().hex[:10]}{extension}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Unable to allocate a unique archive path.")


def robust_ingest_file(system: Any, pdf_path: str | Path) -> dict[str, Any]:
    """Single production ingestion implementation with durable progress and lease fencing."""
    started = time.perf_counter()
    stage_started = started
    stage_timings: dict[str, float] = {}
    file_path = Path(pdf_path)

    def mark(name: str) -> None:
        nonlocal stage_started
        now = time.perf_counter()
        stage_timings[name] = round((now - stage_started) * 1000, 3)
        stage_started = now

    if file_path.suffix.lower() != ".pdf" or not file_path.is_file():
        raise ValueError(f"Unsupported or missing PDF: {file_path}")

    content_hash = system._hash_file(file_path)
    existing = system.state_store.get_by_hash(content_hash)
    previous = system.state_store.get_by_path(str(file_path.resolve()))
    previous_version = previous.get("content_hash") if previous and previous.get("content_hash") != content_hash else None
    current_chunking_config = json.dumps({"size": system.settings.chunk_size, "overlap": system.settings.chunk_overlap}, sort_keys=True)
    current_ocr_config = json.dumps({"engine": "rapidocr", "scale": 2}, sort_keys=True)
    current_version_id = system._ingestion_version_id(content_hash=content_hash, parser_version="pdf-extractor-v2", ocr_config=current_ocr_config, chunking_config=current_chunking_config, embedding_model=system.settings.embedding_model, embedding_profile=None, embedding_dimension=None)

    if existing and system.state_store.is_ready_status(existing.get("status")) and existing.get("version_id") == current_version_id:
        validation = system.vector_store.validate_document_index(existing["document_id"], content_hash)
        if validation.get("valid") and validation.get("count", 0) > 0:
            return {"status": "skipped", "file_name": file_path.name, "document_id": existing["document_id"], "reason": "identical content already indexed and validated"}

    document_id = existing["document_id"] if existing else (previous["document_id"] if previous else content_hash)
    stat = file_path.stat()
    document_values = {
        "document_id": document_id, "content_hash": content_hash, "file_path": str(file_path.resolve()),
        "file_name": file_path.name, "file_size": stat.st_size, "created_at": utc_now(),
        "modified_at": datetime_from_mtime(file_path), "ingestion_started_at": utc_now(),
        "current_stage": "DISCOVERED", "current_page": 0, "total_pages": 0, "status": "RUNNING",
        "parser_version": "pdf-extractor-v2", "ocr_config": current_ocr_config,
        "chunking_config": current_chunking_config, "embedding_model": system.settings.embedding_model,
        "version_id": current_version_id, "index_state": "PENDING",
    }
    worker_id = str(uuid.uuid4())
    if not ensure_and_claim(system.state_store, document_values, worker_id, system.settings.ingestion_lease_seconds):
        return {"status": "busy", "file_name": file_path.name, "document_id": document_id, "reason": "document is currently owned by another ingestion worker"}
    if not system.state_store.claim_document(document_id, worker_id, lease_seconds=system.settings.ingestion_lease_seconds):
        return {"status": "busy", "file_name": file_path.name, "document_id": document_id, "reason": "document lease could not be registered"}

    cancel_flag = system._new_cancel_flag(document_id)
    target = system.settings.processed_dir / file_path.name
    previous_target_backup: Path | None = None
    moved_into_processed = False
    published = False
    try:
        if existing and not system.state_store.is_ready_status(existing.get("status")):
            try:
                system.vector_store.delete_version(document_id, content_hash)
            except Exception:
                system.logger.exception("Failed to clean partial retry index for %s", file_path.name)
        if previous and previous.get("content_hash") != content_hash:
            system.state_store.delete_pages(document_id)

        if not system.state_store.heartbeat_document(document_id, worker_id, lease_seconds=system.settings.ingestion_lease_seconds):
            raise RuntimeError("Ingestion lease was lost before processing started.")

        last_heartbeat = time.perf_counter()
        heartbeat_interval = max(10.0, min(60.0, system.settings.ingestion_lease_seconds / 4))

        def renew(force: bool = False) -> None:
            nonlocal last_heartbeat
            now = time.perf_counter()
            if force or now - last_heartbeat >= heartbeat_interval:
                if not system.state_store.heartbeat_document(document_id, worker_id, lease_seconds=system.settings.ingestion_lease_seconds):
                    raise RuntimeError("Ingestion lease expired during processing.")
                last_heartbeat = now

        def check_cancel() -> None:
            if cancel_flag.cancelled:
                raise RuntimeError("Ingestion cancelled by user.")

        classification = DocumentClassifier.classify(file_path)
        total_pages = int(classification.get("page_count") or 0)
        system.state_store.transition_document_state(document_id, "VALIDATING", total_pages=total_pages, current_page=0)
        system.state_store.record_event(document_id, stage="VALIDATING", status="RUNNING", event_type="classification", message=f"Classified {file_path.name}: {classification.get('document_type', 'unknown')} ({total_pages} pages)", details=classification, total_pages=total_pages, file_name=file_path.name)
        mark("classification")
        renew(force=True)
        check_cancel()

        system.state_store.transition_document_state(document_id, "EXTRACTING", current_page=0, total_pages=total_pages)
        system.state_store.record_event(document_id, stage="EXTRACTING", status="RUNNING", event_type="extract", message=f"Reading {total_pages} PDF pages one by one", details={"file_path": str(file_path)}, current_page=0, total_pages=total_pages, file_name=file_path.name)
        extractor = PDFExtractor(system.state_store, ocr_enabled=getattr(system.settings, "ocr_enabled", False), ocr_confidence_threshold=getattr(system.settings, "ocr_confidence_threshold", 0.55), ocr_min_char_density=getattr(system.settings, "ocr_min_char_density", 0.001), ocr_image_coverage_threshold=getattr(system.settings, "ocr_image_coverage_threshold", 0.55))
        pages = extractor.extract_iter(file_path, document_id)
        chunker = SemanticChunker(system.settings.chunk_size, system.settings.chunk_overlap)
        chunk_batches = chunker.chunk_page_batches(pages, batch_size=system.settings.page_batch_size)
        mark("extract_pipeline_setup")

        chunk_count = 0
        embedding_count = 0
        document_language: str | None = None
        embedding_ms = 0.0
        indexing_ms = 0.0
        current_page = 0
        embedding_stage_started = False
        for batch_number, batch in enumerate(chunk_batches, start=1):
            if not batch:
                continue
            check_cancel()
            renew()
            page_numbers = [page for chunk in batch for page in chunk.page_numbers]
            current_page = min(total_pages, max(page_numbers, default=current_page))
            if not embedding_stage_started:
                system.state_store.transition_document_state(document_id, "CHUNKING", current_page=current_page, total_pages=total_pages)
                system.state_store.record_event(document_id, stage="CHUNKING", status="RUNNING", event_type="chunk", message=f"Prepared first searchable batch through page {current_page} of {total_pages}", details={"page_batch_size": system.settings.page_batch_size, "chunk_size": system.settings.chunk_size, "chunk_overlap": system.settings.chunk_overlap}, current_page=current_page, total_pages=total_pages, file_name=file_path.name)
                system._ensure_embedding_dimension()
                if system.embedding_startup_error is not None:
                    raise RuntimeError(f"FAILED_EMBEDDING: {system.embedding_startup_error}")
                system.vector_store.set_expected_identity(system.embedding_service.identity)
                compatibility = system.vector_store.compatibility_report(system.embedding_service.identity)
                if compatibility.get("status") != "READY":
                    raise RuntimeError(f"FAILED_EMBEDDING: {compatibility.get('message', 'embedding index incompatible')}")
                system.state_store.transition_document_state(document_id, "EMBEDDING", current_page=current_page, total_pages=total_pages)
                system.state_store.record_event(document_id, stage="EMBEDDING", status="RUNNING", event_type="embedding", message="Creating embeddings and committing them in small batches", details={"embedding_batch_size": system.settings.embedding_batch_size}, current_page=current_page, total_pages=total_pages, file_name=file_path.name)
                mark("chunking")
                embedding_stage_started = True
            if document_language is None:
                document_language = detect_language(" ".join(chunk.text for chunk in batch))
            documents = [chunk.text for chunk in batch]
            metadatas: list[dict[str, Any]] = []
            ids: list[str] = []
            for offset, chunk in enumerate(batch):
                global_index = chunk_count + offset
                chunk.chunk_index = global_index
                chunk_id = f"{document_id}-{content_hash[:12]}-{global_index}"
                ids.append(chunk_id)
                metadatas.append({"document_id": chunk.doc_id, "chunk_id": chunk_id, "file_name": chunk.file_name, "page_numbers": chunk.page_numbers, "chunk_index": global_index, "document_type": classification.get("document_type", "unknown"), "language": document_language, "evidence_types": chunk.metadata.get("evidence_types", ["text"]), "index_state": "BUILDING", "version_id": content_hash})
            embedding_started = time.perf_counter()
            try:
                vectors = system.embedding_service.embed_texts(documents)
                if len(vectors) != len(documents):
                    raise RuntimeError(f"Embedding backend returned {len(vectors)} vectors for {len(documents)} chunks")
                system.embedding_service.validate_batch(vectors)
            except Exception as exc:
                embedding_ms += (time.perf_counter() - embedding_started) * 1000
                raise RuntimeError(f"FAILED_EMBEDDING: {exc}") from exc
            embedding_ms += (time.perf_counter() - embedding_started) * 1000
            renew(force=True)
            check_cancel()
            indexing_started = time.perf_counter()
            system.vector_store.add_documents(documents, metadatas, vectors, ids)
            indexing_ms += (time.perf_counter() - indexing_started) * 1000
            chunk_count += len(batch)
            embedding_count += len(vectors)
            system.state_store.update_document(document_id, current_stage="EMBEDDING", current_page=current_page, total_pages=total_pages, embedding_dimension=system.embedding_service.dimension, ingestion_metrics=json.dumps({"stage": "EMBEDDING", "batch": batch_number, "chunks": chunk_count, "embeddings": embedding_count, "page": current_page, "pages": total_pages, "embedding_ms": round(embedding_ms, 3), "indexing_ms": round(indexing_ms, 3), "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)}, sort_keys=True))
            system.state_store.record_event(document_id, stage="EMBEDDING", status="RUNNING", event_type="embedding_batch", message=f"Batch {batch_number}: {len(batch)} chunks indexed through page {current_page}/{total_pages}", details={"batch": batch_number, "chunks_total": chunk_count, "embeddings_total": embedding_count}, current_page=current_page, total_pages=total_pages, file_name=file_path.name)
            renew()

        check_cancel()
        renew(force=True)
        if chunk_count == 0 or embedding_count == 0:
            raise ValueError(f"No extractable searchable content was produced for {file_path.name}.")
        system.state_store.transition_document_state(document_id, "INDEXING", current_page=total_pages, total_pages=total_pages, embedding_dimension=system.embedding_service.dimension)
        system.state_store.record_event(document_id, stage="INDEXING", status="RUNNING", event_type="index_write", message=f"Finalizing {chunk_count} indexed chunks", details={"chunk_count": chunk_count, "embedding_count": embedding_count}, current_page=total_pages, total_pages=total_pages, file_name=file_path.name)
        mark("embedding_and_incremental_index")
        validation = system.vector_store.validate_document_index(document_id, content_hash)
        if not validation.get("valid") or validation.get("count") != embedding_count:
            raise RuntimeError("FAILED_EMBEDDING: committed index validation failed: " + "; ".join(validation.get("issues", [])))

        system.state_store.transition_document_state(document_id, "VALIDATING_INDEX", current_page=total_pages, total_pages=total_pages)
        system.state_store.record_event(document_id, stage="VALIDATING_INDEX", status="RUNNING", event_type="validation", message="Index integrity passed; preparing atomic publication", details={"version_id": content_hash, "chunk_count": chunk_count, "embedding_count": embedding_count}, current_page=total_pages, total_pages=total_pages, file_name=file_path.name)
        mark("validation")
        renew(force=True)
        check_cancel()

        same_target = file_path.resolve() == target.resolve()
        if target.exists() and not same_target:
            previous_target_backup = _unique_archive_path(system.settings.archive_dir, target, content_hash[:12])
            target.replace(previous_target_backup)
        if not same_target:
            file_path.replace(target)
            moved_into_processed = True

        system.vector_store.set_version_index_state(document_id, content_hash, "READY")
        system.state_store.transition_document_state(
            document_id,
            "READY",
            ingestion_completed_at=utc_now(),
            current_page=total_pages,
            total_pages=total_pages,
            file_path=str(target.resolve()),
            ingestion_metrics=json.dumps({"embedding_ms": round(embedding_ms, 3), "indexing_ms": round(indexing_ms, 3), "total": round((time.perf_counter() - started) * 1000, 3), "page_count": total_pages, "chunk_count": chunk_count, "embedding_count": embedding_count}, sort_keys=True),
        )
        published = True

        # Observability is deliberately non-critical after publication. A telemetry/database
        # event failure must never trigger rollback of an already published searchable version.
        try:
            system.state_store.record_event(document_id, stage="READY", status="READY", event_type="completion", message="Document ready for retrieval and grounded questions", details={"page_count": total_pages, "chunk_count": chunk_count, "embedding_count": embedding_count, "vector_store_count": system.vector_store.count()}, current_page=total_pages, total_pages=total_pages, file_name=file_path.name)
        except Exception:
            system.logger.exception("Failed to record completion event for published document %s", file_path.name)

        if previous_version:
            try:
                system.vector_store.set_version_index_state(document_id, previous_version, "FAILED")
                system.vector_store.delete_version(document_id, previous_version)
            except Exception:
                system.logger.exception("Failed to retire previous version for %s", file_path.name)
        if previous_target_backup is not None:
            try:
                previous_target_backup.unlink(missing_ok=True)
            except OSError:
                system.logger.exception("Failed to remove archived previous file for %s", file_path.name)

        total_ms = round((time.perf_counter() - started) * 1000, 3)
        metrics = {"total": total_ms, "page_count": total_pages, "chunk_count": chunk_count, "embedding_count": embedding_count, "embedding_ms": round(embedding_ms, 3), "indexing_ms": round(indexing_ms, 3)}
        system.logger.info("Processed %s incrementally in %.2fs", file_path.name, total_ms / 1000.0)
        return {"status": "success", "document_id": document_id, "file_name": file_path.name, "document_type": classification.get("document_type", "unknown"), "page_count": total_pages, "chunk_count": chunk_count, "embedding_count": embedding_count, "retrieval_mode": "hybrid", "timings_ms": metrics}
    except Exception as exc:
        system.logger.exception("Failed to process %s", file_path.name)
        if published:
            # The durable state already says READY and the new version is visible. Do not
            # destroy a successfully published index because of a late non-critical failure.
            return {"status": "success", "document_id": document_id, "file_name": file_path.name, "warning": "Document was published, but a post-publication operation failed.", "error": type(exc).__name__}
        try:
            system.vector_store.delete_version(document_id, content_hash)
            system.vector_store.set_version_index_state(document_id, content_hash, "FAILED")
        except Exception:
            system.logger.exception("Failed to remove partial index for %s", file_path.name)
        failure_stage = "FAILED_EMBEDDING" if str(exc).startswith("FAILED_EMBEDDING:") else ("FAILED_EXTRACTION" if "extract" in str(exc).casefold() else "FAILED")
        try:
            system.state_store.transition_document_state(document_id, failure_stage, error=str(exc), current_page=current_page if "current_page" in locals() else 0, total_pages=total_pages if "total_pages" in locals() else 0)
        except Exception:
            try:
                system.state_store.update_document(document_id, current_stage=failure_stage, status=failure_stage, error=str(exc))
            except Exception:
                system.logger.exception("Failed to persist ingestion failure state for %s", file_path.name)

        # Restore the previous processed file when a publication step failed.
        if previous_target_backup is not None and previous_target_backup.exists() and not target.exists():
            try:
                previous_target_backup.replace(target)
            except OSError:
                system.logger.exception("Failed to restore previous processed file for %s", file_path.name)

        quarantine_source = target if moved_into_processed and target.exists() else (file_path if file_path.exists() else None)
        failed_path = _unique_archive_path(system.settings.failed_dir, file_path, content_hash[:12])
        if quarantine_source is not None and quarantine_source.exists():
            try:
                failed_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    quarantine_source.resolve().relative_to(system.settings.incoming_dir.resolve())
                    in_incoming = True
                except ValueError:
                    in_incoming = False
                if quarantine_source.resolve() == failed_path.resolve():
                    pass
                elif in_incoming or moved_into_processed:
                    quarantine_source.replace(failed_path)
                else:
                    shutil.copy2(quarantine_source, failed_path)
            except OSError:
                system.logger.exception("Failed to quarantine %s", file_path.name)
        return {"status": "failed", "file_name": file_path.name, "document_id": document_id, "error": str(exc)}
    finally:
        system.state_store.release_document(document_id, worker_id)
        system._remove_cancel_flag(document_id)


def datetime_from_mtime(path: Path) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
