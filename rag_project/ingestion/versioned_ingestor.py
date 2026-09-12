import shutil
import uuid
from pathlib import Path
from typing import Any

from rag_project.ingestion import robust_ingestor


def _unique_archive_path(directory: Path, source: Path, suffix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"{source.stem}-{suffix}{source.suffix}"
    if not candidate.exists():
        return candidate
    for _ in range(100):
        candidate = directory / f"{source.stem}-{suffix}-{uuid.uuid4().hex[:10]}{source.suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Unable to allocate an archive path for a superseded document.")


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    out = dict(result or {})
    status = str(out.get("status") or "").upper()
    if status in {"SUCCESS", "COMPLETED"}:
        out["status"] = "READY"
    elif status == "FAILED":
        out["status"] = "FAILED"
    return out


def _is_success(result: dict[str, Any]) -> bool:
    return str(result.get("status") or "").upper() in {"SUCCESS", "READY", "COMPLETED"}


def _retire_previous_version(system: Any, previous: dict[str, Any], new_document_id: str) -> None:
    previous_document_id = str(previous.get("document_id") or "")
    previous_version = str(previous.get("content_hash") or "")
    if not previous_document_id or not previous_version:
        return

    system.vector_store.set_version_index_state(previous_document_id, previous_version, "FAILED")
    system.vector_store.delete_version(previous_document_id, previous_version)
    system.state_store.delete_pages(previous_document_id)
    try:
        system.state_store.update_document(
            previous_document_id,
            status="SUPERSEDED",
            current_stage="SUPERSEDED",
            index_state="FAILED",
            error=f"Superseded by document version {new_document_id}.",
        )
    finally:
        try:
            system.state_store.record_event(
                previous_document_id,
                stage="SUPERSEDED",
                status="SUPERSEDED",
                event_type="version_retired",
                message=f"Previous document version retired in favor of {new_document_id}.",
                details={"superseded_by": new_document_id, "previous_version": previous_version},
            )
        except Exception:
            pass


def ingest_version_safely(system: Any, pdf_path: str | Path) -> dict[str, Any]:
    """Publish changed content only as READY after durable validation."""
    source = Path(pdf_path)
    resolved = source.resolve()
    previous = system.state_store.get_by_path(str(resolved))
    if not previous or not source.is_file():
        return _public_result(robust_ingestor.robust_ingest_file(system, source))

    content_hash = system._hash_file(source)
    previous_hash = str(previous.get("content_hash") or "")
    if not previous_hash or previous_hash == content_hash:
        return _public_result(robust_ingestor.robust_ingest_file(system, source))

    incoming_dir = Path(system.settings.incoming_dir)
    incoming_dir.mkdir(parents=True, exist_ok=True)
    staging_name = f".reindex-{content_hash[:16]}-{uuid.uuid4().hex[:8]}-{source.name}"
    staging = incoming_dir / staging_name
    shutil.copy2(source, staging)

    result = robust_ingestor.robust_ingest_file(system, staging)
    if not _is_success(result):
        result = dict(result)
        result["versioned_replacement"] = True
        result["previous_document_id"] = previous.get("document_id")
        result["previous_version_preserved"] = True
        return _public_result(result)

    new_document_id = str(result.get("document_id") or "")
    new_record = system.state_store.get_document(new_document_id) if new_document_id else None
    if not new_document_id or not new_record or str(new_record.get("status") or "").upper() != "READY":
        raise RuntimeError("Versioned ingestion reported success without a durable READY document.")

    generated_path = Path(str(new_record.get("file_path") or ""))
    desired_path = Path(system.settings.processed_dir) / source.name
    archived_old_path: Path | None = None
    new_published_path = False
    old_retirement_started = False
    try:
        if generated_path.resolve() != desired_path.resolve():
            if desired_path.exists():
                archived_old_path = _unique_archive_path(system.settings.archive_dir, desired_path, "superseded")
                desired_path.replace(archived_old_path)
            generated_path.replace(desired_path)
            new_published_path = True

        system.state_store.update_document(
            new_document_id,
            file_path=str(desired_path.resolve()),
            file_name=source.name,
        )
        old_retirement_started = True
        _retire_previous_version(system, previous, new_document_id)
    except Exception as exc:
        try:
            system.vector_store.set_version_index_state(
                new_document_id,
                str(new_record.get("content_hash") or content_hash),
                "FAILED",
            )
            system.vector_store.delete_version(
                new_document_id,
                str(new_record.get("content_hash") or content_hash),
            )
            system.state_store.delete_pages(new_document_id)
            system.state_store.update_document(
                new_document_id,
                status="FAILED_INDEXING",
                current_stage="FAILED_INDEXING",
                index_state="FAILED",
                error=f"Version retirement failed: {type(exc).__name__}",
            )
        except Exception:
            pass

        if new_published_path and desired_path.exists():
            try:
                failed_archive = _unique_archive_path(system.settings.failed_dir, desired_path, content_hash[:12])
                desired_path.replace(failed_archive)
            except OSError:
                pass

        if archived_old_path is not None and archived_old_path.exists() and not desired_path.exists():
            try:
                archived_old_path.replace(desired_path)
            except OSError:
                pass

        return {
            "status": "FAILED",
            "file_name": source.name,
            "document_id": new_document_id,
            "previous_document_id": previous.get("document_id"),
            "versioned_replacement": True,
            "previous_version_preserved": not old_retirement_started or str((system.state_store.get_document(str(previous.get("document_id") or "")) or {}).get("status") or "").upper() == "READY",
            "error": f"Versioned replacement could not be published safely: {type(exc).__name__}",
        }

    out = dict(result)
    out.update(
        {
            "status": "READY",
            "versioned_replacement": True,
            "previous_document_id": previous.get("document_id"),
            "previous_version_retired": True,
            "document_id": new_document_id,
            "file_name": source.name,
        }
    )
    return out
