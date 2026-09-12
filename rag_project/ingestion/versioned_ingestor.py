from __future__ import annotations

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


def _is_success(result: dict[str, Any]) -> bool:
    return str(result.get("status") or "").upper() in {"SUCCESS", "READY", "COMPLETED"}


def _retire_previous_version(system: Any, previous: dict[str, Any], new_document_id: str) -> None:
    previous_document_id = str(previous.get("document_id") or "")
    previous_version = str(previous.get("content_hash") or "")
    if not previous_document_id or not previous_version:
        return

    # Fail the old vector version before removing it. Retrieval paths filter on
    # index_state=READY, so even if physical deletion needs recovery, the old
    # version is no longer eligible for search.
    system.vector_store.set_version_index_state(previous_document_id, previous_version, "FAILED")
    system.vector_store.delete_version(previous_document_id, previous_version)
    system.state_store.delete_pages(previous_document_id)
    system.state_store.update_document(
        previous_document_id,
        status="SUPERSEDED",
        current_stage="SUPERSEDED",
        index_state="FAILED",
        error=f"Superseded by document version {new_document_id}.",
    )
    system.state_store.record_event(
        previous_document_id,
        stage="SUPERSEDED",
        status="SUPERSEDED",
        event_type="version_retired",
        message=f"Previous document version retired in favor of {new_document_id}.",
        details={"superseded_by": new_document_id, "previous_version": previous_version},
    )


def ingest_version_safely(system: Any, pdf_path: str | Path) -> dict[str, Any]:
    """Build changed-path content under a new identity and publish only after READY.

    The normal ingestor remains the single extraction/indexing implementation. This
    wrapper only handles replacement identity and retirement of an older READY
    document, preserving the old searchable version across failed re-indexes.
    """
    source = Path(pdf_path)
    resolved = source.resolve()
    previous = system.state_store.get_by_path(str(resolved))
    if not previous or not source.is_file():
        return robust_ingestor.robust_ingest_file(system, source)

    content_hash = system._hash_file(source)
    previous_hash = str(previous.get("content_hash") or "")
    if not previous_hash or previous_hash == content_hash:
        return robust_ingestor.robust_ingest_file(system, source)

    incoming_dir = Path(system.settings.incoming_dir)
    incoming_dir.mkdir(parents=True, exist_ok=True)
    staging_name = f".reindex-{content_hash[:16]}-{uuid.uuid4().hex[:8]}-{source.name}"
    staging = incoming_dir / staging_name
    shutil.copy2(source, staging)

    result = robust_ingestor.robust_ingest_file(system, staging)
    if not _is_success(result):
        # The staging ingest quarantines its own failed artifact. The previous
        # READY document was never touched and remains searchable.
        result = dict(result)
        result["versioned_replacement"] = True
        result["previous_document_id"] = previous.get("document_id")
        result["previous_version_preserved"] = True
        return result

    new_document_id = str(result.get("document_id") or "")
    new_record = system.state_store.get_document(new_document_id) if new_document_id else None
    if not new_document_id or not new_record or str(new_record.get("status") or "").upper() != "READY":
        raise RuntimeError("Versioned ingestion reported success without a durable READY document.")

    generated_path = Path(str(new_record.get("file_path") or ""))
    desired_path = Path(system.settings.processed_dir) / source.name
    archived_old_path: Path | None = None
    try:
        if generated_path.resolve() != desired_path.resolve():
            if desired_path.exists():
                archived_old_path = _unique_archive_path(system.settings.archive_dir, desired_path, "superseded")
                desired_path.replace(archived_old_path)
            generated_path.replace(desired_path)
        system.state_store.update_document(
            new_document_id,
            file_path=str(desired_path.resolve()),
            file_name=source.name,
        )
        _retire_previous_version(system, previous, new_document_id)
    except Exception as exc:
        # Do not expose two searchable versions. Attempt to invalidate the new
        # version if retirement of the old version fails; leave the old state intact.
        try:
            system.vector_store.set_version_index_state(new_document_id, str(new_record.get("content_hash") or content_hash), "FAILED")
            system.vector_store.delete_version(new_document_id, str(new_record.get("content_hash") or content_hash))
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
        if archived_old_path is not None and archived_old_path.exists() and not desired_path.exists():
            try:
                archived_old_path.replace(desired_path)
            except OSError:
                pass
        raise RuntimeError(f"Versioned replacement could not be published safely: {type(exc).__name__}") from exc

    out = dict(result)
    out.update(
        {
            "status": "success",
            "versioned_replacement": True,
            "previous_document_id": previous.get("document_id"),
            "previous_version_retired": True,
            "document_id": new_document_id,
            "file_name": source.name,
        }
    )
    return out
