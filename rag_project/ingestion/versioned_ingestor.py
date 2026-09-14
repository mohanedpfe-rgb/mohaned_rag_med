from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from rag_project.ingestion import robust_ingestor
from rag_project.ingestion.publication_coordinator import PublicationTransaction
from rag_project.ingestion.status_contract import normalize_public_status, public_result


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
    return public_result(result)


def _is_success(result: dict[str, Any]) -> bool:
    return normalize_public_status(result.get("status")) == "READY"


def _retire_previous_version(system: Any, previous: dict[str, Any], new_document_id: str) -> list[str]:
    previous_document_id = str(previous.get("document_id") or "")
    identities: list[str] = []
    for candidate in (previous.get("version_id"), previous.get("content_hash")):
        value = str(candidate or "")
        if value and value not in identities:
            identities.append(value)
    if not previous_document_id or not identities:
        return []
    errors: list[str] = []
    for identity in identities:
        try:
            system.vector_store.set_version_index_state(previous_document_id, identity, "FAILED")
        except Exception as exc:
            errors.append(f"set_old_version_failed:{identity[:12]}:{type(exc).__name__}")
        try:
            system.vector_store.delete_version(previous_document_id, identity)
        except Exception as exc:
            errors.append(f"delete_old_version:{identity[:12]}:{type(exc).__name__}")
    try:
        system.state_store.delete_pages(previous_document_id)
    except Exception as exc:
        errors.append(f"delete_old_pages:{type(exc).__name__}")
    try:
        system.state_store.update_document(
            previous_document_id,
            status="SUPERSEDED",
            current_stage="SUPERSEDED",
            index_state="FAILED",
            error=f"Superseded by document version {new_document_id}." if not errors else (
                f"Superseded by document version {new_document_id}; retirement warnings: " + ", ".join(errors)
            ),
        )
    except Exception as exc:
        errors.append(f"mark_old_superseded:{type(exc).__name__}")
    try:
        system.state_store.record_event(
            previous_document_id,
            stage="RETIRED",
            status="SUPERSEDED",
            event_type="version_retired",
            message=f"Previous document version retired in favor of {new_document_id}.",
            details={"superseded_by": new_document_id, "previous_versions": identities, "retirement_warnings": errors},
        )
    except Exception as exc:
        errors.append(f"record_retirement_event:{type(exc).__name__}")
    return errors


def ingest_version_safely(system: Any, pdf_path: str | Path) -> dict[str, Any]:
    """Run version replacement through one explicit logical publication state machine."""
    source = Path(pdf_path)
    resolved = source.resolve()
    previous = system.state_store.get_by_path(str(resolved))
    if not previous:
        try:
            candidates = [
                row for row in system.state_store.get_all_documents()
                if str(row.get("file_name") or "") == source.name
                and system.state_store.is_ready_status(row.get("status"))
            ]
            if candidates:
                candidates.sort(key=lambda row: str(row.get("modified_at") or ""), reverse=True)
                previous = candidates[0]
        except Exception:
            previous = None

    if not previous or not source.is_file():
        return _public_result(robust_ingestor.robust_ingest_file(system, source))

    content_hash = system._hash_file(source)
    if str(previous.get("content_hash") or "") == content_hash:
        return _public_result(robust_ingestor.robust_ingest_file(system, source))

    incoming_dir = Path(system.settings.incoming_dir)
    incoming_dir.mkdir(parents=True, exist_ok=True)
    staging = incoming_dir / f".reindex-{content_hash[:16]}-{uuid.uuid4().hex[:8]}-{source.name}"
    shutil.copy2(source, staging)

    result = robust_ingestor.robust_ingest_file(system, staging)
    if not _is_success(result):
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            pass
        failed = dict(result)
        failed.update({"versioned_replacement": True, "previous_document_id": previous.get("document_id"), "previous_version_preserved": True})
        return _public_result(failed)

    new_document_id = str(result.get("document_id") or "")
    new_record = system.state_store.get_document(new_document_id) if new_document_id else None
    new_version_id = str((new_record or {}).get("version_id") or (new_record or {}).get("content_hash") or content_hash)
    if not new_document_id or not new_record or str(new_record.get("status") or "").upper() != "READY":
        raise RuntimeError("Versioned ingestion reported success without a durable READY document.")

    transaction = PublicationTransaction(
        system=system,
        document_id=new_document_id,
        version_id=new_version_id,
        previous_document_id=str(previous.get("document_id") or ""),
    )
    generated_path = Path(str(new_record.get("file_path") or ""))
    desired_path = Path(system.settings.processed_dir) / source.name
    archived_old_path: Path | None = None
    new_published_path = False

    try:
        transaction.validated(details={"content_hash": content_hash, "previous_document_id": previous.get("document_id")})
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
            status="READY",
            current_stage="PUBLISHED",
            index_state="READY",
        )
        transaction.published(file_path=str(desired_path.resolve()))
        retirement_warnings = _retire_previous_version(system, previous, new_document_id)
        transaction.warnings.extend(retirement_warnings)
        transaction.retired(details={"retirement_warnings": retirement_warnings})
    except Exception as exc:
        transaction.fail(exc)
        transaction.compensate_new_version()
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
            "previous_version_preserved": str((system.state_store.get_document(str(previous.get("document_id") or "")) or {}).get("status") or "").upper() == "READY",
            "publication_state": transaction.state.value,
            "publication_warnings": transaction.warnings,
            "error": f"Versioned replacement could not be published safely: {type(exc).__name__}: {exc}",
        }

    out = dict(result)
    out.update({
        "status": "READY",
        "versioned_replacement": True,
        "previous_document_id": previous.get("document_id"),
        "previous_version_retired": not transaction.warnings,
        "previous_version_retirement_warnings": list(transaction.warnings),
        "publication_state": transaction.state.value,
        "document_id": new_document_id,
        "file_name": source.name,
    })
    if transaction.warnings:
        out["warning"] = "New version is READY; previous-version retirement completed with recoverable warnings."
    return out
