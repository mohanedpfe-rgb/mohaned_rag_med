from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PublicationState(StrEnum):
    BUILDING = "BUILDING"
    VALIDATED = "VALIDATED"
    PUBLISHED = "PUBLISHED"
    RETIRED = "RETIRED"
    FAILED = "FAILED"


@dataclass
class PublicationTransaction:
    """Durable logical coordinator for a version publication.

    Filesystem, SQLite and Chroma cannot share one physical transaction. This
    object therefore owns the logical state machine, records every transition,
    keeps the previous READY version untouched until validation/publication has
    succeeded, and centralizes compensation for failures.
    """

    system: Any
    document_id: str
    version_id: str
    previous_document_id: str = ""
    state: PublicationState = PublicationState.BUILDING
    warnings: list[str] = field(default_factory=list)

    def transition(self, state: PublicationState, *, message: str, details: dict[str, Any] | None = None) -> None:
        allowed = {
            PublicationState.BUILDING: {PublicationState.VALIDATED, PublicationState.FAILED},
            PublicationState.VALIDATED: {PublicationState.PUBLISHED, PublicationState.FAILED},
            PublicationState.PUBLISHED: {PublicationState.RETIRED},
            PublicationState.RETIRED: set(),
            PublicationState.FAILED: set(),
        }
        if state not in allowed[self.state]:
            raise RuntimeError(f"Invalid publication transition {self.state} -> {state}")
        self.state = state
        try:
            self.system.state_store.record_event(
                self.document_id,
                stage=state.value,
                status=state.value,
                event_type="publication_transition",
                message=message,
                details={"version_id": self.version_id, **(details or {})},
            )
        except Exception as exc:
            self.warnings.append(f"record_transition:{state.value}:{type(exc).__name__}")

    def validated(self, *, details: dict[str, Any] | None = None) -> None:
        self.transition(PublicationState.VALIDATED, message="New version passed durable validation.", details=details)

    def published(self, *, file_path: str) -> None:
        self.transition(PublicationState.PUBLISHED, message="New version is published and READY before retiring the previous version.", details={"file_path": file_path})

    def retired(self, *, details: dict[str, Any] | None = None) -> None:
        self.transition(PublicationState.RETIRED, message="Previous READY version retirement completed.", details=details)

    def fail(self, error: Exception | str) -> None:
        if self.state is PublicationState.FAILED:
            return
        self.state = PublicationState.FAILED
        message = str(error)
        try:
            self.system.state_store.record_event(
                self.document_id,
                stage=PublicationState.FAILED.value,
                status=PublicationState.FAILED.value,
                event_type="publication_failed",
                message=message,
                details={"version_id": self.version_id, "previous_document_id": self.previous_document_id},
            )
        except Exception as exc:
            self.warnings.append(f"record_failure:{type(exc).__name__}")

    def compensate_new_version(self) -> None:
        """Remove only the new version; never delete the previous READY version."""
        try:
            self.system.vector_store.set_version_index_state(self.document_id, self.version_id, "FAILED")
        except Exception as exc:
            self.warnings.append(f"new_version_state:{type(exc).__name__}")
        try:
            self.system.vector_store.delete_version(self.document_id, self.version_id)
        except Exception as exc:
            self.warnings.append(f"new_version_delete:{type(exc).__name__}")
        try:
            self.system.state_store.delete_pages(self.document_id)
        except Exception as exc:
            self.warnings.append(f"new_version_pages:{type(exc).__name__}")
        try:
            self.system.state_store.mark_publication_failed(
                self.document_id,
                error="Publication transaction failed before completion.",
            )
        except Exception as exc:
            self.warnings.append(f"new_version_status:{type(exc).__name__}")


__all__ = ["PublicationState", "PublicationTransaction"]
