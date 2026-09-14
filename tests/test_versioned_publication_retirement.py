from __future__ import annotations

from rag_project.ingestion.versioned_ingestor import _retire_previous_version


class _State:
    def __init__(self):
        self.updated = []
        self.events = []

    def delete_pages(self, document_id):
        self.updated.append(("delete_pages", document_id))

    def update_document(self, document_id, **values):
        self.updated.append(("update_document", document_id, values))

    def record_event(self, document_id, **values):
        self.events.append((document_id, values))


class _Vector:
    def set_version_index_state(self, document_id, version_id, state):
        raise RuntimeError("simulated old-index state update failure")

    def delete_version(self, document_id, version_id):
        raise RuntimeError("simulated old-index delete failure")


class _System:
    def __init__(self):
        self.state_store = _State()
        self.vector_store = _Vector()


def test_old_version_retirement_failure_does_not_raise_or_invalidate_new_version():
    system = _System()
    warnings = _retire_previous_version(
        system,
        {"document_id": "old-doc", "content_hash": "old-hash"},
        "new-doc",
    )

    assert warnings
    assert any("set_old_version_failed" in item for item in warnings)
    assert any("delete_old_version" in item for item in warnings)
    assert any(
        entry[0] == "update_document" and entry[1] == "old-doc"
        for entry in system.state_store.updated
    )
