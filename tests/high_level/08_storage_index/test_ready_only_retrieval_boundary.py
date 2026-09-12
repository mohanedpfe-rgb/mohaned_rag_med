from __future__ import annotations

import pytest

from tests.high_level.conftest import write_minimal_pdf
from tests.high_level.helpers import assert_citations_valid, assert_exact_status, assert_one_of_statuses


@pytest.mark.high_level
@pytest.mark.multi_outcome
def test_storage__building_vector_records_are_not_searchable_until_marked_ready(clean_system, tmp_path):
    pdf = write_minimal_pdf(
        tmp_path / "building_visibility.pdf",
        ["BUILDING_VISIBILITY_MARKER_91 is only visible before publication."],
    )
    ingestion = clean_system.ingest_file(pdf)
    assert_exact_status(ingestion, "READY")
    document_id = str(ingestion.get("document_id") or "")
    assert document_id

    clean_system.vector_store.set_document_index_state(document_id, "BUILDING")
    try:
        result = clean_system.answer("What is BUILDING_VISIBILITY_MARKER_91?")
        assert_one_of_statuses(result, {"NOT_SUPPORTED", "GENERATION_ABSTAIN", "ANSWER_UNAVAILABLE"})
        assert not (result.get("hits") or []), result
        assert result.get("citations") == []
    finally:
        clean_system.vector_store.set_document_index_state(document_id, "READY")

    published = clean_system.answer("What is BUILDING_VISIBILITY_MARKER_91?")
    assert_one_of_statuses(published, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    assert published.get("hits")
    assert_citations_valid(published)
