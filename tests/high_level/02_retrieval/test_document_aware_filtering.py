from __future__ import annotations

import pytest

from tests.high_level.conftest import write_minimal_pdf
from tests.high_level.helpers import assert_status


@pytest.mark.high_level
def test_retrieval__document_aware_answer_uses_requested_document_evidence(clean_system, tmp_path):
    diabetes = write_minimal_pdf(tmp_path / "diabetes_source.pdf", ["DOC_A_UNIQUE: diabetes mellitus is a chronic metabolic disorder."])
    anemia = write_minimal_pdf(tmp_path / "anemia_source.pdf", ["DOC_B_UNIQUE: anemia is a reduction in red blood cell mass."])
    a = clean_system.ingest_file(diabetes)
    b = clean_system.ingest_file(anemia)
    assert_status(a, {"READY"})
    assert_status(b, {"READY"})

    result = clean_system.answer("What does DOC_A_UNIQUE say about diabetes mellitus?")
    assert str(result.get("status") or "").upper() in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}
    hits = result.get("hits") or []
    assert hits
    texts = " ".join(str(getattr(hit, "text", "")) for hit in hits).casefold()
    assert "doc_a_unique" in texts
    assert "doc_b_unique" not in texts
