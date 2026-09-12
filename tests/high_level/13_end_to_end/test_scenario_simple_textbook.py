from __future__ import annotations

import pytest

from tests.high_level.conftest import write_minimal_pdf
from tests.high_level.helpers import assert_citations_valid, assert_grounded, assert_status


@pytest.mark.high_level
def test_e2e_textbook__ingest_retrieve_answer_and_ground_in_one_user_flow(clean_system, tmp_path):
    textbook = write_minimal_pdf(tmp_path / "textbook.pdf", [
        "Diabetes mellitus is a chronic metabolic disorder characterized by hyperglycemia.",
        "HbA1c is used to assess glycemic control. Metformin is commonly used for type 2 diabetes.",
    ])
    ingestion = clean_system.ingest_file(textbook)
    assert_status(ingestion, {"READY"})

    result = clean_system.answer("What is diabetes mellitus and how is HbA1c used to assess glycemic control?")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    assert result.get("hits")
    assert_citations_valid(result)
    assert_grounded(result)
    assert result.get("evidence_first") is True
    assert result.get("canonical_pipeline_executed") is True
