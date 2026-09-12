from __future__ import annotations

import pytest

from tests.high_level.helpers import (
    assert_citations_valid,
    assert_exact_path,
    assert_exact_status,
    assert_grounded,
    assert_latency_under,
    assert_pipeline_authority,
)


@pytest.mark.high_level
def test_e2e_numeric_table__ingest_then_answer_exact_number_and_table_path(clean_system, ready_document):
    # The clean_system fixture already indexed the controlled document; this flow
    # exercises two user-visible evidence requests against that READY corpus.
    before = list(getattr(clean_system.conversation_memory, "history", []) or [])

    dose = clean_system.answer("What dose of metformin is stated in the indexed evidence?")
    assert_exact_status(dose, "SUCCESS")
    assert_exact_path(dose, "PATH_B_TEMPLATE")
    assert "500 mg" in str(dose.get("answer") or "")
    assert "twice daily" in str(dose.get("answer") or "").casefold()
    assert_citations_valid(dose)
    assert_grounded(dose)
    assert_pipeline_authority(dose)
    assert_latency_under(dose, 7.0)

    table = clean_system.answer("Which table contains the HbA1c target?")
    assert_exact_status(table, "SUCCESS")
    assert_exact_path(table, "PATH_B_TEMPLATE")
    table_answer = str(table.get("answer") or "")
    assert "Table 1" in table_answer
    assert "HbA1c" in table_answer
    assert_citations_valid(table)
    assert_grounded(table)
    assert_pipeline_authority(table)
    assert_latency_under(table, 7.0)

    after = list(getattr(clean_system.conversation_memory, "history", []) or [])
    assert len(after) == len(before) + 2
