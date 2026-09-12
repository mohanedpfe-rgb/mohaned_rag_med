from __future__ import annotations


def test_runtime_contract_names_one_canonical_answer_authority_and_entity_paths() -> None:
    from rag_project.application import ANSWER_PIPELINE_AUTHORITY, ACTIVE_ANSWER_PIPELINE_AUTHORITY, runtime_contract
    from rag_project.canonical_runtime import ANSWER_AUTHORITY

    contract = runtime_contract()
    assert ANSWER_PIPELINE_AUTHORITY == ACTIVE_ANSWER_PIPELINE_AUTHORITY
    assert ANSWER_PIPELINE_AUTHORITY == ANSWER_AUTHORITY
    assert contract["answer_pipeline_authority"] == ANSWER_PIPELINE_AUTHORITY
    assert contract["answer_pipeline_execution"] == ANSWER_PIPELINE_AUTHORITY
    assert contract["active_answer_pipeline_authority"] == ANSWER_PIPELINE_AUTHORITY
    assert contract["final_answer_verification"].endswith("final_answer_contract.verify_final_answer")
    assert contract["entity_coverage"].endswith("entity_coverage.score_entity_coverage")
    assert contract["answer_monkey_patch"] is False


def test_final_pipeline_marks_canonical_execution_and_phase5_visibility() -> None:
    from rag_project.intelligence.god_mode_100 import _runtime_phase_implementation

    result = {
        "phases": {
            "phase_1_query_understanding": "complete",
            "phase_2_retrieval_precision": "complete",
            "phase_3_two_stage_generation": "complete",
            "phase_4_verification": "complete",
            "phase_5_intelligence_visibility": "complete",
        },
        "phase_plan": {"planner_source": "deterministic", "planner_confidence": 0.9, "entities": ["diabetes"]},
        "adaptive_retrieval": {"stage": 1, "queries": 1, "final_hits": 1, "escalated": False},
        "two_stage_synthesis": {"used": True, "attempted": True, "required": True, "fallback": False, "verification": {"checked": True, "blocked": False}},
        "confidence_calibration": {"calibrated": 0.9},
    }
    implementation = _runtime_phase_implementation(result, [], {"checked": True, "blocked_claims": 0})
    assert implementation["phase_4_verification"]["final_answer_checked"] is True
    assert implementation["phase_5_intelligence_visibility"]["signals_present"] is False
