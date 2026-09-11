"""Fast sentinels for the most failure-prone cross-module contracts."""

from __future__ import annotations

import inspect

import pytest

from rag_project.intelligence import evidence_guard, god_mode_100, pipeline_integrity, top_level_pipeline


pytestmark = [pytest.mark.fast, pytest.mark.contract, pytest.mark.diagnostic]


def _provenance(obj) -> str:
    try:
        source = inspect.getsourcefile(obj) or "<unknown>"
        line = inspect.getsourcelines(obj)[1]
    except (OSError, TypeError):
        source, line = "<unknown>", "?"
    flags = [name for name in vars(obj) if name.startswith("_runtime_")] if hasattr(obj, "__dict__") else []
    return f"{obj.__module__}.{getattr(obj, '__qualname__', obj)!s} @ {source}:{line} flags={flags}"


@pytest.mark.parametrize(
    "label,obj",
    [
        ("top_level.rewrite_follow_up", top_level_pipeline.rewrite_follow_up),
        ("pipeline_integrity.safe_rewrite_follow_up", pipeline_integrity.safe_rewrite_follow_up),
    ],
)
def test_followup_public_contract_is_clean(label, obj):
    result = obj("What about this?", [("What is diabetes?", "Diabetes is a metabolic disease.")])
    assert "What is diabetes?" in result, f"{label} lost conversational anchor; {_provenance(obj)}"
    assert "Follow-up:" not in result, f"legacy protocol leaked through {label}; {_provenance(obj)}"


def test_numeric_public_contract_is_structured():
    result = evidence_guard.numeric_consistency("Dose is 600 mg.", "The recommended dose is 500 mg.")
    assert isinstance(result, dict), f"numeric_consistency returned {type(result).__name__}; {_provenance(evidence_guard.numeric_consistency)}"
    assert result["checked"] is True
    assert result["mismatch"] is True


def test_god_mode_enhancer_keeps_four_argument_contract():
    parameters = list(inspect.signature(god_mode_100.enhance_result).parameters.values())
    assert len(parameters) >= 4, f"enhance_result signature drifted to {inspect.signature(god_mode_100.enhance_result)}; {_provenance(god_mode_100.enhance_result)}"


def test_followup_contracts_are_semantically_compatible():
    question = "What about this?"
    history = [("What is diabetes?", "Diabetes is a metabolic disease.")]
    top = top_level_pipeline.rewrite_follow_up(question, history)
    safe = pipeline_integrity.safe_rewrite_follow_up(question, history)
    assert "What is diabetes?" in top
    assert "What is diabetes?" in safe
    assert "Follow-up:" not in top
    assert "Follow-up:" not in safe


def test_legacy_adapter_is_not_inside_authoritative_modules():
    for obj in (top_level_pipeline.rewrite_follow_up, pipeline_integrity.safe_rewrite_follow_up):
        provenance = _provenance(obj)
        assert "runtime_final_contracts_v6.py" not in provenance
        assert "runtime_final_contracts_v7.py" not in provenance
        assert "runtime_final_contracts_v8.py" not in provenance
