from rag_project.runtime_functionality_deep_fix import _wrap_contradiction_detection


def test_contradiction_guard_keeps_real_shared_context_conflict():
    wrapped = _wrap_contradiction_detection(lambda claim, blocks: True)
    assert wrapped(
        "Metformin is not recommended in pregnancy.",
        ["Metformin is recommended during pregnancy."],
    ) is True


def test_contradiction_guard_drops_unrelated_negation_overlap():
    wrapped = _wrap_contradiction_detection(lambda claim, blocks: True)
    assert wrapped(
        "Metformin is not recommended in pregnancy.",
        ["Metformin is recommended for type 2 diabetes."],
    ) is False
