from __future__ import annotations

from types import SimpleNamespace

from rag_project.runtime_functionality_comparison_template_fix import _wrap_comparison_template


def test_comparison_template_does_not_falsely_assign_claims_to_items():
    original = lambda self, compiled, route: "unused"
    compiled = {
        "claims": [
            SimpleNamespace(text="Metformin is commonly used.", source_numbers=(1,)),
            SimpleNamespace(text="Metformin may cause gastrointestinal effects.", source_numbers=(2,)),
        ]
    }
    route = SimpleNamespace(template_type="comparison")
    result = _wrap_comparison_template(original)(SimpleNamespace(), compiled, route)
    assert "first item" not in result.lower()
    assert "second/comparison item" not in result.lower()
    assert "Comparison evidence:" in result
    assert "[S1]" in result and "[S2]" in result
