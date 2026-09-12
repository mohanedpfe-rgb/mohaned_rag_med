from rag_project.testing.implementation_contracts import validate_runtime_ownership


def test_all_17_phase_implementations_have_authoritative_runtime_ownership():
    report = validate_runtime_ownership()
    assert report["phase_count"] == 17
    assert report["phase_numbers"] == list(range(1, 18))
    assert report["dispatch_numbers"] == list(range(1, 18))
    assert report["hardened_runtime_bindings_verified"] is True
    assert report["pass"] is True, report["failures"]
