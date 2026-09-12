from rag_project.testing.architecture_contracts import analyze


def test_architecture_semantic_contract_is_fail_closed():
    report = analyze()
    assert report["parse_failures"] == []
    assert report["dependency_cycles"] == []
    assert report["forbidden_edges"] == []
    assert report["production_test_harness_imports"] == []
    assert report["ownership_failures"] == []
    assert report["contract_pass"] is True
