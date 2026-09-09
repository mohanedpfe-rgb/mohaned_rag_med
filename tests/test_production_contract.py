from rag_project.intelligence.production_contract import (
    FEATURES,
    redact_sensitive_text,
    production_readiness,
    validate_feature_contract,
)


def test_exactly_44_executable_features_resolve() -> None:
    report = validate_feature_contract()
    assert len(FEATURES) == 44
    assert report["feature_count"] == 44
    assert report["unique_names"] is True
    assert report["duplicates"] == []
    assert report["all_resolved"] is True, report


def test_diagnostic_redaction_is_deterministic() -> None:
    value = redact_sensitive_text("Call +213 555 123 456 or email doctor@example.com, patient 123456789.")
    assert "doctor@example.com" not in value
    assert "123456789" not in value
    assert "[REDACTED_EMAIL]" in value
    assert "[REDACTED_ID]" in value


def test_release_gate_requires_every_independent_gate() -> None:
    assert production_readiness({"feature_contract": True, "tests_green": True, "index_ready": True, "privacy_controls": True, "medical_safety": True})["release_ready"] is True
    assert production_readiness({"feature_contract": True, "tests_green": False, "index_ready": True, "privacy_controls": True, "medical_safety": True})["release_ready"] is False
