from pathlib import Path
from rag_project.intelligence.cloud_hybrid import (
    CloudConfig, ConsentStore, CostTracker, HybridRouter, PIIAnonymizer,
    has_permission,
)


def test_pii_redaction_removes_common_identifiers():
    text = "Email test@example.com phone +213 555 123 456 ID 1234567890123"
    redacted = PIIAnonymizer.redact(text)
    assert "test@example.com" not in redacted
    assert "1234567890123" not in redacted


def test_hybrid_router_is_local_first_and_consent_gated(tmp_path: Path):
    consent = ConsentStore(tmp_path / "consent.json")
    cfg = CloudConfig(enabled=True, require_consent=True, cloud_threshold=0.70)
    router = HybridRouter(cfg, consent=consent)
    assert router.should_escalate(confidence=0.20, subject="u")[0] is False
    consent.set("u", True)
    assert router.should_escalate(confidence=0.20, subject="u")[0] is True
    assert router.should_escalate(confidence=0.95, emergency=True, subject="u")[0] is True


def test_cost_tracker_enforces_daily_budget():
    tracker = CostTracker(0.01)
    assert tracker.reserve(0.006)
    assert tracker.reserve(0.004)
    assert not tracker.reserve(0.001)


def test_enterprise_roles_are_scoped():
    assert has_permission("admin", "adjust_settings")
    assert has_permission("clinician", "verify_answers")
    assert not has_permission("researcher", "adjust_settings")
