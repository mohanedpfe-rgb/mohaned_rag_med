from pathlib import Path

from rag_project.intelligence.production_ops import ABTestManager, CircuitBreaker, MetricsService, OperationsStore, RetryPolicy
from rag_project.quality.human_test_readiness import HumanTestReadinessGate


def test_ab_assignment_is_deterministic_and_balanced_shape(tmp_path: Path):
    store = OperationsStore(tmp_path / "ops.sqlite3")
    manager = ABTestManager(store)
    values = [manager.assignment("exp", f"q-{i}") for i in range(200)]
    assert set(values) <= {"A", "B"}
    assert 60 < values.count("A") < 140
    assert manager.assignment("exp", "q-7") == manager.assignment("exp", "q-7")


def test_circuit_breaker_opens_after_threshold():
    breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=60)
    assert breaker.allow()
    breaker.record_failure()
    assert breaker.allow()
    breaker.record_failure()
    assert not breaker.allow()


def test_retry_policy_is_exponential_and_bounded():
    policy = RetryPolicy(retries=4, base_delay_seconds=0.1, max_delay_seconds=0.25)
    assert list(policy.delays()) == [0.1, 0.2, 0.25, 0.25]


def test_metrics_endpoint_data_model(tmp_path: Path):
    store = OperationsStore(tmp_path / "ops.sqlite3")
    store.record_result("q1", "What is hypertension?", {
        "route": {"intent": "factual", "complexity": 0.1},
        "generation_path": "PATH_A_EXTRACTIVE",
        "status": "SUCCESS",
        "answer": "Hypertension is high blood pressure.",
        "confidence": {"evidence_confidence": 0.95},
        "verification": {"supported_ratio": 0.95},
    }, 120.0)
    store.add_feedback("q1", "helpful")
    snap = MetricsService(store).snapshot()
    assert snap["latency_percentiles"]["p50"] == 120.0
    assert snap["path_usage"]["PATH_A_EXTRACTIVE"] == 1
    assert snap["feedback"]["helpful"] == 1


def test_human_test_gate_is_fail_closed_without_real_kb(tmp_path: Path):
    report = HumanTestReadinessGate(tmp_path).evaluate()
    assert report.ready is False
    assert any(not check.passed for check in report.checks)
