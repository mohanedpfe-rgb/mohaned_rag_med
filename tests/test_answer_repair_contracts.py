from __future__ import annotations

from dataclasses import dataclass

from rag_project.intelligence.answer_repair import repair_and_verify, repair_needed
from rag_project.intelligence.evidence_guard import verify_claims


@dataclass
class FakeLLM:
    repaired: str

    def generate(self, **_: object) -> str:
        return self.repaired


def test_repair_needed_only_for_blocked_claims():
    checks = verify_claims("Diabetes is common.", ["Diabetes is common."], ["S1"])
    assert not repair_needed(checks)
    bad = verify_claims("Diabetes is common. It is caused by salt.", ["Diabetes is common."], ["S1"])
    assert repair_needed(bad)


def test_repair_removes_unsupported_claim_and_reverifies():
    evidence = ["Diabetes is common."]
    draft = "Diabetes is common. It is caused by salt."
    checks = verify_claims(draft, evidence, ["S1"])
    llm = FakeLLM("Diabetes is common. [S1]")
    repaired, fresh, used = repair_and_verify(llm, "What is diabetes?", draft, evidence, ["S1"], checks)
    assert used is True
    assert "salt" not in repaired.casefold()
    assert all(c.status in {"SUPPORTED", "PARTIAL"} for c in fresh)


def test_repair_cannot_bypass_verifier():
    evidence = ["Diabetes is common."]
    draft = "Diabetes is common. It is caused by salt."
    checks = verify_claims(draft, evidence, ["S1"])
    llm = FakeLLM("Diabetes is common. Insulin cures every cancer.")
    repaired, fresh, used = repair_and_verify(llm, "What is diabetes?", draft, evidence, ["S1"], checks)
    assert used is True
    assert any(c.status == "UNSUPPORTED" for c in fresh)
    assert "cancer" in repaired.casefold()


def test_repair_failure_preserves_deterministic_guard():
    evidence = ["Diabetes is common."]
    draft = "Diabetes is common. It is caused by salt."
    checks = verify_claims(draft, evidence, ["S1"])
    repaired, fresh, used = repair_and_verify(None, "What is diabetes?", draft, evidence, ["S1"], checks)
    assert repaired == draft
    assert fresh == checks
    assert used is False
