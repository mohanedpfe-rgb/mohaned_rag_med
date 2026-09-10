from __future__ import annotations

from typing import Any, Sequence

from rag_project.intelligence.evidence_entailment import build_claim_evidence_matrix
from rag_project.intelligence.evidence_guard import verify_claims

_BLOCKED = {"UNSUPPORTED", "WEAK", "NUMERIC_MISMATCH", "CONTRADICTED"}


def verify_final_answer(answer: str, hits: Sequence[Any], *, require_entailment: bool = False) -> dict[str, Any]:
    """Verify the exact answer that will be returned to the user against final evidence."""
    evidence = [str(getattr(hit, "text", "") or "") for hit in hits]
    source_ids = [f"S{i + 1}" for i in range(len(evidence))]
    checks = verify_claims(answer, evidence, source_ids) if answer.strip() and evidence else []
    matrix = build_claim_evidence_matrix([check.claim for check in checks], hits, source_ids) if checks and hits else ()
    blocked_checks = [check for check in checks if check.status in _BLOCKED or check.contradiction]
    blocked_matrix = [record for record in matrix if record.status != "ENTAILED"]
    supported = sum(check.status in {"SUPPORTED", "PARTIAL"} and not check.contradiction for check in checks)
    support_ratio = supported / max(1, len(checks))
    matrix_strong = bool(matrix) and not blocked_matrix
    allow = bool(checks) and not blocked_checks and support_ratio >= 0.60 and (matrix_strong if require_entailment else True)
    if not checks:
        reason = "no_verifiable_claims"
    elif blocked_checks:
        reason = "blocked_claims"
    elif support_ratio < 0.60:
        reason = "support_ratio_below_threshold"
    elif require_entailment and not matrix_strong:
        reason = "final_matrix_not_fully_entailed"
    else:
        reason = "verified"
    return {
        "checked": bool(checks),
        "allow": allow,
        "reason": reason,
        "claim_count": len(checks),
        "blocked_claims": len(blocked_checks),
        "supported_ratio": round(support_ratio, 4),
        "matrix_claim_count": len(matrix),
        "matrix_all_entailed": matrix_strong,
        "claim_checks": [check.to_dict() for check in checks],
        "evidence_claim_matrix": [record.to_dict() for record in matrix],
    }


__all__ = ["verify_final_answer"]
