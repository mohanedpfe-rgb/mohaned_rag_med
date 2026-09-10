from __future__ import annotations

import re
from typing import Any, Sequence

from rag_project.intelligence.evidence_guard import ClaimCheck, verify_claims

_REPAIR_SYSTEM = (
    "You are the final answer repair component of a medical document RAG system. "
    "You do not have authority to add facts. Rewrite the draft using ONLY the supplied evidence. "
    "Remove unsupported, contradicted, or numerically inconsistent claims. Preserve supported claims, qualifiers, "
    "negation, uncertainty, population, phase, units, and citations such as [S1]. "
    "Do not mention this repair process. Return only the repaired answer."
)


def _compact(text: str, limit: int = 7000) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:limit]


def repair_needed(checks: Sequence[ClaimCheck]) -> bool:
    return any(c.status in {"UNSUPPORTED", "NUMERIC_MISMATCH", "CONTRADICTED"} or c.contradiction for c in checks)


def build_repair_prompt(question: str, draft: str, evidence_blocks: Sequence[str], checks: Sequence[ClaimCheck]) -> str:
    blocked = [c for c in checks if c.status in {"UNSUPPORTED", "NUMERIC_MISMATCH", "CONTRADICTED"} or c.contradiction]
    evidence = "\n\n".join(f"[S{i + 1}] {_compact(block, 1200)}" for i, block in enumerate(evidence_blocks[:12]) if str(block).strip())
    failures = "\n".join(f"- {c.status}: {c.claim} ({c.reason})" for c in blocked[:12])
    return (
        f"Question: {_compact(question, 1800)}\n\n"
        f"Draft answer:\n{_compact(draft, 5000)}\n\n"
        f"Claims that failed verification:\n{failures}\n\n"
        f"Verified evidence:\n{evidence}\n\n"
        "Rewrite conservatively. A claim must be omitted when the evidence does not support it. "
        "Do not infer a missing causal, treatment, dosage, or safety link."
    )


def repair_answer(llm: Any, question: str, draft: str, evidence_blocks: Sequence[str], checks: Sequence[ClaimCheck]) -> str | None:
    if llm is None or not draft or not repair_needed(checks):
        return None
    prompt = build_repair_prompt(question, draft, evidence_blocks, checks)
    try:
        repaired = llm.generate(prompt=prompt, system_prompt=_REPAIR_SYSTEM, temperature=0.0)
    except Exception:
        return None
    value = str(repaired or "").strip()
    return value[:9000] if value else None


def repair_and_verify(llm: Any, question: str, draft: str, evidence_blocks: Sequence[str], source_ids: Sequence[str], checks: Sequence[ClaimCheck]) -> tuple[str, list[ClaimCheck], bool]:
    repaired = repair_answer(llm, question, draft, evidence_blocks, checks)
    if not repaired or repaired == draft:
        return draft, list(checks), False
    fresh = verify_claims(repaired, evidence_blocks, source_ids)
    return repaired, fresh, True
