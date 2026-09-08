from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Sequence

from rag_project.utils.text_utils import keyword_overlap_score, meaningful_tokens


@dataclass(frozen=True)
class ClaimCheck:
    claim: str
    support: float
    status: str
    sources: tuple[str, ...]
    numeric_mismatch: bool = False
    contradiction: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_NUM_RE = re.compile(r"[-+]?\d+(?:[\.,]\d+)?")
_UNIT_RE = re.compile(r"[-+]?\d+(?:[\.,]\d+)?\s*(?:mg|g|kg|mcg|µg|ug|ml|l|mmhg|%|bpm|cm|mm|mL)\b", re.I)
_NEGATION = re.compile(r"\b(no|not|without|never|none|contraindicated|avoid|cannot|does not|non|aucun|sans|jamais|ne\s+pas|لا|ليس|دون|ممنوع)\b", re.I)
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")


def split_claims(answer: str) -> list[str]:
    raw = (answer or "").strip()
    if not raw:
        return []
    sentences = [x.strip() for x in _SENTENCE_RE.split(raw) if x.strip()]
    claims = []
    for sentence in sentences:
        if len(meaningful_tokens(sentence)) >= 3:
            claims.append(sentence)
    return claims[:30]


def _numbers(text: str) -> set[str]:
    return {m.group(0).replace(",", ".") for m in _NUM_RE.finditer(text or "")}


def numeric_consistency(claim: str, evidence: str) -> dict[str, Any]:
    claim_units = {m.group(0).lower().replace(",", ".") for m in _UNIT_RE.finditer(claim or "")}
    evidence_units = {m.group(0).lower().replace(",", ".") for m in _UNIT_RE.finditer(evidence or "")}
    if not claim_units:
        return {"checked": False, "mismatch": False, "claim_values": [], "evidence_values": []}
    return {
        "checked": True,
        "mismatch": bool(claim_units - evidence_units),
        "claim_values": sorted(claim_units),
        "evidence_values": sorted(evidence_units),
    }


def detect_contradiction(claim: str, evidence_blocks: Sequence[str]) -> bool:
    claim_low = (claim or "").casefold()
    claim_neg = bool(_NEGATION.search(claim_low))
    claim_tokens = set(meaningful_tokens(claim_low))
    for evidence in evidence_blocks:
        ev_low = evidence.casefold()
        overlap = keyword_overlap_score(claim, evidence)
        if overlap < 0.45:
            continue
        ev_neg = bool(_NEGATION.search(ev_low))
        if claim_neg != ev_neg and len(claim_tokens & set(meaningful_tokens(ev_low))) >= 2:
            return True
    return False


def verify_claims(answer: str, evidence_blocks: Sequence[str], source_ids: Sequence[str]) -> list[ClaimCheck]:
    checks: list[ClaimCheck] = []
    for claim in split_claims(answer):
        supports: list[tuple[float, str]] = []
        for index, evidence in enumerate(evidence_blocks):
            score = keyword_overlap_score(claim, evidence)
            if score > 0:
                source = source_ids[index] if index < len(source_ids) else f"S{index + 1}"
                supports.append((score, source))
        supports.sort(reverse=True)
        best = supports[0][0] if supports else 0.0
        top_sources = tuple(item[1] for item in supports[:3])
        evidence_joined = "\n".join(evidence_blocks)
        num = numeric_consistency(claim, evidence_joined)
        contradiction = detect_contradiction(claim, evidence_blocks)
        if contradiction:
            status = "CONTRADICTED"
        elif num.get("mismatch"):
            status = "NUMERIC_MISMATCH"
        elif best >= 0.55:
            status = "SUPPORTED"
        elif best >= 0.30:
            status = "PARTIAL"
        else:
            status = "UNSUPPORTED"
        checks.append(ClaimCheck(claim, round(best, 4), status, top_sources, bool(num.get("mismatch")), contradiction))
    return checks


def evidence_confidence(*, retrieval: float, rerank: float, entailment: float, quality: float, contradiction: float = 0.0, ocr_penalty: float = 0.0) -> float:
    value = 0.25 * retrieval + 0.25 * rerank + 0.30 * entailment + 0.20 * quality
    value -= 0.35 * contradiction
    value -= 0.20 * ocr_penalty
    return round(max(0.0, min(1.0, value)), 4)


def citation_firewall(answer: str, claim_checks: Iterable[ClaimCheck]) -> tuple[str, bool]:
    checks = list(claim_checks)
    if not checks:
        return answer, False
    unsafe = [c for c in checks if c.status in {"UNSUPPORTED", "NUMERIC_MISMATCH", "CONTRADICTED"}]
    if not unsafe:
        return answer, False
    supported = [c for c in checks if c.status in {"SUPPORTED", "PARTIAL"}]
    lines = []
    if supported:
        lines.append("Supported evidence:")
        lines.extend(f"- {c.claim} {' '.join('[' + s + ']' for s in c.sources)}" for c in supported)
    lines.append("\nI could not safely verify every generated claim against the indexed evidence, so unsupported details were withheld.")
    return "\n".join(lines), True
