from __future__ import annotations

import math
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
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")
_NUM_RE = re.compile(r"[-+]?\d+(?:[\.,]\d+)?")
_VALUE_UNIT_RE = re.compile(
    r"(?P<value>[-+]?\d+(?:[\.,]\d+)?(?:\s*[-–]\s*\d+(?:[\.,]\d+)?)?)\s*"
    r"(?P<unit>mg|g|kg|mcg|µg|ug|ml|l|mmhg|cmh2o|%|bpm|°c|c|mm|cm|m|hz|khz|m/s|h|min|s|day|days|week|weeks|month|months|year|years)\b",
    re.I,
)
_NEGATION = re.compile(
    r"\b(no|not|without|never|none|contraindicated|avoid|cannot|does not|doesn't|non|aucun|sans|jamais|ne\s+pas|لا|ليس|دون|ممنوع|منع)\b",
    re.I,
)
_MODAL = re.compile(r"\b(may|might|can|could|should|recommended|suggested|possibly|likely|probably|must|shall|may not|should not)\b", re.I)

_UNIT_SCALE = {
    "ug": ("mass", 1e-6), "mg": ("mass", 1e-3), "g": ("mass", 1.0), "kg": ("mass", 1000.0),
    "ml": ("volume", 1.0), "l": ("volume", 1000.0), "mmhg": ("pressure", 1.0), "cmh2o": ("pressure", 0.735559),
    "%": ("percent", 1.0), "bpm": ("rate", 1.0), "c": ("temperature", 1.0), "mm": ("length", 1.0),
    "cm": ("length", 10.0), "m": ("length", 1000.0), "hz": ("frequency", 1.0), "khz": ("frequency", 1000.0),
    "m/s": ("velocity", 1.0), "s": ("time", 1.0), "min": ("time", 60.0), "h": ("time", 3600.0),
    "day": ("time", 86400.0), "days": ("time", 86400.0), "week": ("time", 604800.0), "weeks": ("time", 604800.0),
    "month": ("time", 2592000.0), "months": ("time", 2592000.0), "year": ("time", 31536000.0), "years": ("time", 31536000.0),
}


def split_claims(answer: str) -> list[str]:
    raw = (answer or "").strip()
    if not raw:
        return []
    # Treat explicit Markdown/plain-text bullets as hard claim boundaries.
    bullet_lines = re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+(.+?)\s*$", raw)
    if bullet_lines:
        claims: list[str] = []
        for line in bullet_lines:
            line = line.strip()
            if len(meaningful_tokens(line)) >= 3:
                claims.append(line)
        if claims:
            return claims[:40]
    claims = []
    for sentence in _SENTENCE_RE.split(raw):
        sentence = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", sentence.strip())
        if len(meaningful_tokens(sentence)) >= 3:
            claims.append(sentence)
    return claims[:40]


def _norm_num(value: str) -> str:
    return value.replace(",", ".").replace("–", "-").replace(" ", "")


def _unit(value: str) -> str:
    aliases = {"µg": "ug", "mcg": "ug", "milligram": "mg", "milliliter": "ml", "litre": "l", "liter": "l", "°c": "c"}
    return aliases.get(value.casefold(), value.casefold())


def extract_measurements(text: str) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for match in _VALUE_UNIT_RE.finditer(text or ""):
        pair = (_norm_num(match.group("value")), _unit(match.group("unit")))
        if pair not in result:
            result.append(pair)
    return result


def _to_float(value: str) -> float | None:
    try:
        return float(value.replace(",", "."))
    except (ValueError, TypeError):
        return None


def _measurement_compatible(claim: tuple[str, str], evidence: tuple[str, str], tolerance: float = 1e-6) -> bool:
    cv, cu = claim
    ev, eu = evidence
    if "-" in cv or "-" in ev:
        return cv == ev and cu == eu
    c, e = _to_float(cv), _to_float(ev)
    if c is None or e is None:
        return cv == ev and cu == eu
    source = _UNIT_SCALE.get(cu)
    target = _UNIT_SCALE.get(eu)
    if source and target and source[0] == target[0]:
        converted = c * source[1] / target[1]
        return math.isclose(converted, e, rel_tol=0.0, abs_tol=tolerance)
    if cu != eu:
        return False
    return math.isclose(c, e, rel_tol=0.0, abs_tol=tolerance)


def numeric_consistency(claim: str, evidence: str) -> dict[str, Any]:
    claim_values = extract_measurements(claim)
    evidence_values = extract_measurements(evidence)
    if not claim_values:
        return {"checked": False, "mismatch": False, "claim_values": [], "evidence_values": [], "unsupported_numeric": []}
    unsupported = [item for item in claim_values if not any(_measurement_compatible(item, ev) for ev in evidence_values)]
    return {
        "checked": True,
        "mismatch": bool(unsupported),
        "claim_values": [f"{v} {u}" for v, u in claim_values],
        "evidence_values": [f"{v} {u}" for v, u in evidence_values],
        "unsupported_numeric": [f"{v} {u}" for v, u in unsupported],
    }


def _polarity(text: str) -> int:
    if not text:
        return 0
    return -1 if _NEGATION.search(text) else 1


def semantic_support(claim: str, evidence: str) -> float:
    claim_tokens = set(meaningful_tokens(claim))
    evidence_tokens = set(meaningful_tokens(evidence))
    if not claim_tokens or not evidence_tokens:
        return 0.0
    overlap = len(claim_tokens & evidence_tokens) / max(len(claim_tokens), 1)
    token_jaccard = len(claim_tokens & evidence_tokens) / max(len(claim_tokens | evidence_tokens), 1)
    char_hint = keyword_overlap_score(claim, evidence)
    neg_penalty = 0.35 if _polarity(claim) != _polarity(evidence) else 0.0
    modal_bonus = 0.04 if bool(_MODAL.search(claim)) == bool(_MODAL.search(evidence)) else 0.0
    return max(0.0, min(1.0, 0.48 * overlap + 0.22 * token_jaccard + 0.25 * char_hint + modal_bonus - neg_penalty))


def detect_contradiction(claim: str, evidence_blocks: Sequence[str]) -> bool:
    claim_tokens = set(meaningful_tokens(claim.casefold()))
    claim_pol = _polarity(claim)
    if not claim_tokens or claim_pol == 0:
        return False
    for evidence in evidence_blocks:
        evidence_tokens = set(meaningful_tokens(evidence.casefold()))
        overlap_tokens = claim_tokens & evidence_tokens
        if not overlap_tokens or _polarity(evidence) in (0, claim_pol):
            continue
        score = semantic_support(claim, evidence)
        # Short binary claims such as "treatment is contraindicated" vs
        # "treatment is indicated" have only one shared content token.
        min_score = 0.18 if len(claim_tokens) <= 3 else 0.35
        if score >= min_score:
            return True
    return False


def _best_support(claim: str, evidence_blocks: Sequence[str], source_ids: Sequence[str]) -> tuple[float, tuple[str, ...]]:
    ranked: list[tuple[float, str]] = []
    for index, evidence in enumerate(evidence_blocks):
        score = semantic_support(claim, evidence)
        if score <= 0:
            continue
        source = source_ids[index] if index < len(source_ids) else f"S{index + 1}"
        ranked.append((score, source))
    ranked.sort(reverse=True)
    return (ranked[0][0] if ranked else 0.0, tuple(source for _, source in ranked[:3]))


def verify_claims(answer: str, evidence_blocks: Sequence[str], source_ids: Sequence[str]) -> list[ClaimCheck]:
    checks: list[ClaimCheck] = []
    joined = "\n".join(evidence_blocks)
    for claim in split_claims(answer):
        best, sources = _best_support(claim, evidence_blocks, source_ids)
        num = numeric_consistency(claim, joined)
        contradiction = detect_contradiction(claim, evidence_blocks)
        if contradiction:
            status, reason = "CONTRADICTED", "A high-overlap source has opposing polarity/negation."
        elif num["mismatch"]:
            status, reason = "NUMERIC_MISMATCH", "A stated measurement is not present with a compatible unit/value in evidence."
        elif best >= 0.62:
            status, reason = "SUPPORTED", "Claim has strong lexical/semantic evidence support."
        elif best >= 0.38:
            status, reason = "PARTIAL", "Claim has partial evidence support."
        elif best > 0:
            status, reason = "WEAK", "Claim overlaps evidence weakly."
        else:
            status, reason = "UNSUPPORTED", "No meaningful evidence support found."
        checks.append(ClaimCheck(claim, round(best, 4), status, sources, bool(num["mismatch"]), contradiction, reason))
    return checks


def evidence_confidence(*, retrieval: float, rerank: float, entailment: float, quality: float, contradiction: float = 0.0, ocr_penalty: float = 0.0) -> float:
    value = 0.24 * retrieval + 0.26 * rerank + 0.30 * entailment + 0.20 * quality
    value -= 0.40 * contradiction
    value -= 0.20 * ocr_penalty
    return round(max(0.0, min(1.0, value)), 4)


def contradiction_report(claims: Sequence[ClaimCheck]) -> dict[str, Any]:
    contradicted = [c for c in claims if c.contradiction or c.status == "CONTRADICTED"]
    return {"has_contradiction": bool(contradicted), "count": len(contradicted), "claims": [c.to_dict() for c in contradicted]}


def citation_firewall(answer: str, claim_checks: Iterable[ClaimCheck]) -> tuple[str, bool]:
    checks = list(claim_checks)
    if not checks:
        return answer, False
    safe = [c for c in checks if c.status in {"SUPPORTED", "PARTIAL"} and not c.contradiction]
    unsafe = [c for c in checks if c.status in {"UNSUPPORTED", "NUMERIC_MISMATCH", "CONTRADICTED"}]
    if not unsafe:
        return answer, False
    lines: list[str] = []
    if safe:
        lines.append("Verified findings:")
        for c in safe:
            refs = " ".join(f"[{s}]" for s in c.sources)
            lines.append(f"- {c.claim} {refs}".strip())
    lines.append("\nSome generated details were withheld because they could not be verified against the indexed evidence.")
    return "\n".join(lines), True


def grounding_decision(claims: Sequence[ClaimCheck], *, min_supported_ratio: float = 0.60) -> dict[str, Any]:
    if not claims:
        return {"allow": False, "reason": "No claims were extracted from the generated answer.", "supported_ratio": 0.0}
    safe = sum(1 for c in claims if c.status in {"SUPPORTED", "PARTIAL"} and not c.contradiction)
    blocked = sum(1 for c in claims if c.status in {"UNSUPPORTED", "NUMERIC_MISMATCH", "CONTRADICTED"} or c.contradiction)
    ratio = safe / max(len(claims), 1)
    return {"allow": ratio >= min_supported_ratio and blocked == 0, "reason": "Grounding threshold passed." if ratio >= min_supported_ratio and blocked == 0 else "Grounding threshold failed.", "supported_ratio": round(ratio, 4), "blocked_claims": blocked}
