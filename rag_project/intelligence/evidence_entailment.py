from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Sequence

from rag_project.intelligence.advanced_clinical_reasoner import extract_clinical_facts
from rag_project.intelligence.evidence_guard import extract_measurements, semantic_support
from rag_project.utils.text_utils import meaningful_tokens


@dataclass(frozen=True)
class EvidenceSpan:
    source_id: str
    document_id: str
    chunk_id: str
    page_numbers: tuple[Any, ...]
    start: int
    end: int
    text: str
    support: float
    entailment: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimEvidenceRecord:
    claim: str
    status: str
    support: float
    evidence: tuple[EvidenceSpan, ...]
    fact_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _negated(text: str) -> bool:
    return bool(re.search(r"\b(no|not|never|without|cannot|contraindicated|avoid|does not|doesn't|non|sans|ne pas|لا|ليس|دون|ممنوع)\b", text or "", re.I | re.UNICODE))


def _numbers_compatible(claim: str, evidence: str) -> bool:
    claim_values = extract_measurements(claim)
    if not claim_values:
        return True
    evidence_values = extract_measurements(evidence)
    if not evidence_values:
        return False
    from rag_project.intelligence.evidence_guard import _measurement_compatible
    return all(any(_measurement_compatible(c, e) for e in evidence_values) for c in claim_values)


def _entailment_score(claim: str, evidence: str) -> float:
    support = semantic_support(claim, evidence)
    claim_terms = set(meaningful_tokens(claim.casefold()))
    evidence_terms = set(meaningful_tokens(evidence.casefold()))
    coverage = len(claim_terms & evidence_terms) / max(1, len(claim_terms))
    polarity_bonus = 0.10 if _negated(claim) == _negated(evidence) else -0.25
    numeric = 0.20 if _numbers_compatible(claim, evidence) else -0.40
    return max(0.0, min(1.0, 0.55 * support + 0.25 * coverage + polarity_bonus + numeric))


def build_claim_evidence_matrix(claims: Sequence[str], hits: Sequence[Any], source_ids: Sequence[str]) -> tuple[ClaimEvidenceRecord, ...]:
    records: list[ClaimEvidenceRecord] = []
    for claim in claims:
        candidates: list[EvidenceSpan] = []
        for index, hit in enumerate(hits):
            text = str(getattr(hit, "text", "") or "")
            if not text.strip():
                continue
            source = source_ids[index] if index < len(source_ids) else f"S{index + 1}"
            meta = getattr(hit, "metadata", {}) or {}
            best_local: EvidenceSpan | None = None
            offset = 0
            for sentence in re.split(r"(?<=[.!?。！？])\s+|\n+", text):
                sentence = sentence.strip()
                if not sentence:
                    continue
                start = text.find(sentence, offset)
                if start < 0:
                    start = offset
                offset = start + len(sentence)
                score = _entailment_score(claim, sentence)
                if score <= 0:
                    continue
                fact_ids = tuple(f"{f.node_id}:{i}" for i, f in enumerate(extract_clinical_facts(sentence, node_id=str(meta.get("chunk_id") or source), document_id=str(meta.get("document_id") or getattr(hit, "doc_id", ""))), 1))
                candidate = EvidenceSpan(source, str(meta.get("document_id") or getattr(hit, "doc_id", "")), str(meta.get("chunk_id") or getattr(hit, "doc_id", "")), tuple(meta.get("page_numbers") or ()), start, start + len(sentence), sentence[:1600], round(semantic_support(claim, sentence), 4), round(score, 4))
                if best_local is None or candidate.entailment > best_local.entailment:
                    best_local = candidate
            if best_local:
                candidates.append(best_local)
        candidates.sort(key=lambda item: (item.entailment, item.support), reverse=True)
        selected = tuple(candidates[:3])
        best = selected[0].entailment if selected else 0.0
        if best >= 0.72:
            status = "ENTAILED"
        elif best >= 0.52:
            status = "PARTIALLY_ENTAILED"
        else:
            status = "NOT_ENTAILED"
        fact_ids = tuple(fid for item in selected for fid in extract_fact_ids(item.text, item.chunk_id))
        records.append(ClaimEvidenceRecord(claim, status, round(best, 4), selected, fact_ids[:12]))
    return tuple(records)


def extract_fact_ids(text: str, node_id: str) -> tuple[str, ...]:
    facts = extract_clinical_facts(text, node_id=node_id)
    return tuple(f"{fact.node_id}:{index}" for index, fact in enumerate(facts, 1))


def matrix_has_strong_support(matrix: Sequence[ClaimEvidenceRecord]) -> bool:
    return bool(matrix) and all(record.status == "ENTAILED" for record in matrix)
