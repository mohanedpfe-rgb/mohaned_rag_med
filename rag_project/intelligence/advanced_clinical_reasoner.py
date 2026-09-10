from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Sequence

from rag_project.intelligence.semantic_reasoning import ClinicalEntity, QueryUnderstanding, extract_clinical_entities


_RELATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("causes", ("causes", "cause", "caused by", "due to", "leads to", "results in", "responsible for", "provoque", "provoquent", "entraîne", "entraine", "est dû à", "est du a")),
    ("association", ("associated with", "associated", "related to", "linked to", "association", "lié à", "lie a", "associé à", "associe a", "مرتبط", "علاقة")),
    ("diagnoses", ("diagnosed by", "diagnostic criteria", "criteria", "supports the diagnosis", "diagnosed with", "diagnostique", "critères diagnostiques", "معايير التشخيص")),
    ("treated_with", ("treated with", "managed with", "management includes", "treated using", "indicated for", "traité par", "prise en charge", "يُعالج بـ", "يعالج بـ")),
    ("contraindicated", ("contraindicated", "should not", "avoid", "not recommended", "contre-indiqué", "contre indique", "éviter", "ممنوع", "تجنب")),
    ("precedes", ("before", "prior to", "precedes", "preceded by", "avant", "précède", "precede", "قبل")),
    ("follows", ("after", "subsequently", "then", "followed by", "après", "ensuite", "puis", "بعد", "ثم")),
)

_OPPOSITE: dict[str, str] = {
    "causes": "not_causes",
    "association": "not_association",
    "diagnoses": "not_diagnoses",
    "treated_with": "not_treated_with",
    "contraindicated": "recommended",
    "precedes": "follows",
    "follows": "precedes",
}

_NEGATED_RELATION_PATTERNS: tuple[str, ...] = (
    r"\bnot\s+(?:a\s+)?cause\b",
    r"\bdoes\s+not\s+cause\b",
    r"\bnot\s+associated\b",
    r"\bnot\s+related\b",
    r"\bnot\s+recommended\b",
    r"\bshould\s+not\s+be\s+treated\s+with\b",
    r"\bne\s+cause\s+pas\b",
    r"\bnon\s+associé\b",
    r"\bne\s+doit\s+pas\s+être\s+traité\b",
)


@dataclass(frozen=True)
class ClinicalFact:
    subject: str
    predicate: str
    object: str
    polarity: int
    confidence: float
    node_id: str
    sentence: str
    document_id: str


@dataclass(frozen=True)
class ReasoningPath:
    nodes: tuple[str, ...]
    relations: tuple[str, ...]
    fact_ids: tuple[str, ...]
    support: float
    explicit: bool
    cross_document: bool


@dataclass(frozen=True)
class ClinicalReasoningAssessment:
    allow_generation: bool
    mode: str
    depth: int
    entity_coverage: float
    direct_support: float
    path_support: float
    source_agreement: float
    contradiction: float
    safety_conflict: float
    confidence: float
    supported_paths: tuple[ReasoningPath, ...]
    blocked_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _is_negated(sentence: str, start: int) -> bool:
    prefix = sentence[max(0, start - 90): start]
    return bool(re.search(r"\b(no|not|never|without|cannot|does not|doesn't|non|ne pas|sans|aucun|ممنوع|ليس|لا)\b", prefix, re.I | re.UNICODE))


def _relation_hits(sentence: str) -> list[tuple[str, int]]:
    hits: list[tuple[str, int]] = []
    lowered = _normal(sentence)
    for relation, cues in _RELATIONS:
        positions = [lowered.find(_normal(cue)) for cue in cues if _normal(cue) in lowered]
        if positions:
            hits.append((relation, min(positions)))
    return sorted(hits, key=lambda item: item[1])


def _entity_mentions(sentence: str) -> list[ClinicalEntity]:
    return list(extract_clinical_entities(sentence))


def extract_clinical_facts(text: str, *, node_id: str = "", document_id: str = "") -> tuple[ClinicalFact, ...]:
    facts: list[ClinicalFact] = []
    clean = str(text or "").strip()
    if not clean:
        return ()
    sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", clean)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        entities = _entity_mentions(sentence)
        relation_hits = _relation_hits(sentence)
        if len(entities) < 2 or not relation_hits:
            continue
        for rel_index, (relation, position) in enumerate(relation_hits[:3]):
            ordered = sorted(entities, key=lambda entity: sentence.casefold().find(entity.text.casefold()))
            if len(ordered) < 2:
                continue
            if rel_index == 0:
                pairs = list(combinations(ordered, 2))
            else:
                pairs = list(combinations(ordered, 2))[:3]
            for pair_index, (left, right) in enumerate(pairs[:6]):
                left_pos = sentence.casefold().find(left.text.casefold())
                right_pos = sentence.casefold().find(right.text.casefold())
                subject, obj = (left, right) if left_pos <= right_pos else (right, left)
                polarity = -1 if _is_negated(sentence, position) or subject.negated or obj.negated else 1
                confidence = min(0.97, 0.55 + 0.08 * min(len(entities), 3) + 0.06 * min(pair_index, 2))
                facts.append(ClinicalFact(
                    subject=subject.normalized,
                    predicate=relation,
                    object=obj.normalized,
                    polarity=polarity,
                    confidence=confidence,
                    node_id=node_id or f"node_{len(facts) + 1}",
                    sentence=sentence,
                    document_id=document_id,
                ))
    return tuple(facts[:24])


def _query_edges(understanding: QueryUnderstanding) -> set[tuple[str, str]]:
    entities = [entity.normalized for entity in understanding.entities]
    return set(zip(entities, entities[1:]))


def _fact_support(fact: ClinicalFact, query_entities: set[str]) -> float:
    score = fact.confidence
    if fact.subject in query_entities:
        score += 0.10
    if fact.object in query_entities:
        score += 0.10
    if fact.polarity < 0:
        score -= 0.25
    return max(0.0, min(1.0, score))


def _find_paths(facts: Sequence[ClinicalFact], understanding: QueryUnderstanding, *, max_depth: int = 3) -> tuple[ReasoningPath, ...]:
    query_entities = [entity.normalized for entity in understanding.entities]
    if len(query_entities) < 2:
        return ()
    adjacency: dict[str, list[ClinicalFact]] = {}
    for fact in facts:
        if fact.polarity > 0:
            adjacency.setdefault(fact.subject, []).append(fact)
    results: list[ReasoningPath] = []
    target_pairs = list(zip(query_entities, query_entities[1:]))
    for start, target in target_pairs[:6]:
        stack: list[tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...], float, set[str]]] = [(start, (start,), (), (), 1.0, set())]
        while stack:
            node, nodes, relations, fact_ids, score, documents = stack.pop()
            if len(relations) > max_depth:
                continue
            if node == target and relations:
                results.append(ReasoningPath(
                    nodes=nodes,
                    relations=relations,
                    fact_ids=fact_ids,
                    support=round(score, 3),
                    explicit=True,
                    cross_document=len(documents) > 1,
                ))
                continue
            for fact in adjacency.get(node, ()):
                if fact.object in nodes:
                    continue
                next_score = score * (0.72 + 0.28 * fact.confidence)
                next_docs = set(documents)
                if fact.document_id:
                    next_docs.add(fact.document_id)
                stack.append((fact.object, nodes + (fact.object,), relations + (fact.predicate,), fact_ids + (f"{fact.node_id}:{len(fact_ids)}",), next_score, next_docs))
    unique: dict[tuple[str, ...], ReasoningPath] = {}
    for path in sorted(results, key=lambda p: p.support, reverse=True):
        unique.setdefault(path.nodes + path.relations, path)
    return tuple(unique.values())[:8]


def _same_claim_conflict(facts: Sequence[ClinicalFact]) -> float:
    grouped: dict[tuple[str, str, str], set[int]] = {}
    for fact in facts:
        grouped.setdefault((fact.subject, fact.predicate, fact.object), set()).add(fact.polarity)
    conflicts = sum(1 for polarities in grouped.values() if len(polarities) > 1)
    return min(1.0, conflicts / max(1, len(grouped)))


def _safety_conflict(understanding: QueryUnderstanding, facts: Sequence[ClinicalFact]) -> float:
    if "safety" not in understanding.constraints:
        return 0.0
    positive_contra = sum(1 for fact in facts if fact.predicate == "contraindicated" and fact.polarity > 0)
    positive_treatment = sum(1 for fact in facts if fact.predicate == "treated_with" and fact.polarity > 0)
    if positive_contra and positive_treatment:
        return 1.0
    return 0.5 if positive_contra else 0.0


def assess_clinical_reasoning(understanding: QueryUnderstanding, hits: Sequence[Any], *, max_depth: int = 3) -> ClinicalReasoningAssessment:
    all_facts: list[ClinicalFact] = []
    for index, hit in enumerate(hits):
        meta = getattr(hit, "metadata", {}) or {}
        all_facts.extend(extract_clinical_facts(
            str(getattr(hit, "text", "") or ""),
            node_id=str(meta.get("chunk_id") or f"N{index + 1}"),
            document_id=str(meta.get("document_id") or getattr(hit, "doc_id", "")),
        ))
    query_entities = {entity.normalized for entity in understanding.entities}
    covered = {fact.subject for fact in all_facts} | {fact.object for fact in all_facts}
    entity_coverage = len(query_entities & covered) / max(1, len(query_entities))
    positive_direct = [fact for fact in all_facts if fact.polarity > 0 and fact.subject in query_entities and fact.object in query_entities]
    direct_support = max((_fact_support(fact, query_entities) for fact in positive_direct), default=0.0)
    paths = _find_paths(all_facts, understanding, max_depth=max_depth)
    path_support = max((path.support for path in paths), default=0.0)
    source_agreement = 0.0
    if paths:
        path = paths[0]
        path_docs = {fact.document_id for fact in all_facts if fact.node_id in path.nodes and fact.document_id}
        source_agreement = min(1.0, 0.5 + 0.25 * len(path_docs))
    contradiction = _same_claim_conflict(all_facts)
    safety_conflict = _safety_conflict(understanding, all_facts)
    relation_needed = bool(understanding.relations or understanding.primary_intent in {"etiology", "mechanism", "association", "comparison"})
    if positive_direct:
        mode = "DIRECT"
        depth = 1
    elif paths and relation_needed:
        mode = "MULTI_HOP" if len(paths[0].relations) > 1 else "ONE_HOP"
        depth = len(paths[0].relations)
    else:
        mode = "INSUFFICIENT"
        depth = 0
    blocked: list[str] = []
    if entity_coverage < 0.5:
        blocked.append("insufficient_entity_coverage")
    if relation_needed and not positive_direct and not paths:
        blocked.append("no_explicit_reasoning_path")
    if contradiction >= 0.5:
        blocked.append("conflicting_evidence")
    if safety_conflict >= 1.0:
        blocked.append("safety_conflict")
    confidence = max(0.0, min(1.0, 0.35 * direct_support + 0.35 * path_support + 0.15 * entity_coverage + 0.15 * source_agreement - 0.30 * contradiction - 0.35 * safety_conflict))
    allow = not blocked and confidence >= (0.42 if mode == "DIRECT" else 0.50)
    return ClinicalReasoningAssessment(
        allow_generation=allow,
        mode=mode,
        depth=depth,
        entity_coverage=round(entity_coverage, 3),
        direct_support=round(direct_support, 3),
        path_support=round(path_support, 3),
        source_agreement=round(source_agreement, 3),
        contradiction=round(contradiction, 3),
        safety_conflict=round(safety_conflict, 3),
        confidence=round(confidence, 3),
        supported_paths=paths,
        blocked_reasons=tuple(blocked),
    )


def build_reasoning_instruction(understanding: QueryUnderstanding, assessment: ClinicalReasoningAssessment) -> str:
    if assessment.mode == "MULTI_HOP":
        path_text = " -> ".join(assessment.supported_paths[0].nodes) if assessment.supported_paths else ""
        return (
            f"Use the explicit evidence path only: {path_text}. "
            "Each relationship must be supported by a retrieved source. Do not invent a missing link. "
            "Separate what is directly stated from what is only inferred. Preserve population, timing, contraindications, and numeric qualifiers. "
            "Do not reveal private reasoning steps; provide only the evidence-supported conclusion."
        )
    if assessment.mode == "ONE_HOP":
        return "Use the explicit relationship supported by the evidence. Do not extend beyond the retrieved fact or introduce unstated clinical conclusions."
    if "safety" in understanding.constraints:
        return "Prioritize safety-relevant contraindications and population/phase qualifiers. Never convert a contraindication into a recommendation."
    return "Answer from directly supported evidence, preserving uncertainty, negation, numbers, units, and qualifiers."
