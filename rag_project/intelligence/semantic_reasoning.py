from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from rag_project.utils.text_utils import meaningful_tokens


@dataclass(frozen=True)
class ClinicalEntity:
    text: str
    normalized: str
    kind: str
    confidence: float
    negated: bool = False


@dataclass(frozen=True)
class QueryUnderstanding:
    normalized: str
    intents: tuple[str, ...]
    primary_intent: str
    entities: tuple[ClinicalEntity, ...]
    relations: tuple[str, ...]
    constraints: tuple[str, ...]
    answer_shape: str
    semantic_terms: tuple[str, ...]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceNode:
    node_id: str
    text: str
    score: float
    document_id: str
    chunk_id: str
    page_numbers: tuple[Any, ...]


@dataclass(frozen=True)
class ReasoningEdge:
    source: str
    target: str
    relation: str
    confidence: float
    evidence: tuple[str, ...]


_MEDICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "diabetes mellitus": ("diabetes", "diabète", "diabete", "dm", "d2", "type 2 diabetes", "diabète de type 2"),
    "diabetic ketoacidosis": ("dka", "acidocétose diabétique", "acidocetose diabetique", "ketoacidosis"),
    "diabetic nephropathy": ("nephropathie diabetique", "néphropathie diabétique", "diabetic kidney disease", "dkd"),
    "albuminuria": ("albuminurie", "microalbuminuria", "microalbuminurie", "urinary albumin"),
    "hypertension": ("hta", "high blood pressure", "hypertension artérielle", "hypertension arterielle"),
    "hypokalemia": ("hypokaliémie", "hypokaliemia", "low potassium", "dyskaliémie", "dyskaliemia"),
    "hyperaldosteronism": ("hyperaldostéronisme", "hyperaldosteronisme", "primary aldosteronism", "hpa"),
    "thyroid cancer": ("cancer thyroïde", "cancer de la thyroïde", "thyroid carcinoma", "thyroid neoplasm"),
    "insulin resistance": ("insulin resistance", "insulinorésistance", "insulinorésistance", "insulin resistance syndrome"),
    "albumin": ("albumine",),
    "potassium": ("kaliémie", "kalemie", "k+", "potassium"),
    "insulin": ("insuline", "insulin"),
}

_ENTITY_KIND = {
    "diabetes mellitus": "disease",
    "diabetic ketoacidosis": "disease",
    "diabetic nephropathy": "complication",
    "albuminuria": "finding",
    "hypertension": "disease",
    "hypokalemia": "finding",
    "hyperaldosteronism": "disease",
    "thyroid cancer": "disease",
    "insulin resistance": "pathophysiology",
    "albumin": "biomarker",
    "potassium": "analyte",
    "insulin": "hormone",
}

_INTENTS = {
    "definition": ("what is", "define", "definition", "meaning", "qu'est-ce que", "définition", "ما هو", "ما هي", "تعريف"),
    "comparison": ("compare", "comparison", "difference", "differences", "versus", "vs", "between", "différence", "comparaison", "مقارنة", "فرق", "بين"),
    "diagnosis": ("diagnosis", "diagnostic", "criteria", "diagnostic criteria", "diagnostiquer", "diagnostique", "تشخيص", "معايير التشخيص"),
    "management": ("treatment", "management", "therapy", "treat", "prise en charge", "traitement", "prise charge", "علاج", "التدبير"),
    "etiology": ("cause", "causes", "etiology", "aetiology", "why", "risk factor", "facteur", "étiologie", "سبب", "أسباب"),
    "mechanism": ("mechanism", "physiopathology", "pathophysiology", "how does", "mécanisme", "physiopathologie", "آلية", "فيزيولوجيا مرضية"),
    "prognosis": ("prognosis", "outcome", "survival", "prognostic", "pronostic", "مآل", "التكهن"),
    "numeric": ("dose", "dosage", "mg", "ml", "mmhg", "percentage", "how many", "how much", "range", "threshold", "قيمة", "جرعة", "نسبة"),
    "association": ("related", "relationship", "associated", "association", "linked", "lien", "relation", "علاقة", "مرتبط"),
    "navigation": ("which page", "page number", "section", "where", "source", "citation", "quelle page", "où", "أين", "أي صفحة"),
    "table_lookup": ("table", "row", "column", "tableau", "جدول", "صف", "عمود"),
    "figure_lookup": ("figure", "diagram", "chart", "graph", "image", "figure", "schéma", "رسم", "شكل"),
}

_RELATION_PATTERNS = (
    ("causality", r"\b(cause|causes|caused by|due to|leads to|responsible for|provoque|entra[iî]ne|سبب)\b"),
    ("association", r"\b(associated with|associated|related to|linked to|association|lié à|associé à|مرتبط)\b"),
    ("comparison", r"\b(compare|versus|vs|difference|différence|مقارنة|فرق)\b"),
    ("sequence", r"\b(then|after|before|subsequently|ensuite|après|avant|ثم|بعد|قبل)\b"),
)

_NEGATION = re.compile(r"\b(no|not|without|never|none|cannot|does not|doesn't|contraindicated|avoid|aucun|sans|jamais|ne pas|ممنوع|منع|لا|ليس|دون)\b", re.I)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _phrase_match(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(phrase.casefold())}(?!\w)", text, re.I | re.UNICODE))


def normalize_medical_term(text: str) -> str:
    value = _norm(text)
    for canonical, aliases in _MEDICAL_ALIASES.items():
        if value == canonical or any(_phrase_match(value, alias) for alias in aliases):
            return canonical
    return value


def extract_clinical_entities(text: str) -> tuple[ClinicalEntity, ...]:
    value = _norm(text)
    found: dict[str, ClinicalEntity] = {}
    for canonical, aliases in _MEDICAL_ALIASES.items():
        matched_alias = next((alias for alias in aliases if _phrase_match(value, alias)), None)
        if matched_alias:
            before = value[: value.find(matched_alias.casefold())]
            negated = bool(_NEGATION.search(before[-80:]))
            found[canonical] = ClinicalEntity(
                text=matched_alias,
                normalized=canonical,
                kind=_ENTITY_KIND.get(canonical, "concept"),
                confidence=0.96 if len(matched_alias.split()) > 1 else 0.88,
                negated=negated,
            )
    # Keep important domain terms that do not have a dictionary alias.
    tokens = meaningful_tokens(value)
    for token in tokens:
        if token in {e.normalized for e in found.values()}:
            continue
        if token.endswith(("itis", "osis", "emia", "pathy", "carcinoma")) and token not in found:
            found[token] = ClinicalEntity(token, token, "medical_concept", 0.72, False)
    return tuple(found.values())[:24]


def _intent_hits(text: str) -> list[tuple[str, int]]:
    hits: list[tuple[str, int]] = []
    for intent, cues in _INTENTS.items():
        count = sum(1 for cue in cues if _phrase_match(text, cue))
        if count:
            hits.append((intent, count))
    return sorted(hits, key=lambda item: (-item[1], item[0]))


def understand_query(query: str, *, conversation_context: str = "") -> QueryUnderstanding:
    normalized = _norm(query)
    combined = normalized
    if conversation_context:
        combined = f"{normalized} {conversation_context[-1200:]}"
    hits = _intent_hits(combined)
    intents = [intent for intent, _ in hits]
    if not intents:
        intents = ["factual"]
    if len(intents) > 1 and "numeric" in intents and len(intents) > 2:
        # Numeric is an attribute of another intent, not necessarily the primary task.
        intents.remove("numeric")
        intents.append("numeric")
    primary = intents[0]
    relations = tuple(sorted({kind for kind, pattern in _RELATION_PATTERNS if re.search(pattern, combined, re.I | re.UNICODE)}))
    constraints: list[str] = []
    if any(re.search(r"\b(exact|precise|exactly|strictly|exacte|précis)\b", combined, re.I) for _ in [0]):
        constraints.append("exactness")
    if any(re.search(r"\b(adult|child|pediatric|pregnan|grossesse|enfant|adulte)\b", combined, re.I) for _ in [0]):
        constraints.append("population")
    if any(re.search(r"\b(first[- ]line|second[- ]line|initial|maintenance|acute|chronic|aigu|chronique)\b", combined, re.I) for _ in [0]):
        constraints.append("clinical_phase")
    answer_shape = "comparison" if primary == "comparison" else "list" if primary in {"diagnosis", "management", "etiology", "prognosis", "numeric"} else "definition" if primary == "definition" else "explanation"
    entities = extract_clinical_entities(combined)
    semantic_terms = tuple(dict.fromkeys([e.normalized for e in entities] + meaningful_tokens(normalized)))[:32]
    cue_conf = min(1.0, 0.40 + 0.12 * len(hits) + 0.04 * len(entities))
    if len(tokens := meaningful_tokens(normalized)) >= 5:
        cue_conf = min(1.0, cue_conf + 0.10)
    return QueryUnderstanding(
        normalized=normalized,
        intents=tuple(intents),
        primary_intent=primary,
        entities=entities,
        relations=relations,
        constraints=tuple(constraints),
        answer_shape=answer_shape,
        semantic_terms=semantic_terms,
        confidence=round(cue_conf, 3),
    )


def build_evidence_graph(hits: Sequence[Any], understanding: QueryUnderstanding) -> tuple[tuple[EvidenceNode, ...], tuple[ReasoningEdge, ...]]:
    nodes: list[EvidenceNode] = []
    edges: list[ReasoningEdge] = []
    entity_to_nodes: dict[str, list[str]] = {}
    for index, hit in enumerate(hits):
        meta = getattr(hit, "metadata", {}) or {}
        node_id = str(meta.get("chunk_id") or f"N{index + 1}")
        entities = extract_clinical_entities(str(getattr(hit, "text", "")))
        nodes.append(EvidenceNode(
            node_id=node_id,
            text=str(getattr(hit, "text", "")),
            score=max(0.0, min(1.0, float(getattr(hit, "score", 0.0)))),
            document_id=str(meta.get("document_id") or getattr(hit, "doc_id", "")),
            chunk_id=node_id,
            page_numbers=tuple(meta.get("page_numbers") or ()),
        ))
        for entity in entities:
            entity_to_nodes.setdefault(entity.normalized, []).append(node_id)
    for relation in understanding.relations:
        labels = list(understanding.entities)
        if len(labels) < 2:
            continue
        for left, right in zip(labels, labels[1:]):
            left_nodes = entity_to_nodes.get(left.normalized, [])
            right_nodes = entity_to_nodes.get(right.normalized, [])
            for source in left_nodes[:3]:
                for target in right_nodes[:3]:
                    if source != target:
                        edges.append(ReasoningEdge(source, target, relation, 0.62, tuple(sorted(set(left_nodes + right_nodes)))[:3]))
    return tuple(nodes), tuple(edges)


def clinical_reasoning_ready(understanding: QueryUnderstanding, nodes: Sequence[EvidenceNode], edges: Sequence[ReasoningEdge]) -> dict[str, Any]:
    direct = bool(nodes)
    multi_hop = bool(edges) and (understanding.primary_intent in {"etiology", "mechanism", "association", "comparison"} or len(understanding.relations) > 0)
    return {
        "direct_evidence": direct,
        "multi_hop": multi_hop,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "entity_coverage": round(sum(1 for entity in understanding.entities if any(entity.normalized in _norm(node.text) for node in nodes)) / max(len(understanding.entities), 1), 3),
        "reasoning_depth": 2 if multi_hop else 1,
    }
