from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

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
    "diabetes mellitus": ("diabetes mellitus", "diabetes", "diabète", "diabete", "dm", "d2", "type 2 diabetes", "diabète de type 2", "داء السكري", "السكري"),
    "diabetic ketoacidosis": ("diabetic ketoacidosis", "dka", "acidocétose diabétique", "acidocetose diabetique", "ketoacidosis", "acidocétose", "الحماض الكيتوني السكري", "الحماض الكيتوني"),
    "diabetic nephropathy": ("diabetic nephropathy", "nephropathie diabetique", "néphropathie diabétique", "diabetic kidney disease", "dkd", "maladie rénale diabétique", "maladie renale diabetique", "اعتلال الكلية السكري"),
    "albuminuria": ("albuminuria", "albuminurie", "microalbuminuria", "microalbuminurie", "urinary albumin", "albumin in urine", "بيلة الألبومين", "زلال البول"),
    "hypertension": ("hypertension", "hta", "high blood pressure", "hypertension artérielle", "hypertension arterielle", "ارتفاع ضغط الدم"),
    "hypokalemia": ("hypokalemia", "hypokaliémie", "hypokaliemia", "low potassium", "dyskaliémie", "dyskaliemia", "نقص بوتاسيوم الدم"),
    "hyperaldosteronism": ("hyperaldosteronism", "hyperaldostéronisme", "hyperaldosteronisme", "primary aldosteronism", "hpa", "فرط الألدوستيرونية"),
    "thyroid cancer": ("thyroid cancer", "cancer thyroïde", "cancer de la thyroïde", "thyroid carcinoma", "thyroid neoplasm", "سرطان الغدة الدرقية"),
    "insulin resistance": ("insulin resistance", "insulinorésistance", "insulinorésistance", "insulin resistance syndrome", "مقاومة الأنسولين"),
    "albumin": ("albumin", "albumine", "الألبومين"),
    "potassium": ("potassium", "kaliémie", "kalemie", "k+", "البوتاسيوم"),
    "insulin": ("insulin", "insuline", "الأنسولين", "الإنسولين"),
}

_ENTITY_KIND = {
    "diabetes mellitus": "disease", "diabetic ketoacidosis": "disease", "diabetic nephropathy": "complication",
    "albuminuria": "finding", "hypertension": "disease", "hypokalemia": "finding", "hyperaldosteronism": "disease",
    "thyroid cancer": "disease", "insulin resistance": "pathophysiology", "albumin": "biomarker", "potassium": "analyte", "insulin": "hormone",
}

_INTENTS = {
    "definition": ("what is", "define", "definition", "meaning", "qu'est-ce que", "définition", "ما هو", "ما هي", "تعريف"),
    "comparison": ("compare", "comparison", "difference", "differences", "versus", "vs", "between", "différence", "comparaison", "مقارنة", "فرق", "بين"),
    "diagnosis": ("diagnosis", "diagnostic", "criteria", "diagnostic criteria", "diagnostiquer", "diagnostique", "تشخيص", "معايير التشخيص"),
    "management": ("treatment", "treated", "management", "therapy", "treat", "prise en charge", "traitement", "prise charge", "علاج", "التدبير"),
    "etiology": ("cause", "causes", "etiology", "aetiology", "why", "risk factor", "facteur", "étiologie", "سبب", "أسباب"),
    "mechanism": ("mechanism", "mechanisms", "physiopathology", "pathophysiology", "how does", "mécanisme", "physiopathologie", "آلية", "فيزيولوجيا مرضية"),
    "prognosis": ("prognosis", "outcome", "survival", "prognostic", "pronostic", "مآل", "التكهن"),
    "numeric": ("dose", "dosage", "mg", "ml", "mmhg", "percentage", "how many", "how much", "range", "threshold", "قيمة", "جرعة", "نسبة"),
    "association": ("related", "relationship", "associated", "association", "linked", "lien", "relation", "علاقة", "مرتبط"),
    "navigation": ("which page", "page number", "section", "where", "source", "citation", "quelle page", "où", "أين", "أي صفحة"),
    "table_lookup": ("table", "row", "column", "tableau", "جدول", "صف", "عمود"),
    "figure_lookup": ("figure", "diagram", "chart", "graph", "image", "schéma", "رسم", "شكل"),
}

_RELATION_PATTERNS = (
    ("causality", r"\b(cause|causes|caused by|due to|leads to|responsible for|provoque|entra[iî]ne|سبب|يؤدي)\b"),
    ("association", r"\b(associated with|associated|related to|linked to|association|lié à|associé à|مرتبط|علاقة)\b"),
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
            negated = bool(_NEGATION.search(before[-100:]))
            found[canonical] = ClinicalEntity(matched_alias, canonical, _ENTITY_KIND.get(canonical, "concept"), 0.96 if len(matched_alias.split()) > 1 else 0.88, negated)
    normalized_found = set(found)
    for token in meaningful_tokens(value):
        if token in normalized_found:
            continue
        if token.endswith(("itis", "osis", "emia", "pathy", "carcinoma")):
            found[token] = ClinicalEntity(token, token, "medical_concept", 0.72, False)
    return tuple(found.values())[:24]


def _intent_hits(text: str) -> list[tuple[str, int]]:
    hits: list[tuple[str, int]] = []
    for intent, cues in _INTENTS.items():
        count = sum(1 for cue in cues if _phrase_match(text, cue))
        if count:
            hits.append((intent, count))
    return sorted(hits, key=lambda item: (-item[1], item[0]))


def _primary_intent(current_hits: list[tuple[str, int]], context_hits: list[tuple[str, int]]) -> str:
    if not current_hits:
        return context_hits[0][0] if context_hits else "factual"
    current = {name: score for name, score in current_hits}
    specialized = ["comparison", "management", "diagnosis", "numeric", "mechanism", "etiology", "prognosis", "association", "navigation", "table_lookup", "figure_lookup"]
    for intent in specialized:
        if intent in current:
            return intent
    if "definition" in current:
        for intent in specialized:
            if intent in {name for name, _ in context_hits}:
                return intent
    return current_hits[0][0]


def understand_query(query: str, *, conversation_context: str = "") -> QueryUnderstanding:
    normalized = _norm(query)
    current_hits = _intent_hits(normalized)
    combined = f"{normalized} {conversation_context[-1200:]}".strip() if conversation_context else normalized
    context_hits = _intent_hits(conversation_context[-1200:]) if conversation_context else []
    all_hits = _intent_hits(combined)
    intents = [intent for intent, _ in all_hits] or ["factual"]
    primary = _primary_intent(current_hits, context_hits or all_hits)
    if primary not in intents:
        intents.insert(0, primary)
    relations = tuple(sorted({kind for kind, pattern in _RELATION_PATTERNS if re.search(pattern, combined, re.I | re.UNICODE)}))
    constraints: list[str] = []
    if re.search(r"\b(exact|precise|exactly|strictly|exacte|précis)\b", combined, re.I):
        constraints.append("exactness")
    if re.search(r"\b(adult|child|children|pediatric|pregnan|grossesse|enfant|enfants|adulte|pédiatrique)\b", combined, re.I):
        constraints.append("population")
    if re.search(r"\b(first[- ]line|second[- ]line|initial|maintenance|acute|chronic|aigu|chronique|initiale|entretien)\b", combined, re.I):
        constraints.append("clinical_phase")
    if re.search(r"\b(contraindication|contraindications|contraindicated|avoid|contre-indication|ممنوع|تجنب)\b", combined, re.I):
        constraints.append("safety")
    answer_shape = "comparison" if primary == "comparison" else "list" if primary in {"diagnosis", "management", "etiology", "prognosis", "numeric"} else "definition" if primary == "definition" else "explanation"
    entities = extract_clinical_entities(combined)
    semantic_terms = tuple(dict.fromkeys([e.normalized for e in entities] + meaningful_tokens(normalized)))[:32]
    confidence = min(1.0, 0.40 + 0.12 * len(intents) + 0.04 * len(entities) + (0.10 if len(meaningful_tokens(normalized)) >= 5 else 0.0))
    return QueryUnderstanding(normalized, tuple(intents), primary, entities, relations, tuple(constraints), answer_shape, tuple(semantic_terms), round(confidence, 3))


def build_evidence_graph(hits: Sequence[Any], understanding: QueryUnderstanding) -> tuple[tuple[EvidenceNode, ...], tuple[ReasoningEdge, ...]]:
    nodes: list[EvidenceNode] = []
    edges: list[ReasoningEdge] = []
    for index, hit in enumerate(hits):
        meta = getattr(hit, "metadata", {}) or {}
        node_id = str(meta.get("chunk_id") or f"N{index + 1}")
        text = str(getattr(hit, "text", ""))
        nodes.append(EvidenceNode(node_id, text, max(0.0, min(1.0, float(getattr(hit, "score", 0.0)))), str(meta.get("document_id") or getattr(hit, "doc_id", "")), node_id, tuple(meta.get("page_numbers") or ())))
        entities = extract_clinical_entities(text)
        relation_hits = [kind for kind, pattern in _RELATION_PATTERNS if re.search(pattern, text, re.I | re.UNICODE)]
        wanted = [e.normalized for e in understanding.entities]
        present = {e.normalized for e in entities}
        if len(present.intersection(wanted)) >= 2 and relation_hits:
            relation = next((r for r in relation_hits if r in understanding.relations), relation_hits[0])
            edges.append(ReasoningEdge(node_id, node_id, relation, 0.78, (node_id,)))
    return tuple(nodes), tuple(edges)


def clinical_reasoning_ready(understanding: QueryUnderstanding, nodes: Sequence[EvidenceNode], edges: Sequence[ReasoningEdge]) -> dict[str, Any]:
    direct = bool(nodes)
    cross_node_edges = [edge for edge in edges if edge.source != edge.target]
    multi_hop = bool(cross_node_edges) and (understanding.primary_intent in {"etiology", "mechanism", "association", "comparison"} or len(understanding.relations) > 0)
    entity_coverage = sum(1 for entity in understanding.entities if any(entity.normalized in _norm(node.text) for node in nodes)) / max(len(understanding.entities), 1)
    return {"direct_evidence": direct, "multi_hop": multi_hop, "node_count": len(nodes), "edge_count": len(edges), "entity_coverage": round(entity_coverage, 3), "reasoning_depth": 2 if multi_hop else 1}


def _token_overlap(left: str, right: str) -> float:
    a = set(meaningful_tokens(left)); b = set(meaningful_tokens(right))
    return len(a & b) / max(1, len(a))


def _entity_surface_score(query_entities: Sequence[ClinicalEntity], text: str) -> float:
    if not query_entities:
        return 0.0
    evidence_entities = {entity.normalized for entity in extract_clinical_entities(text)}
    return sum(1 for entity in query_entities if entity.normalized in evidence_entities) / len(query_entities)


def semantic_evidence_alignment(question: str, hits: Sequence[Any], *, conversation_context: str = "") -> dict[str, Any]:
    understanding = understand_query(question, conversation_context=conversation_context)
    nonempty = [hit for hit in hits if str(getattr(hit, "text", "") or "").strip()]
    if not nonempty:
        return {"score": 0.0, "entity_coverage": 0.0, "semantic_overlap": 0.0, "relation_coverage": 0.0, "best_hit_score": 0.0, "decision": "NOT_SUPPORTED", "reason": "No non-empty evidence was available for semantic alignment.", "understanding": understanding.to_dict()}
    per_hit: list[dict[str, Any]] = []
    for hit in nonempty:
        text = str(getattr(hit, "text", "") or "")
        entity_score = _entity_surface_score(understanding.entities, text)
        lexical_score = _token_overlap(understanding.normalized, text)
        evidence_relations = {kind for kind, pattern in _RELATION_PATTERNS if re.search(pattern, text, re.I | re.UNICODE)}
        relation_score = len(set(understanding.relations) & evidence_relations) / max(len(understanding.relations), 1) if understanding.relations else 0.0
        retrieval_score = max(0.0, min(1.0, float(getattr(hit, "score", 0.0))))
        score = min(1.0, 0.55 * entity_score + 0.25 * lexical_score + 0.10 * relation_score + 0.10 * retrieval_score)
        per_hit.append({"score": round(score, 3), "entity_score": round(entity_score, 3), "lexical_score": round(lexical_score, 3), "relation_score": round(relation_score, 3)})
    best = max(per_hit, key=lambda item: item["score"])
    top_scores = sorted((item["score"] for item in per_hit), reverse=True)[:3]
    score = max(best["score"], (sum(top_scores) / max(1, len(top_scores))) * 0.85)
    decision = "DIRECTLY_SUPPORTED" if score >= 0.55 else "PARTIALLY_SUPPORTED" if score >= 0.25 else "RELATED_BUT_NOT_ANSWERING" if score > 0.05 else "NOT_SUPPORTED"
    return {"score": round(score, 3), "entity_coverage": round(best["entity_score"], 3), "semantic_overlap": round(best["lexical_score"], 3), "relation_coverage": round(best["relation_score"], 3), "best_hit_score": best["score"], "decision": decision, "reason": "Semantic entity, concept, and relation alignment was computed across the retrieved evidence.", "understanding": understanding.to_dict()}
