from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any

from rag_project.intelligence.query_intelligence import QueryPlan
from rag_project.intelligence.semantic_reasoning import QueryUnderstanding, extract_clinical_entities

_ALLOWED_INTENTS = {
    "definition", "comparison", "diagnosis", "management", "etiology", "mechanism",
    "prognosis", "numeric", "relationship", "navigation", "table_lookup", "figure_lookup", "factual",
}
_ALLOWED_RELATIONS = {"causality", "association", "comparison", "sequence"}
_ALLOWED_CONSTRAINTS = {"exactness", "population", "clinical_phase", "safety"}

_SYSTEM = (
    "You are a constrained medical query analyst inside a document RAG system. "
    "Do not answer the medical question. Return ONLY one JSON object. "
    "Identify the user's task, entities, relationships, clinical constraints, and the smallest set of retrieval subquestions. "
    "Do not invent facts. Preserve uncertainty. Keep every array <= 6 items."
)

_HARD_INTENTS = {"comparison", "diagnosis", "management", "etiology", "mechanism", "prognosis", "relationship", "numeric", "table_lookup", "figure_lookup"}
_RELATION_CUES = ("cause", "caused by", "due to", "leads to", "results in", "related", "relationship", "associated", "linked", "association", "why", "because", "pourquoi", "caus", "lié", "associé", "سبب", "علاقة", "مرتبط")
_SAFETY_CUES = ("contraindication", "contraindicated", "avoid", "not recommended", "contre-indication", "contre-indiqué", "ممنوع", "تجنب")
_EXACTNESS_CUES = ("exact", "exactly", "precise", "strictly", "exacte", "précis")
_POPULATION_CUES = ("adult", "child", "children", "pediatric", "pregnan", "grossesse", "enfant", "adulte", "neonate", "newborn")
_PHASE_CUES = ("first-line", "second-line", "initial", "maintenance", "acute", "chronic", "aigu", "chronique", "initiale", "entretien")


def _contains_any(text: str, cues: tuple[str, ...]) -> bool:
    low = str(text or "").casefold()
    return any(cue.casefold() in low for cue in cues)


def should_use_small_model(question: str, understanding: QueryUnderstanding) -> bool:
    """Escalate only queries whose structure genuinely benefits from model analysis."""
    text = str(question or "").strip()
    tokens = re.findall(r"\S+", text)
    lowered = text.casefold()
    ambiguity = bool(re.search(r"\b(this|that|it|they|them|the latter|the former|what about|how about)\b", lowered))
    hard_reasoning = bool(understanding.relations) or understanding.primary_intent in _HARD_INTENTS
    # Zero entities is normal for generic requests such as "What are the main findings?".
    # It must not, by itself, promote every simple factual question to an LLM path.
    return ambiguity or hard_reasoning or len(understanding.intents) >= 2 or len(tokens) >= 28


def _extract_json(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    candidates = [raw]
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _clean_list(value: Any, *, limit: int = 6, max_chars: int = 180) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        text = re.sub(r"\s+", " ", str(item or "")).strip()[:max_chars]
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _question_entities(question: str) -> set[str]:
    allowed: set[str] = set()
    try:
        allowed.update(str(entity.normalized).casefold() for entity in extract_clinical_entities(question) if entity.normalized)
    except Exception:
        pass
    return allowed


def _sanitize_assist(question: str, understanding: QueryUnderstanding, parsed: dict[str, Any]) -> dict[str, Any]:
    deterministic_entities = _question_entities(question)
    entities: list[str] = []
    for item in _clean_list(parsed.get("entities"), limit=8):
        norm = item.casefold().strip()
        if norm in deterministic_entities or norm in {str(e.normalized).casefold() for e in understanding.entities}:
            entities.append(item)
    intent = str(parsed.get("intent") or "").strip().casefold()
    if intent not in _ALLOWED_INTENTS:
        intent = understanding.primary_intent
    # A model may refine a deterministic factual query, but it must not invent a hard
    # clinical intent that is absent from the user's wording.
    if understanding.primary_intent == "factual" and intent in _HARD_INTENTS:
        hard_cue = _contains_any(question, (
            "compare", "difference", "versus", "between", "diagnos", "criteri", "treatment", "therapy",
            "cause", "why", "mechanism", "prognosis", "survival", "dose", "dosage", "mg", "ml",
            "table", "figure", "relationship", "related", "associated", "link", "علاج", "تشخيص", "سبب", "جرعة", "مقارنة", "علاقة",
        ))
        if not hard_cue:
            intent = "factual"
    relation_items = [item for item in _clean_list(parsed.get("relations")) if item in _ALLOWED_RELATIONS]
    relations = relation_items if _contains_any(question, _RELATION_CUES) else []
    constraint_items = [item for item in _clean_list(parsed.get("constraints")) if item in _ALLOWED_CONSTRAINTS]
    constraints: list[str] = []
    if "safety" in constraint_items and _contains_any(question, _SAFETY_CUES):
        constraints.append("safety")
    if "exactness" in constraint_items and _contains_any(question, _EXACTNESS_CUES):
        constraints.append("exactness")
    if "population" in constraint_items and _contains_any(question, _POPULATION_CUES):
        constraints.append("population")
    if "clinical_phase" in constraint_items and _contains_any(question, _PHASE_CUES):
        constraints.append("clinical_phase")
    return {
        "intent": intent,
        "entities": entities,
        "relations": relations,
        "constraints": constraints,
        "subquestions": _clean_list(parsed.get("subquestions")),
        "retrieval_terms": _clean_list(parsed.get("retrieval_terms")),
        "answer_strategy": re.sub(r"\s+", " ", str(parsed.get("answer_strategy") or "")).strip()[:400],
    }


def analyze_with_small_model(llm: Any, question: str, understanding: QueryUnderstanding, conversation_context: str = "") -> dict[str, Any] | None:
    if llm is None or not should_use_small_model(question, understanding):
        return None
    prompt = (
        "Return JSON with exactly these keys:\n"
        '{"intent":"...","entities":[],"relations":[],"constraints":[],"subquestions":[],"retrieval_terms":[],"answer_strategy":"..."}\n\n'
        f"Current deterministic interpretation: {json.dumps(understanding.to_dict(), ensure_ascii=False)}\n"
        f"Recent conversation reference: {str(conversation_context or '')[-900:]}\n"
        f"User question: {str(question or '')[:1800]}"
    )
    try:
        raw = llm.generate(prompt=prompt, system_prompt=_SYSTEM, temperature=0.0)
    except Exception:
        return None
    parsed = _extract_json(raw)
    if not parsed:
        return None
    return _sanitize_assist(question, understanding, parsed)


def augment_query_plan(plan: QueryPlan, assist: dict[str, Any] | None) -> QueryPlan:
    if not assist:
        return plan
    entities = list(plan.entities)
    for item in assist.get("entities", []):
        if item and item not in entities:
            entities.append(item)
        if len(entities) >= 16:
            break
    variants = list(plan.variants)
    for item in assist.get("subquestions", []):
        if item and item not in variants:
            variants.append(item)
        if len(variants) >= 10:
            break
    for item in assist.get("retrieval_terms", []):
        if item:
            candidate = f"{plan.normalized} {item}".strip()
            if candidate not in variants:
                variants.append(candidate)
        if len(variants) >= 10:
            break
    intent = str(assist.get("intent") or plan.intent)
    if intent not in _ALLOWED_INTENTS:
        intent = plan.intent
    return replace(
        plan,
        intent=intent,
        entities=tuple(entities[:16]),
        variants=tuple(variants[:10]),
        needs_multi_hop=plan.needs_multi_hop or bool(assist.get("relations")) or len(assist.get("subquestions", [])) > 1,
    )


def merge_understanding(understanding: QueryUnderstanding, assist: dict[str, Any] | None) -> QueryUnderstanding:
    if not assist:
        return understanding
    intents = list(understanding.intents)
    intent = str(assist.get("intent") or "")
    if intent and intent not in intents:
        intents.append(intent)
    relations = list(understanding.relations)
    for relation in assist.get("relations", []):
        if relation not in relations:
            relations.append(relation)
    constraints = list(understanding.constraints)
    for constraint in assist.get("constraints", []):
        if constraint not in constraints:
            constraints.append(constraint)
    hard_intent_conflict = understanding.primary_intent == "factual" and intent in _HARD_INTENTS and not _contains_any(understanding.normalized, (
        "compare", "difference", "versus", "between", "diagnos", "criteri", "treatment", "therapy", "cause", "why",
        "mechanism", "prognosis", "dose", "dosage", "mg", "ml", "table", "figure", "relationship", "related", "associated",
        "علاج", "تشخيص", "سبب", "جرعة", "مقارنة", "علاقة",
    ))
    if hard_intent_conflict:
        intent = understanding.primary_intent
        intents = [x for x in intents if x != str(assist.get("intent") or "")]
    return replace(
        understanding,
        intents=tuple(intents[:8]),
        primary_intent=intent or understanding.primary_intent,
        relations=tuple(relations[:6]),
        constraints=tuple(constraints[:8]),
        confidence=min(1.0, understanding.confidence + (0.08 if assist else 0.0)),
    )
