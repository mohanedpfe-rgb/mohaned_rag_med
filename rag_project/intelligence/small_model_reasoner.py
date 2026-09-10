from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any

from rag_project.intelligence.query_intelligence import QueryPlan
from rag_project.intelligence.semantic_reasoning import QueryUnderstanding

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


def should_use_small_model(question: str, understanding: QueryUnderstanding) -> bool:
    """Escalate only hard cases so normal questions stay fast and deterministic."""
    text = str(question or "").strip()
    tokens = text.split()
    lowered = text.casefold()
    ambiguity = bool(re.search(r"\b(this|that|it|they|them|the latter|the former|and the treatment|what about|how about)\b", lowered))
    multi_intent = len(understanding.intents) >= 2
    hard_reasoning = bool(understanding.relations) or understanding.primary_intent in {
        "comparison", "diagnosis", "management", "etiology", "mechanism", "prognosis", "relationship"
    }
    low_entity_signal = len(understanding.entities) == 0
    return ambiguity or multi_intent or (hard_reasoning and understanding.confidence < 0.90) or low_entity_signal or len(tokens) >= 28


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
    intent = str(parsed.get("intent") or "").strip().casefold()
    if intent not in _ALLOWED_INTENTS:
        intent = understanding.primary_intent
    relations = [item for item in _clean_list(parsed.get("relations")) if item in _ALLOWED_RELATIONS]
    constraints = [item for item in _clean_list(parsed.get("constraints")) if item in _ALLOWED_CONSTRAINTS]
    return {
        "intent": intent,
        "entities": _clean_list(parsed.get("entities")),
        "relations": relations,
        "constraints": constraints,
        "subquestions": _clean_list(parsed.get("subquestions")),
        "retrieval_terms": _clean_list(parsed.get("retrieval_terms")),
        "answer_strategy": re.sub(r"\s+", " ", str(parsed.get("answer_strategy") or "")).strip()[:400],
    }


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
    return replace(
        plan,
        intent=str(assist.get("intent") or plan.intent),
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
    return replace(
        understanding,
        intents=tuple(intents[:8]),
        primary_intent=intent or understanding.primary_intent,
        relations=tuple(relations[:6]),
        constraints=tuple(constraints[:8]),
        confidence=min(1.0, understanding.confidence + (0.08 if assist else 0.0)),
    )
