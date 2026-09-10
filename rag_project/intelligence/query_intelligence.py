from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

from rag_project.intelligence.semantic_reasoning import understand_query

_COMPARISON = ("compare", "comparison", "difference", "differences", "versus", " vs ", "between", "différence", "comparaison", "مقارنة", "فرق", "بين")
_NUMERIC = ("dose", "dosage", "mg", "ml", "percent", "percentage", "how many", "how much", "value", "range", "عدد", "جرعة", "ملغ", "نسبة", "قيمة")
_DEFINITION = ("what is", "define", "definition", "meaning", "qu'est-ce", "définition", "ما هو", "ما هي", "تعريف")
_RELATION = ("related", "relationship", "relation", "associated", "association", "linked", "lien", "rapport", "علاقة", "مرتبط")
_NAV = ("where", "which page", "section", "page number", "citation", "source", "où", "quelle page", "أين", "أي صفحة")
_TABLE = ("table", "tables", "row", "column", "tabular", "tableau", "جدول", "صف", "عمود")
_FIGURE = ("figure", "fig.", "image", "images", "diagram", "chart", "graph", "illustration", "رسم", "شكل", "صورة", "مخطط")
_STOP = {"what", "does", "the", "and", "for", "with", "which", "from", "that", "this", "about", "have", "into", "dans", "avec", "pour", "les", "des", "est", "sont", "une", "sur", "ما", "ماذا", "كيف", "هل", "عن", "من", "هذا", "هذه"}


def _contains_term(text: str, term: str) -> bool:
    haystack = (text or "").casefold(); needle = (term or "").casefold().strip()
    return bool(needle and re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack, flags=re.UNICODE))


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(_contains_term(text, term) for term in terms)


@dataclass(frozen=True)
class QueryPlan:
    original: str
    normalized: str
    intent: str
    subqueries: tuple[str, ...]
    variants: tuple[str, ...]
    entities: tuple[str, ...]
    needs_numeric: bool
    needs_table: bool
    needs_figure: bool
    needs_multi_hop: bool
    needs_neighborhood: bool
    score_boosts: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_query(query: str) -> str:
    raw = str(query or "").strip()
    text = re.sub(r"\s+", " ", raw).replace("–", "-").replace("—", "-")
    text = re.sub(r"\bwhat's\b", "what is", text, flags=re.I)
    text = re.sub(r"\bvs\.?\b", "versus", text, flags=re.I)
    if raw and raw[0].isupper() and text: text = text[0].upper() + text[1:]
    return text[:3000]


def extract_query_entities(query: str) -> tuple[str, ...]:
    understanding = understand_query(normalize_query(query))
    out: list[str] = []
    for entity in understanding.entities:
        term = str(entity.normalized or entity.text or "").strip().casefold()
        if term and term not in out: out.append(term)
    for quoted in re.findall(r"[\"“]([^\"”]+)[\"”]", query or ""):
        q = re.sub(r"\s+", " ", quoted.strip()).casefold()
        if q and q not in out: out.append(q)
    return tuple(out[:16])


def _split_top_level(query: str) -> list[str]:
    parts = [p.strip(" ?!;,." ) for p in re.split(r"\?|;|\band\b|\bet\b|\bو\b|\balso\b|\bthen\b|\bwhile\b|\bmais\b", query, flags=re.I) if p.strip()]
    return list(dict.fromkeys(parts))


def decompose_query(query: str) -> tuple[str, ...]:
    q = normalize_query(query)
    if not q: return ()
    pieces = _split_top_level(q)
    if len(pieces) == 1 and re.search(r"\bversus\b|\bbetween\b", q, re.I):
        pieces = re.split(r"\bversus\b|\bbetween\b|\band\b", q, flags=re.I)
        pieces = [p.strip(" ?!;,." ) for p in pieces if p.strip()]
    return tuple(dict.fromkeys(pieces))[:8]


def classify_intent(normalized: str, subqueries: tuple[str, ...]) -> str:
    semantic = understand_query(normalized)
    primary = semantic.primary_intent
    if primary == "association": return "relationship"
    if primary == "comparison" or _contains_any(normalized, _COMPARISON): return "comparison"
    if primary in {"diagnosis", "management", "etiology", "mechanism", "prognosis"}: return primary
    if _contains_any(normalized, _NUMERIC) or "numeric" in semantic.intents: return "numeric"
    if primary == "factual":
        if _contains_any(normalized, _TABLE): return "table_lookup"
        if _contains_any(normalized, _FIGURE): return "figure_lookup"
    if len(subqueries) > 1 and primary == "definition": return "multi_part"
    if primary == "definition": return "factual"
    return primary


def _make_variants(normalized: str, intent: str, entities: tuple[str, ...], *, numeric: bool, table: bool, figure: bool) -> tuple[str, ...]:
    variants = [normalized]
    if entities: variants.append(" ".join(entities[:8]))
    if intent in {"numeric", "diagnosis", "management"} or numeric:
        variants.extend([normalized + " exact numeric value units range", normalized + " dosage measurement quantity threshold"])
    if intent == "comparison": variants.extend([normalized + " differences similarities compare", normalized + " each item evidence"])
    if intent == "relationship": variants.extend([normalized + " association relationship linked evidence", normalized + " common mechanism connection"])
    if intent == "etiology": variants.extend([normalized + " causes risk factors etiology", normalized + " differential causes"])
    if intent == "mechanism": variants.extend([normalized + " mechanism physiopathology pathway", normalized + " how why biological process"])
    if intent == "diagnosis": variants.append(normalized + " diagnostic criteria signs tests thresholds")
    if intent == "management": variants.append(normalized + " treatment management first line contraindications")
    if intent == "prognosis": variants.append(normalized + " prognosis outcomes risk predictors follow-up")
    if table: variants.append(normalized + " table rows columns values")
    if figure: variants.append(normalized + " figure diagram chart caption")
    return tuple(dict.fromkeys(v for v in variants if v))[:10]


def plan_query(query: str, conversation_context: str = "") -> QueryPlan:
    original = query or ""; normalized = normalize_query(original); semantic = understand_query(normalized, conversation_context=conversation_context)
    subqueries = decompose_query(normalized); entities = extract_query_entities(normalized)
    numeric = "numeric" in semantic.intents or _contains_any(normalized, _NUMERIC)
    # A multi-part numeric request (for example dose + contraindications) benefits
    # from the table-oriented retrieval branch even when the user did not literally
    # say "table". This keeps the established planner contract backward compatible.
    table = "table_lookup" in semantic.intents or _contains_any(normalized, _TABLE) or (numeric and len(subqueries) > 1)
    figure = "figure_lookup" in semantic.intents or _contains_any(normalized, _FIGURE)
    relation = "relationship" in semantic.intents or "association" in semantic.intents or _contains_any(normalized, _RELATION)
    navigation = "navigation" in semantic.intents or _contains_any(normalized, _NAV)
    intent = classify_intent(normalized, subqueries)
    multi_hop = len(subqueries) > 1 or relation or intent in {"etiology", "mechanism", "diagnosis", "management", "prognosis"} or bool(semantic.relations) or _contains_any(normalized, ("why", "because", "lead to", "causes", "then", "result", "نتيجة", "سبب", "pourquoi"))
    variants = _make_variants(normalized, intent, entities, numeric=numeric, table=table, figure=figure)
    boosts = {"lexical": 1.20 if numeric or navigation else 1.0, "vector": 1.15 if intent in {"factual", "definition", "relationship", "etiology", "mechanism", "diagnosis", "management"} else 1.0, "table": 1.45 if table else 1.0, "figure": 1.35 if figure else 1.0, "section": 1.20 if navigation or intent in {"definition", "factual"} else 1.0, "entity": 1.25 if entities else 1.0, "parent": 1.15 if multi_hop or navigation else 1.0, "semantic": 1.30 if semantic.confidence >= 0.70 else 1.10}
    return QueryPlan(original, normalized, intent, subqueries, variants, entities, numeric, table, figure, multi_hop, True, boosts)
