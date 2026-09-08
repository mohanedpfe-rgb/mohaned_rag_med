from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any


_COMPARISON = ("compare", "comparison", "difference", "differences", "versus", " vs ", "between", "différence", "comparaison", "مقارنة", "فرق", "بين")
_NUMERIC = ("dose", "dosage", "mg", "ml", "percent", "percentage", "how many", "how much", "value", "range", "عدد", "جرعة", "ملغ", "نسبة", "قيمة")
_DEFINITION = ("what is", "define", "definition", "meaning", "qu'est-ce", "définition", "ما هو", "ما هي", "تعريف")
_RELATION = ("related", "relationship", "relation", "associated", "association", "linked", "lien", "rapport", "علاقة", "مرتبط")
_NAV = ("where", "which page", "section", "page number", "citation", "source", "où", "quelle page", "أين", "أي صفحة")
_TABLE = ("table", "tables", "row", "column", "tabular", "tableau", "جدول", "صف", "عمود")
_FIGURE = ("figure", "fig.", "image", "images", "diagram", "chart", "graph", "illustration", "figure", "رسم", "شكل", "صورة", "مخطط")

_STOP = {"what", "does", "the", "and", "for", "with", "which", "from", "that", "this", "about", "have", "into", "dans", "avec", "pour", "les", "des", "est", "sont", "une", "sur", "ما", "ماذا", "كيف", "هل", "عن", "من", "هذا", "هذه"}


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
    text = re.sub(r"\s+", " ", (query or "").strip())
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\bwhat's\b", "what is", text, flags=re.I)
    text = re.sub(r"\bvs\.?\b", "versus", text, flags=re.I)
    return text[:3000]


def extract_query_entities(query: str) -> tuple[str, ...]:
    raw = normalize_query(query)
    tokens = re.findall(r"[\wÀ-ÿ'/-]{3,}", raw.casefold())
    out: list[str] = []
    # Preserve multi-word quoted terms and biomedical-ish token forms.
    for quoted in re.findall(r"[\"“]([^\"”]+)[\"”]", raw):
        q = re.sub(r"\s+", " ", quoted.strip())
        if q and q.casefold() not in out:
            out.append(q.casefold())
    for token in tokens:
        if token in _STOP or token in out or re.fullmatch(r"\d+", token):
            continue
        if len(token) >= 4 or any(ch.isdigit() for ch in token) or "/" in token:
            out.append(token)
        if len(out) >= 16:
            break
    return tuple(out)


def _split_top_level(query: str) -> list[str]:
    parts = [p.strip(" ?!;,.") for p in re.split(
        r"\?|;|\band\b|\bet\b|\bو\b|\balso\b|\bthen\b|\bwhile\b",
        query,
        flags=re.I,
    ) if p.strip()]
    return list(dict.fromkeys(parts))


def decompose_query(query: str) -> tuple[str, ...]:
    q = normalize_query(query)
    if not q:
        return ()
    pieces = _split_top_level(q)
    # Comparison questions frequently hide two entity retrieval targets in one sentence.
    if len(pieces) == 1 and re.search(r"\bversus\b|\bbetween\b", q, re.I):
        pieces = re.split(r"\bversus\b|\bbetween\b|\band\b", q, flags=re.I)
        pieces = [p.strip(" ?!;,.") for p in pieces if p.strip()]
    return tuple(dict.fromkeys(pieces))[:8]


def classify_intent(normalized: str, subqueries: tuple[str, ...]) -> str:
    low = normalized.casefold()
    if any(term in low for term in _COMPARISON):
        return "comparison"
    if any(term in low for term in _NUMERIC):
        return "numeric"
    if any(term in low for term in _DEFINITION):
        return "definition"
    if any(term in low for term in _RELATION):
        return "relationship"
    if any(term in low for term in _NAV):
        return "navigation"
    if any(term in low for term in _FIGURE):
        return "figure_lookup"
    if any(term in low for term in _TABLE):
        return "table_lookup"
    if len(subqueries) > 1:
        return "multi_part"
    return "factual"


def _make_variants(normalized: str, intent: str, entities: tuple[str, ...], *, numeric: bool, table: bool, figure: bool) -> tuple[str, ...]:
    variants = [normalized]
    if entities:
        variants.append(" ".join(entities[:8]))
    if numeric:
        variants.extend([normalized + " exact numeric value units range", normalized + " dosage measurement quantity"])
    if intent == "comparison":
        variants.extend([normalized + " differences similarities compare", normalized + " each item evidence"])
    if intent == "relationship":
        variants.extend([normalized + " association relationship linked evidence", normalized + " common mechanism connection"])
    if table:
        variants.append(normalized + " table rows columns values")
    if figure:
        variants.append(normalized + " figure diagram chart caption")
    return tuple(dict.fromkeys(v for v in variants if v))[:8]


def plan_query(query: str) -> QueryPlan:
    original = query or ""
    normalized = normalize_query(original)
    low = normalized.casefold()
    subqueries = decompose_query(normalized)
    entities = extract_query_entities(normalized)
    numeric = any(term in low for term in _NUMERIC)
    table = numeric or any(term in low for term in _TABLE)
    figure = any(term in low for term in _FIGURE)
    comparison = any(term in low for term in _COMPARISON)
    relation = any(term in low for term in _RELATION)
    navigation = any(term in low for term in _NAV)
    intent = classify_intent(normalized, subqueries)
    multi_hop = len(subqueries) > 1 or relation or comparison or any(k in low for k in ("why", "because", "lead to", "causes", "then", "result", "نتيجة", "سبب"))
    variants = _make_variants(normalized, intent, entities, numeric=numeric, table=table, figure=figure)
    boosts = {
        "lexical": 1.20 if numeric or navigation else 1.0,
        "vector": 1.10 if intent in {"definition", "relationship"} else 1.0,
        "table": 1.45 if table else 1.0,
        "figure": 1.35 if figure else 1.0,
        "section": 1.20 if navigation or intent == "definition" else 1.0,
        "entity": 1.25 if entities else 1.0,
        "parent": 1.15 if multi_hop or navigation else 1.0,
    }
    return QueryPlan(
        original=original,
        normalized=normalized,
        intent=intent,
        subqueries=subqueries,
        variants=variants,
        entities=entities,
        needs_numeric=numeric,
        needs_table=table,
        needs_figure=figure,
        needs_multi_hop=multi_hop,
        needs_neighborhood=True,
        score_boosts=boosts,
    )
