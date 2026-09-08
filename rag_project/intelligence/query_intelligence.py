from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any


_COMPARISON = ("compare", "difference", "versus", " vs ", "between", "différence", "مقارنة", "فرق", "بين")
_NUMERIC = ("dose", "dosage", "mg", "ml", "percent", "percentage", "how many", "how much", "جرعة", "ملغ", "نسبة")
_DEFINITION = ("what is", "define", "definition", "meaning", "qu'est-ce", "définition", "ما هو", "ما هي")
_RELATION = ("related", "relationship", "relation", "associated", "lien", "rapport", "علاقة", "مرتبط")
_NAV = ("where", "which page", "section", "page number", "où", "quelle page", "أين", "أي صفحة")


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
    needs_multi_hop: bool
    needs_neighborhood: bool
    score_boosts: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_query(query: str) -> str:
    text = re.sub(r"\s+", " ", (query or "").strip())
    return text.replace("–", "-").replace("—", "-")


def extract_query_entities(query: str) -> tuple[str, ...]:
    tokens = re.findall(r"[\wÀ-ÿ']{3,}", (query or "").casefold())
    stop = {"what", "does", "the", "and", "for", "with", "which", "from", "dans", "avec", "pour", "les", "des", "est", "sont", "ما", "ماذا", "كيف", "هل", "عن", "من"}
    out: list[str] = []
    for token in tokens:
        if token in stop or token in out:
            continue
        out.append(token)
        if len(out) >= 12:
            break
    return tuple(out)


def decompose_query(query: str) -> tuple[str, ...]:
    q = normalize_query(query)
    if not q:
        return ()
    pieces = [p.strip(" ?!;,.") for p in re.split(r"\?|;|\band\b|\bet\b|\bو\b", q, flags=re.I) if p.strip()]
    if len(pieces) <= 1:
        return (q,)
    return tuple(dict.fromkeys(pieces))[:6]


def plan_query(query: str) -> QueryPlan:
    normalized = normalize_query(query)
    low = normalized.casefold()
    subqueries = decompose_query(normalized)
    entities = extract_query_entities(normalized)
    numeric = any(term in low for term in _NUMERIC)
    table = numeric or any(term in low for term in ("table", "tabular", "row", "column", "جدول", "tableau"))
    comparison = any(term in low for term in _COMPARISON)
    definition = any(term in low for term in _DEFINITION)
    relation = any(term in low for term in _RELATION)
    navigation = any(term in low for term in _NAV)
    if comparison:
        intent = "comparison"
    elif numeric:
        intent = "numeric"
    elif definition:
        intent = "definition"
    elif relation:
        intent = "relationship"
    elif navigation:
        intent = "navigation"
    elif len(subqueries) > 1:
        intent = "multi_part"
    else:
        intent = "factual"
    variants = [normalized]
    if entities:
        variants.append(" ".join(entities))
    if numeric:
        variants.append(normalized + " exact value units dose dosage")
    if comparison:
        variants.append(normalized + " differences similarities compared")
    if relation:
        variants.append(normalized + " association correlation relationship")
    variants = tuple(dict.fromkeys(v for v in variants if v))[:5]
    boosts = {
        "lexical": 1.15 if numeric or navigation else 1.0,
        "vector": 1.10 if definition or relation else 1.0,
        "table": 1.35 if table else 1.0,
        "section": 1.20 if navigation or definition else 1.0,
        "entity": 1.25 if entities else 1.0,
    }
    return QueryPlan(
        original=query or "",
        normalized=normalized,
        intent=intent,
        subqueries=subqueries,
        variants=variants,
        entities=entities,
        needs_numeric=numeric,
        needs_table=table,
        needs_multi_hop=len(subqueries) > 1 or relation or comparison,
        needs_neighborhood=True,
        score_boosts=boosts,
    )
