"""Document-aware, query-routed retrieval engine for production BookRAG.

The engine is deliberately deterministic-first: it can improve retrieval quality
without requiring a larger local LLM, and every derived decision is exposed in a
machine-readable trace for evaluation and debugging.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from rag_project.intelligence.query_intelligence import plan_query
from rag_project.utils.text_utils import keyword_overlap_score, meaningful_tokens


@dataclass(frozen=True)
class QueryRoute:
    kind: str
    scope: str
    needs_table: bool = False
    needs_numeric: bool = False
    needs_figure: bool = False
    multi_hop: bool = False
    summary: bool = False
    comparison: bool = False
    exact_lookup: bool = False
    expected_slots: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    confidence: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalDecision:
    query: str
    route: QueryRoute
    queries: list[str] = field(default_factory=list)
    candidates: int = 0
    final_hits: int = 0
    rerank_ms: float = 0.0
    retrieval_ms: float = 0.0
    self_corrections: int = 0
    strategy: list[str] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    contradiction: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["route"] = self.route.to_dict()
        return data


_MEDICAL_SYNONYMS = {
    "mi": ("myocardial infarction", "heart attack"),
    "htn": ("hypertension", "high blood pressure"),
    "dm": ("diabetes mellitus", "diabetes"),
    "copd": ("chronic obstructive pulmonary disease",),
    "ckd": ("chronic kidney disease",),
    "aki": ("acute kidney injury",),
    "cva": ("stroke", "cerebrovascular accident"),
    "pe": ("pulmonary embolism",),
    "dvt": ("deep vein thrombosis",),
    "acs": ("acute coronary syndrome",),
    "af": ("atrial fibrillation",),
    "ecg": ("electrocardiogram",),
    "tsh": ("thyroid stimulating hormone", "thyrotropin"),
    "hba1c": ("hemoglobin a1c", "glycated hemoglobin"),
}

_SUMMARY = re.compile(r"\b(summary|summarize|summarise|overview|main findings|key findings|main points|key points)\b|résumé|synthèse|aperçu|points principaux|résultats principaux|ملخص|خلاصة|نظرة عامة|النقاط الرئيسية", re.I | re.UNICODE)
_COMPARE = re.compile(r"\b(compare|comparison|difference|differences|versus|vs\.?|contrast|distinguish)\b|comparer|différence|différences|مقارنة|الفرق", re.I | re.UNICODE)
_TABLE = re.compile(r"\b(table|tabular|criteria|score|classification|staging|reference range|normal range|cut[- ]off|threshold|criteria)\b|tableau|critères|classification|stade|plage normale|جدول|معايير|تصنيف", re.I | re.UNICODE)
_NUMERIC = re.compile(r"\b(value|values|number|percent|percentage|rate|range|dose|duration|time|age|years?|days?|hours?|mg|mcg|g/dl|mmhg|bpm|mmol|mol|cm|kg)\b|valeur|pourcentage|dose|durée|âge|سنة|جرعة|نسبة", re.I | re.UNICODE)
_MULTI = re.compile(r"\b(and|also|as well as|respectively|first.*then|cause.*effect)\b|et|ainsi que|puis|ثم", re.I | re.UNICODE)
_EXACT = re.compile(r"\b(what is|define|definition|who is|which drug|which test|where is|page|exact|specifically)\b|définir|définition|quel médicament|quel test|exactement|ما هو|تعريف", re.I | re.UNICODE)


def _clean(value: Any, limit: int = 3500) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _entities_from_plan(question: str) -> tuple[str, ...]:
    try:
        plan = plan_query(question, conversation_context="")
        raw = getattr(plan, "entities", ()) or ()
    except Exception:
        raw = ()
    values = []
    for value in raw:
        text = _clean(value, 160)
        if text and text.casefold() not in {v.casefold() for v in values}:
            values.append(text)
    return tuple(values[:12])


def route_query(question: str, planner: Any | None = None) -> QueryRoute:
    q = _clean(question)
    entities = _entities_from_plan(q) if planner is None else tuple(_clean(x, 160) for x in (getattr(planner, "entities", ()) or ()) if _clean(x, 160))[:12]
    summary = bool(_SUMMARY.search(q))
    comparison = bool(_COMPARE.search(q))
    needs_table = bool(_TABLE.search(q))
    needs_numeric = bool(_NUMERIC.search(q)) or bool(re.search(r"\d", q))
    multi_hop = bool(_MULTI.search(q)) or comparison or len(entities) >= 2
    exact = bool(_EXACT.search(q)) or len(meaningful_tokens(q)) <= 5

    if summary:
        kind = "summary"
        slots = ("overview", "key_points", "supporting_details")
    elif comparison:
        kind = "comparison"
        slots = ("definition", "entity_a", "entity_b", "differences", "similarities")
    elif needs_table and needs_numeric:
        kind = "table_numeric"
        slots = ("target_value", "unit_or_condition", "context")
    elif needs_table:
        kind = "table_lookup"
        slots = ("criteria_or_rows", "context")
    elif multi_hop:
        kind = "multi_hop"
        slots = ("premise", "relationship", "conclusion")
    elif exact:
        kind = "exact_fact"
        slots = ("direct_answer", "supporting_context")
    else:
        kind = "factual"
        slots = ("answer", "supporting_context")

    confidence = 0.92 if summary or comparison else 0.86 if needs_table or multi_hop else 0.82
    scope = "document" if summary else "question"
    return QueryRoute(
        kind=kind,
        scope=scope,
        needs_table=needs_table,
        needs_numeric=needs_numeric,
        needs_figure=False,
        multi_hop=multi_hop,
        summary=summary,
        comparison=comparison,
        exact_lookup=exact,
        expected_slots=slots,
        entities=entities,
        confidence=confidence,
    )


def expand_query_variants(question: str, route: QueryRoute, base_plan: Any | None = None) -> list[str]:
    q = _clean(question)
    values: list[str] = [q]
    if base_plan is None:
        try:
            base_plan = plan_query(q, conversation_context="")
        except Exception:
            base_plan = None
    if base_plan is not None:
        values.extend(_clean(x) for x in (getattr(base_plan, "variants", ()) or ()))
        values.extend(_clean(x) for x in (getattr(base_plan, "subqueries", ()) or ()))
    if route.kind == "summary":
        values += [f"{q} overview", f"{q} introduction", f"{q} key points", f"{q} conclusion", f"{q} chapter sections"]
    elif route.kind == "comparison" and len(route.entities) >= 2:
        a, b = route.entities[:2]
        values += [f"{a} {b} differences", f"{a} {b} comparison", f"{a} versus {b}"]
    elif route.kind in {"table_lookup", "table_numeric"}:
        values += [f"{q} table", f"{q} criteria", f"{q} reference range"]
    elif route.kind == "exact_fact":
        values += [q, f"definition {q}", f"exact {q}"]
    for token in meaningful_tokens(q):
        synonyms = _MEDICAL_SYNONYMS.get(token.casefold())
        if synonyms:
            for synonym in synonyms:
                values.append(re.sub(re.escape(token), synonym, q, flags=re.I))
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = _clean(value)
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            unique.append(value)
    return unique[:16]


def _page(meta: dict[str, Any]) -> str:
    value = meta.get("page_numbers") or meta.get("page_number") or meta.get("page") or "?"
    if isinstance(value, (list, tuple)):
        return str(value[0] if value else "?")
    return str(value)


def _type_bonus(meta: dict[str, Any], route: QueryRoute) -> float:
    types = {str(x).casefold() for x in (meta.get("evidence_types") or ())}
    page_type = str(meta.get("page_type") or "").casefold()
    bonus = 0.0
    if route.needs_table and ("table" in types or meta.get("table_id") or "table" in page_type):
        bonus += 0.22
    if route.needs_figure and ("figure" in types or meta.get("figure_id") or "figure" in page_type):
        bonus += 0.16
    if meta.get("section") or meta.get("chapter"):
        bonus += 0.04
    if meta.get("quality_score") is not None:
        try:
            bonus += 0.04 * max(0.0, min(1.0, float(meta.get("quality_score"))))
        except (TypeError, ValueError):
            pass
    return bonus


def _semantic_score(hit: Any) -> float:
    return max(0.0, min(1.0, float(getattr(hit, "vector_score", 0.0) or getattr(hit, "score", 0.0) or 0.0)))


def rerank_hits(question: str, hits: Iterable[Any], route: QueryRoute) -> list[Any]:
    q = " ".join(meaningful_tokens(question))
    q_tokens = set(meaningful_tokens(question))
    rows: list[tuple[float, Any]] = []
    for hit in hits:
        text = str(getattr(hit, "text", "") or "")
        meta = dict(getattr(hit, "metadata", {}) or {})
        semantic = _semantic_score(hit)
        lexical = max(0.0, min(1.0, float(getattr(hit, "lexical_score", 0.0) or keyword_overlap_score(q, text) or 0.0)))
        exact = 1.0 if q.casefold() in text.casefold() and q else 0.0
        token_overlap = len(q_tokens & set(meaningful_tokens(text))) / max(1, len(q_tokens)) if q_tokens else 0.0
        section_text = " ".join(str(meta.get(k) or "") for k in ("chapter", "section", "subsection", "headings"))
        section_overlap = keyword_overlap_score(q, section_text) if section_text else 0.0
        score = 0.34 * semantic + 0.25 * lexical + 0.17 * token_overlap + 0.10 * exact + 0.10 * section_overlap + _type_bonus(meta, route)
        score = max(0.0, min(1.5, score))
        try:
            hit.score = score
        except Exception:
            pass
        rows.append((score, hit))
    rows.sort(key=lambda x: x[0], reverse=True)
    return [hit for _, hit in rows]


def diversify_hits(hits: list[Any], route: QueryRoute, limit: int) -> list[Any]:
    limit = max(1, int(limit))
    selected: list[Any] = []
    seen_keys: set[str] = set()
    seen_pages: set[str] = set()
    seen_sections: set[str] = set()
    for hit in hits:
        meta = dict(getattr(hit, "metadata", {}) or {})
        doc = str(meta.get("document_id") or getattr(hit, "doc_id", ""))
        page = _page(meta)
        section = str(meta.get("section_id") or meta.get("section") or "")
        key = str(meta.get("chunk_id") or f"{doc}:{page}:{str(getattr(hit, 'text', ''))[:80]}")
        if key in seen_keys:
            continue
        if route.summary and len(selected) < min(limit, 16):
            # For a summary, maximize document/section coverage before duplicates.
            if f"{doc}:{page}" in seen_pages:
                continue
            seen_pages.add(f"{doc}:{page}")
        elif route.kind == "comparison" and len(selected) < limit:
            # Prefer evidence from both sides of the comparison before repeats.
            entity_text = str(getattr(hit, "text", "")) + " " + section
            covered = sum(1 for entity in route.entities if entity.casefold() in entity_text.casefold())
            if covered == 0 and selected:
                continue
        elif section and section in seen_sections and len(selected) < limit // 2:
            continue
        seen_keys.add(key)
        if section:
            seen_sections.add(section)
        selected.append(hit)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        for hit in hits:
            if hit not in selected:
                selected.append(hit)
                if len(selected) >= limit:
                    break
    return selected


def evidence_coverage(question: str, route: QueryRoute, hits: list[Any]) -> dict[str, Any]:
    texts = [str(getattr(h, "text", "") or "") for h in hits]
    combined = " ".join(texts).casefold()
    slots = list(route.expected_slots)
    coverage: dict[str, float] = {}
    for slot in slots:
        if slot in {"overview", "answer", "direct_answer", "supporting_context"}:
            tokens = set(meaningful_tokens(question)) - {"what", "are", "the", "is", "this", "that", "about", "how", "why"}
        elif slot in {"entity_a", "entity_b", "differences", "similarities"}:
            idx = 0 if slot == "entity_a" else 1 if slot == "entity_b" else None
            tokens = set(meaningful_tokens(route.entities[idx])) if idx is not None and idx < len(route.entities) else set(meaningful_tokens(question))
        elif slot in {"target_value", "unit_or_condition", "criteria_or_rows"}:
            tokens = set(meaningful_tokens(question)) | set(re.findall(r"\b\d+(?:\.\d+)?\b", question))
        else:
            tokens = set(meaningful_tokens(question))
        tokens = {t for t in tokens if len(t) > 2}
        matched = sum(1 for token in tokens if token in combined)
        coverage[slot] = min(1.0, matched / max(1, len(tokens)))
    overall = sum(coverage.values()) / max(1, len(coverage))
    entity_coverage = min(1.0, sum(1 for entity in route.entities if entity.casefold() in combined) / max(1, len(route.entities)))
    return {"slots": coverage, "overall": round(overall, 4), "entity_coverage": round(entity_coverage, 4), "sufficient": overall >= 0.32 or (route.summary and overall >= 0.20)}


def detect_contradictions(hits: list[Any]) -> dict[str, Any]:
    """Detect obvious numeric disagreements across sources without pretending to solve clinical semantics."""
    groups: dict[str, list[tuple[str, str, str]]] = {}
    number_pattern = re.compile(r"(?<!\w)(\d+(?:\.\d+)?\s*(?:%|mg|mcg|g|kg|mmHg|bpm|years?|days?|hours?|mmol/L|mg/dL)?)(?!\w)", re.I)
    for hit in hits:
        meta = dict(getattr(hit, "metadata", {}) or {})
        text = str(getattr(hit, "text", "") or "")
        anchors = set(meaningful_tokens(text))
        if not anchors:
            continue
        for number in number_pattern.findall(text):
            anchor = " ".join(sorted(list(anchors)[:8]))
            groups.setdefault(anchor, []).append((number.casefold(), _page(meta), str(meta.get("document_id") or getattr(hit, "doc_id", ""))))
    conflicts = []
    for anchor, values in groups.items():
        unique = {value[0] for value in values}
        if len(unique) > 1 and len(values) >= 2:
            conflicts.append({"anchor": anchor, "values": sorted(unique), "sources": [{"value": v, "page": p, "document_id": d} for v, p, d in values[:8]]})
    return {"has_contradiction": bool(conflicts), "conflicts": conflicts[:12], "count": len(conflicts)}


def _add_parent_context(selected: list[Any], all_hits: list[Any], limit: int) -> list[Any]:
    by_parent: dict[str, Any] = {}
    for hit in all_hits:
        meta = dict(getattr(hit, "metadata", {}) or {})
        parent = str(meta.get("parent_id") or meta.get("section_id") or "")
        if parent and parent not in by_parent:
            by_parent[parent] = hit
    output: list[Any] = []
    for hit in selected:
        output.append(hit)
        meta = dict(getattr(hit, "metadata", {}) or {})
        parent = str(meta.get("parent_id") or meta.get("section_id") or "")
        if parent in by_parent and by_parent[parent] not in output:
            output.append(by_parent[parent])
        if len(output) >= limit:
            break
    return output[:limit]


def retrieve_document_aware(system: Any, question: str, metadata_filter: dict[str, Any] | None = None, *, top_k: int = 8) -> tuple[list[Any], dict[str, Any]]:
    started = time.perf_counter()
    route = route_query(question)
    try:
        plan = plan_query(question, conversation_context="")
    except Exception:
        plan = None
    queries = expand_query_variants(question, route, plan)[:16]
    strategy = [route.kind, "semantic+lexical", "medical_synonym_expansion"]
    if route.summary:
        strategy += ["document_coverage", "section_diversification", "parent_context"]
    if route.needs_table:
        strategy += ["table_priority"]
    if route.needs_numeric:
        strategy += ["numeric_exactness"]
    if route.multi_hop:
        strategy += ["decomposed_queries"]

    candidates: dict[str, Any] = {}
    attempts = 0
    for query in queries:
        try:
            raw = system.retriever.retrieve(query, top_k=max(24, min(96, int(top_k) * 6)), where=_metadata_filter(metadata_filter))
        except Exception:
            raw = []
        attempts += 1
        for hit in raw or ():
            if hit is None or not str(getattr(hit, "text", "") or "").strip():
                continue
            meta = dict(getattr(hit, "metadata", {}) or {})
            key = str(meta.get("chunk_id") or f"{getattr(hit, 'doc_id', '')}:{_page(meta)}:{str(getattr(hit, 'text', ''))[:120]}")
            existing = candidates.get(key)
            if existing is None or _semantic_score(hit) > _semantic_score(existing):
                candidates[key] = hit

    rerank_start = time.perf_counter()
    ranked = rerank_hits(question, list(candidates.values()), route)
    selected = diversify_hits(ranked, route, max(int(top_k) * 3, 16))
    selected = _add_parent_context(selected, ranked, max(int(top_k) * 3, 16))
    correction_count = 0

    coverage = evidence_coverage(question, route, selected)
    # Self-correct once with stronger lexical/structural queries when evidence is weak.
    if not coverage["sufficient"] and attempts < 16:
        correction_count = 1
        correction_queries = list(queries[-4:]) + [f"{question} section", f"{question} clinical findings", f"{question} definition"]
        for query in correction_queries:
            try:
                raw = system.retriever.retrieve(query, top_k=max(32, int(top_k) * 8), where=_metadata_filter(metadata_filter))
            except Exception:
                raw = []
            for hit in raw or ():
                if hit is None or not str(getattr(hit, "text", "") or "").strip():
                    continue
                meta = dict(getattr(hit, "metadata", {}) or {})
                key = str(meta.get("chunk_id") or f"{getattr(hit, 'doc_id', '')}:{_page(meta)}:{str(getattr(hit, 'text', ''))[:120]}")
                if key not in candidates:
                    candidates[key] = hit
        ranked = rerank_hits(question, list(candidates.values()), route)
        selected = _add_parent_context(diversify_hits(ranked, route, max(int(top_k) * 3, 16)), ranked, max(int(top_k) * 3, 16))
        coverage = evidence_coverage(question, route, selected)
        strategy.append("self_correction")

    contradiction = detect_contradictions(selected)
    elapsed_ms = (time.perf_counter() - started) * 1000
    decision = RetrievalDecision(
        query=_clean(question),
        route=route,
        queries=queries,
        candidates=len(candidates),
        final_hits=len(selected),
        rerank_ms=round((time.perf_counter() - rerank_start) * 1000, 2),
        retrieval_ms=round(elapsed_ms, 2),
        self_corrections=correction_count,
        strategy=strategy,
        coverage=coverage,
        contradiction=contradiction,
    )
    return selected, decision.to_dict()


def _metadata_filter(metadata_filter: dict[str, Any] | None) -> Any:
    try:
        from rag_project.retrieval.metadata_filter import MetadataFilter
        return MetadataFilter.build(metadata_filter)
    except Exception:
        return None


def build_answer_plan(question: str, route: QueryRoute, coverage: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_type": route.kind,
        "scope": route.scope,
        "required_slots": list(route.expected_slots),
        "entities": list(route.entities),
        "covered_slots": [slot for slot, value in (coverage.get("slots") or {}).items() if float(value) >= 0.35],
        "missing_slots": [slot for slot, value in (coverage.get("slots") or {}).items() if float(value) < 0.35],
        "coverage": float(coverage.get("overall", 0.0) or 0.0),
        "generation_rule": "evidence_only",
    }


__all__ = [
    "QueryRoute",
    "RetrievalDecision",
    "route_query",
    "expand_query_variants",
    "rerank_hits",
    "diversify_hits",
    "evidence_coverage",
    "detect_contradictions",
    "retrieve_document_aware",
    "build_answer_plan",
]
