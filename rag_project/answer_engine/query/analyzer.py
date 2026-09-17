"""Query analyzer for deterministic medical answering."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import Counter

from rag_project.utils.text_utils import meaningful_tokens, detect_language


@dataclass(frozen=True)
class QueryPlan:
    """Complete analysis of a medical question."""
    question: str
    normalized: str
    intent: str
    subquestions: List[str]
    entities: List[str]
    numeric_values: List[Dict[str, Any]]
    is_follow_up: bool
    confidence: float
    language: str
    question_type: str
    requires_multi_hop: bool
    requires_table: bool
    requires_figure: bool
    temporal_signs: List[str]
    comparison_signs: List[str]


def _extract_numeric_values(text: str) -> List[Dict[str, Any]]:
    """Extract numeric values from text with units."""
    patterns = [
        (r"(\d+(?:\.\d+)?)\s*(mg|mcg|µg|g|kg|ml|l|mmhg|mmol/l|%)", "dosage"),
        (r"(\d+(?:\.\d+)?)\s*(mg/ml|mcg/ml)", "concentration"),
        (r"(\d+(?:\.\d+)?)\s*(bpm|°c|°f)", "vital"),
        (r"(\d+(?:\.\d+)?)\s*(years?|months?|weeks?|days?|hours?)", "duration"),
        (r"(\d+(?:\.\d+)?-\d+(?:\.\d+)?)\s*(mg|g|ml)", "range"),
    ]
    results = []
    for pattern, unit_type in patterns:
        for match in re.finditer(pattern, text, re.I):
            value_str = match.group(1)
            unit = match.group(2).lower()
            try:
                value = float(value_str.replace(",", "."))
                results.append({
                    "value": value,
                    "unit": unit,
                    "unit_type": unit_type,
                    "original": match.group(0),
                })
            except ValueError:
                continue
    return results


def _is_follow_up(question: str) -> bool:
    """Detect if question is a follow-up."""
    patterns = [
        r"^(what about|how about|and|also|then|et|puis|و|ثم)",
        r"\b(it|this|that|they|them|those|these)\b",
        r"what\s+about\s+\w+",
        r"and\s+about",
    ]
    lowered = question.lower()
    return any(re.search(p, lowered, re.I) for p in patterns)


def _detect_comparison(question: str) -> List[str]:
    """Detect comparison indicators in question."""
    indicators = [
        "vs", "versus", "versus", "compared with", "compare",
        "difference", "differences", "difference between",
        "better", "worse", "higher", "lower", "greater", "less",
        "more", "less", "most", "least", "highest", "lowest",
    ]
    lowered = question.lower()
    return [ind for ind in indicators if ind in lowered]


def _detect_temporal(question: str) -> List[str]:
    """Detect temporal indicators in question."""
    indicators = [
        "before", "after", "during", "initially", "subsequently",
        "first", "second", "then", "later", "currently", "now",
        "previously", "formerly", "formerly", "recent", "latest",
        "new", "old", "previous", "next", "duration", "interval",
    ]
    lowered = question.lower()
    return [ind for ind in indicators if ind in lowered]


def analyze_query(question: str, context: str = "") -> QueryPlan:
    """Perform comprehensive query analysis."""
    if not question:
        question = ""
    
    # Normalize question
    normalized = re.sub(r"\s+", " ", str(question).strip())
    language = detect_language(normalized)
    
    # Extract tokens and entities
    tokens = meaningful_tokens(normalized)
    
    # Extract numeric values
    numeric_values = _extract_numeric_values(normalized)
    
    # Detect question characteristics
    is_follow = _is_follow_up(normalized)
    comparisons = _detect_comparison(normalized)
    temporal = _detect_temporal(normalized)
    
    # Determine intent
    intent = _classify_intent(normalized, tokens, comparisons, temporal)
    
    # Detect question type
    question_type = _determine_question_type(normalized, intent, numeric_values, comparisons)
    
    # Determine multi-hop requirement
    requires_multi_hop = (
        intent in {"multi_hop", "causal", "mechanism", "comparison"} or
        "and" in normalized.lower() or
        len(temporal) > 1 or
        "then" in normalized.lower() or
        "result" in normalized.lower()
    )
    
    # Detect table/figure requirements
    requires_table = any(term in normalized.lower() for term in ("table", "chart", "list", "values", "rows", "columns"))
    requires_figure = any(term in normalized.lower() for term in ("figure", "diagram", "image", "photo", "scheme"))
    
    # Extract subquestions for multi-part questions
    subquestions = _decompose_question(normalized)
    
    # Estimate confidence
    confidence = _estimate_confidence(normalized, tokens, numeric_values)
    
    return QueryPlan(
        question=question,
        normalized=normalized,
        intent=intent,
        subquestions=subquestions,
        entities=tokens[:20],
        numeric_values=numeric_values,
        is_follow_up=is_follow,
        confidence=confidence,
        language=language,
        question_type=question_type,
        requires_multi_hop=requires_multi_hop,
        requires_table=requires_table,
        requires_figure=requires_figure,
        temporal_signs=temporal,
        comparison_signs=comparisons,
    )


def _classify_intent(normalized: str, tokens: List[str], comparisons: List[str], temporal: List[str]) -> str:
    """Classify the intent of the question."""
    lowered = normalized.lower()
    
    if any(term in lowered for term in ("what is", "define", "definition")):
        return "definition"
    if any(term in lowered for term in ("how much", "how many", "dosage", "dose", "mg", "ml")):
        return "numeric"
    if any(term in lowered for term in ("compare", "vs", "versus", "difference")):
        return "comparison"
    if any(term in lowered for term in ("cause", "causes", "caused by", "risk factor")):
        return "causal"
    if any(term in lowered for term in ("mechanism", "pathway", "pathophysiology", "how it works")):
        return "mechanism"
    if any(term in lowered for term in ("treat", "treatment", "therapy", "management", "drug")):
        return "management"
    if any(term in lowered for term in ("diagnose", "diagnosis", "diagnostic", "test", "criteria")):
        return "diagnostic"
    if any(term in lowered for term in ("prognosis", "outcome", "survival", "prognostic")):
        return "prognosis"
    if any(term in lowered for term in ("why", "reason", "because")):
        return "causal"
    if any(term in lowered for term in ("list", "all", "enumerate")):
        return "list"
    if any(term in lowered for term in ("relationship", "association", "linked", "correlat")):
        return "relationship"
    
    return "factual"


def _determine_question_type(normalized: str, intent: str, numeric_values: List[Dict], comparisons: List[str]) -> str:
    """Determine specific question type."""
    if comparisons:
        return "comparison"
    if intent == "numeric" and numeric_values:
        return "numeric_value"
    if intent == "numeric":
        return "dosage"
    if intent == "definition":
        return "definition"
    if intent == "list":
        return "list"
    if intent == "management":
        return "management"
    if intent == "diagnostic":
        return "diagnostic_criteria"
    if intent == "causal":
        return "causal"
    if intent == "mechanism":
        return "mechanism"
    return "fact"


def _decompose_question(normalized: str) -> List[str]:
    """Decompose multi-part questions into subquestions."""
    parts = re.split(r"\s+(?:and|ou|w|puis|then)\s+", normalized, flags=re.I)
    if len(parts) > 1:
        return [p.strip() for p in parts if len(p.strip()) > 10]
    
    parts = re.split(r"\s*;\s*", normalized)
    if len(parts) > 1:
        return [p.strip() for p in parts if len(p.strip()) > 10]
    
    return [normalized]


def _estimate_confidence(normalized: str, tokens: List[str], numeric_values: List[Dict]) -> float:
    """Estimate confidence in question clarity."""
    if not normalized or len(normalized) < 5:
        return 0.3
    
    confidence = 0.5
    
    if normalized.endswith("?"):
        confidence += 0.1
    
    if numeric_values:
        confidence += 0.1
    
    if len(tokens) >= 3:
        confidence += 0.1
    
    vague_words = ["thing", "stuff", "it", "this", "that", "what"]
    if any(w in normalized.lower() for w in vague_words):
        confidence -= 0.2
    
    return max(0.0, min(1.0, confidence))
