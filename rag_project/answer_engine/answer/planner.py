"""Answer planner for deterministic medical answering."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class AnswerStructure(str, Enum):
    """Types of answer structures."""
    SIMPLE_STATEMENT = "simple_statement"
    PARAGRAPH = "paragraph"
    BULLETED_LIST = "bulleted_list"
    NUMBERED_LIST = "numbered_list"
    TABLE_FORMAT = "table_format"
    COMPARISON_TABLE = "comparison_table"
    DEFINITION_FORMAT = "definition_format"
    STEP_BY_STEP = "step_by_step"
    CAUSAL_CHAIN = "causal_chain"
    PROBLEM_SOLUTION = "problem_solution"


@dataclass(frozen=True)
class AnswerPlan:
    """Complete plan for generating a deterministic answer."""
    structure: AnswerStructure
    sections: List[Dict[str, Any]]
    emphasis: List[str]  # sections to emphasize
    citations_needed: bool
    numeric_format: str  # decimal, fraction, range
    language_style: str  # formal, clinical, patient
    depth: str  # brief, moderate, comprehensive
    examples: bool
    warnings: bool
    references: List[str]


@dataclass(frozen=True)
class SectionBuilder:
    """Configuration for building an answer section."""
    title: str
    content_type: str  # text, list, table, definition
    required_claims: List[str]  # claim IDs required for this section
    optional_claims: List[str]  # claim IDs that could enhance this section
    source_requirements: List[str]  # source types required
    max_length: int  # max characters
    formatting: str  # bullet, number, paragraph, definition


def plan_answer(
    question: str,
    answer_type: str,
    claims: List[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
    intent: str,
    language: str = "en",
) -> AnswerPlan:
    """Create an answer plan based on question and claims."""
    
    # Determine structure based on answer type
    structure = _determine_structure(answer_type, intent, claims)
    
    # Build sections
    sections = _build_sections(answer_type, intent, claims, evidence)
    
    # Identify emphasis areas
    emphasis = _identify_emphasis(claims)
    
    # Determine formatting preferences
    citations_needed = len(evidence) > 2
    numeric_format = _determine_numeric_format(claims)
    language_style = "clinical" if intent in {"diagnostic", "treatment", "management"} else "formal"
    depth = _determine_depth(question, claims, evidence)
    
    # Examples and warnings
    examples = intent in {"definition", "management", "treatment", "diagnostic_criteria"}
    warnings = intent in {"treatment", "contraindication", "adverse_effect"}
    
    # Build references
    references = [e.get("source_id", "") for e in evidence if e.get("source_id")]
    
    return AnswerPlan(
        structure=structure,
        sections=sections,
        emphasis=emphasis,
        citations_needed=citations_needed,
        numeric_format=numeric_format,
        language_style=language_style,
        depth=depth,
        examples=examples,
        warnings=warnings,
        references=list(set(references)),
    )


def _determine_structure(answer_type: str, intent: str, claims: List[Dict]) -> AnswerStructure:
    """Determine the best answer structure."""
    
    # Multi-part questions need structured lists
    if len(claims) > 3:
        return AnswerStructure.BULLETED_LIST
    
    # Comparison questions need tables
    if answer_type == "comparison" or intent == "comparison":
        return AnswerStructure.COMPARISON_TABLE
    
    # Numeric answers
    if answer_type == "numeric" or answer_type == "dosage":
        if len(claims) == 1:
            return AnswerStructure.SIMPLE_STATEMENT
        return AnswerStructure.TABLE_FORMAT
    
    # Definition questions
    if answer_type == "definition":
        return AnswerStructure.DEFINITION_FORMAT
    
    # Management/treatment questions
    if intent in {"management", "treatment", "prevention"}:
        return AnswerStructure.NUMBERED_LIST
    
    # Causal/mechanism questions
    if intent in {"causal", "mechanism", "pathophysiology"}:
        return AnswerStructure.CAUSAL_CHAIN
    
    # Default to paragraph
    return AnswerStructure.PARAGRAPH


def _build_sections(
    answer_type: str,
    intent: str,
    claims: List[Dict],
    evidence: List[Dict],
) -> List[Dict[str, Any]]:
    """Build answer sections from claims."""
    
    sections = []
    seen_sections = set()
    
    # Group claims by category
    definition_claims = []
    numeric_claims = []
    list_claims = []
    relationship_claims = []
    
    for claim in claims:
        claim_type = claim.get("claim_type", "")
        status = claim.get("status", "")
        
        if status not in {"SUPPORTED", "PARTIALLY_SUPPORTED"}:
            continue
        
        if claim_type == "definition":
            definition_claims.append(claim)
        elif "numeric" in claim_type or claim.get("numeric_value") is not None:
            numeric_claims.append(claim)
        elif claim_type in {"list", "enumeration"}:
            list_claims.append(claim)
        else:
            relationship_claims.append(claim)
    
    # Build section for definitions
    if definition_claims and "definition" not in seen_sections:
        sections.append({
            "title": "Definition",
            "type": "definition",
            "claims": [c["id"] for c in definition_claims],
            "evidence_needed": [e["id"] for e in evidence if e.get("semantic_relevance", 0) > 0.5],
        })
        seen_sections.add("definition")
    
    # Build section for numeric values
    if numeric_claims and "numeric" not in seen_sections:
        sections.append({
            "title": "Values",
            "type": "numeric",
            "claims": [c["id"] for c in numeric_claims],
            "evidence_needed": [e["id"] for e in evidence if e.get("numeric_match")],
        })
        seen_sections.add("numeric")
    
    # Build section for lists
    if list_claims and "list" not in seen_sections:
        sections.append({
            "title": "Details",
            "type": "list",
            "claims": [c["id"] for c in list_claims],
            "evidence_needed": [e["id"] for e in evidence],
        })
        seen_sections.add("list")
    
    # Build section for relationships
    if relationship_claims and "relationships" not in seen_sections:
        sections.append({
            "title": "Key Information",
            "type": "text",
            "claims": [c["id"] for c in relationship_claims],
            "evidence_needed": [e["id"] for e in evidence],
        })
        seen_sections.add("relationships")
    
    # Add summary section if multiple sections
    if len(sections) > 1:
        sections.append({
            "title": "Summary",
            "type": "summary",
            "claims": [],
            "evidence_needed": [],
        })
    
    # Ensure at least one section
    if not sections:
        sections.append({
            "title": "Answer",
            "type": "text",
            "claims": [],
            "evidence_needed": [e["id"] for e in evidence],
        })
    
    return sections


def _identify_emphasis(claims: List[Dict]) -> List[str]:
    """Identify claims that should be emphasized."""
    emphasis = []
    
    for claim in claims:
        status = claim.get("status", "")
        if status == "SUPPORTED":
            confidence = claim.get("support_ratio", 0)
            if confidence >= 0.8:
                emphasis.append(claim.get("id", ""))
    
    return emphasis


def _determine_numeric_format(claims: List[Dict]) -> str:
    """Determine how to format numeric values."""
    has_ranges = any(c.get("numeric_range") for c in claims)
    has_exact = any(c.get("numeric_value") is not None and not c.get("numeric_range") for c in claims)
    
    if has_ranges and has_exact:
        return "mixed"
    if has_ranges:
        return "range"
    return "decimal"


def _determine_depth(question: str, claims: List[Dict], evidence: List[Dict]) -> str:
    """Determine answer depth based on question and available evidence."""
    
    # Check question length and complexity
    question_lower = question.lower()
    if len(question_lower) < 10:
        return "brief"
    
    # Check for depth indicators
    depth_indicators = [
        "detailed", "comprehensive", "complete", "thorough",
        "all", "entire", "full", "extensive", "in depth",
    ]
    
    if any(ind in question_lower for ind in depth_indicators):
        return "comprehensive"
    
    # Check evidence quantity
    if len(evidence) >= 5 and len(claims) >= 3:
        return "moderate"
    
    # Default
    if len(evidence) >= 2:
        return "moderate"
    
    return "brief"


def build_section_builder(
    section_title: str,
    section_type: str,
    required_claim_ids: List[str],
    optional_claim_ids: List[str],
    source_types: List[str],
    max_length: int = 1000,
    formatting: str = "paragraph",
) -> SectionBuilder:
    """Build a section configuration."""
    return SectionBuilder(
        title=section_title,
        content_type=section_type,
        required_claims=required_claim_ids,
        optional_claims=optional_claim_ids,
        source_requirements=source_types,
        max_length=max_length,
        formatting=formatting,
    )


def create_causal_chain(claims: List[Dict]) -> List[Dict[str, Any]]:
    """Create a causal chain from claims for mechanism questions."""
    chain = []
    
    # Sort claims by some temporal or logical order
    sorted_claims = sorted(
        claims,
        key=lambda c: (
            c.get("context", {}).get("temporal_order", 999),
            c.get("support_ratio", 0)
        ),
        reverse=True
    )
    
    for i, claim in enumerate(sorted_claims):
        if claim.get("status") not in {"SUPPORTED", "PARTIALLY_SUPPORTED"}:
            continue
        
        # Find supporting claims
        supports = []
        for other in sorted_claims:
            if other["id"] == claim["id"]:
                continue
            if other.get("status") not in {"SUPPORTED", "PARTIALLY_SUPPORTED"}:
                continue
            if claim.get("entities") and other.get("entities"):
                # Check for connection
                if any(e in other.get("entities", []) for e in claim.get("entities", [])):
                    supports.append(other["id"])
        
        chain.append({
            "step": i + 1,
            "claim_id": claim["id"],
            "text": claim.get("text", ""),
            "supports": supports,
            "explanation": _generate_step_explanation(claim, supports),
        })
    
    return chain


def _generate_step_explanation(claim: Dict, supports: List[str]) -> str:
    """Generate a step explanation for causal chains."""
    if not supports:
        return "This finding is directly supported by evidence."
    
    support_text = ", ".join(f"claim_{s}" for s in supports[:3])
    return f"Based on: {support_text}"