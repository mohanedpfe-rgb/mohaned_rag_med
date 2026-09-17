"""Query decomposer for multi-part questions."""
from __future__ import annotations

import re
from typing import List, Dict, Any


def decompose_question(question: str) -> List[Dict[str, str]]:
    """Decompose a complex question into simpler subquestions.
    
    Handles:
    - Multi-part questions with "and", "or", semicolons
    - Questions with explicit numbering
    - Questions asking for multiple distinct pieces of information
    """
    if not question:
        return []
    
    original = question.strip()
    decomposed = []
    
    # Pattern 1: Semicolon-separated
    if ";" in question and not question.startswith("What is"):
        parts = [p.strip() for p in question.split(";") if len(p.strip()) > 10]
        if len(parts) > 1:
            return [{"question": p, "type": "decomposed"} for p in parts]
    
    # Pattern 2: Numbered list (1. 2. 3.)
    numbered = re.findall(r"\d+\.\s*([^\n]+)", question)
    if len(numbered) >= 2:
        return [{"question": p.strip(), "type": "numbered"} for p in numbered]
    
    # Pattern 3: "What is X, what is Y, what is Z"
    multi_what = re.findall(r"(what is [^,]+(?:,|and|ou|w)?\s*(?:what is)?[^,]*)", question, re.I)
    if len(multi_what) >= 2:
        return [{"question": p.strip(), "type": "multi_what"} for p in multi_what]
    
    # Pattern 4: "and" separated (but not in medical terms)
    if " and " in question.lower():
        # Check if it's a complex question, not a medical term
        parts = re.split(r"\s+and\s+", question, maxsplit=3)
        if len(parts) >= 2:
            # Verify each part looks like a question
            valid_parts = [p.strip() for p in parts if "?" in p or len(p.strip().split()) > 3]
            if len(valid_parts) >= 2:
                return [{"question": p, "type": "conjunction"} for p in valid_parts]
    
    # Pattern 5: Multi-hop patterns (then, subsequently, result)
    if any(term in question.lower() for term in ("then", "subsequently", "result", "outcome")):
        # Try to split on these terms
        parts = re.split(r"\s+(?:then|subsequently|result)\s+", question, flags=re.I)
        if len(parts) >= 2:
            return [{"question": p.strip(), "type": "multi_hop"} for p in parts]
    
    # No decomposition needed
    return [{"question": original, "type": "atomic"}]


def extract_subquestions(question: str) -> List[str]:
    """Extract just the subquestion strings."""
    decomposed = decompose_question(question)
    return [d["question"] for d in decomposed]


def has_multi_part_structure(question: str) -> bool:
    """Check if question has multi-part structure."""
    decomposed = decompose_question(question)
    return len(decomposed) > 1


def is_single_subquestion(question: str) -> bool:
    """Check if question is a single, atomic question."""
    decomposed = decompose_question(question)
    return len(decomposed) == 1 and decomposed[0]["type"] == "atomic"
