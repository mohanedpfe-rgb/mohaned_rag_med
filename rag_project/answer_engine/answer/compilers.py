"""Answer compilers for deterministic medical answers."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from rag_project.answer_engine.schemas import AnswerType, ConfidenceLevel


@dataclass(frozen=True)
class CompilationResult:
    """Result of answer compilation."""
    success: bool
    answer: str
    citations: List[str]
    confidence: float
    components: Dict[str, Any]


class BaseAnswerCompiler(ABC):
    """Base class for answer compilers."""
    
    def __init__(self, language: str = "en"):
        self.language = language
    
    @abstractmethod
    def can_compile(self, answer_type: AnswerType, claims: List[Dict]) -> bool:
        """Check if this compiler can handle the given answer type."""
        pass
    
    @abstractmethod
    def compile(
        self,
        question: str,
        claims: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> CompilationResult:
        """Compile an answer from claims and evidence."""
        pass
    
    def _build_citations(self, evidence: List[Dict]) -> List[str]:
        """Build citation strings from evidence."""
        citations = []
        seen = set()
        
        for e in evidence:
            source_id = e.get("source_id", "")
            if source_id and source_id not in seen:
                seen.add(source_id)
                citations.append(source_id)
        
        return citations


class DefinitionAnswerCompiler(BaseAnswerCompiler):
    """Compiler for definition answers."""
    
    def can_compile(self, answer_type: AnswerType, claims: List[Dict]) -> bool:
        return answer_type == AnswerType.DEFINITION
    
    def compile(
        self,
        question: str,
        claims: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> CompilationResult:
        # Filter supported claims
        supported = [c for c in claims if c.get("status") == "SUPPORTED"]
        
        if not supported:
            return CompilationResult(False, "", [], 0.0, {})
        
        # Get the best definition
        best = max(supported, key=lambda c: c.get("support_ratio", 0))
        text = best.get("text", "")
        
        # Add qualifiers if present
        qualifiers = best.get("qualifiers", [])
        if qualifiers:
            text += f" ({', '.join(qualifiers)})"
        
        return CompilationResult(
            success=True,
            answer=text,
            citations=self._build_citations(evidence),
            confidence=best.get("support_ratio", 0.0),
            components={"definition_claim_id": best.get("id")},
        )


class FactAnswerCompiler(BaseAnswerCompiler):
    """Compiler for factual answers."""
    
    def can_compile(self, answer_type: AnswerType, claims: List[Dict]) -> bool:
        return answer_type in {AnswerType.FACT, AnswerType.RELATIONSHIP}
    
    def compile(
        self,
        question: str,
        claims: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> CompilationResult:
        supported = [c for c in claims if c.get("status") == "SUPPORTED"]
        
        if not supported:
            return CompilationResult(False, "", [], 0.0, {})
        
        # Build combined fact
        facts = []
        for claim in supported[:3]:
            text = claim.get("text", "")
            if text:
                facts.append(text)
        
        answer = " ".join(facts) if facts else ""
        
        return CompilationResult(
            success=bool(answer),
            answer=answer,
            citations=self._build_citations(evidence),
            confidence=sum(c.get("support_ratio", 0) for c in supported) / len(supported) if supported else 0.0,
            components={"fact_claims": [c.get("id") for c in supported]},
        )


class NumericAnswerCompiler(BaseAnswerCompiler):
    """Compiler for numeric answers."""
    
    def can_compile(self, answer_type: AnswerType, claims: List[Dict]) -> bool:
        return answer_type == AnswerType.NUMERIC
    
    def compile(
        self,
        question: str,
        claims: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> CompilationResult:
        numeric_claims = [
            c for c in claims
            if c.get("numeric_value") is not None or c.get("numeric_range")
        ]
        
        if not numeric_claims:
            return CompilationResult(False, "", [], 0.0, {})
        
        parts = []
        for claim in numeric_claims:
            value = claim.get("numeric_value")
            unit = claim.get("numeric_unit", "")
            range_val = claim.get("numeric_range")
            
            if range_val:
                parts.append(f"{range_val[0]} to {range_val[1]} {unit}".strip())
            elif value is not None:
                parts.append(f"{value} {unit}".strip())
        
        answer = "; ".join(parts) if parts else ""
        
        return CompilationResult(
            success=bool(answer),
            answer=answer,
            citations=self._build_citations(evidence),
            confidence=sum(c.get("support_ratio", 0) for c in numeric_claims) / len(numeric_claims) if numeric_claims else 0.0,
            components={"numeric_claims": [c.get("id") for c in numeric_claims]},
        )


class ListAnswerCompiler(BaseAnswerCompiler):
    """Compiler for list answers."""
    
    def can_compile(self, answer_type: AnswerType, claims: List[Dict]) -> bool:
        return answer_type == AnswerType.LIST
    
    def compile(
        self,
        question: str,
        claims: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> CompilationResult:
        supported = [c for c in claims if c.get("status") == "SUPPORTED"]
        
        if not supported:
            return CompilationResult(False, "", [], 0.0, {})
        
        items = []
        for claim in supported[:5]:
            text = claim.get("text", "")
            if text:
                items.append(text)
        
        if not items:
            return CompilationResult(False, "", [], 0.0, {})
        
        answer = "\n".join(f"- {item}" for item in items)
        
        return CompilationResult(
            success=True,
            answer=answer,
            citations=self._build_citations(evidence),
            confidence=sum(c.get("support_ratio", 0) for c in supported) / len(supported),
            components={"list_items": items},
        )


class ComparisonAnswerCompiler(BaseAnswerCompiler):
    """Compiler for comparison answers."""
    
    def can_compile(self, answer_type: AnswerType, claims: List[Dict]) -> bool:
        return answer_type == AnswerType.COMPARISON
    
    def compile(
        self,
        question: str,
        claims: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> CompilationResult:
        supported = [c for c in claims if c.get("status") == "SUPPORTED"]
        
        if not supported:
            return CompilationResult(False, "", [], 0.0, {})
        
        # Group by comparison type
        parts = []
        for claim in supported[:3]:
            text = claim.get("text", "")
            if text:
                parts.append(text)
        
        if not parts:
            return CompilationResult(False, "", [], 0.0, {})
        
        answer = "Comparison:\n" + "\n".join(f"- {p}" for p in parts)
        
        return CompilationResult(
            success=True,
            answer=answer,
            citations=self._build_citations(evidence),
            confidence=sum(c.get("support_ratio", 0) for c in supported) / len(supported),
            components={"comparison_items": parts},
        )


class CausalAnswerCompiler(BaseAnswerCompiler):
    """Compiler for causal/mechanism answers."""
    
    def can_compile(self, answer_type: AnswerType, claims: List[Dict]) -> bool:
        return answer_type in {AnswerType.CAUSE, AnswerType.MECHANISM}
    
    def compile(
        self,
        question: str,
        claims: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> CompilationResult:
        supported = [c for c in claims if c.get("status") == "SUPPORTED"]
        
        if not supported:
            return CompilationResult(False, "", [], 0.0, {})
        
        parts = []
        for claim in supported[:5]:
            text = claim.get("text", "")
            if text:
                parts.append(text)
        
        if not parts:
            return CompilationResult(False, "", [], 0.0, {})
        
        answer = "\n".join(parts)
        
        return CompilationResult(
            success=True,
            answer=answer,
            citations=self._build_citations(evidence),
            confidence=sum(c.get("support_ratio", 0) for c in supported) / len(supported),
            components={"causal_steps": parts},
        )


class ManagementAnswerCompiler(BaseAnswerCompiler):
    """Compiler for management/treatment answers."""
    
    def can_compile(self, answer_type: AnswerType, claims: List[Dict]) -> bool:
        return answer_type == AnswerType.MANAGEMENT
    
    def compile(
        self,
        question: str,
        claims: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> CompilationResult:
        supported = [c for c in claims if c.get("status") == "SUPPORTED"]
        
        if not supported:
            return CompilationResult(False, "", [], 0.0, {})
        
        parts = []
        for claim in supported[:5]:
            text = claim.get("text", "")
            if text:
                parts.append(text)
        
        if not parts:
            return CompilationResult(False, "", [], 0.0, {})
        
        answer = "\n".join(f"{i+1}. {p}" for i, p in enumerate(parts))
        
        return CompilationResult(
            success=True,
            answer=answer,
            citations=self._build_citations(evidence),
            confidence=sum(c.get("support_ratio", 0) for c in supported) / len(supported),
            components={"management_steps": parts},
        )


class DiagnosticAnswerCompiler(BaseAnswerCompiler):
    """Compiler for diagnostic answers."""
    
    def can_compile(self, answer_type: AnswerType, claims: List[Dict]) -> bool:
        return answer_type == AnswerType.DIAGNOSTIC_CRITERIA
    
    def compile(
        self,
        question: str,
        claims: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> CompilationResult:
        supported = [c for c in claims if c.get("status") == "SUPPORTED"]
        
        if not supported:
            return CompilationResult(False, "", [], 0.0, {})
        
        parts = []
        for claim in supported[:5]:
            text = claim.get("text", "")
            if text:
                parts.append(text)
        
        if not parts:
            return CompilationResult(False, "", [], 0.0, {})
        
        answer = "\n".join(f"- {p}" for p in parts)
        
        return CompilationResult(
            success=True,
            answer=answer,
            citations=self._build_citations(evidence),
            confidence=sum(c.get("support_ratio", 0) for c in supported) / len(supported),
            components={"diagnostic_criteria": parts},
        )


class PrognosisAnswerCompiler(BaseAnswerCompiler):
    """Compiler for prognosis answers."""
    
    def can_compile(self, answer_type: AnswerType, claims: List[Dict]) -> bool:
        return answer_type == AnswerType.PROGNOSIS
    
    def compile(
        self,
        question: str,
        claims: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> CompilationResult:
        supported = [c for c in claims if c.get("status") == "SUPPORTED"]
        
        if not supported:
            return CompilationResult(False, "", [], 0.0, {})
        
        parts = []
        for claim in supported[:3]:
            text = claim.get("text", "")
            if text:
                parts.append(text)
        
        if not parts:
            return CompilationResult(False, "", [], 0.0, {})
        
        answer = " ".join(parts)
        
        return CompilationResult(
            success=True,
            answer=answer,
            citations=self._build_citations(evidence),
            confidence=sum(c.get("support_ratio", 0) for c in supported) / len(supported),
            components={"prognosis_factors": parts},
        )


class MultiHopAnswerCompiler(BaseAnswerCompiler):
    """Compiler for multi-hop reasoning answers."""
    
    def can_compile(self, answer_type: AnswerType, claims: List[Dict]) -> bool:
        return answer_type == AnswerType.MULTI_HOP
    
    def compile(
        self,
        question: str,
        claims: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> CompilationResult:
        supported = [c for c in claims if c.get("status") == "SUPPORTED"]
        
        if not supported:
            return CompilationResult(False, "", [], 0.0, {})
        
        # Build chain
        parts = []
        for claim in supported[:5]:
            text = claim.get("text", "")
            if text:
                parts.append(text)
        
        if not parts:
            return CompilationResult(False, "", [], 0.0, {})
        
        answer = "\n\n".join(f"Step {i+1}: {p}" for i, p in enumerate(parts))
        
        return CompilationResult(
            success=True,
            answer=answer,
            citations=self._build_citations(evidence),
            confidence=sum(c.get("support_ratio", 0) for c in supported) / len(supported),
            components={"reasoning_steps": parts},
        )


class SummaryAnswerCompiler(BaseAnswerCompiler):
    """Compiler for summary answers."""
    
    def can_compile(self, answer_type: AnswerType, claims: List[Dict]) -> bool:
        return answer_type == AnswerType.SUMMARY
    
    def compile(
        self,
        question: str,
        claims: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> CompilationResult:
        supported = [c for c in claims if c.get("status") == "SUPPORTED"]
        
        if not supported:
            return CompilationResult(False, "", [], 0.0, {})
        
        # Get top claims by support ratio
        sorted_claims = sorted(supported, key=lambda c: c.get("support_ratio", 0), reverse=True)
        
        parts = []
        for claim in sorted_claims[:3]:
            text = claim.get("text", "")
            if text:
                parts.append(text)
        
        if not parts:
            return CompilationResult(False, "", [], 0.0, {})
        
        answer = " ".join(parts)
        
        return CompilationResult(
            success=True,
            answer=answer,
            citations=self._build_citations(evidence),
            confidence=sum(c.get("support_ratio", 0) for c in supported) / len(supported),
            components={"summary_points": parts},
        )


class FollowUpAnswerCompiler(BaseAnswerCompiler):
    """Compiler for follow-up answers."""
    
    def can_compile(self, answer_type: AnswerType, claims: List[Dict]) -> bool:
        return answer_type == AnswerType.FOLLOW_UP
    
    def compile(
        self,
        question: str,
        claims: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> CompilationResult:
        supported = [c for c in claims if c.get("status") == "SUPPORTED"]
        
        if not supported:
            return CompilationResult(False, "", [], 0.0, {})
        
        parts = []
        for claim in supported[:3]:
            text = claim.get("text", "")
            if text:
                parts.append(text)
        
        if not parts:
            return CompilationResult(False, "", [], 0.0, {})
        
        answer = " ".join(parts)
        
        return CompilationResult(
            success=True,
            answer=answer,
            citations=self._build_citations(evidence),
            confidence=sum(c.get("support_ratio", 0) for c in supported) / len(supported),
            components={"follow_up_info": parts},
        )


class CrossReferenceAnswerCompiler(BaseAnswerCompiler):
    """Compiler for cross-reference answers."""
    
    def can_compile(self, answer_type: AnswerType, claims: List[Dict]) -> bool:
        return answer_type == AnswerType.CROSS_REFERENCE
    
    def compile(
        self,
        question: str,
        claims: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> CompilationResult:
        supported = [c for c in claims if c.get("status") == "SUPPORTED"]
        
        if not supported:
            return CompilationResult(False, "", [], 0.0, {})
        
        parts = []
        for claim in supported[:3]:
            text = claim.get("text", "")
            if text:
                parts.append(text)
        
        if not parts:
            return CompilationResult(False, "", [], 0.0, {})
        
        answer = " ".join(parts)
        
        return CompilationResult(
            success=True,
            answer=answer,
            citations=self._build_citations(evidence),
            confidence=sum(c.get("support_ratio", 0) for c in supported) / len(supported),
            components={"cross_references": parts},
        )


class TableAnswerCompiler(BaseAnswerCompiler):
    """Compiler for table answers."""
    
    def can_compile(self, answer_type: AnswerType, claims: List[Dict]) -> bool:
        return answer_type == AnswerType.TABLE
    
    def compile(
        self,
        question: str,
        claims: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> CompilationResult:
        supported = [c for c in claims if c.get("status") == "SUPPORTED"]
        
        if not supported:
            return CompilationResult(False, "", [], 0.0, {})
        
        parts = []
        for claim in supported[:5]:
            text = claim.get("text", "")
            if text:
                parts.append(text)
        
        if not parts:
            return CompilationResult(False, "", [], 0.0, {})
        
        # Build table format
        answer = "Value\n" + "\n".join(f"- {p}" for p in parts)
        
        return CompilationResult(
            success=True,
            answer=answer,
            citations=self._build_citations(evidence),
            confidence=sum(c.get("support_ratio", 0) for c in supported) / len(supported),
            components={"table_values": parts},
        )


class ChainOfThoughtAnswerCompiler(BaseAnswerCompiler):
    """Compiler for chain-of-thought reasoning answers."""
    
    def can_compile(self, answer_type: AnswerType, claims: List[Dict]) -> bool:
        return answer_type in {AnswerType.MECHANISM, AnswerType.CAUSE}
    
    def compile(
        self,
        question: str,
        claims: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> CompilationResult:
        supported = [c for c in claims if c.get("status") == "SUPPORTED"]
        
        if not supported:
            return CompilationResult(False, "", [], 0.0, {})
        
        # Sort by some logical order
        parts = []
        for claim in supported[:5]:
            text = claim.get("text", "")
            if text:
                parts.append(text)
        
        if not parts:
            return CompilationResult(False, "", [], 0.0, {})
        
        # Build chain
        answer = "\n".join(f"{i+1}. {p}" for i, p in enumerate(parts))
        
        return CompilationResult(
            success=True,
            answer=answer,
            citations=self._build_citations(evidence),
            confidence=sum(c.get("support_ratio", 0) for c in supported) / len(supported),
            components={"chain_steps": parts},
        )


def get_all_compilers() -> List[BaseAnswerCompiler]:
    """Get all available answer compilers."""
    return [
        DefinitionAnswerCompiler(),
        FactAnswerCompiler(),
        NumericAnswerCompiler(),
        ListAnswerCompiler(),
        ComparisonAnswerCompiler(),
        CausalAnswerCompiler(),
        ManagementAnswerCompiler(),
        DiagnosticAnswerCompiler(),
        PrognosisAnswerCompiler(),
        MultiHopAnswerCompiler(),
        SummaryAnswerCompiler(),
        FollowUpAnswerCompiler(),
        CrossReferenceAnswerCompiler(),
        TableAnswerCompiler(),
    ]


def compile_answer(
    answer_type: AnswerType,
    question: str,
    claims: List[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
) -> CompilationResult:
    """Compile an answer using the appropriate compiler."""
    compilers = get_all_compilers()
    
    for compiler in compilers:
        if compiler.can_compile(answer_type, claims):
            return compiler.compile(question, claims, evidence)
    
    # Fallback
    return CompilationResult(False, "", [], 0.0, {})