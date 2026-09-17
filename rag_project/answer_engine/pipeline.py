"""Complete pipeline for deterministic medical answering with all gates."""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time


class PipelineStage(str, Enum):
    """Stages of the answer pipeline."""
    QUERY_ANALYSIS = "query_analysis"
    EVIDENCE_RETRIEVAL = "evidence_retrieval"
    EVIDENCE_NORMALIZATION = "evidence_normalization"
    EVIDENCE_DEDUPLICATION = "evidence_deduplication"
    EVIDENCE_SCORING = "evidence_scoring"
    EVIDENCE_GROUPING = "evidence_grouping"
    EVIDENCE_GRAPH = "evidence_graph"
    CLAIM_EXTRACTION = "claim_extraction"
    CLAIM_NORMALIZATION = "claim_normalization"
    CLAIM_VERIFICATION = "claim_verification"
    NUMERIC_VERIFICATION = "numeric_verification"
    CONTRADICTION_DETECTION = "contradiction_detection"
    SAFETY_CHECK = "safety_check"
    CONFIDENCE_CALCULATION = "confidence_calculation"
    ABSTENTION_CHECK = "abstention_check"
    ANSWER_COMPILATION = "answer_compilation"


@dataclass(frozen=True)
class GateResult:
    """Result of a pipeline gate."""
    passed: bool
    reason: str
    details: Dict[str, Any]


@dataclass(frozen=True)
class PipelineResult:
    """Complete result of the pipeline."""
    success: bool
    answer: Dict[str, Any]
    gates_passed: List[str]
    gates_failed: List[str]
    processing_time_ms: float
    steps: List[Dict[str, Any]]
    warnings: List[str]
    errors: List[str]


class PipelineGate:
    """A gate in the answer pipeline."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def check(self, context: Dict[str, Any]) -> GateResult:
        """Check if the gate passes. Override in subclasses."""
        return GateResult(True, "No checks defined", {})


class EvidenceGate(PipelineGate):
    """Gate: Check evidence quality and quantity."""
    
    def __init__(self):
        super().__init__("evidence", "Evidence quality and quantity check")
        self.min_evidence_count = 1
        self.min_quality_score = 0.30
    
    def check(self, context: Dict[str, Any]) -> GateResult:
        evidence = context.get("evidence", [])
        
        # Check count
        if len(evidence) < self.min_evidence_count:
            return GateResult(
                False,
                f"Insufficient evidence: {len(evidence)} < {self.min_evidence_count}",
                {"actual": len(evidence), "minimum": self.min_evidence_count},
            )
        
        # Check quality scores
        low_quality = [
            e for e in evidence
            if e.get("composite_score", 1.0) < self.min_quality_score
        ]
        
        if len(low_quality) > len(evidence) * 0.5:
            return GateResult(
                False,
                "More than 50% of evidence is low quality",
                {"low_quality_count": len(low_quality), "total": len(evidence)},
            )
        
        return GateResult(True, "Evidence check passed", {"count": len(evidence)})


class ClaimGate(PipelineGate):
    """Gate: Check claim support and quality."""
    
    def __init__(self):
        super().__init__("claim", "Claim support and quality check")
        self.min_support_ratio = 0.50
        self.max_unsupported_ratio = 0.50
    
    def check(self, context: Dict[str, Any]) -> GateResult:
        claims = context.get("claims", [])
        
        if not claims:
            return GateResult(
                False,
                "No claims extracted",
                {},
            )
        
        # Check supported claims
        supported = [c for c in claims if c.get("status") == "SUPPORTED"]
        unsupported = [c for c in claims if c.get("status") == "UNSUPPORTED"]
        
        unsupported_ratio = len(unsupported) / len(claims) if claims else 1.0
        
        if unsupported_ratio > self.max_unsupported_ratio:
            return GateResult(
                False,
                f"Too many unsupported claims: {unsupported_ratio:.1%}",
                {
                    "supported": len(supported),
                    "unsupported": len(unsupported),
                    "total": len(claims),
                },
            )
        
        # Check support ratios
        for claim in supported:
            if claim.get("support_ratio", 0) < self.min_support_ratio:
                return GateResult(
                    False,
                    f"Claim support ratio below threshold: {claim.get('id')}",
                    {"claim_id": claim.get("id"), "support_ratio": claim.get("support_ratio")},
                )
        
        return GateResult(True, "Claim check passed", {"supported": len(supported)})


class NumericGate(PipelineGate):
    """Gate: Check numeric values for accuracy."""
    
    def __init__(self):
        super().__init__("numeric", "Numeric value accuracy check")
        self.require_units = True
        self.min_precision = 2
    
    def check(self, context: Dict[str, Any]) -> GateResult:
        claims = context.get("claims", [])
        
        numeric_claims = [
            c for c in claims
            if c.get("numeric_value") is not None or c.get("numeric_range")
        ]
        
        if not numeric_claims:
            return GateResult(True, "No numeric claims to check", {})
        
        # Check units for numeric claims
        if self.require_units:
            claims_without_units = [
                c for c in numeric_claims
                if not c.get("numeric_unit")
            ]
            
            if claims_without_units:
                return GateResult(
                    False,
                    f"Numeric claims without units: {len(claims_without_units)}",
                    {"count": len(claims_without_units)},
                )
        
        return GateResult(True, "Numeric check passed", {"count": len(numeric_claims)})


class CitationGate(PipelineGate):
    """Gate: Check citation coverage."""
    
    def __init__(self):
        super().__init__("citation", "Citation coverage check")
        self.min_citation_ratio = 0.7
    
    def check(self, context: Dict[str, Any]) -> GateResult:
        claims = context.get("claims", [])
        evidence = context.get("evidence", [])
        
        # Check if claims have evidence
        claims_with_evidence = sum(
            1 for c in claims
            if c.get("evidence") and len(c.get("evidence", [])) > 0
        )
        
        total_claims = len(claims)
        if total_claims == 0:
            return GateResult(True, "No claims to cite", {})
        
        citation_ratio = claims_with_evidence / total_claims
        
        if citation_ratio < self.min_citation_ratio:
            return GateResult(
                False,
                f"Citation ratio below threshold: {citation_ratio:.1%}",
                {
                    "cited": claims_with_evidence,
                    "total": total_claims,
                    "ratio": citation_ratio,
                },
            )
        
        return GateResult(True, "Citation check passed", {"ratio": citation_ratio})


class ContradictionGate(PipelineGate):
    """Gate: Check for contradictions."""
    
    def __init__(self):
        super().__init__("contradiction", "Contradiction check")
    
    def check(self, context: Dict[str, Any]) -> GateResult:
        claims = context.get("claims", [])
        
        contradictory = [
            c for c in claims
            if c.get("contradiction")
        ]
        
        if contradictory:
            return GateResult(
                False,
                f"Found {len(contradictory)} contradictory claims",
                {"count": len(contradictory)},
            )
        
        return GateResult(True, "No contradictions found", {})


class CompletenessGate(PipelineGate):
    """Gate: Check answer completeness."""
    
    def __init__(self):
        super().__init__("completeness", "Answer completeness check")
        self.min_answer_length = 10
    
    def check(self, context: Dict[str, Any]) -> GateResult:
        answer = context.get("direct_answer", "")
        question = context.get("question", "")
        
        # Check answer length
        if len(answer) < self.min_answer_length:
            return GateResult(
                False,
                f"Answer too short: {len(answer)} < {self.min_answer_length}",
                {"length": len(answer), "minimum": self.min_answer_length},
            )
        
        # Check if answer addresses question
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        overlap = question_words & answer_words
        
        if len(overlap) < 3:
            return GateResult(
                False,
                "Answer may not address question adequately",
                {"overlap_words": len(overlap)},
            )
        
        return GateResult(True, "Completeness check passed", {"answer_length": len(answer)})


class SafetyGate(PipelineGate):
    """Gate: Check safety concerns."""
    
    def __init__(self):
        super().__init__("safety", "Safety concerns check")
    
    def check(self, context: Dict[str, Any]) -> GateResult:
        safety_status = context.get("safety_status", "")
        
        if safety_status in {"rejected", "review_required"}:
            return GateResult(
                False,
                f"Safety gate {safety_status}",
                {"status": safety_status},
            )
        
        return GateResult(True, "Safety check passed", {"status": safety_status})


class AnswerPipeline:
    """Complete pipeline with all gates for deterministic medical answering."""
    
    def __init__(self):
        self.gates = [
            EvidenceGate(),
            ClaimGate(),
            NumericGate(),
            CitationGate(),
            ContradictionGate(),
            CompletenessGate(),
            SafetyGate(),
        ]
        
        self.steps: List[Dict[str, Any]] = []
    
    def run(
        self,
        question: str,
        evidence: List[Dict[str, Any]],
        claims: List[Dict[str, Any]],
        answer: str = "",
        safety_status: str = "approved",
    ) -> PipelineResult:
        """Run the pipeline with all gates."""
        start_time = time.time()
        
        context = {
            "question": question,
            "evidence": evidence,
            "claims": claims,
            "direct_answer": answer,
            "safety_status": safety_status,
        }
        
        gates_passed = []
        gates_failed = []
        warnings = []
        errors = []
        
        # Run each gate
        for gate in self.gates:
            try:
                result = gate.check(context)
                
                step = {
                    "gate": gate.name,
                    "passed": result.passed,
                    "reason": result.reason,
                    "details": result.details,
                }
                self.steps.append(step)
                
                if result.passed:
                    gates_passed.append(gate.name)
                else:
                    gates_failed.append(gate.name)
                    errors.append(f"{gate.name}: {result.reason}")
                    
                    # If critical gate fails, stop
                    if gate.name in {"evidence", "claim", "safety"}:
                        break
            
            except Exception as e:
                step = {
                    "gate": gate.name,
                    "passed": False,
                    "reason": f"Error: {str(e)}",
                    "error": str(e),
                }
                self.steps.append(step)
                gates_failed.append(gate.name)
                errors.append(str(e))
        
        processing_time = (time.time() - start_time) * 1000
        
        success = len(gates_failed) == 0
        
        return PipelineResult(
            success=success,
            answer={
                "question": question,
                "evidence": evidence,
                "claims": claims,
                "direct_answer": answer,
                "safety_status": safety_status,
            },
            gates_passed=gates_passed,
            gates_failed=gates_failed,
            processing_time_ms=processing_time,
            steps=self.steps,
            warnings=warnings,
            errors=errors,
        )
    
    def check_all_gates(
        self,
        context: Dict[str, Any],
    ) -> List[Tuple[PipelineGate, GateResult]]:
        """Check all gates and return results."""
        results = []
        for gate in self.gates:
            result = gate.check(context)
            results.append((gate, result))
        return results


def run_deterministic_pipeline(
    question: str,
    evidence: List[Dict[str, Any]],
    claims: List[Dict[str, Any]],
    answer: str = "",
    safety_status: str = "approved",
) -> PipelineResult:
    """Convenience function to run the full pipeline."""
    pipeline = AnswerPipeline()
    return pipeline.run(question, evidence, claims, answer, safety_status)


def build_pipeline_context(
    question: str,
    evidence: List[Dict[str, Any]],
    claims: List[Dict[str, Any]],
    answer: str = "",
    safety_status: str = "approved",
) -> Dict[str, Any]:
    """Build pipeline context from components."""
    return {
        "question": question,
        "evidence": evidence,
        "claims": claims,
        "direct_answer": answer,
        "safety_status": safety_status,
    }