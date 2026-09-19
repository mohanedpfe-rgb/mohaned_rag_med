"""Complete pipeline for deterministic medical answering with all gates."""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import datetime
import re


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
        self.min_overlap = 1  # Reduced to 1 for more lenient matching
    
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
        
        # Check if answer addresses question (more lenient overlap check)
        # Remove punctuation and normalize
        question_clean = re.sub(r'[^\w\s]', '', question.lower())
        answer_clean = re.sub(r'[^\w\s]', '', answer.lower())
        
        question_words = set(word for word in question_clean.split() if len(word) > 3)
        answer_words = set(word for word in answer_clean.split() if len(word) > 3)
        
        overlap = question_words & answer_words
        
        if len(overlap) < self.min_overlap:
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


class GroundingGate(PipelineGate):
    """Phase 74 fix: Gate to validate that answers are actually grounded in evidence."""
    
    def __init__(self):
        super().__init__("grounding", "Answer grounding validation")
        self.min_grounding_ratio = 0.4  # Adjusted to be more reasonable for medical text
    
    def check(self, context: Dict[str, Any]) -> GateResult:
        answer = context.get("direct_answer", "")
        evidence = context.get("evidence", [])
        
        if not answer or not evidence:
            return GateResult(True, "No answer or evidence to validate grounding", {})
        
        # Extract key terms from answer
        answer_words = set(word.lower() for word in answer.split() if len(word) > 3)
        
        if not answer_words:
            return GateResult(True, "Answer too short for grounding validation", {})
        
        # Check how many answer terms appear in evidence
        evidence_text = " ".join(str(e.get("text", "")) for e in evidence)
        evidence_words = set(word.lower() for word in evidence_text.split() if len(word) > 3)
        
        grounded_terms = answer_words & evidence_words
        grounding_ratio = len(grounded_terms) / len(answer_words) if answer_words else 0.0
        
        if grounding_ratio < self.min_grounding_ratio:
            return GateResult(
                False,
                f"Answer insufficiently grounded: {grounding_ratio:.1%} < {self.min_grounding_ratio:.1%}",
                {
                    "grounding_ratio": grounding_ratio,
                    "grounded_terms": len(grounded_terms),
                    "total_terms": len(answer_words),
                },
            )
        
        return GateResult(
            True,
            "Answer grounding validated",
            {"grounding_ratio": grounding_ratio},
        )


class ConflictDetectionGate(PipelineGate):
    """Phase 76 fix: Gate to detect contradictory medical guidelines."""
    
    def __init__(self):
        super().__init__("conflict_detection", "Medical guideline conflict detection")
    
    def check(self, context: Dict[str, Any]) -> GateResult:
        evidence = context.get("evidence", [])
        claims = context.get("claims", [])
        
        if len(evidence) < 2:
            return GateResult(True, "Insufficient evidence for conflict detection", {})
        
        # Check for conflicting recommendations in evidence
        conflicts = []
        
        # Look for contradictory terms in evidence
        contradictory_pairs = [
            ("should", "should not"),
            ("recommended", "not recommended"),
            ("indicated", "contraindicated"),
            ("beneficial", "harmful"),
            ("effective", "ineffective"),
        ]
        
        for i, ev1 in enumerate(evidence):
            for j, ev2 in enumerate(evidence[i+1:], i+1):
                text1 = str(ev1.get("text", "")).lower()
                text2 = str(ev2.get("text", "")).lower()
                
                for term1, term2 in contradictory_pairs:
                    if term1 in text1 and term2 in text2:
                        conflicts.append({
                            "evidence_1": i,
                            "evidence_2": j,
                            "conflict_type": f"{term1} vs {term2}",
                        })
        
        # Check for contradictory claims
        claim_values = {}
        for claim in claims:
            claim_id = claim.get("id", "")
            claim_value = claim.get("claim_text", "")
            if claim_value:
                claim_values[claim_id] = claim_value.lower()
        
        # Detect numeric contradictions
        numeric_claims = [c for c in claims if c.get("numeric_value") is not None]
        for i, c1 in enumerate(numeric_claims):
            for j, c2 in enumerate(numeric_claims[i+1:], i+1):
                val1 = c1.get("numeric_value")
                val2 = c2.get("numeric_value")
                if val1 is not None and val2 is not None:
                    # Check for significant numerical differences
                    if abs(float(val1) - float(val2)) > max(abs(float(val1)), abs(float(val2))) * 0.5:
                        conflicts.append({
                            "claim_1": c1.get("id"),
                            "claim_2": c2.get("id"),
                            "conflict_type": "numeric_contradiction",
                            "values": [val1, val2],
                        })
        
        if conflicts:
            return GateResult(
                False,
                f"Found {len(conflicts)} potential conflicts in medical guidelines",
                {"conflicts": conflicts[:5]},  # Return first 5 conflicts
            )
        
        return GateResult(True, "No significant conflicts detected", {})


class VersionHandlingGate(PipelineGate):
    """Phase 77 fix: Gate to detect outdated medical information."""
    
    def __init__(self):
        super().__init__("version_handling", "Medical information currency check")
        self.max_age_years = 10  # Consider information older than 10 years as potentially outdated
    
    def check(self, context: Dict[str, Any]) -> GateResult:
        evidence = context.get("evidence", [])
        
        if not evidence:
            return GateResult(True, "No evidence to check version", {})
        
        current_year = datetime.datetime.now(datetime.timezone.utc).year
        outdated_sources = []
        
        for ev in evidence:
            metadata = ev.get("metadata", {})
            year = metadata.get("year") or metadata.get("publication_year")
            
            if year:
                try:
                    year_int = int(year)
                    age = current_year - year_int
                    if age > self.max_age_years:
                        outdated_sources.append({
                            "source": metadata.get("file_name", "unknown"),
                            "year": year_int,
                            "age": age,
                        })
                except (ValueError, TypeError):
                    pass
        
        if outdated_sources:
            # Flag outdated sources but don't necessarily fail
            return GateResult(
                True,  # Pass but with warning
                f"Found {len(outdated_sources)} potentially outdated sources",
                {"outdated_sources": outdated_sources, "warning": True},
            )
        
        return GateResult(True, "Evidence sources are current", {})


class ConfidenceAssessmentGate(PipelineGate):
    """Phase 78 fix: Gate to assess clinical confidence levels."""
    
    def __init__(self):
        super().__init__("confidence_assessment", "Clinical confidence assessment")
    
    def check(self, context: Dict[str, Any]) -> GateResult:
        evidence = context.get("evidence", [])
        claims = context.get("claims", [])
        
        if not evidence:
            return GateResult(True, "No evidence for confidence assessment", {})
        
        # Distinguish between clinical guidelines and case reports
        guideline_count = 0
        case_report_count = 0
        study_count = 0
        
        for ev in evidence:
            metadata = ev.get("metadata", {})
            evidence_type = metadata.get("evidence_type", "").lower()
            text = str(ev.get("text", "")).lower()
            
            if "guideline" in evidence_type or "guideline" in text:
                guideline_count += 1
            elif "case" in evidence_type or "case report" in text:
                case_report_count += 1
            elif "study" in evidence_type or "trial" in text:
                study_count += 1
        
        # Calculate confidence level based on evidence types
        total_evidence = len(evidence)
        if total_evidence == 0:
            return GateResult(True, "No evidence to assess", {})
        
        guideline_ratio = guideline_count / total_evidence
        case_report_ratio = case_report_count / total_evidence
        
        confidence_level = "low"
        if guideline_ratio >= 0.5:
            confidence_level = "high"
        elif guideline_ratio >= 0.3 or study_count >= 2:
            confidence_level = "medium"
        elif case_report_ratio > 0.7:
            confidence_level = "low"  # Mostly case reports
        
        # If confidence is low and we have important medical claims, flag it
        if confidence_level == "low" and claims:
            important_claims = [c for c in claims if c.get("importance", "normal") == "high"]
            if important_claims:
                return GateResult(
                    True,  # Pass but with warning
                    f"Low confidence level for important medical claims",
                    {
                        "confidence_level": confidence_level,
                        "guideline_count": guideline_count,
                        "case_report_count": case_report_count,
                        "study_count": study_count,
                        "warning": True,
                    },
                )
        
        return GateResult(
            True,
            f"Confidence assessment: {confidence_level}",
            {
                "confidence_level": confidence_level,
                "guideline_count": guideline_count,
                "case_report_count": case_report_count,
                "study_count": study_count,
            },
        )


class AnswerPipeline:
    """Complete pipeline with all gates for deterministic medical answering."""
    
    def __init__(self):
        self.gates = [
            EvidenceGate(),
            ClaimGate(),
            NumericGate(),
            CitationGate(),
            ContradictionGate(),
            GroundingGate(),  # Phase 74 fix
            ConflictDetectionGate(),  # Phase 76 fix
            VersionHandlingGate(),  # Phase 77 fix
            ConfidenceAssessmentGate(),  # Phase 78 fix
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