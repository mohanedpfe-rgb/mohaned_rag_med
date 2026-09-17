"""Safety gate for deterministic medical answers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class SafetyDecision(str, Enum):
    """Decision made by the safety gate."""
    APPROVED = "approved"
    REJECTED = "rejected"
    APPROVED_WITH_CAUTION = "approved_with_caution"
    REVIEW_REQUIRED = "review_required"


@dataclass(frozen=True)
class SafetyGateResult:
    """Result of a safety gate check."""
    decision: SafetyDecision
    passed: bool
    blocked_reasons: List[str]
    warnings: List[str]
    safety_score: float
    recommendations: List[str]


@dataclass(frozen=True)
class SafetyCheck:
    """A single safety check."""
    name: str
    passed: bool
    details: Dict[str, Any]


class SafetyGate:
    """Main safety gate for medical answer generation."""
    
    # Safety category keywords
    DANGEROUS_CONDITIONS = {
        "cancer", "cancer", "tumor", "malignant", "carcinoma",
        "infarction", "stroke", "seizure", " coma",
        "hemorrhage", "rupture", "dissection", "embolism",
        "failure", "shock", "arrest", "crisis",
    }
    
    DANGEROUS_INTERVENTIONS = {
        "surgery", "operation", "resection", "resection",
        "transplant", "bypass", "stent", "catheter",
        "chemotherapy", "radiation", "radiation",
        "biopsy", "puncture", "cannulation", "intubation",
    }
    
    LEGAL_SENSITIVE = {
        "malpractice", "lawsuit", "settlement", "compensation",
        "informed consent", "consent", "consent",
    }
    
    FINANCIAL = {
        "cost", "price", "insurance", "coverage", "bill",
        "copay", "deductible", "premium", "reimbursement",
        "payment", "fee", "charge", "expense",
    }
    
    NON_MEDICAL = {
        "opinion", "recommend", "personal", "advice",
        "buy", "purchase", "order", "avail",
    }
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self._checks: List[SafetyCheck] = []
    
    def check_answer(
        self,
        answer_data: Dict[str, Any],
        claims: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
        question: str,
    ) -> SafetyGateResult:
        """Run all safety checks on an answer."""
        self._checks = []
        
        # Run individual checks
        dangerous_content_check = self._check_dangerous_content(answer_data, claims)
        safety_critical_claims_check = self._check_safety_critical_claims(claims)
        legal_sensitive_check = self._check_legal_sensitive(answer_data)
        financial_check = self._check_financial(answer_data)
        non_medical_check = self._check_non_medical(question, answer_data)
        consistency_check = self._check_consistency(claims)
        source_quality_check = self._check_source_quality(evidence)
        
        self._checks.extend([
            dangerous_content_check,
            safety_critical_claims_check,
            legal_sensitive_check,
            financial_check,
            non_medical_check,
            consistency_check,
            source_quality_check,
        ])
        
        # Calculate overall safety score
        safety_score = self._calculate_safety_score()
        
        # Make decision
        decision, passed, blocked_reasons, warnings, recommendations = self._make_decision(
            safety_score, answer_data, claims, evidence
        )
        
        return SafetyGateResult(
            decision=decision,
            passed=passed,
            blocked_reasons=blocked_reasons,
            warnings=warnings,
            safety_score=safety_score,
            recommendations=recommendations,
        )
    
    def _check_dangerous_content(
        self,
        answer_data: Dict[str, Any],
        claims: List[Dict[str, Any]],
    ) -> SafetyCheck:
        """Check for dangerous medical content."""
        text = answer_data.get("direct_answer", "")
        for claim in claims:
            text += " " + claim.get("text", "")
        
        found_dangerous = []
        for term in self.DANGEROUS_CONDITIONS:
            if re.search(r'\b' + term + r'\b', text, re.I):
                found_dangerous.append(term)
        
        passed = not found_dangerous or not self.strict_mode
        
        return SafetyCheck(
            name="dangerous_content",
            passed=passed,
            details={
                "found_terms": found_dangerous,
                "strict_mode": self.strict_mode,
            },
        )
    
    def _check_safety_critical_claims(self, claims: List[Dict[str, Any]]) -> SafetyCheck:
        """Check for safety-critical claims."""
        critical_claims = []
        
        for claim in claims:
            text = claim.get("text", "").lower()
            status = claim.get("status", "")
            
            # Check if claim is safety-critical
            if self._is_safety_critical_claim(text, status):
                critical_claims.append(claim.get("id", "unknown"))
        
        passed = len(critical_claims) == 0 or not self.strict_mode
        
        return SafetyCheck(
            name="safety_critical_claims",
            passed=passed,
            details={
                "critical_claims": critical_claims,
                "strict_mode": self.strict_mode,
            },
        )
    
    def _is_safety_critical_claim(self, text: str, status: str) -> bool:
        """Check if a claim is safety-critical."""
        safety_indicators = [
            "fatal", "deadly", "lethal", "dangerous", "hazardous",
            "contraindicated", "should not", "avoid", "emergency",
            "immediate", "urgent", "critical",
            "first-line", "preferred", "recommended", "only",
        ]
        
        return (
            any(indicator in text.lower() for indicator in safety_indicators) and
            status in {"SUPPORTED", "PARTIALLY_SUPPORTED"}
        )
    
    def _check_legal_sensitive(self, answer_data: Dict[str, Any]) -> SafetyCheck:
        """Check for legal-sensitive content."""
        text = answer_data.get("direct_answer", "")
        
        found_legal = []
        for term in self.LEGAL_SENSITIVE:
            if re.search(r'\b' + term + r'\b', text, re.I):
                found_legal.append(term)
        
        passed = not found_legal
        
        return SafetyCheck(
            name="legal_sensitive",
            passed=passed,
            details={
                "found_terms": found_legal,
            },
        )
    
    def _check_financial(self, answer_data: Dict[str, Any]) -> SafetyCheck:
        """Check for financial content."""
        text = answer_data.get("direct_answer", "")
        
        found_financial = []
        for term in self.FINANCIAL:
            if re.search(r'\b' + term + r'\b', text, re.I):
                found_financial.append(term)
        
        passed = not found_financial
        
        return SafetyCheck(
            name="financial",
            passed=passed,
            details={
                "found_terms": found_financial,
            },
        )
    
    def _check_non_medical(self, question: str, answer_data: Dict[str, Any]) -> SafetyCheck:
        """Check for non-medical content."""
        text = answer_data.get("direct_answer", "")
        
        found_non_medical = []
        for term in self.NON_MEDICAL:
            if re.search(r'\b' + term + r'\b', text, re.I):
                found_non_medical.append(term)
        
        passed = not found_non_medical
        
        return SafetyCheck(
            name="non_medical",
            passed=passed,
            details={
                "found_terms": found_non_medical,
            },
        )
    
    def _check_consistency(self, claims: List[Dict[str, Any]]) -> SafetyCheck:
        """Check for consistency."""
        contradictions = [
            c for c in claims
            if c.get("contradiction")
        ]
        
        passed = len(contradictions) == 0
        
        return SafetyCheck(
            name="consistency",
            passed=passed,
            details={
                "contradictions": len(contradictions),
            },
        )
    
    def _check_source_quality(self, evidence: List[Dict[str, Any]]) -> SafetyCheck:
        """Check evidence source quality."""
        low_quality = [
            e for e in evidence
            if e.get("page_quality", 1.0) < 0.3 or e.get("text_extraction_quality", 1.0) < 0.3
        ]
        
        passed = len(low_quality) == 0
        
        return SafetyCheck(
            name="source_quality",
            passed=passed,
            details={
                "low_quality_evidence": len(low_quality),
            },
        )
    
    def _calculate_safety_score(self) -> float:
        """Calculate overall safety score."""
        if not self._checks:
            return 0.5
        
        scores = []
        for check in self._checks:
            scores.append(1.0 if check.passed else 0.0)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _make_decision(
        self,
        safety_score: float,
        answer_data: Dict[str, Any],
        claims: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> tuple[SafetyDecision, bool, List[str], List[str], List[str]]:
        """Make safety decision based on checks."""
        blocked_reasons = []
        warnings = []
        recommendations = []
        passed = True
        
        # Check each check
        for check in self._checks:
            if not check.passed:
                passed = False
                reason = f"{check.name} failed"
                blocked_reasons.append(reason)
                
                # Add warnings and recommendations based on check type
                if check.name == "dangerous_content":
                    warnings.append("Contains potentially dangerous medical content")
                    recommendations.append("Verify dangerous content with clinical specialist")
                elif check.name == "safety_critical_claims":
                    warnings.append("Contains safety-critical claims")
                    recommendations.append("Ensure safety-critical claims have high support ratio")
                elif check.name == "legal_sensitive":
                    warnings.append("Contains legal-sensitive content")
                    recommendations.append("Remove legal-sensitive content")
                elif check.name == "financial":
                    warnings.append("Contains financial content")
                    recommendations.append("Remove financial content")
                elif check.name == "non_medical":
                    warnings.append("Contains non-medical content")
                    recommendations.append("Focus on medical content")
                elif check.name == "consistency":
                    warnings.append("Contains contradictions")
                    recommendations.append("Resolve contradictions in evidence")
                elif check.name == "source_quality":
                    warnings.append("Contains low-quality evidence")
                    recommendations.append("Remove low-quality evidence or find better sources")
        
        # Determine decision
        if not passed and self.strict_mode:
            decision = SafetyDecision.REJECTED
        elif not passed and not self.strict_mode:
            decision = SafetyDecision.APPROVED_WITH_CAUTION
        elif safety_score >= 0.8:
            decision = SafetyDecision.APPROVED
        else:
            decision = SafetyDecision.REVIEW_REQUIRED
        
        return decision, passed, blocked_reasons, warnings, recommendations


def run_safety_gate(
    answer_data: Dict[str, Any],
    claims: List[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
    question: str,
    strict_mode: bool = True,
) -> SafetyGateResult:
    """Convenience function to run safety gate."""
    gate = SafetyGate(strict_mode=strict_mode)
    return gate.check_answer(answer_data, claims, evidence, question)