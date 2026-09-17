"""Risk detector for medical answers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class RiskLevel(str, Enum):
    """Risk levels for medical information."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class RiskAssessment:
    """Risk assessment for medical content."""
    level: RiskLevel
    risk_score: float  # 0-1
    risk_factors: List[str]
    mitigation_suggestions: List[str]


@dataclass(frozen=True)
class RiskCategory:
    """A risk category."""
    name: str
    score: float
    details: List[str]


class RiskDetector:
    """Detects and assesses risks in medical information."""
    
    # Risk categories and their indicators
    RISK_INDICATORS = {
        "safety_critical": [
            r"\bfatal\b", r"\bdeadly\b", r"\blethal\b", r"\bdangerous\b",
            r"\bcontraindicated\b", r"\bshould not\b", r"\bemergency\b",
            r"\bfatality\b", r"\bmortality\b", r"\bfatal risk\b",
        ],
        "absolute_recommendation": [
            r"\bfirst[- ]?line\b", r"\bpreferred\b", r"\bonly\b",
            r"\balways\b", r"\bnever\b", r"\bgold standard\b",
            r"\bmost effective\b", r"\brecommended\b",
        ],
        "treatment_risk": [
            r"\bsurgery\b", r"\boperation\b", r"\bresection\b",
            r"\btransplant\b", r"\bchemotherapy\b", r"\bradiation\b",
            r"\bbiopsy\b", r"\bside effect\b", r"\badverse effect\b",
        ],
        "diagnostic_risk": [
            r"\bdiagnose\b", r"\bdiagnosis\b", r"\bdifferential\b",
            r"\bcriteria\b", r"\bdiagnostic\b", r"\bconfirm\b",
        ],
        "prognostic_risk": [
            r"\bsurvival\b", r"\bprognosis\b", r"\boutcome\b",
            r"\bmortality\b", r"\bmorbidity\b",
        ],
        "absolute_statement": [
            r"\bcure\b", r"\bguarantee\b", r"\bguaranteed\b",
            r"\bperfect\b", r"\bcomplete\b", r"\btotal\b",
        ],
        "personalized_risk": [
            r"\bmy\b", r"\byour\b", r"\bmyself\b", r"\byourself\b",
            r"\bI\b", r"\bme\b", r"\byou\b",
        ],
        "legal_risk": [
            r"\blaw\b", r"\blegal\b", r"\blawsuit\b", r"\bmalpractice\b",
            r"\bconsent\b", r"\binformed consent\b",
        ],
        "financial_risk": [
            r"\bcost\b", r"\bprice\b", r"\binsurance\b", r"\bcoverage\b",
            r"\bbill\b", r"\bpayment\b",
        ],
    }
    
    # Absolute terms that require high confidence
    ABSOLUTE_PHRASES = [
        "always", "never", "everyone", "nobody", "all", "none",
        "completely", "entirely", "perfectly", "definitely",
        "cure", "guarantee", "guaranteed",
    ]
    
    def __init__(self):
        self._risk_categories: List[RiskCategory] = []
    
    def assess_risk(
        self,
        text: str,
        claims: List[Dict[str, Any]] = None,
        evidence: List[Dict[str, Any]] = None,
    ) -> RiskAssessment:
        """Assess risk level for content."""
        self._risk_categories = []
        
        # Calculate risk for each category
        safety_critical = self._assess_category("safety_critical", text, claims)
        treatment_risk = self._assess_category("treatment_risk", text, claims)
        diagnostic_risk = self._assess_category("diagnostic_risk", text, claims)
        absolute_statement = self._assess_category("absolute_statement", text, claims)
        personalized = self._assess_category("personalized_risk", text, claims)
        legal = self._assess_category("legal_risk", text, claims)
        financial = self._assess_category("financial_risk", text, claims)
        
        self._risk_categories.extend([
            safety_critical,
            treatment_risk,
            diagnostic_risk,
            absolute_statement,
            personalized,
            legal,
            financial,
        ])
        
        # Calculate overall risk score
        risk_score = self._calculate_overall_risk()
        
        # Determine risk level
        if risk_score >= 0.75:
            level = RiskLevel.CRITICAL
        elif risk_score >= 0.50:
            level = RiskLevel.HIGH
        elif risk_score >= 0.30:
            level = RiskLevel.MEDIUM
        elif risk_score >= 0.15:
            level = RiskLevel.LOW
        else:
            level = RiskLevel.NONE
        
        # Generate mitigation suggestions
        mitigation = self._generate_mitigation(level, risk_score)
        
        return RiskAssessment(
            level=level,
            risk_score=risk_score,
            risk_factors=[c.name for c in self._risk_categories if c.score > 0],
            mitigation_suggestions=mitigation,
        )
    
    def _assess_category(
        self,
        category: str,
        text: str,
        claims: List[Dict[str, Any]] = None,
    ) -> RiskCategory:
        """Assess risk for a specific category."""
        text_lower = text.lower()
        score = 0.0
        details = []
        
        # Check indicators
        indicators = self.RISK_INDICATORS.get(category, [])
        
        for pattern in indicators:
            matches = re.findall(pattern, text_lower, re.I)
            if matches:
                score += 0.2 * len(matches)
                details.append(f"Found {len(matches)} '{pattern}' pattern(s)")
        
        # Check for absolute phrases
        if category == "absolute_statement":
            for phrase in self.ABSOLUTE_PHRASES:
                if re.search(r'\b' + phrase + r'\b', text_lower):
                    score += 0.15
        
        # Check claims for risk indicators
        if claims:
            for claim in claims:
                claim_text = claim.get("text", "").lower()
                for pattern in indicators:
                    if re.search(pattern, claim_text, re.I):
                        score += 0.1
                        break
        
        # Cap score at 1.0
        score = min(1.0, score)
        
        return RiskCategory(
            name=category,
            score=score,
            details=details,
        )
    
    def _calculate_overall_risk(self) -> float:
        """Calculate overall risk score."""
        if not self._risk_categories:
            return 0.0
        
        # Weight different categories
        weights = {
            "safety_critical": 1.5,
            "absolute_statement": 1.2,
            "treatment_risk": 1.3,
            "diagnostic_risk": 0.8,
            "personalized_risk": 1.0,
            "legal_risk": 1.1,
            "financial_risk": 0.9,
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for category in self._risk_categories:
            weight = weights.get(category.name, 1.0)
            weighted_sum += category.score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _generate_mitigation(
        self,
        level: RiskLevel,
        risk_score: float,
    ) -> List[str]:
        """Generate risk mitigation suggestions."""
        suggestions = []
        
        if level == RiskLevel.CRITICAL:
            suggestions.append("Consider adding safety disclaimer")
            suggestions.append("Review all absolute statements")
            suggestions.append("Verify all treatment recommendations")
            suggestions.append("Add citations to all safety-critical claims")
        elif level == RiskLevel.HIGH:
            suggestions.append("Review safety-critical statements")
            suggestions.append("Add qualifying language where appropriate")
            suggestions.append("Verify treatment claims")
        elif level == RiskLevel.MEDIUM:
            suggestions.append("Consider adding 'consult healthcare provider' note")
            suggestions.append("Review absolute language")
        elif level == RiskLevel.LOW:
            suggestions.append("Standard disclaimer may be sufficient")
        
        # Add general suggestions
        if risk_score > 0.3:
            suggestions.append("Ensure all claims have supporting evidence")
            suggestions.append("Maintain appropriate level of certainty")
        
        return suggestions
    
    def assess_claim_risk(
        self,
        claim: Dict[str, Any],
    ) -> RiskAssessment:
        """Assess risk for a single claim."""
        text = claim.get("text", "")
        status = claim.get("status", "")
        
        risk = self.assess_risk(text, [claim])
        
        # Adjust risk based on status
        if status == "UNSUPPORTED":
            risk.risk_score = min(1.0, risk.risk_score + 0.2)
            risk.risk_factors.append("unsupported")
        elif status == "CONTRADICTED":
            risk.risk_score = min(1.0, risk.risk_score + 0.3)
            risk.risk_factors.append("contradicted")
        
        # Re-determine level
        if risk.risk_score >= 0.75:
            risk.level = RiskLevel.CRITICAL
        elif risk.risk_score >= 0.50:
            risk.level = RiskLevel.HIGH
        elif risk.risk_score >= 0.30:
            risk.level = RiskLevel.MEDIUM
        elif risk.risk_score >= 0.15:
            risk.level = RiskLevel.LOW
        else:
            risk.level = RiskLevel.NONE
        
        # Re-generate mitigation
        risk.mitigation_suggestions = self._generate_mitigation(risk.level, risk.risk_score)
        
        return risk
    
    def get_high_risk_claims(
        self,
        claims: List[Dict[str, Any]],
    ) -> List[tuple[Dict[str, Any], RiskAssessment]]:
        """Get claims with high or critical risk."""
        high_risk = []
        
        for claim in claims:
            risk = self.assess_claim_risk(claim)
            if risk.level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
                high_risk.append((claim, risk))
        
        return high_risk
    
    def get_risk_summary(
        self,
        risk_assessment: RiskAssessment,
    ) -> Dict[str, Any]:
        """Get a summary of the risk assessment."""
        return {
            "level": risk_assessment.level,
            "risk_score": risk_assessment.risk_score,
            "risk_factors": risk_assessment.risk_factors,
            "mitigations": risk_assessment.mitigation_suggestions,
            "needs_review": risk_assessment.level in {RiskLevel.HIGH, RiskLevel.CRITICAL},
        }


def assess_risk(
    text: str,
    claims: List[Dict[str, Any]] = None,
    evidence: List[Dict[str, Any]] = None,
) -> RiskAssessment:
    """Convenience function to assess risk."""
    detector = RiskDetector()
    return detector.assess_risk(text, claims, evidence)