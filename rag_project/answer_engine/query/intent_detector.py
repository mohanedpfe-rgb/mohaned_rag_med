"""Intent detector for medical questions."""
from __future__ import annotations

import re
from typing import List, Dict, Any, Set
from dataclasses import dataclass


@dataclass(frozen=True)
class IntentClassification:
    """Classification of a question's intent."""
    primary_intent: str
    secondary_intents: List[str]
    confidence: float
    keywords: List[str]
    question_type: str
    entities_detected: List[str]


class IntentDetector:
    """Detects the intent of medical questions."""
    
    INTENT_KEYWORDS = {
        "definition": {
            "words": ["what is", "define", "definition", "means", "refers to"],
            "keywords": ["definition", "meaning", "definition"],
            "confidence_boost": 0.9,
        },
        "fact": {
            "words": ["is", "are", "was", "were", "does", "do", "did"],
            "keywords": [],
            "confidence_boost": 0.8,
        },
        "numeric": {
            "words": ["how much", "how many", "what is the value", "value", "range"],
            "keywords": ["mg", "ml", "mmhg", "mmol/l", "%", "dose", "dosage"],
            "confidence_boost": 0.95,
        },
        "dosage": {
            "words": ["dose", "dosage", "how much", "take", "tablet"],
            "keywords": ["mg", "ml", "daily", "twice daily", "once daily"],
            "confidence_boost": 0.9,
        },
        "frequency": {
            "words": ["how often", "frequency", "when", "schedule"],
            "keywords": ["daily", "weekly", "monthly", "every", "hourly"],
            "confidence_boost": 0.85,
        },
        "duration": {
            "words": ["how long", "duration", "long", "period"],
            "keywords": ["weeks", "months", "years", "continuous", "ongoing"],
            "confidence_boost": 0.8,
        },
        "threshold": {
            "words": ["threshold", "cut-off", "cut off", "level", "value"],
            "keywords": ["above", "below", "greater than", "less than"],
            "confidence_boost": 0.85,
        },
        "range": {
            "words": ["normal range", "reference range", "values", "from", "to"],
            "keywords": ["between", "range"],
            "confidence_boost": 0.8,
        },
        "comparison": {
            "words": ["compare", "versus", "vs", "difference", "different"],
            "keywords": ["better", "worse", "higher", "lower"],
            "confidence_boost": 0.9,
        },
        "causal": {
            "words": ["cause", "causes", "caused by", "due to", "reason"],
            "keywords": ["risk factor", "risk for", "predispose"],
            "confidence_boost": 0.85,
        },
        "mechanism": {
            "words": ["mechanism", "pathway", "pathophysiology", "how it works"],
            "keywords": ["how does", "mechanism of"],
            "confidence_boost": 0.9,
        },
        "diagnostic": {
            "words": ["diagnose", "diagnosis", "diagnostic", "test", "criteria"],
            "keywords": ["diagnostic criteria", "workup", "evaluation"],
            "confidence_boost": 0.85,
        },
        "treatment": {
            "words": ["treat", "treatment", "therapy", "drug", "medication"],
            "keywords": ["management", "pharmacologic", "therapy"],
            "confidence_boost": 0.9,
        },
        "management": {
            "words": ["management", "treat", "handle", "approach"],
            "keywords": [],
            "confidence_boost": 0.85,
        },
        "prognosis": {
            "words": ["prognosis", "outcome", "survival", "prognostic"],
            "keywords": ["forecast", "expected", "expect"],
            "confidence_boost": 0.85,
        },
        "list": {
            "words": ["list", "all", "enumerate", "which"],
            "keywords": ["examples", "types", "categories"],
            "confidence_boost": 0.7,
        },
        "multi_hop": {
            "words": ["then", "subsequently", "result", "outcome"],
            "keywords": ["and then", "what happens"],
            "confidence_boost": 0.8,
        },
    }
    
    def __init__(self):
        self._build_intent_index()
    
    def _build_intent_index(self):
        """Build keyword index for fast lookup."""
        self._keyword_to_intents: Dict[str, List[str]] = {}
        
        for intent, data in self.INTENT_KEYWORDS.items():
            for word in data["words"]:
                self._keyword_to_intents.setdefault(word, []).append(intent)
            for kw in data["keywords"]:
                self._keyword_to_intents.setdefault(kw, []).append(intent)
    
    def detect_intent(self, question: str) -> IntentClassification:
        """Detect the intent of a question."""
        if not question:
            return IntentClassification(
                primary_intent="unknown",
                secondary_intents=[],
                confidence=0.0,
                keywords=[],
                question_type="unknown",
                entities_detected=[],
            )
        
        question_lower = question.lower()
        detected_intents: Dict[str, float] = {}
        matched_keywords: List[str] = []
        
        # Check all intents
        for intent, data in self.INTENT_KEYWORDS.items():
            score = 0.0
            
            for word in data["words"]:
                if word in question_lower:
                    score += data["confidence_boost"]
                    matched_keywords.append(word)
            
            for kw in data["keywords"]:
                if kw in question_lower:
                    score += data["confidence_boost"] * 0.7
                    matched_keywords.append(kw)
            
            if score > 0:
                detected_intents[intent] = score
        
        if not detected_intents:
            # Default to factual
            detected_intents["fact"] = 0.5
            matched_keywords.append("factual")
        
        # Sort by score
        sorted_intents = sorted(detected_intents.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_intents[0][0]
        primary_score = sorted_intents[0][1]
        
        secondary = []
        if len(sorted_intents) > 1:
            for intent, score in sorted_intents[1:]:
                if score >= primary_score * 0.7:
                    secondary.append(intent)
        
        # Determine question type
        question_type = self._determine_question_type(primary, secondary, matched_keywords)
        
        return IntentClassification(
            primary_intent=primary,
            secondary_intents=secondary,
            confidence=min(1.0, primary_score),
            keywords=matched_keywords,
            question_type=question_type,
            entities_detected=[],
        )
    
    def _determine_question_type(self, primary: str, secondary: List[str], keywords: List[str]) -> str:
        """Determine the question type."""
        if primary == "definition":
            return "definition"
        if primary == "numeric":
            if "dosage" in keywords:
                return "dosage"
            if "range" in keywords:
                return "range"
            return "numeric_value"
        if primary == "comparison":
            return "comparison"
        if primary == "list":
            return "list"
        if primary == "causal":
            return "causal"
        if primary == "mechanism":
            return "mechanism"
        if primary == "treatment" or primary == "management":
            return "management"
        if primary == "diagnostic":
            return "diagnostic_criteria"
        if primary == "prognosis":
            return "prognosis"
        if primary == "multi_hop":
            return "multi_hop"
        return "fact"


def detect_intent(question: str) -> IntentClassification:
    """Convenience function to detect question intent."""
    detector = IntentDetector()
    return detector.detect_intent(question)
