"""Question classifier with 45 question types."""
from __future__ import annotations

import re
from typing import List, Dict, Any


class QuestionClassifier:
    """Classifies medical questions into 45 explicit types."""
    
    # Intent categories with keywords
    INTENT_CATEGORIES = {
        "definition": [
            "what is", "define", "definition", "means", "meaning",
            "is defined as", "refers to", "what does",
        ],
        "fact": [
            "what is", "what are", "is", "are", "was", "were",
            "does", "do", "did", "can", "could",
        ],
        "numeric": [
            "how much", "how many", "what is the dose", "dose",
            "dosage", "concentration", "concentration", "what is the value",
            "what is the range", "what is the normal", "normal value",
            "threshold", "cut-off", "cut off",
        ],
        "dosage": [
            "dose", "dosage", "how much", "mg", "mcg", "ml", "g",
            "how many mg", "how many ml", "tablet", "tablet",
            "daily dose", "daily dosage", "frequency",
        ],
        "frequency": [
            "how often", "frequency", "when", "when to", "schedule",
            "every", "daily", "weekly", "monthly", "hourly",
        ],
        "duration": [
            "how long", "duration", "long", "time", "period",
            "for how long", "continued", "ongoing",
        ],
        "threshold": [
            "threshold", "cut-off", "cut off", "level", "value",
            "above", "below", "greater than", "less than",
        ],
        "range": [
            "normal range", "reference range", "values", "values",
            "between", "from", "to",
        ],
        "classification": [
            "types", "type", "categories", "category", "classification",
            "stages", "stage", "degrees", "grade", "grading",
        ],
        "staging": [
            "stage", "staging", " TNM", "TNM", "grade", "grading",
            "I", "II", "III", "IV", "stage I", "stage II",
        ],
        "comparison": [
            "vs", "versus", "versus", "compare", "comparison",
            "difference", "different", "better", "worse",
        ],
        "difference": [
            "difference", "differences", "different from", "vs",
            "versus", "versus", "compare",
        ],
        "similarity": [
            "similar", "same as", "similar to", "like",
            "alike", "alike",
        ],
        "list": [
            "list", "list all", "enumerate", "all", "what are",
            "which", "examples", "example",
        ],
        "enumeration": [
            "list", "enumerate", "all", "each", "every",
        ],
        "relationship": [
            "relationship", "relationship", "associated with",
            "link", "linked to", "correlation", "correlate",
        ],
        "cause": [
            "cause", "causes", "caused by", "cause of",
            "due to", "because", "reason",
        ],
        "risk_factor": [
            "risk factor", "risk factor", "risk for", "predisposing",
            "predispose", "factor", "associated with",
        ],
        "mechanism": [
            "mechanism", "mechanism", "pathway", "pathophysiology",
            "how it works", "how does", "mechanism of",
        ],
        "pathophysiology": [
            "pathophysiology", "pathophysiology", "mechanism", "pathway",
        ],
        "diagnostic_criteria": [
            "diagnostic criteria", "diagnostic criteria", "diagnosis",
            "diagnose", "diagnostic", "criteria",
        ],
        "diagnostic_workup": [
            "workup", "work up", "work-up", "evaluation",
            "assess", "assess", "evaluate", "investigation",
        ],
        "treatment": [
            "treat", "treatment", "therapy", "treatment", "management",
            "drug", "medication", "pharmacologic",
        ],
        "management": [
            "management", "treatment", "treatment", "therapy",
            "manage", "handling", "approach",
        ],
        "prevention": [
            "prevent", "prevention", "preventive", "prophylaxis",
            "avoid", "precaution",
        ],
        "contraindication": [
            "contraindication", "contraindicated", "contraindicated",
            "should not", "not for", "avoid",
        ],
        "interaction": [
            "interaction", "interact", "interact with",
            "drug interaction", "interaction", "interacts with",
        ],
        "adverse_effect": [
            "side effect", "side effect", "adverse effect",
            "adverse reaction", "reaction", "toxicity",
        ],
        "prognosis": [
            "prognosis", "outcome", "prognostic", "survival",
            "forecast", "expect", "expected",
        ],
        "epidemiology": [
            "epidemiology", "epidemiology", "prevalence", "incidence",
            "frequency", "rate", "mortality", "morbidity",
        ],
        "temporal": [
            "before", "after", "during", "initially", "subsequently",
            "then", "later", "first", "second",
        ],
        "sequence": [
            "sequence", "order", "first", "then", "after",
            "before", "subsequently",
        ],
        "procedure": [
            "procedure", "surgery", "operation", "intervention",
            "perform", "technique",
        ],
        "table_lookup": [
            "table", "chart", "table", "chart", "list",
        ],
        "figure_lookup": [
            "figure", "figure", "diagram", "image", "scheme",
        ],
        "summary": [
            "summary", "summarize", "overview", "main points",
            "main findings", "main findings",
        ],
        "multi_hop": [
            "multi-hop", "multiple steps", "then", "then",
            "and then", "result", "result",
        ],
        "follow_up": [
            "what about", "how about", "then", "then",
            "next", "subsequent",
        ],
        "cross_reference": [
            "see also", "see also", "refer to", "refer to",
            "related", "related to",
        ],
    }
    
    def __init__(self):
        # Build reverse lookup from keywords to intents
        self.keyword_to_intent: Dict[str, str] = {}
        for intent, keywords in self.INTENT_CATEGORIES.items():
            for kw in keywords:
                self.keyword_to_intent[kw] = intent
    
    def classify(self, question: str) -> List[str]:
        """Classify question into zero or more types."""
        question_lower = question.lower()
        detected = []
        
        # Check all keywords
        for keyword, intent in self.keyword_to_intent.items():
            if keyword in question_lower:
                if intent not in detected:
                    detected.append(intent)
        
        # Default to fact if nothing matched
        if not detected:
            detected = ["fact"]
        
        return detected
    
    def get_primary_intent(self, question: str) -> str:
        """Get the primary (most likely) intent."""
        classifications = self.classify(question)
        
        # Priority ordering
        priority = [
            "definition", "numeric", "dosage", "list", "comparison",
            "treatment", "diagnostic_criteria", "mechanism", "cause",
            "management", "prognosis", "epidemiology", "table_lookup",
            "summary", "multi_hop", "follow_up", "cross_reference",
        ]
        
        for p in priority:
            if p in classifications:
                return p
        
        return classifications[0] if classifications else "fact"


def classify_question(question: str) -> str:
    """Convenience function to classify a question."""
    classifier = QuestionClassifier()
    return classifier.get_primary_intent(question)
