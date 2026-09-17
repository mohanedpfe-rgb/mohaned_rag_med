"""Controlled medical language generation for deterministic answers."""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional
from enum import Enum


class LanguageMode(str, Enum):
    """Language modes for answer generation."""
    FORMAL = "formal"
    CLINICAL = "clinical"
    PATIENT = "patient"
    SCIENTIFIC = "scientific"
    MIXED = "mixed"  # Clinical for medical terms, formal for rest


class LanguageControl:
    """Controls language style and vocabulary for deterministic answers."""
    
    # Medical term dictionaries
    CLINICAL_TERMS = {
        "pain": "pain",
        "swelling": "swelling",
        "inflammation": "inflammation",
        "fever": "fever",
        "headache": "headache",
        "nausea": "nausea",
        "vomiting": "vomiting",
        "diarrhea": "diarrhea",
        "constipation": "constipation",
        "breathlessness": "dyspnea",
        "chest_pain": "chest_pain",
        "palpitations": "palpitations",
        "dizziness": "dizziness",
        "syncope": "syncope",
        "fatigue": "fatigue",
        "weakness": "weakness",
        "weight_loss": "weight_loss",
        "weight_gain": "weight_gain",
        "appetite": "appetite",
        "sleep": "sleep",
        "energy": "energy",
    }
    
    PATIENT_TERMS = {
        "dyspnea": "breathlessness",
        "tachycardia": "fast heart rate",
        "bradycardia": "slow heart rate",
        "hypertension": "high blood pressure",
        "hypotension": "low blood pressure",
        "hyperglycemia": "high blood sugar",
        "hypoglycemia": "low blood sugar",
        "hyperlipidemia": "high cholesterol",
        "arrhythmia": "irregular heartbeat",
        "edema": "swelling",
        "pruritus": "itching",
        "urticaria": "hives",
        "macule": "flat spot",
        "papule": "small bump",
        "nodule": "larger bump",
        "vesicle": "small blister",
        "pustule": "pus-filled bump",
        "ulcer": "open sore",
        "erosion": "shallow open sore",
        "fissure": "crack",
        "scale": "flaky skin",
        "crust": "dried serum or blood",
        "excoriation": "scratch",
        "abscess": "collection of pus",
        "cyst": "sac containing fluid or semi-solid material",
        "polyp": "growth projecting from a surface",
        "wart": "small rough growth",
        "mole": "colored spot on skin",
        "lentigo": "flat brown spot",
        "melanoma": "skin cancer",
        "carcinoma": "cancer",
        "sarcoma": "cancer",
        "metastasis": "spread",
        "remission": "improvement",
        "relapse": "return",
        "recurrence": "return",
        "complication": "problem",
        "sequelae": "consequences",
        "prognosis": "outcome",
        "morbidity": "illness",
        "mortality": "death",
        "etiology": "cause",
        "pathogenesis": "development",
        "pathophysiology": "mechanism",
        "diagnosis": "identification",
        "differential_diagnosis": "possible_conditions",
        "prognosis": "expected_outcome",
        "survival": "living",
        "mortality": "death_rate",
        "prevalence": "how_many_have",
        "incidence": "how_many_get",
        "risk_factor": "thing_that_increases_risk",
        "prophylaxis": "prevention",
        "contraindication": "reason_to_avoid",
        "adverse_effect": "negative_effect",
        "side_effect": "side_effect",
        "interaction": "interaction",
        "toxicity": "harmful_effect",
        "therapeutic": "treatment",
        "pharmacokinetics": "how_body_handles_drug",
        "pharmacodynamics": "how_drug_affects_body",
        "indication": "reason_to_use",
        "dosage": "amount",
        "administration": "how_to_give",
        "route": "method",
        "injection": "injection",
        "infusion": "infusion",
        "oral": "by_mouth",
        "intravenous": "through_vein",
        "intramuscular": "into_muscle",
        "subcutaneous": "under_skin",
        "topical": "on_skin",
        "inhalation": "inhale",
        "instillation": "drop_in",
    }
    
    # Avoid these words in formal/clinical contexts
    AVOID_WORDS = {
        "prove", "proven", "cure", "guarantee", "definitely",
        "always", "never", "everyone", "nobody", "perfect",
        "best", "worst", "perfectly", "completely",
    }
    
    # Safety phrases
    SAFETY_PHRASES = [
        "consult_a_healthcare_provider",
        "seek_medical_advice",
        "professional_medical_attention",
        "not_a_substitute_for_professional_advice",
        "individual_circumstances_may_varry",
    ]
    
    def __init__(self, mode: LanguageMode = LanguageMode.CLINICAL):
        self.mode = mode
        self._build_term_maps()
    
    def _build_term_maps(self):
        """Build term mapping dictionaries."""
        self.patient_to_clinical = {v: k for k, v in self.PATIENT_TERMS.items()}
        self.clinical_to_patient = self.PATIENT_TERMS.copy()
    
    def translate_to_language(self, text: str, target_mode: LanguageMode) -> str:
        """Translate text to target language mode."""
        if target_mode == self.mode:
            return text
        
        if target_mode == LanguageMode.PATIENT:
            return self._to_patient_language(text)
        elif target_mode == LanguageMode.CLINICAL:
            return self._to_clinical_language(text)
        else:
            return text
    
    def _to_patient_language(self, text: str) -> str:
        """Convert clinical terms to patient-friendly terms."""
        result = text
        
        # Replace clinical terms with patient terms
        for clinical, patient in self.clinical_to_patient.items():
            pattern = r'\b' + re.escape(clinical) + r'\b'
            result = re.sub(pattern, patient, result, flags=re.I)
        
        return result
    
    def _to_clinical_language(self, text: str) -> str:
        """Convert patient terms to clinical terms."""
        result = text
        
        # Replace patient terms with clinical terms
        for patient, clinical in self.patient_to_clinical.items():
            pattern = r'\b' + re.escape(patient) + r'\b'
            result = re.sub(pattern, clinical, result, flags=re.I)
        
        return result
    
    def apply_language_rules(self, text: str) -> str:
        """Apply language style rules to text."""
        result = text
        
        # Remove avoid words
        for word in self.AVOID_WORDS:
            pattern = r'\b' + re.escape(word) + r'\b'
            result = re.sub(pattern, "", result, flags=re.I)
        
        # Clean up extra spaces
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def format_sentence(
        self,
        sentence: str,
        mode: Optional[LanguageMode] = None,
        emphasize: bool = False,
        cautionary: bool = False,
    ) -> str:
        """Format a sentence according to language rules."""
        if mode is None:
            mode = self.mode
        
        result = sentence
        
        # Apply language translation
        result = self.translate_to_language(result, mode)
        
        # Apply style rules
        result = self.apply_language_rules(result)
        
        # Add cautionary note if needed
        if cautionary and mode in {LanguageMode.CLINICAL, LanguageMode.FORMAL}:
            safety_phrase = self.SAFETY_PHRASES[0]
            result = f"{result} ({safety_phrase})"
        
        return result
    
    def generate_list_item(
        self,
        item_text: str,
        number: Optional[int] = None,
        bullet: str = "•",
    ) -> str:
        """Generate a formatted list item."""
        if number is not None:
            return f"{number}. {item_text}"
        return f"{bullet} {item_text}"
    
    def generate_table_cell(
        self,
        content: str,
        align: str = "left",
    ) -> str:
        """Generate a table cell with content."""
        return f"| {content} |"
    
    def bold_text(self, text: str) -> str:
        """Make text bold (Markdown)."""
        return f"**{text}**"
    
    def italic_text(self, text: str) -> str:
        """Make text italic (Markdown)."""
        return f"*{text}*"
    
    def code_text(self, text: str) -> str:
        """Make text code-formatted."""
        return f"`{text}`"
    
    def quote_text(self, text: str) -> str:
        """Quote text."""
        return f"> {text}"
    
    def link_text(self, text: str, url: str) -> str:
        """Create a markdown link."""
        return f"[{text}]({url})"
    
    def generate_citation(
        self,
        source_id: str,
        number: Optional[int] = None,
    ) -> str:
        """Generate a citation reference."""
        if number is not None:
            return f"[{number}]"
        return f"[{source_id}]"
    
    def generate_section_header(
        self,
        title: str,
        level: int = 2,
    ) -> str:
        """Generate a section header."""
        return f"{'#' * level} {title}"
    
    def generate_summary(
        self,
        points: List[str],
        mode: Optional[LanguageMode] = None,
    ) -> str:
        """Generate a summary from points."""
        if mode is None:
            mode = self.mode
        
        summary_lines = []
        for i, point in enumerate(points, 1):
            formatted_point = self.format_sentence(point, mode)
            summary_lines.append(self.generate_list_item(formatted_point, i))
        
        return "\n".join(summary_lines)


def get_default_language_control() -> LanguageControl:
    """Get a default language control instance."""
    return LanguageControl(LanguageMode.CLINICAL)


def get_patient_language_control() -> LanguageControl:
    """Get a patient-friendly language control instance."""
    return LanguageControl(LanguageMode.PATIENT)


def get_scientific_language_control() -> LanguageControl:
    """Get a scientific language control instance."""
    return LanguageControl(LanguageMode.SCIENTIFIC)


def apply_language_transform(
    text: str,
    mode: str = "clinical",
) -> str:
    """Apply language transformation to text."""
    if mode == "patient":
        control = get_patient_language_control()
    elif mode == "scientific":
        control = get_scientific_language_control()
    else:
        control = get_default_language_control()
    
    return control.format_sentence(text, LanguageMode(mode))