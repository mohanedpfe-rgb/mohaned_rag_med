"""Medical entity resolution with multilingual support."""
from __future__ import annotations

import re
from typing import List, Dict, Any, Set
from dataclasses import dataclass


@dataclass(frozen=True)
class EntityResolution:
    """Resolved entity with normalization."""
    original: str
    canonical: str
    type: str  # disease, drug, symptom, test, etc.
    synonyms: List[str]
    confidence: float


class MedicalEntityResolver:
    """Resolves medical entities across languages."""
    
    # Synonym mappings (disease -> canonical)
    DISEASE_SYNONYMS = {
        "diabetes": ["diabète", "diabete", "diabetes mellitus", "dm"],
        "hypertension": ["hta", "hypertensive disease", "high blood pressure"],
        "obesity": ["obésité", "morbid obesity"],
        "hyperlipidemia": ["hypercholesterolemia", "high cholesterol"],
        "heart failure": ["cardiac failure", "congestive heart failure", "chf"],
        "chronic kidney disease": ["ckd", "renal failure", "kidney failure"],
        "copd": ["chronic obstructive pulmonary disease"],
        "stroke": ["cerebrovascular accident", "cva", "brain attack"],
        "myocardial infarction": ["heart attack", "mi", "cardiac infarction"],
        "cancer": ["malignancy", "malignant tumor", "tumor", "neoplasm"],
    }
    
    # Drug synonyms
    DRUG_SYNONYMS = {
        "metformin": ["glucophage"],
        "lisinopril": ["prinivil", "zestril"],
        "simvastatin": ["zocor"],
        "amlodipine": ["norvasc"],
        "aspirin": ["acetylsalicylic acid", "asa"],
        "insulin": ["human insulin", "rapid-acting insulin"],
    }
    
    # Symptom synonyms
    SYMPTOM_SYNONYMS = {
        "pain": ["douleur", "malaise", "discomfort"],
        "fever": ["féver", "pyrexia", "elevated temperature"],
        "headache": ["céphalée", "cephalic pain"],
        "nausea": ["nausée", "vomiting", "queasy"],
        "fatigue": ["fatigue", "tiredness", "weakness"],
    }
    
    # Test/lab synonyms
    TEST_SYNONYMS = {
        "hba1c": ["hemoglobin a1c", "glycated hemoglobin", "a1c"],
        "creatinine": ["crea", "serum creatinine"],
        "bmi": ["body mass index", "mass index"],
        "cholesterol": ["ldl", "hdl", "total cholesterol"],
        "blood glucose": ["glucose", "sugar", "glycemia"],
    }
    
    def __init__(self):
        # Build reverse lookup
        self._build_lookup_tables()
    
    def _build_lookup_tables(self):
        """Build reverse lookup tables."""
        self._entity_to_canonical: Dict[str, str] = {}
        self._entity_to_type: Dict[str, str] = {}
        
        # Add disease mappings
        for canonical, synonyms in self.DISEASE_SYNONYMS.items():
            self._entity_to_canonical[canonical] = canonical
            self._entity_to_type[canonical] = "disease"
            for syn in synonyms:
                self._entity_to_canonical[syn] = canonical
                self._entity_to_type[syn] = "disease"
        
        # Add drug mappings
        for canonical, synonyms in self.DRUG_SYNONYMS.items():
            self._entity_to_canonical[canonical] = canonical
            self._entity_to_type[canonical] = "drug"
            for syn in synonyms:
                self._entity_to_canonical[syn] = canonical
                self._entity_to_type[syn] = "drug"
        
        # Add symptom mappings
        for canonical, synonyms in self.SYMPTOM_SYNONYMS.items():
            self._entity_to_canonical[canonical] = canonical
            self._entity_to_type[canonical] = "symptom"
            for syn in synonyms:
                self._entity_to_canonical[syn] = canonical
                self._entity_to_type[syn] = "symptom"
        
        # Add test mappings
        for canonical, synonyms in self.TEST_SYNONYMS.items():
            self._entity_to_canonical[canonical] = canonical
            self._entity_to_type[canonical] = "test"
            for syn in synonyms:
                self._entity_to_canonical[syn] = canonical
                self._entity_to_type[syn] = "test"
    
    def resolve(self, text: str) -> List[EntityResolution]:
        """Extract and resolve entities from text."""
        entities = []
        found_canonicals: Set[str] = set()
        
        # Tokenize text
        words = text.lower().split()
        
        # Check for multi-word entities first
        for i in range(len(words)):
            for length in range(3, 0, -1):  # Check 3-word, 2-word, 1-word
                if i + length > len(words):
                    continue
                
                phrase = " ".join(words[i:i+length])
                
                if phrase in self._entity_to_canonical:
                    canonical = self._entity_to_canonical[phrase]
                    entity_type = self._entity_to_type[phrase]
                    
                    if canonical not in found_canonicals:
                        entities.append(EntityResolution(
                            original=phrase,
                            canonical=canonical,
                            type=entity_type,
                            synonyms=[phrase],
                            confidence=0.95,
                        ))
                        found_canonicals.add(canonical)
        
        return entities
    
    def resolve_entity(self, entity: str) -> EntityResolution | None:
        """Resolve a single entity."""
        if not entity:
            return None
        
        entity_lower = entity.lower().strip()
        
        if entity_lower in self._entity_to_canonical:
            canonical = self._entity_to_canonical[entity_lower]
            entity_type = self._entity_to_type.get(entity_lower, "unknown")
            
            return EntityResolution(
                original=entity,
                canonical=canonical,
                type=entity_type,
                synonyms=[entity_lower],
                confidence=0.95,
            )
        
        return None
    
    def normalize(self, text: str) -> str:
        """Normalize text by replacing synonyms with canonical forms."""
        result = text
        
        for synonym, canonical in self._entity_to_canonical.items():
            if synonym != canonical:
                pattern = r"\b" + re.escape(synonym) + r"\b"
                result = re.sub(pattern, canonical, result, flags=re.I)
        
        return result


def resolve_entities(text: str) -> List[EntityResolution]:
    """Convenience function to resolve entities from text."""
    resolver = MedicalEntityResolver()
    return resolver.resolve(text)


def normalize_medical_text(text: str) -> str:
    """Normalize medical text by replacing synonyms with canonical forms."""
    resolver = MedicalEntityResolver()
    return resolver.normalize(text)
