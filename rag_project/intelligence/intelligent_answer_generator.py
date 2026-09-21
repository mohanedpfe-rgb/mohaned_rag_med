"""
Intelligent Answer Generator - Advanced deterministic answer generation without AI/LLMs.
Creates sophisticated, contextually aware, and professionally structured answers.
"""

import re
from typing import Any, List, Dict


@dataclass
class AnswerStructure:
    """Represents the structure of an intelligent answer."""
    introduction: str
    main_content: List[str]
    conclusion: str
    has_examples: bool
    has_procedure: bool
    has_definition: bool
    has_statistics: bool


class IntelligentAnswerGenerator:
    """Advanced deterministic answer generation for medical queries."""
    
    # Medical intent patterns for intelligent classification
    DEFINITION_PATTERNS = [
        r'\bwhat\s+is\b', r'\bdefine\b', r'\bexplain\b.*\bwhat\b', 
        r'\bmeaning\s+of\b', r'\bdefinition\b', r'\bdescribe\b.*\bwhat\b'
    ]
    
    PROCEDURE_PATTERNS = [
        r'\bhow\s+to\b', r'\bhow\s+do\b', r'\bhow\s+does\b', 
        r'\bprocedure\b', r'\bmethod\b', r'\bsteps\b', r'\bprocess\b'
    ]
    
    COMPARISON_PATTERNS = [
        r'\bcompare\b', r'\bdifference\b', r'\bvs\b', r'\bversus\b',
        r'\bbetter\b', r'\bworse\b', r'\bcontrast\b'
    ]
    
    SYMPTOM_PATTERNS = [
        r'\bsymptom', r'\bsign', r'\bmanifestation\b', r'\bwhat\s+are\s+the',
        r'\bindications\b', r'\bclinical.*feature'
    ]
    
    CAUSE_PATTERNS = [
        r'\bcause', r'\bcaused\s+by\b', r'\breason\b', r'\bwhy\b',
        r'\betiology\b', r'\borigination\b', r'\btrigger'
    ]
    
    TREATMENT_PATTERNS = [
        r'\btreat', r'\btherapy\b', r'\bmedication\b', r'\bdrug\b',
        r'\bcure\b', r'\bmanagement\b', r'\bintervention\b'
    ]
    
    @staticmethod
    def classify_question_intent(question: str) -> str:
        """Classify the primary intent of the medical question."""
        question_lower = question.lower()
        
        if any(re.search(pattern, question_lower) for pattern in IntelligentAnswerGenerator.DEFINITION_PATTERNS):
            return "definition"
        elif any(re.search(pattern, question_lower) for pattern in IntelligentAnswerGenerator.PROCEDURE_PATTERNS):
            return "procedure"
        elif any(re.search(pattern, question_lower) for pattern in IntelligentAnswerGenerator.COMPARISON_PATTERNS):
            return "comparison"
        elif any(re.search(pattern, question_lower) for pattern in IntelligentAnswerGenerator.SYMPTOM_PATTERNS):
            return "symptoms"
        elif any(re.search(pattern, question_lower) for pattern in IntelligentAnswerGenerator.CAUSE_PATTERNS):
            return "causes"
        elif any(re.search(pattern, question_lower) for pattern in IntelligentAnswerGenerator.TREATMENT_PATTERNS):
            return "treatment"
        else:
            return "general"
    
    @staticmethod
    def extract_key_terms(question: str) -> List[str]:
        """Extract key medical terms from the question."""
        # Remove common stop words and extract potential medical terms
        stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 3]
        return key_terms
    
    @staticmethod
    def clean_claim_text(text: str) -> str:
        """Clean claim text for intelligent processing."""
        text = str(text or "").strip()
        text = re.sub(r"\[RAG-STRUCTURE[^\]]*\]", "", text, flags=re.I)
        text = re.sub(r"\[FIGURE[^\]]*\]", "", text, flags=re.I)
        text = re.sub(r"\[Section[^\]]*\]", "", text, flags=re.I)
        text = re.sub(r"\[[^\]]+\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    @staticmethod
    def create_introduction(question: str, intent: str, key_terms: List[str]) -> str:
        """Create an intelligent introduction based on question intent."""
        if intent == "definition":
            if key_terms:
                main_term = key_terms[0].capitalize()
                return f"{main_term} is a medical condition characterized by specific clinical features and manifestations."
            return "This medical condition is characterized by specific clinical features and manifestations."
        
        elif intent == "procedure":
            if key_terms:
                main_term = key_terms[0].capitalize()
                return f"The procedure for {main_term} involves several key steps and considerations."
            return "This procedure involves several key steps and considerations for proper implementation."
        
        elif intent == "symptoms":
            return "The clinical presentation of this condition includes various characteristic symptoms and signs."
        
        elif intent == "causes":
            return "This condition can develop due to multiple etiological factors and underlying mechanisms."
        
        elif intent == "treatment":
            return "Management of this condition requires a comprehensive therapeutic approach based on clinical guidelines."
        
        else:
            return "Based on the available medical evidence, the following information addresses this query."
    
    @staticmethod
    def organize_evidence_by_theme(claims: List[Any]) -> Dict[str, List[Any]]:
        """Organize evidence claims into thematic groups for intelligent structuring."""
        themes = {
            "definition_explanation": [],
            "symptoms_manifestations": [],
            "causes_risk_factors": [],
            "treatment_management": [],
            "prognosis_outcomes": [],
            "statistics_data": [],
            "general_evidence": []
        }
        
        for claim in claims:
            text = claim.text.lower()
            
            # Categorize based on content patterns
            if any(word in text for word in ['defined', 'characterized', 'classified', 'type', 'form']):
                themes["definition_explanation"].append(claim)
            elif any(word in text for word in ['symptom', 'sign', 'manifestation', 'clinical', 'present']):
                themes["symptoms_manifestations"].append(claim)
            elif any(word in text for word in ['cause', 'risk', 'factor', 'etiology', 'reason', 'trigger']):
                themes["causes_risk_factors"].append(claim)
            elif any(word in text for word in ['treat', 'therapy', 'medication', 'drug', 'manage', 'intervention']):
                themes["treatment_management"].append(claim)
            elif any(word in text for word in ['prognosis', 'outcome', 'course', 'evolution']):
                themes["prognosis_outcomes"].append(claim)
            elif any(word in text for word in ['%', 'rate', 'percent', 'number', 'statistic', 'prevalence']):
                themes["statistics_data"].append(claim)
            else:
                themes["general_evidence"].append(claim)
        
        return themes
    
    @staticmethod
    def synthesize_theme(theme_claims: List[Any], theme_name: str) -> str:
        """Synthesize claims within a theme into coherent content."""
        if not theme_claims:
            return ""
        
        synthesized_parts = []
        for claim in theme_claims[:4]:  # Limit to top 4 claims per theme
            cleaned = IntelligentAnswerGenerator.clean_claim_text(claim.text)
            if cleaned:
                synthesized_parts.append(cleaned)
        
        if not synthesized_parts:
            return ""
        
        # Join sentences intelligently
        synthesized = " ".join(synthesized_parts)
        # Ensure proper sentence endings
        synthesized = re.sub(r'([.!?])\s+([A-Z])', r'\1 \2', synthesized)
        
        return synthesized
    
    @staticmethod
    def create_conclusion(intent: str, evidence_count: int) -> str:
        """Create an intelligent conclusion based on intent and evidence."""
        if intent == "definition":
            return "This definition encompasses the core characteristics and clinical features of the condition."
        elif intent == "procedure":
            return "Following these steps ensures proper implementation and optimal outcomes."
        elif intent == "symptoms":
            return "Recognition of these symptoms is essential for accurate diagnosis and timely intervention."
        elif intent == "causes":
            return "Understanding these etiological factors is crucial for prevention and management strategies."
        elif intent == "treatment":
            return "The therapeutic approach should be individualized based on patient-specific factors and clinical presentation."
        else:
            return f"This information is based on {evidence_count} evidence sources from the medical literature."
    
    @staticmethod
    def generate_intelligent_answer(question: str, claims: List[Any], route: Any = None) -> str:
        """Generate an intelligent, structured answer without AI/LLMs."""
        if not claims:
            return "I could not find sufficient evidence in the indexed documents to provide a comprehensive answer to this question."
        
        # Classify question intent
        intent = IntelligentAnswerGenerator.classify_question_intent(question)
        key_terms = IntelligentAnswerGenerator.extract_key_terms(question)
        
        # Create intelligent introduction
        introduction = IntelligentAnswerGenerator.create_introduction(question, intent, key_terms)
        
        # Organize evidence by themes
        themed_evidence = IntelligentAnswerGenerator.organize_evidence_by_theme(claims)
        
        # Build main content based on intent and available themes
        main_content = []
        
        if intent == "definition":
            if themed_evidence["definition_explanation"]:
                content = IntelligentAnswerGenerator.synthesize_theme(themed_evidence["definition_explanation"], "definition")
                if content:
                    main_content.append(content)
            if themed_evidence["symptoms_manifestations"]:
                content = IntelligentAnswerGenerator.synthesize_theme(themed_evidence["symptoms_manifestations"], "symptoms")
                if content:
                    main_content.append(content)
        
        elif intent == "symptoms":
            if themed_evidence["symptoms_manifestations"]:
                content = IntelligentAnswerGenerator.synthesize_theme(themed_evidence["symptoms_manifestations"], "symptoms")
                if content:
                    main_content.append(content)
            if themed_evidence["definition_explanation"]:
                content = IntelligentAnswerGenerator.synthesize_theme(themed_evidence["definition_explanation"], "definition")
                if content:
                    main_content.append(content)
        
        elif intent == "causes":
            if themed_evidence["causes_risk_factors"]:
                content = IntelligentAnswerGenerator.synthesize_theme(themed_evidence["causes_risk_factors"], "causes")
                if content:
                    main_content.append(content)
            if themed_evidence["definition_explanation"]:
                content = IntelligentAnswerGenerator.synthesize_theme(themed_evidence["definition_explanation"], "definition")
                if content:
                    main_content.append(content)
        
        elif intent == "treatment":
            if themed_evidence["treatment_management"]:
                content = IntelligentAnswerGenerator.synthesize_theme(themed_evidence["treatment_management"], "treatment")
                if content:
                    main_content.append(content)
            if themed_evidence["general_evidence"]:
                content = IntelligentAnswerGenerator.synthesize_theme(themed_evidence["general_evidence"], "procedure")
                if content:
                    main_content.append(content)
        
        elif intent == "procedure":
            if themed_evidence["treatment_management"]:
                content = IntelligentAnswerGenerator.synthesize_theme(themed_evidence["treatment_management"], "treatment")
                if content:
                    main_content.append(content)
            if themed_evidence["general_evidence"]:
                content = IntelligentAnswerGenerator.synthesize_theme(themed_evidence["general_evidence"], "procedure")
                if content:
                    main_content.append(content)
        
        else:  # general
            # Add most relevant themes first
            for theme in ["definition_explanation", "symptoms_manifestations", "causes_risk_factors", 
                          "treatment_management", "statistics_data", "general_evidence"]:
                if themed_evidence[theme]:
                    content = IntelligentAnswerGenerator.synthesize_theme(themed_evidence[theme], theme)
                    if content:
                        main_content.append(content)
                    if len(main_content) >= 3:  # Limit to 3 themes for general questions
                        break
        
        # If no themed content, use general evidence
        if not main_content and themed_evidence["general_evidence"]:
            content = IntelligentAnswerGenerator.synthesize_theme(themed_evidence["general_evidence"], "general")
            if content:
                main_content.append(content)
        
        # Create conclusion
        conclusion = IntelligentAnswerGenerator.create_conclusion(intent, len(claims))
        
        # Build final answer with proper structure
        answer_parts = []
        
        # Add introduction
        answer_parts.append(introduction)
        
        # Add main content with proper transitions
        if main_content:
            for i, content in enumerate(main_content):
                if i == 0:
                    answer_parts.append(content)
                else:
                    # Add transition phrases for better flow
                    transitions = ["Additionally,", "Furthermore,", "Moreover,", "In addition,"]
                    transition = transitions[i % len(transitions)]
                    answer_parts.append(f"{transition} {content.lower()}")
        
        # Add conclusion
        answer_parts.append(conclusion)
        
        # Build final answer
        final_answer = " ".join(answer_parts)
        
        # Add citations intelligently - add them at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', final_answer)
        cited_sentences = []
        
        for i, sentence in enumerate(sentences):
            if i < len(claims):
                source_num = claims[i].source_numbers[0] if claims[i].source_numbers else (i + 1)
                # Remove trailing punctuation if present
                clean_sentence = sentence.rstrip('.!?')
                cited_sentences.append(f"{clean_sentence} [S{source_num}].")
            else:
                cited_sentences.append(sentence)
        
        final_answer = " ".join(cited_sentences)
        
        return final_answer
    
    @staticmethod
    def create_professional_summary(claims: List[Any], question: str) -> str:
        """Create a professional summary from claims without AI."""
        if not claims:
            return ""
        
        # Extract and clean claim texts
        cleaned_claims = []
        for claim in claims[:5]:
            cleaned = IntelligentAnswerGenerator.clean_claim_text(claim.text)
            if cleaned and len(cleaned) > 10:
                cleaned_claims.append(cleaned)
        
        if not cleaned_claims:
            return ""
        
        # Create a coherent narrative from the claims
        # Remove duplicates and similar sentences
        unique_claims = []
        seen_hashes = set()
        
        for claim in cleaned_claims:
            claim_hash = hash(claim.lower()[:50])  # Simple deduplication
            if claim_hash not in seen_hashes:
                seen_hashes.add(claim_hash)
                unique_claims.append(claim)
        
        # Build professional narrative
        narrative_parts = []
        
        # Add opening statement
        if len(unique_claims) > 0:
            first_claim = unique_claims[0]
            if first_claim[0].islower():
                first_claim = first_claim[0].upper() + first_claim[1:]
            narrative_parts.append(first_claim)
        
        # Add additional claims with transitions
        transitions = ["Additionally,", "Furthermore,", "Moreover,", "In addition,", "Also,"]
        for i, claim in enumerate(unique_claims[1:], 1):
            if i < len(transitions):
                transition = transitions[i]
                narrative_parts.append(f"{transition} {claim.lower()}")
            else:
                narrative_parts.append(f"Also, {claim.lower()}")
        
        # Join into coherent text
        narrative = " ".join(narrative_parts)
        
        # Ensure proper sentence endings
        narrative = re.sub(r'([.!?])\s+([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), narrative)
        
        return narrative