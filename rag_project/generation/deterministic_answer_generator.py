"""
Advanced Deterministic Answer Generator without LLM dependency.
Uses sophisticated NLP techniques to generate high-quality answers from retrieved evidence.
"""
from __future__ import annotations

import re
import math
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
import difflib

from rag_project.utils.text_utils import meaningful_tokens, keyword_overlap_score
from rag_project.retrieval.hybrid_retriever import RetrievalHit


class QuestionType(Enum):
    """Types of questions for specialized handling."""
    FACTUAL = "factual"  # What is X? When did Y happen?
    DEFINITION = "definition"  # Define X, What does X mean?
    COMPARISON = "comparison"  # Compare X and Y, Differences between X and Y
    PROCEDURAL = "procedural"  # How to do X, Steps for X
    CAUSAL = "causal"  # Why X, What causes X
    NUMERIC = "numeric"  # How much, What is the value of X
    LIST = "list"  # List X, Examples of X
    YES_NO = "yes_no"  # Is X true, Does X cause Y
    TEMPORAL = "temporal"  # When, timeline related
    LOCATION = "location"  # Where, location related
    RELATIONSHIP = "relationship"  # Relationship between X and Y
    UNKNOWN = "unknown"


@dataclass
class Sentence:
    """Represents a sentence with enhanced metadata for extraction."""
    text: str
    source_doc: str
    page_numbers: List[int]
    chunk_id: str
    relevance_score: float = 0.0
    position_in_chunk: int = 0
    contains_numbers: bool = False
    contains_entities: bool = False
    is_definition: bool = False
    is_comparison: bool = False
    is_procedural: bool = False
    is_causal: bool = False
    is_temporal: bool = False
    is_numeric: bool = False
    semantic_role: str = "unknown"  # subject, predicate, object, etc.
    named_entities: List[str] = field(default_factory=list)
    dependency_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedAnswer:
    """An extracted answer component with enhanced metadata."""
    primary_answer: str
    supporting_sentences: List[str]
    confidence: float
    source_document: str
    page_numbers: List[int]
    answer_type: QuestionType
    evidence_summary: str
    citations: List[str] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    evidence_quality: float = 0.0
    answer_coherence: float = 0.0


@dataclass
class SynthesizedAnswer:
    """Final synthesized answer with comprehensive metadata."""
    question: str
    answer_text: str
    answer_type: QuestionType
    confidence: float
    sources_used: List[str]
    evidence_count: int
    reasoning_trace: List[str]
    quality_score: float
    citations: List[Dict[str, Any]]
    extraction_method: str
    coherence_score: float
    factual_accuracy: float
    completeness_score: float
    multi_document: bool = False
    language_detected: str = "unknown"


class AdvancedQuestionAnalyzer:
    """Advanced question analysis with semantic understanding."""
    
    QUESTION_PATTERNS = {
        QuestionType.DEFINITION: [
            r'\bdefine\b', r'\bwhat does .+ mean\b', r'\bmeaning of\b', 
            r'\bdefinition\b', r'\bexplain .+ is\b', r'\bwhat is .+ (defined as|called)\b',
            r'\brefers to\b', r'\bcan be defined as\b'
        ],
        QuestionType.COMPARISON: [
            r'\bcompare\b', r'\bdifference\b', r'\bversus\b', r'\bvs\b', 
            r'\bcontrast\b', r'\bsimilar\b', r'\bbetter\b', r'\bworse\b',
            r'\bdifferent from\b', r'\blike\b', r'\bunlike\b'
        ],
        QuestionType.PROCEDURAL: [
            r'\bhow to\b', r'\bsteps?\b', r'\bprocedure\b', r'\bprocess\b',
            r'\bmethod\b', r'\bway to\b', r'\binstructions?\b', r'\bguide\b',
            r'\bapproach\b', r'\btechnique\b'
        ],
        QuestionType.CAUSAL: [
            r'\bwhy\b', r'\bcause\b', r'\breason\b', r'\blead to\b',
            r'\bresult in\b', r'\bdue to\b', r'\bbecause\b', r'\btrigger\b',
            r'\bbring about\b', r'\bproduce\b'
        ],
        QuestionType.NUMERIC: [
            r'\bhow (much|many)\b', r'\bwhat (value|amount|number|count)\b',
            r'\b\d+\b', r'\b(mg|g|ml|kg|cm|mm|%|units?)\b', r'\bquantity\b',
            r'\bmeasure\b', r'\bcost\b', r'\bprice\b'
        ],
        QuestionType.LIST: [
            r'\blist\b', r'\bexamples?\b', r'\btypes?\b', r'\bkinds?\b',
            r'\binclude\b', r'\bwhat are .+ (types|kinds|examples)\b', r'\bmention\b',
            r'\bidentify\b', r'\bname\b'
        ],
        QuestionType.YES_NO: [
            r'\b(is|are|do|does|can|could|will|would|should)\b .+ \?\s*$',
            r'\b(true|false|correct|right)\b', r'\bexist\b', r'\bpossible\b'
        ],
        QuestionType.TEMPORAL: [
            r'\bwhen\b', r'\btime\b', r'\bdate\b', r'\bbefore\b', r'\bafter\b',
            r'\bduring\b', r'\bperiod\b', r'\b(age|duration|timeline)\b', r'\bwhile\b',
            r'\bsince\b', r'\buntil\b'
        ],
        QuestionType.LOCATION: [
            r'\bwhere\b', r'\blocation\b', r'\bplace\b', r'\bposition\b',
            r'\b(area|region|site)\b', r'\bfound\b', r'\blocated\b'
        ],
        QuestionType.RELATIONSHIP: [
            r'\brelationship\b', r'\brelated\b', r'\bassociated\b', 
            r'\bconnection\b', r'\blink\b', r'\bbetween\b', r'\bwith\b',
            r'\binteract\b', r'\baffect\b'
        ],
    }
    
    def __init__(self):
        self.entity_patterns = {
            'medical': r'\b(?:diabetes|hypertension|insulin|glucose|metformin|cortisol|acth)\b',
            'numeric': r'\b\d+(?:\.\d+)?\s*(?:mg|g|ml|kg|cm|mm|%|units?|mcg|µg)\b',
            'temporal': r'\b\d{4}\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
        }
    
    def analyze(self, question: str) -> Dict[str, Any]:
        """Comprehensive question analysis."""
        question_lower = question.lower()
        
        # Detect question type
        question_type = self._detect_type(question_lower)
        
        # Extract key entities
        entities = self._extract_entities(question)
        
        # Extract key terms
        key_terms = self._extract_key_terms(question)
        
        # Detect complexity
        complexity = self._assess_complexity(question)
        
        # Detect if multi-part question
        is_multi_part = self._is_multi_part(question)
        
        return {
            'question_type': question_type,
            'entities': entities,
            'key_terms': key_terms,
            'complexity': complexity,
            'is_multi_part': is_multi_part,
            'normalized_question': self._normalize_question(question),
        }
    
    def _detect_type(self, question_lower: str) -> QuestionType:
        """Detect the type of question."""
        for qtype, patterns in self.QUESTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return qtype
        
        # Default to factual for simple "what is" questions
        if re.search(r'\bwhat\b', question_lower):
            return QuestionType.FACTUAL
        
        return QuestionType.UNKNOWN
    
    def _extract_entities(self, question: str) -> Dict[str, List[str]]:
        """Extract named entities from question."""
        entities = {}
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, question, re.IGNORECASE)
            if matches:
                entities[entity_type] = list(set(matches))
        return entities
    
    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract key terms from question."""
        tokens = meaningful_tokens(question)
        # Filter out stop words and keep meaningful terms
        stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from'}
        key_terms = [token for token in tokens if token.lower() not in stop_words and len(token) > 2]
        return key_terms
    
    def _assess_complexity(self, question: str) -> str:
        """Assess question complexity."""
        tokens = meaningful_tokens(question)
        sentence_count = len(re.split(r'[.!?]', question))
        
        if len(tokens) <= 5 and sentence_count == 1:
            return "simple"
        elif len(tokens) <= 10 and sentence_count == 1:
            return "moderate"
        else:
            return "complex"
    
    def _is_multi_part(self, question: str) -> bool:
        """Check if question has multiple parts."""
        multi_part_indicators = ['and', 'or', 'but', 'also', 'additionally', 'furthermore']
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in multi_part_indicators)
    
    def _normalize_question(self, question: str) -> str:
        """Normalize question for processing."""
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', question).strip()
        # Remove trailing punctuation
        normalized = re.sub(r'[?!.,;:]+$', '', normalized)
        return normalized


class AdvancedSentenceExtractor:
    """Advanced sentence extraction with NLP techniques."""
    
    def __init__(self):
        self.definition_patterns = [
            r'\b.+\s+is\s+(?:a|an|the|defined\s+as|considered)\s+.+',
            r'\b.+\s+refers\s+to\s+.+',
            r'\b.+\s+means\s+.+',
            r'\bdefinition:\s*.+',
            r'\b.+\s+can\s+be\s+defined\s+as\s+.+',
        ]
        self.comparison_patterns = [
            r'\b(?:compared|similar|different|unlike|like)\s+.+',
            r'\b(?:whereas|while|although|however|conversely)\s+.+',
            r'\b(?:both|either|neither)\s+.+',
            r'\b(?:more|less)\s+.+\s+than\s+.+',
        ]
        self.procedural_patterns = [
            r'\b(?:first|second|third|finally|next|then|initially)\b',
            r'\b(?:step|stage|phase)\s+\d+',
            r'\b(?:should|must|need to|required to)\s+.+',
            r'\b(?:follow|perform|execute|implement)\s+.+',
        ]
        self.causal_patterns = [
            r'\b(?:cause|lead to|result in|trigger|produce)\s+.+',
            r'\b(?:due to|because of|owing to)\s+.+',
            r'\b(?:consequently|therefore|thus|hence)\s+.+',
        ]
        self.temporal_patterns = [
            r'\b\d{4}\b',
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(?:morning|afternoon|evening|night|day|week|month|year)\b',
            r'\b(?:before|after|during|while|since|until)\b',
        ]
        self.numeric_patterns = [
            r'\b\d+(?:\.\d+)?\s*(?:mg|g|ml|kg|cm|mm|%|units?|mcg|µg)\b',
            r'\b\d+(?:\.\d+)?\s*(?:times?|days?|hours?|minutes?)\b',
            r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',
        ]
    
    def extract_sentences(self, hits: List[RetrievalHit]) -> List[Sentence]:
        """Extract sentences with enhanced metadata."""
        sentences = []
        
        for hit in hits:
            text = hit.text or ""
            metadata = hit.metadata or {}
            
            # Split into sentences with advanced splitting
            raw_sentences = self._advanced_sentence_split(text)
            
            for i, sent_text in enumerate(raw_sentences):
                if len(sent_text.strip()) < 15:  # Skip very short sentences
                    continue
                
                sentence = Sentence(
                    text=sent_text.strip(),
                    source_doc=metadata.get('file_name', hit.doc_id),
                    page_numbers=metadata.get('page_numbers', []),
                    chunk_id=metadata.get('chunk_id', ''),
                    position_in_chunk=i,
                    contains_numbers=bool(re.search(r'\d+', sent_text)),
                    contains_entities=bool(re.search(r'\b[A-Z][a-z]+\b', sent_text)),
                    is_definition=self._is_definition(sent_text),
                    is_comparison=self._is_comparison(sent_text),
                    is_procedural=self._is_procedural(sent_text),
                    is_causal=self._is_causal(sent_text),
                    is_temporal=self._is_temporal(sent_text),
                    is_numeric=self._is_numeric(sent_text),
                    semantic_role=self._detect_semantic_role(sent_text),
                    named_entities=self._extract_named_entities(sent_text),
                    dependency_info=self._analyze_dependencies(sent_text),
                )
                sentences.append(sentence)
        
        return sentences
    
    def _advanced_sentence_split(self, text: str) -> List[str]:
        """Advanced sentence splitting."""
        # Split on common sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Handle abbreviations and special cases
        refined_sentences = []
        for sent in sentences:
            # Don't split on abbreviations
            if re.search(r'\b(?:Mr|Dr|Prof|etc|e\.g|i\.e)\.$', sent):
                # Check if next sentence starts with lowercase (continuation)
                # For now, keep as is
                pass
            refined_sentences.append(sent)
        
        return [s for s in refined_sentences if s.strip()]
    
    def _is_definition(self, text: str) -> bool:
        """Check if sentence is a definition."""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.definition_patterns)
    
    def _is_comparison(self, text: str) -> bool:
        """Check if sentence contains comparison."""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.comparison_patterns)
    
    def _is_procedural(self, text: str) -> bool:
        """Check if sentence is procedural."""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.procedural_patterns)
    
    def _is_causal(self, text: str) -> bool:
        """Check if sentence is causal."""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.causal_patterns)
    
    def _is_temporal(self, text: str) -> bool:
        """Check if sentence is temporal."""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.temporal_patterns)
    
    def _is_numeric(self, text: str) -> bool:
        """Check if sentence contains numeric information."""
        return any(re.search(pattern, text) for pattern in self.numeric_patterns)
    
    def _detect_semantic_role(self, text: str) -> str:
        """Detect semantic role of sentence."""
        text_lower = text.lower()
        
        # Check for different roles
        if any(word in text_lower for word in ['define', 'definition', 'means', 'refers']):
            return 'definition'
        elif any(word in text_lower for word in ['cause', 'because', 'due to', 'result']):
            return 'causal'
        elif any(word in text_lower for word in ['compare', 'similar', 'different', 'versus']):
            return 'comparison'
        elif any(word in text_lower for word in ['first', 'second', 'step', 'then', 'next']):
            return 'procedural'
        elif any(word in text_lower for word in ['example', 'such as', 'including']):
            return 'example'
        elif any(word in text_lower for word in ['however', 'although', 'but', 'conversely']):
            return 'contrast'
        elif any(word in text_lower for word in ['therefore', 'thus', 'consequently', 'hence']):
            return 'conclusion'
        else:
            return 'statement'
    
    def _extract_named_entities(self, text: str) -> List[str]:
        """Extract potential named entities."""
        # Simple pattern-based entity extraction
        medical_entities = re.findall(r'\b(?:diabetes|hypertension|insulin|glucose|metformin|cortisol|acth|pancreas)\b', text, re.IGNORECASE)
        numeric_entities = re.findall(r'\b\d+(?:\.\d+)?\s*(?:mg|g|ml|kg|cm|mm|%|units?|mcg|µg)\b', text, re.IGNORECASE)
        
        return list(set(medical_entities + numeric_entities))
    
    def _analyze_dependencies(self, text: str) -> Dict[str, Any]:
        """Simple dependency analysis."""
        return {
            'has_subject': bool(re.search(r'\b(?:the|a|an)\s+[a-z]+\b', text, re.IGNORECASE)),
            'has_predicate': bool(re.search(r'\b(?:is|are|was|were|be|been|being)\b', text, re.IGNORECASE)),
            'has_object': bool(re.search(r'\b(?:to|for|with|by|from)\s+[a-z]+\b', text, re.IGNORECASE)),
            'sentence_length': len(text.split()),
        }


class AdvancedEvidenceRanker:
    """Advanced evidence ranking with multiple signals."""
    
    def __init__(self):
        self.weights = {
            'keyword_overlap': 0.3,
            'semantic_relevance': 0.25,
            'position_boost': 0.15,
            'type_match': 0.2,
            'source_diversity': 0.1,
        }
    
    def rank_sentences(self, sentences: List[Sentence], question: str, 
                      question_analysis: Dict[str, Any]) -> List[Sentence]:
        """Rank sentences using multiple signals."""
        question_tokens = set(meaningful_tokens(question.lower()))
        question_type = question_analysis.get('question_type', QuestionType.FACTUAL)
        key_terms = set(question_analysis.get('key_terms', []))
        
        for sentence in sentences:
            sentence_tokens = set(meaningful_tokens(sentence.text.lower()))
            
            # Signal 1: Keyword overlap
            keyword_score = self._calculate_keyword_overlap(question_tokens, sentence_tokens, key_terms)
            
            # Signal 2: Semantic relevance
            semantic_score = self._calculate_semantic_relevance(sentence, question_type)
            
            # Signal 3: Position boost
            position_score = self._calculate_position_boost(sentence)
            
            # Signal 4: Type match
            type_score = self._calculate_type_match(sentence, question_type)
            
            # Signal 5: Entity match
            entity_score = self._calculate_entity_match(sentence, question_analysis)
            
            # Combined score
            combined_score = (
                keyword_score * self.weights['keyword_overlap'] +
                semantic_score * self.weights['semantic_relevance'] +
                position_score * self.weights['position_boost'] +
                type_score * self.weights['type_match'] +
                entity_score * 0.15  # Additional entity weight
            )
            
            sentence.relevance_score = combined_score
        
        return sorted(sentences, key=lambda s: s.relevance_score, reverse=True)
    
    def _calculate_keyword_overlap(self, question_tokens: Set[str], 
                                   sentence_tokens: Set[str], 
                                   key_terms: Set[str]) -> float:
        """Calculate keyword overlap score."""
        if not question_tokens:
            return 0.0
        
        # Regular overlap
        overlap = len(question_tokens & sentence_tokens)
        base_score = overlap / len(question_tokens)
        
        # Key terms bonus
        key_overlap = len(key_terms & sentence_tokens)
        key_bonus = key_overlap / max(len(key_terms), 1) if key_terms else 0.0
        
        return min(1.0, base_score + key_bonus * 0.5)
    
    def _calculate_semantic_relevance(self, sentence: Sentence, 
                                      question_type: QuestionType) -> float:
        """Calculate semantic relevance based on sentence features."""
        score = 0.0
        
        # Type-specific relevance
        if question_type == QuestionType.DEFINITION and sentence.is_definition:
            score += 0.8
        elif question_type == QuestionType.COMPARISON and sentence.is_comparison:
            score += 0.8
        elif question_type == QuestionType.PROCEDURAL and sentence.is_procedural:
            score += 0.8
        elif question_type == QuestionType.CAUSAL and sentence.is_causal:
            score += 0.8
        elif question_type == QuestionType.NUMERIC and sentence.is_numeric:
            score += 0.8
        elif question_type == QuestionType.TEMPORAL and sentence.is_temporal:
            score += 0.8
        
        # General relevance signals
        if sentence.contains_entities:
            score += 0.2
        if sentence.semantic_role in ['definition', 'causal', 'comparison']:
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_position_boost(self, sentence: Sentence) -> float:
        """Calculate position-based boost."""
        # Earlier sentences often more relevant
        return 1.0 / (1.0 + sentence.position_in_chunk * 0.15)
    
    def _calculate_type_match(self, sentence: Sentence, 
                             question_type: QuestionType) -> float:
        """Calculate type-specific match score."""
        type_mapping = {
            QuestionType.DEFINITION: sentence.is_definition,
            QuestionType.COMPARISON: sentence.is_comparison,
            QuestionType.PROCEDURAL: sentence.is_procedural,
            QuestionType.CAUSAL: sentence.is_causal,
            QuestionType.NUMERIC: sentence.is_numeric,
            QuestionType.TEMPORAL: sentence.is_temporal,
        }
        
        return 1.0 if type_mapping.get(question_type, False) else 0.3
    
    def _calculate_entity_match(self, sentence: Sentence, 
                                question_analysis: Dict[str, Any]) -> float:
        """Calculate entity match score."""
        question_entities = question_analysis.get('entities', {})
        sentence_entities = set(sentence.named_entities)
        
        if not question_entities:
            return 0.5  # Neutral score when no entities to match
        
        # Check if sentence contains entities from question
        all_question_entities = set()
        for entity_list in question_entities.values():
            all_question_entities.update(entity_list)
        
        if not all_question_entities:
            return 0.5
        
        overlap = len(sentence_entities & all_question_entities)
        return overlap / len(all_question_entities)


class AdvancedEvidenceSynthesizer:
    """Advanced evidence synthesis with multi-document capabilities."""
    
    def __init__(self):
        self.similarity_threshold = 0.7
        self.deduplication_threshold = 0.85
        self.min_evidence_quality = 0.3
    
    def synthesize(self, sentences: List[Sentence], question: str, 
                  question_analysis: Dict[str, Any],
                  max_sentences: int = 7) -> ExtractedAnswer:
        """Synthesize evidence into coherent answer."""
        if not sentences:
            return self._empty_answer(question, question_analysis)
        
        # Filter by quality
        quality_filtered = [s for s in sentences if s.relevance_score >= self.min_evidence_quality]
        
        if not quality_filtered:
            quality_filtered = sentences[:5]  # Fallback to top 5
        
        # Advanced deduplication
        unique_sentences = self._advanced_deduplication(quality_filtered)
        
        # Group by semantic role
        grouped = self._group_by_semantic_role(unique_sentences)
        
        # Select best from each group
        selected = self._select_diverse_evidence(grouped, max_sentences)
        
        # Build answer
        primary_answer = self._build_primary_answer(selected, question_analysis)
        supporting_sentences = [s.text for s in selected[1:]]
        
        # Calculate comprehensive metrics
        confidence = self._calculate_confidence(selected, question)
        quality = self._assess_evidence_quality(selected)
        coherence = self._assess_coherence(selected)
        
        # Build reasoning steps
        reasoning_steps = self._build_reasoning_steps(selected, question_analysis)
        
        # Build citations
        citations = self._build_citations(selected)
        
        # Extract source info
        source_doc = selected[0].source_doc if selected else "unknown"
        page_numbers = list(set([p for s in selected for p in s.page_numbers]))
        
        return ExtractedAnswer(
            primary_answer=primary_answer,
            supporting_sentences=supporting_sentences,
            confidence=confidence,
            source_document=source_doc,
            page_numbers=page_numbers,
            answer_type=question_analysis.get('question_type', QuestionType.UNKNOWN),
            evidence_summary=self._build_evidence_summary(selected),
            citations=citations,
            reasoning_steps=reasoning_steps,
            evidence_quality=quality,
            answer_coherence=coherence,
        )
    
    def _advanced_deduplication(self, sentences: List[Sentence]) -> List[Sentence]:
        """Advanced deduplication using multiple similarity measures."""
        unique = []
        
        for sentence in sentences:
            is_duplicate = False
            
            for existing in unique:
                # Text similarity
                text_sim = self._text_similarity(sentence.text, existing.text)
                
                # Semantic similarity (based on features)
                semantic_sim = self._semantic_similarity(sentence, existing)
                
                # Combined similarity
                combined_sim = (text_sim + semantic_sim) / 2
                
                if combined_sim > self.deduplication_threshold:
                    is_duplicate = True
                    # Keep the higher scored one
                    if sentence.relevance_score > existing.relevance_score:
                        unique.remove(existing)
                        unique.append(sentence)
                    break
            
            if not is_duplicate:
                unique.append(sentence)
        
        return unique
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using sequence matching."""
        return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _semantic_similarity(self, sent1: Sentence, sent2: Sentence) -> float:
        """Calculate semantic similarity based on features."""
        feature_matches = 0
        total_features = 5
        
        if sent1.is_definition == sent2.is_definition:
            feature_matches += 1
        if sent1.is_comparison == sent2.is_comparison:
            feature_matches += 1
        if sent1.is_procedural == sent2.is_procedural:
            feature_matches += 1
        if sent1.semantic_role == sent2.semantic_role:
            feature_matches += 1
        if set(sent1.named_entities) & set(sent2.named_entities):
            feature_matches += 1
        
        return feature_matches / total_features
    
    def _group_by_semantic_role(self, sentences: List[Sentence]) -> Dict[str, List[Sentence]]:
        """Group sentences by semantic role."""
        grouped = defaultdict(list)
        for sentence in sentences:
            grouped[sentence.semantic_role].append(sentence)
        return dict(grouped)
    
    def _select_diverse_evidence(self, grouped: Dict[str, List[Sentence]], 
                                 max_sentences: int) -> List[Sentence]:
        """Select diverse evidence from different groups."""
        selected = []
        
        # Prioritize important semantic roles
        role_priority = ['definition', 'causal', 'comparison', 'procedural', 'statement', 'example']
        
        for role in role_priority:
            if role in grouped and len(selected) < max_sentences:
                # Take top sentence from this role
                top_sentence = max(grouped[role], key=lambda s: s.relevance_score)
                selected.append(top_sentence)
        
        # Fill remaining slots with highest scored sentences
        if len(selected) < max_sentences:
            remaining = [s for group in grouped.values() for s in group if s not in selected]
            remaining_sorted = sorted(remaining, key=lambda s: s.relevance_score, reverse=True)
            
            slots_remaining = max_sentences - len(selected)
            selected.extend(remaining_sorted[:slots_remaining])
        
        return sorted(selected, key=lambda s: s.relevance_score, reverse=True)
    
    def _build_primary_answer(self, sentences: List[Sentence], 
                            question_analysis: Dict[str, Any]) -> str:
        """Build primary answer from selected sentences."""
        if not sentences:
            return "No relevant information found."
        
        question_type = question_analysis.get('question_type', QuestionType.FACTUAL)
        key_terms = set(question_analysis.get('key_terms', []))
        
        # Type-specific answer building with key term matching
        if question_type == QuestionType.DEFINITION:
            definition_sentences = [s for s in sentences if s.is_definition]
            if definition_sentences:
                # Pick definition with best key term overlap
                best_def = max(definition_sentences, key=lambda s: self._key_term_overlap(s.text, list(key_terms)))
                return best_def.text
            return sentences[0].text
        
        elif question_type == QuestionType.NUMERIC:
            numeric_sentences = [s for s in sentences if s.is_numeric]
            if numeric_sentences:
                # Pick numeric with best key term overlap
                best_num = max(numeric_sentences, key=lambda s: self._key_term_overlap(s.text, list(key_terms)))
                return best_num.text
            return sentences[0].text
        
        elif question_type == QuestionType.PROCEDURAL:
            procedural_sentences = [s for s in sentences if s.is_procedural]
            if procedural_sentences:
                # Pick procedural with best key term overlap
                best_proc = max(procedural_sentences, key=lambda s: self._key_term_overlap(s.text, list(key_terms)))
                return best_proc.text
            return sentences[0].text
        
        elif question_type == QuestionType.COMPARISON:
            comparison_sentences = [s for s in sentences if s.is_comparison]
            if comparison_sentences:
                # Pick comparison with best key term overlap
                best_comp = max(comparison_sentences, key=lambda s: self._key_term_overlap(s.text, list(key_terms)))
                return best_comp.text
            return sentences[0].text
        
        # Default: pick sentence with best key term overlap
        best_sentence = max(sentences, key=lambda s: self._key_term_overlap(s.text, list(key_terms)))
        return best_sentence.text
    
    def _key_term_overlap(self, text: str, key_terms: List[str]) -> float:
        """Calculate key term overlap for sentence selection."""
        if not key_terms:
            return 0.0
        
        text_lower = text.lower()
        overlap = sum(1 for term in key_terms if term.lower() in text_lower)
        return overlap / len(key_terms)
    
    def _calculate_confidence(self, sentences: List[Sentence], question: str) -> float:
        """Calculate comprehensive confidence score."""
        if not sentences:
            return 0.0
        
        # Base confidence from relevance scores
        avg_relevance = sum(s.relevance_score for s in sentences) / len(sentences)
        
        # Evidence diversity boost
        unique_sources = len(set(s.source_doc for s in sentences))
        diversity_boost = min(0.2, unique_sources * 0.1)
        
        # Semantic role diversity boost
        unique_roles = len(set(s.semantic_role for s in sentences))
        role_boost = min(0.15, unique_roles * 0.05)
        
        # Entity coverage boost
        total_entities = sum(len(s.named_entities) for s in sentences)
        entity_boost = min(0.1, total_entities * 0.02)
        
        confidence = avg_relevance + diversity_boost + role_boost + entity_boost
        return min(0.95, max(0.1, confidence))
    
    def _assess_evidence_quality(self, sentences: List[Sentence]) -> float:
        """Assess overall evidence quality."""
        if not sentences:
            return 0.0
        
        quality_factors = []
        
        # Average relevance
        avg_relevance = sum(s.relevance_score for s in sentences) / len(sentences)
        quality_factors.append(avg_relevance)
        
        # Entity presence
        entity_ratio = sum(1 for s in sentences if s.named_entities) / len(sentences)
        quality_factors.append(entity_ratio)
        
        # Semantic diversity
        role_diversity = len(set(s.semantic_role for s in sentences)) / max(len(sentences), 1)
        quality_factors.append(role_diversity)
        
        return sum(quality_factors) / len(quality_factors)
    
    def _assess_coherence(self, sentences: List[Sentence]) -> float:
        """Assess answer coherence."""
        if len(sentences) < 2:
            return 0.7  # Single sentence is moderately coherent
        
        # Check for logical flow
        coherence_score = 0.7
        
        # Check for connecting patterns
        all_text = " ".join(s.text for s in sentences)
        connecting_words = ['however', 'therefore', 'furthermore', 'moreover', 
                          'consequently', 'additionally', 'also', 'thus', 'hence']
        has_connections = any(word in all_text.lower() for word in connecting_words)
        
        if has_connections:
            coherence_score += 0.2
        
        # Check for thematic consistency
        themes = set()
        for sent in sentences:
            themes.update(sent.named_entities)
        
        if len(themes) > 0:
            coherence_score += 0.1
        
        return min(1.0, coherence_score)
    
    def _build_reasoning_steps(self, sentences: List[Sentence], 
                              question_analysis: Dict[str, Any]) -> List[str]:
        """Build reasoning trace."""
        steps = []
        
        question_type = question_analysis.get('question_type', QuestionType.UNKNOWN)
        steps.append(f"Question identified as {question_type.value} type")
        
        if sentences:
            steps.append(f"Found {len(sentences)} relevant evidence segments")
            steps.append(f"Top evidence score: {sentences[0].relevance_score:.3f}")
            
            semantic_roles = set(s.semantic_role for s in sentences)
            if len(semantic_roles) > 1:
                steps.append(f"Evidence covers {len(semantic_roles)} semantic categories")
            
            unique_sources = len(set(s.source_doc for s in sentences))
            if unique_sources > 1:
                steps.append(f"Information synthesized from {unique_sources} sources")
        
        return steps
    
    def _build_citations(self, sentences: List[Sentence]) -> List[str]:
        """Build citation references."""
        citations = []
        for i, sent in enumerate(sentences):
            citation = f"[S{i+1}] {sent.source_doc} (pages {sent.page_numbers})"
            citations.append(citation)
        return citations
    
    def _build_evidence_summary(self, sentences: List[Sentence]) -> str:
        """Build evidence summary."""
        if not sentences:
            return "No evidence available."
        
        sources = set(s.source_doc for s in sentences)
        roles = set(s.semantic_role for s in sentences)
        
        summary_parts = []
        
        if len(sources) == 1:
            summary_parts.append(f"Based on evidence from {list(sources)[0]}")
        else:
            summary_parts.append(f"Based on evidence from {len(sources)} sources")
        
        if roles:
            role_list = ", ".join(sorted(roles))
            summary_parts.append(f"covering semantic roles: {role_list}")
        
        return ". ".join(summary_parts) + "."
    
    def _empty_answer(self, question: str, question_analysis: Dict[str, Any]) -> ExtractedAnswer:
        """Return empty answer."""
        return ExtractedAnswer(
            primary_answer="No relevant information was found in the indexed documents to answer this question.",
            supporting_sentences=[],
            confidence=0.0,
            source_document="none",
            page_numbers=[],
            answer_type=question_analysis.get('question_type', QuestionType.UNKNOWN),
            evidence_summary="No evidence available.",
            citations=[],
            reasoning_steps=["No evidence could be retrieved for this question"],
            evidence_quality=0.0,
            answer_coherence=0.0,
        )


class AdvancedTemplateEngine:
    """Advanced template engine with context-aware patterns."""
    
    TEMPLATES = {
        QuestionType.FACTUAL: {
            "direct": "{answer}",
            "contextual": "According to the available evidence, {answer}",
            "attribution": "Based on the documents, {answer}",
            "qualified": "The evidence suggests that {answer}",
        },
        QuestionType.DEFINITION: {
            "standard": "{term} is {definition}",
            "formal": "{term} can be defined as {definition}",
            "descriptive": "{term} refers to {definition}",
            "contextual": "According to the sources, {term} is {definition}",
        },
        QuestionType.COMPARISON: {
            "direct": "When comparing {items}, {comparison}",
            "structured": "The comparison between {items} shows {comparison}",
            "balanced": "Both {items} share similarities in {similarities}, but differ in {differences}",
            "analytical": "Analysis of {items} reveals {comparison}",
        },
        QuestionType.PROCEDURAL: {
            "direct": "To {action}, {steps}",
            "structured": "The process for {action} involves: {steps}",
            "stepwise": "Steps for {action}: {steps}",
            "instructional": "To {action}, follow these steps: {steps}",
        },
        QuestionType.CAUSAL: {
            "direct": "{answer}",
            "analytical": "The primary cause of {effect} is {cause}",
            "explanatory": "{cause} leads to {effect}",
            "mechanistic": "The mechanism for {effect} involves {cause}",
            "simple": "{answer}",
        },
        QuestionType.NUMERIC: {
            "direct": "The {quantity} is {value} {unit}",
            "precise": "According to the data, the {quantity} is {value} {unit}",
            "contextual": "The {quantity} measures {value} {unit}",
            "qualified": "The available evidence indicates the {quantity} is {value} {unit}",
        },
        QuestionType.LIST: {
            "direct": "The {items} include: {list}",
            "enumerated": "Examples of {items} are: {list}",
            "comprehensive": "{items} consist of: {list}",
            "categorized": "The main {items} are: {list}",
        },
        QuestionType.YES_NO: {
            "direct": "{answer}. {explanation}",
            "evidenced": "The evidence suggests that {answer}. {explanation}",
            "qualified": "{answer}, based on the available information. {explanation}",
            "contextual": "According to the sources, {answer}. {explanation}",
        },
        QuestionType.TEMPORAL: {
            "direct": "{event} occurred {timeframe}",
            "specific": "The {event} took place {timeframe}",
            "contextual": "{timeframe}, {event} occurred",
            "narrative": "The records indicate that {event} happened {timeframe}",
        },
        QuestionType.LOCATION: {
            "direct": "{subject} is located {location}",
            "specific": "The location of {subject} is {location}",
            "contextual": "{location} is where {subject} is found",
            "descriptive": "{subject} can be found {location}",
        },
        QuestionType.RELATIONSHIP: {
            "direct": "{item1} and {item2} are related through {relationship}",
            "analytical": "The relationship between {item1} and {item2} involves {relationship}",
            "descriptive": "{item1} has a {relationship} relationship with {item2}",
            "contextual": "According to the evidence, {item1} and {item2} are connected via {relationship}",
        },
    }
    
    def generate(self, question_type: QuestionType, extracted_data: Dict[str, Any],
                context: Dict[str, Any]) -> str:
        """Generate answer using context-aware template selection."""
        templates = self.TEMPLATES.get(question_type, self.TEMPLATES[QuestionType.FACTUAL])
        
        # For causal questions, prefer simple direct answer
        if question_type == QuestionType.CAUSAL:
            if "answer" in extracted_data and len(extracted_data["answer"]) > 20:
                return extracted_data["answer"]
        
        # Select template based on context
        template_key = self._select_template(context)
        template = templates.get(template_key, templates.get("direct", templates.get("simple", list(templates.values())[0])))
        
        try:
            answer = template.format(**extracted_data)
            if len(answer) > 20:  # Ensure non-empty answer
                return answer
        except KeyError:
            pass
        
        # Try other templates
        for key, template in templates.items():
            try:
                answer = template.format(**extracted_data)
                if len(answer) > 20:
                    return answer
            except KeyError:
                continue
        
        # Fallback to direct answer
        if "answer" in extracted_data:
            return extracted_data["answer"]
        
        # Final fallback
        return self._fallback_answer(extracted_data)
    
    def _select_template(self, context: Dict[str, Any]) -> str:
        """Select appropriate template based on context."""
        confidence = context.get('confidence', 0.5)
        evidence_count = context.get('evidence_count', 1)
        
        if confidence > 0.8 and evidence_count > 2:
            return "contextual"
        elif confidence > 0.6:
            return "qualified"
        else:
            return "direct"
    
    def _fallback_answer(self, data: Dict[str, Any]) -> str:
        """Generate fallback answer."""
        parts = []
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 10:
                parts.append(value)
        return " ".join(parts) if parts else "No specific answer could be generated."


class AdvancedDataExtractor:
    """Advanced data extraction for template population."""
    
    def extract(self, question: str, question_analysis: Dict[str, Any],
               extracted_answer: ExtractedAnswer) -> Dict[str, Any]:
        """Extract structured data for template generation."""
        question_type = question_analysis.get('question_type', QuestionType.FACTUAL)
        
        base_data = {
            "answer": extracted_answer.primary_answer,
            "definition": extracted_answer.primary_answer,
            "explanation": " ".join(extracted_answer.supporting_sentences[:2]),
        }
        
        # Type-specific extraction
        type_specific = self._extract_type_specific(question, question_type, extracted_answer)
        
        # General extraction
        general = self._extract_general(question, question_analysis)
        
        return {**base_data, **type_specific, **general}
    
    def _extract_type_specific(self, question: str, question_type: QuestionType,
                              extracted_answer: ExtractedAnswer) -> Dict[str, Any]:
        """Extract type-specific data."""
        data = {}
        
        if question_type == QuestionType.DEFINITION:
            data["term"] = self._extract_term(question)
        elif question_type == QuestionType.COMPARISON:
            items = self._extract_comparison_items(question)
            data["items"] = items
            data["item1"] = items.split(" and ")[0] if " and " in items else items
            data["item2"] = items.split(" and ")[1] if " and " in items else ""
            data["comparison"] = extracted_answer.primary_answer
            data["differences"] = extracted_answer.primary_answer
            data["similarities"] = extracted_answer.evidence_summary
        elif question_type == QuestionType.PROCEDURAL:
            data["action"] = self._extract_action(question)
            data["steps"] = extracted_answer.primary_answer
        elif question_type == QuestionType.CAUSAL:
            data["effect"] = self._extract_effect(question)
            data["cause"] = extracted_answer.primary_answer
        elif question_type == QuestionType.NUMERIC:
            data["quantity"] = self._extract_quantity(question)
            data["value"] = self._extract_value(extracted_answer.primary_answer)
            data["unit"] = self._extract_unit(extracted_answer.primary_answer)
        elif question_type == QuestionType.LIST:
            data["items"] = self._extract_list_subject(question)
            data["list"] = extracted_answer.primary_answer
        elif question_type == QuestionType.YES_NO:
            data["answer"] = self._extract_yes_no(extracted_answer.primary_answer)
        elif question_type == QuestionType.TEMPORAL:
            data["event"] = self._extract_event(question)
            data["timeframe"] = self._extract_timeframe(extracted_answer.primary_answer)
        elif question_type == QuestionType.LOCATION:
            data["subject"] = self._extract_subject(question)
            data["location"] = self._extract_location(extracted_answer.primary_answer)
        elif question_type == QuestionType.RELATIONSHIP:
            items = self._extract_relationship_items(question)
            data["item1"] = items[0] if len(items) > 0 else "item 1"
            data["item2"] = items[1] if len(items) > 1 else "item 2"
            data["relationship"] = extracted_answer.primary_answer
        
        return data
    
    def _extract_general(self, question: str, question_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract general data."""
        return {
            "term": self._extract_term(question),
            "subject": self._extract_subject(question),
            "items": self._extract_items_general(question),
        }
    
    def _extract_term(self, question: str) -> str:
        """Extract the term being defined."""
        patterns = [
            r'(?:what is|define)\s+(.+?)(?:\?|$)',
            r'(?:definition of|meaning of)\s+(.+?)(?:\?|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return "the term"
    
    def _extract_comparison_items(self, question: str) -> str:
        """Extract items being compared."""
        patterns = [
            r'(.+?)\s+(?:and|vs|versus)\s+(.+)',
            r'compare\s+(.+?)\s+and\s+(.+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return f"{match.group(1)} and {match.group(2)}"
        return "the items"
    
    def _extract_action(self, question: str) -> str:
        """Extract the action from procedural questions."""
        match = re.search(r'how to\s+(.+?)(?:\?|$)', question, re.IGNORECASE)
        return match.group(1).strip() if match else "perform the action"
    
    def _extract_effect(self, question: str) -> str:
        """Extract the effect from causal questions."""
        match = re.search(r'why\s+(?:is|are|was|were)?\s*(.+?)(?:\?|$)', question, re.IGNORECASE)
        if match:
            effect = match.group(1).strip()
            # Clean up common patterns
            effect = re.sub(r'^(?:is|are|was|were)\s+', '', effect)
            return effect
        return "the effect"
    
    def _extract_quantity(self, question: str) -> str:
        """Extract the quantity being asked about."""
        match = re.search(r'how (?:much|many)\s+(.+?)(?:\?|$)', question, re.IGNORECASE)
        return match.group(1).strip() if match else "the quantity"
    
    def _extract_value(self, text: str) -> str:
        """Extract numeric value from text."""
        match = re.search(r'\d+(?:\.\d+)?', text)
        return match.group(0) if match else "unknown"
    
    def _extract_unit(self, text: str) -> str:
        """Extract unit from text."""
        match = re.search(r'\b(mg|g|ml|kg|cm|mm|%|units?|mcg|µg)\b', text, re.IGNORECASE)
        return match.group(1) if match else ""
    
    def _extract_list_subject(self, question: str) -> str:
        """Extract subject for list questions."""
        match = re.search(r'(?:list|examples?|types?)\s+(?:of|for)?\s*(.+?)(?:\?|$)', question, re.IGNORECASE)
        return match.group(1).strip() if match else "the items"
    
    def _extract_yes_no(self, text: str) -> str:
        """Extract yes/no answer from text."""
        text_lower = text.lower()
        if any(word in text_lower for word in ['yes', 'true', 'correct', 'confirmed']):
            return "Yes"
        elif any(word in text_lower for word in ['no', 'false', 'incorrect', 'not']):
            return "No"
        else:
            return "The evidence is inconclusive"
    
    def _extract_event(self, question: str) -> str:
        """Extract event from temporal questions."""
        match = re.search(r'when\s+(?:did|was|were|is)\s+(.+?)(?:\?|$)', question, re.IGNORECASE)
        return match.group(1).strip() if match else "the event"
    
    def _extract_timeframe(self, text: str) -> str:
        """Extract timeframe from text."""
        temporal_patterns = [
            r'\b\d{4}\b',
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)\b',
        ]
        for pattern in temporal_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return "unknown time"
    
    def _extract_subject(self, question: str) -> str:
        """Extract subject from location questions."""
        match = re.search(r'where\s+(?:is|are|was|were)\s+(.+?)(?:\?|$)', question, re.IGNORECASE)
        return match.group(1).strip() if match else "the subject"
    
    def _extract_location(self, text: str) -> str:
        """Extract location from text."""
        location_patterns = [
            r'\bin\s+(.+?)(?:\.|,|$)',
            r'\bat\s+(.+?)(?:\.|,|$)',
            r'\blocated\s+(?:in|at)\s+(.+?)(?:\.|,|$)',
        ]
        for pattern in location_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return "unknown location"
    
    def _extract_relationship_items(self, question: str) -> List[str]:
        """Extract items from relationship questions."""
        patterns = [
            r'(.+?)\s+(?:and|with)\s+(.+)',
            r'relationship\s+between\s+(.+?)\s+and\s+(.+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return [match.group(1).strip(), match.group(2).strip()]
        return ["item 1", "item 2"]
    
    def _extract_items_general(self, question: str) -> str:
        """Extract items in general context."""
        match = re.search(r'(.+?)\s+(?:and|or)\s+(.+)', question, re.IGNORECASE)
        if match:
            return f"{match.group(1)} and {match.group(2)}"
        return "the items"


class AdvancedQualityAssessor:
    """Advanced quality assessment for generated answers."""
    
    def __init__(self):
        self.optimal_length_range = (60, 400)
        self.quality_weights = {
            'relevance': 0.25,
            'coherence': 0.20,
            'completeness': 0.15,
            'factual_accuracy': 0.20,
            'evidence_support': 0.10,
            'length_appropriateness': 0.10,
        }
    
    def assess(self, answer: str, question: str, extracted_answer: ExtractedAnswer,
              question_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive quality assessment."""
        scores = {}
        
        # Relevance score
        scores['relevance'] = self._assess_relevance(answer, question)
        
        # Coherence score
        scores['coherence'] = self._assess_coherence(answer)
        
        # Completeness score
        scores['completeness'] = self._assess_completeness(answer, question, question_analysis)
        
        # Factual accuracy score
        scores['factual_accuracy'] = self._assess_factual_accuracy(answer, extracted_answer)
        
        # Evidence support score
        scores['evidence_support'] = self._assess_evidence_support(extracted_answer)
        
        # Length appropriateness score
        scores['length_appropriateness'] = self._assess_length(answer)
        
        # Overall quality score
        scores['overall'] = sum(
            scores[key] * self.quality_weights[key] 
            for key in self.quality_weights
        )
        
        return scores
    
    def _assess_relevance(self, answer: str, question: str) -> float:
        """Assess answer relevance to question."""
        return keyword_overlap_score(answer, question)
    
    def _assess_coherence(self, answer: str) -> float:
        """Assess answer coherence."""
        sentences = re.split(r'[.!?]', answer)
        
        if len(sentences) < 2:
            return 0.6  # Single sentence
        
        # Check for connecting patterns
        connecting_words = ['however', 'therefore', 'furthermore', 'moreover', 
                          'consequently', 'additionally', 'also', 'thus', 'hence']
        has_connections = any(word in answer.lower() for word in connecting_words)
        
        base_score = 0.7
        if has_connections:
            base_score += 0.2
        
        # Check sentence variety
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if sentence_lengths and max(sentence_lengths) > 2 * min(sentence_lengths):
            base_score += 0.1  # Good variety
        
        return min(1.0, base_score)
    
    def _assess_completeness(self, answer: str, question: str, 
                           question_analysis: Dict[str, Any]) -> float:
        """Assess answer completeness."""
        question_tokens = set(meaningful_tokens(question.lower()))
        answer_tokens = set(meaningful_tokens(answer.lower()))
        
        # Key term coverage
        key_terms = set(question_analysis.get('key_terms', []))
        key_coverage = len(key_terms & answer_tokens) / max(len(key_terms), 1)
        
        # General term coverage
        general_coverage = len(question_tokens & answer_tokens) / max(len(question_tokens), 1)
        
        # Combined score
        completeness = (key_coverage * 0.7 + general_coverage * 0.3)
        return min(1.0, completeness * 1.2)  # Boost for high coverage
    
    def _assess_factual_accuracy(self, answer: str, extracted_answer: ExtractedAnswer) -> float:
        """Assess factual accuracy based on evidence alignment."""
        # Use evidence quality as proxy for factual accuracy
        evidence_quality = extracted_answer.evidence_quality
        
        # Check for hallucinations (answer parts not in evidence)
        answer_tokens = set(meaningful_tokens(answer.lower()))
        evidence_tokens = set()
        for sent in extracted_answer.supporting_sentences:
            evidence_tokens.update(meaningful_tokens(sent.lower()))
        
        # High overlap with evidence suggests factual accuracy
        overlap = len(answer_tokens & evidence_tokens) / max(len(answer_tokens), 1)
        
        factual_score = (evidence_quality * 0.6 + overlap * 0.4)
        return min(1.0, factual_score)
    
    def _assess_evidence_support(self, extracted_answer: ExtractedAnswer) -> float:
        """Assess evidence support."""
        if not extracted_answer.supporting_sentences:
            return 0.0
        
        # More supporting sentences = better support
        support_score = min(1.0, len(extracted_answer.supporting_sentences) / 3.0)
        
        # Higher confidence = better support
        confidence_boost = extracted_answer.confidence * 0.2
        
        return min(1.0, support_score + confidence_boost)
    
    def _assess_length(self, answer: str) -> float:
        """Assess answer length appropriateness."""
        length = len(answer.split())
        
        if length < self.optimal_length_range[0]:
            # Too short
            ratio = length / self.optimal_length_range[0]
            return ratio * 0.7
        elif length > self.optimal_length_range[1]:
            # Too long
            ratio = self.optimal_length_range[1] / length
            return ratio * 0.8
        else:
            # Optimal
            return 1.0


class DeterministicAnswerGenerator:
    """Main deterministic answer generator with advanced architecture."""
    
    def __init__(self):
        # Initialize components
        self.question_analyzer = AdvancedQuestionAnalyzer()
        self.sentence_extractor = AdvancedSentenceExtractor()
        self.evidence_ranker = AdvancedEvidenceRanker()
        self.evidence_synthesizer = AdvancedEvidenceSynthesizer()
        self.template_engine = AdvancedTemplateEngine()
        self.data_extractor = AdvancedDataExtractor()
        self.quality_assessor = AdvancedQualityAssessor()
    
    def generate_answer(self, question: str, hits: List[RetrievalHit]) -> SynthesizedAnswer:
        """Generate high-quality deterministic answer."""
        reasoning_trace = []
        
        # Step 1: Advanced question analysis
        question_analysis = self.question_analyzer.analyze(question)
        reasoning_trace.append(f"Question analyzed: {question_analysis['question_type'].value} type, {question_analysis['complexity']} complexity")
        
        # Step 2: Advanced sentence extraction
        sentences = self.sentence_extractor.extract_sentences(hits)
        reasoning_trace.append(f"Extracted {len(sentences)} sentences with semantic features")
        
        if not sentences:
            reasoning_trace.append("No sentences extracted - returning empty answer")
            return self._empty_answer(question, question_analysis, reasoning_trace)
        
        # Step 3: Advanced evidence ranking
        ranked_sentences = self.evidence_ranker.rank_sentences(sentences, question, question_analysis)
        reasoning_trace.append(f"Ranked sentences using multi-signal scoring (top score: {ranked_sentences[0].relevance_score:.3f})")
        
        # Step 4: Advanced evidence synthesis
        extracted_answer = self.evidence_synthesizer.synthesize(
            ranked_sentences, question, question_analysis
        )
        reasoning_trace.extend(extracted_answer.reasoning_steps)
        
        # Step 5: Template data extraction
        template_data = self.data_extractor.extract(question, question_analysis, extracted_answer)
        
        # Step 6: Context-aware template generation
        context = {
            'confidence': extracted_answer.confidence,
            'evidence_count': len(hits),
            'question_type': question_analysis['question_type'],
        }
        final_answer = self.template_engine.generate(
            question_analysis['question_type'], template_data, context
        )
        reasoning_trace.append("Generated answer using context-aware template engine")
        
        # Step 7: Advanced quality assessment
        quality_scores = self.quality_assessor.assess(
            final_answer, question, extracted_answer, question_analysis
        )
        reasoning_trace.append(f"Quality assessment: overall score {quality_scores['overall']:.3f}")
        
        # Step 8: Build comprehensive citations
        citations = self._build_comprehensive_citations(hits, ranked_sentences)
        
        # Step 9: Determine multi-document status
        multi_document = len(set(h.metadata.get('file_name', h.doc_id) for h in hits)) > 1
        
        return SynthesizedAnswer(
            question=question,
            answer_text=final_answer,
            answer_type=question_analysis['question_type'],
            confidence=extracted_answer.confidence,
            sources_used=list(set(h.metadata.get('file_name', h.doc_id) for h in hits)),
            evidence_count=len(hits),
            reasoning_trace=reasoning_trace,
            quality_score=quality_scores['overall'],
            citations=citations,
            extraction_method="advanced_deterministic",
            coherence_score=quality_scores['coherence'],
            factual_accuracy=quality_scores['factual_accuracy'],
            completeness_score=quality_scores['completeness'],
            multi_document=multi_document,
            language_detected=self._detect_language(question),
        )
    
    def _build_comprehensive_citations(self, hits: List[RetrievalHit], 
                                     sentences: List[Sentence]) -> List[Dict[str, Any]]:
        """Build comprehensive citation information."""
        citations = []
        
        # Create mapping from chunk_id to hit metadata
        hit_metadata = {}
        for hit in hits:
            chunk_id = hit.metadata.get('chunk_id', '')
            if chunk_id:
                hit_metadata[chunk_id] = hit.metadata
        
        for i, sentence in enumerate(sentences[:7]):  # Top 7 sentences
            metadata = hit_metadata.get(sentence.chunk_id, {})
            citation = {
                "id": f"S{i+1}",
                "text": sentence.text,
                "source": metadata.get('file_name', sentence.source_doc),
                "pages": metadata.get('page_numbers', sentence.page_numbers),
                "chunk_id": sentence.chunk_id,
                "relevance": sentence.relevance_score,
                "semantic_role": sentence.semantic_role,
                "named_entities": sentence.named_entities,
            }
            citations.append(citation)
        
        return citations
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection."""
        # Check for French characters
        french_chars = set('àâäéèêëïîôùûüÿç')
        if any(char in text.lower() for char in french_chars):
            return "french"
        return "english"
    
    def _empty_answer(self, question: str, question_analysis: Dict[str, Any],
                     reasoning_trace: List[str]) -> SynthesizedAnswer:
        """Return empty answer when no evidence is available."""
        return SynthesizedAnswer(
            question=question,
            answer_text="I could not find relevant information in the indexed documents to answer this question.",
            answer_type=question_analysis['question_type'],
            confidence=0.0,
            sources_used=[],
            evidence_count=0,
            reasoning_trace=reasoning_trace,
            quality_score=0.0,
            citations=[],
            extraction_method="empty_fallback",
            coherence_score=0.0,
            factual_accuracy=0.0,
            completeness_score=0.0,
            multi_document=False,
            language_detected=self._detect_language(question),
        )