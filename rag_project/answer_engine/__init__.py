"""Answer engine package for deterministic medical answering."""
from __future__ import annotations

from rag_project.answer_engine.schemas import (
    AnswerEnvelope,
    AnswerType,
    ConfidenceLevel,
    Claim,
    ClaimStatus,
    ClaimEvidence,
    EvidenceItem,
    EvidenceQuality,
    SourceLocation,
    ReasoningTrace,
    Relationship,
    Conflict,
)

from rag_project.answer_engine.query.analyzer import analyze_query, QueryPlan
from rag_project.answer_engine.query.classifier import QuestionClassifier, classify_question
from rag_project.answer_engine.query.decomposer import decompose_question, extract_subquestions
from rag_project.answer_engine.query.entity_resolver import MedicalEntityResolver, EntityResolution
from rag_project.answer_engine.query.intent_detector import IntentDetector, IntentClassification

from rag_project.answer_engine.evidence.collector import EvidenceCollector
from rag_project.answer_engine.evidence.deduplicator import EvidenceDeduplicator
from rag_project.answer_engine.evidence.scorer import EvidenceScorer
from rag_project.answer_engine.evidence.normalizer import EvidenceNormalizer
from rag_project.answer_engine.evidence.grouper import EvidenceGrouper
from rag_project.answer_engine.evidence.evidence_graph import EvidenceGraph

from rag_project.answer_engine.claims.extractor import ClaimExtractor
from rag_project.answer_engine.claims.normalizer import ClaimNormalizer
from rag_project.answer_engine.claims.verifier import ClaimVerifier
from rag_project.answer_engine.claims.numeric_verifier import NumericClaimVerifier as NumericVerifier
from rag_project.answer_engine.claims.contradiction_detector import ContradictionDetector

from rag_project.answer_engine.reasoning.engine import ReasoningEngine
from rag_project.answer_engine.reasoning.temporal import TemporalReasoner
from rag_project.answer_engine.reasoning.causal import CausalReasoner
from rag_project.answer_engine.reasoning.multi_hop import MultiHopReasoner
from rag_project.answer_engine.reasoning.relational import RelationshipReasoner as RelationalReasoner
from rag_project.answer_engine.reasoning.comparators import ComparatorEngine
from rag_project.answer_engine.reasoning.aggregators import EvidenceAggregator as AggregatorEngine
from rag_project.answer_engine.reasoning.hierarchy import HierarchyReasoner

from rag_project.answer_engine.answer.planner import plan_answer, AnswerStructure, SectionBuilder
from rag_project.answer_engine.answer.controlled_language import LanguageControl, LanguageMode
from rag_project.answer_engine.answer.citation_builder import Citation, CitationGroup, build_citations
from rag_project.answer_engine.answer.confidence import (
    calculate_answer_confidence,
    ConfidenceComponents,
    ConfidenceResult,
)
from rag_project.answer_engine.answer.abstention import (
    check_abstention,
    AbstentionReason,
    format_abstention_message,
    AbstentionDecision,
)
from rag_project.answer_engine.answer.compilers import (
    BaseAnswerCompiler,
    CompilationResult,
    get_all_compilers,
    compile_answer,
    DefinitionAnswerCompiler,
    FactAnswerCompiler,
    NumericAnswerCompiler,
    ListAnswerCompiler,
    ComparisonAnswerCompiler,
    CausalAnswerCompiler,
    ManagementAnswerCompiler,
    DiagnosticAnswerCompiler,
    PrognosisAnswerCompiler,
    MultiHopAnswerCompiler,
    SummaryAnswerCompiler,
    FollowUpAnswerCompiler,
    CrossReferenceAnswerCompiler,
    TableAnswerCompiler,
    ChainOfThoughtAnswerCompiler,
)

from rag_project.answer_engine.safety.gate import SafetyGate, SafetyGateResult, SafetyDecision
from rag_project.answer_engine.safety.unsupported_claims import UnsupportedClaimsDetector, UnsupportedClaim
from rag_project.answer_engine.safety.risk_detector import RiskDetector, RiskAssessment, RiskLevel

from rag_project.answer_engine.engine import DeterministicAnswerEngine, create_deterministic_engine
from rag_project.answer_engine.pipeline import AnswerPipeline, PipelineResult, PipelineGate, GateResult

from rag_project.answer_engine.benchmark import (
    BenchmarkHarness,
    BenchmarkResult,
    BenchmarkSummary,
    run_sample_benchmark,
)

__all__ = [
    # Schemas
    "AnswerEnvelope",
    "AnswerType",
    "ConfidenceLevel",
    "Claim",
    "ClaimStatus",
    "ClaimEvidence",
    "EvidenceItem",
    "EvidenceQuality",
    "SourceLocation",
    "ReasoningTrace",
    "Relationship",
    "Conflict",
    # Query
    "analyze_query",
    "QueryPlan",
    "QuestionClassifier",
    "classify_question",
    "decompose_question",
    "extract_subquestions",
    "MedicalEntityResolver",
    "EntityResolution",
    "IntentDetector",
    "IntentClassification",
    # Evidence
    "EvidenceCollector",
    "EvidenceDeduplicator",
    "EvidenceScorer",
    "EvidenceNormalizer",
    "EvidenceGrouper",
    "EvidenceGraph",
    # Claims
    "ClaimExtractor",
    "ClaimNormalizer",
    "ClaimVerifier",
    "NumericVerifier",
    "ContradictionDetector",
    # Reasoning
    "ReasoningEngine",
    "TemporalReasoner",
    "CausalReasoner",
    "MultiHopReasoner",
    "RelationalReasoner",
    "ComparatorEngine",
    "AggregatorEngine",
    "HierarchyReasoner",
    # Answer generation
    "plan_answer",
    "AnswerStructure",
    "SectionBuilder",
    "LanguageControl",
    "LanguageMode",
    "Citation",
    "CitationGroup",
    "build_citations",
    "calculate_answer_confidence",
    "ConfidenceComponents",
    "ConfidenceResult",
    "check_abstention",
    "AbstentionReason",
    "format_abstention_message",
    "AbstentionDecision",
    # Compilers
    "BaseAnswerCompiler",
    "CompilationResult",
    "get_all_compilers",
    "compile_answer",
    "DefinitionAnswerCompiler",
    "FactAnswerCompiler",
    "NumericAnswerCompiler",
    "ListAnswerCompiler",
    "ComparisonAnswerCompiler",
    "CausalAnswerCompiler",
    "ManagementAnswerCompiler",
    "DiagnosticAnswerCompiler",
    "PrognosisAnswerCompiler",
    "MultiHopAnswerCompiler",
    "SummaryAnswerCompiler",
    "FollowUpAnswerCompiler",
    "CrossReferenceAnswerCompiler",
    "TableAnswerCompiler",
    "ChainOfThoughtAnswerCompiler",
    # Safety
    "SafetyGate",
    "SafetyGateResult",
    "SafetyDecision",
    "UnsupportedClaimsDetector",
    "UnsupportedClaim",
    "RiskDetector",
    "RiskAssessment",
    "RiskLevel",
    # Engine
    "DeterministicAnswerEngine",
    "create_deterministic_engine",
    # Pipeline
    "AnswerPipeline",
    "PipelineResult",
    "PipelineGate",
    "GateResult",
    # Benchmark
    "BenchmarkHarness",
    "BenchmarkResult",
    "BenchmarkSummary",
    "run_sample_benchmark",
]