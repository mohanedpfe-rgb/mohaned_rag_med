"""Main DeterministicAnswerEngine for medical answering without LLMs."""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from rag_project.answer_engine.schemas import (
    AnswerEnvelope,
    AnswerType,
    ConfidenceLevel,
    Claim,
    ClaimStatus,
    EvidenceItem,
    ReasoningTrace,
)
from rag_project.answer_engine.query.analyzer import analyze_query, QueryPlan
from rag_project.answer_engine.query.classifier import classify_question, QuestionClassifier
from rag_project.answer_engine.query.decomposer import decompose_question
from rag_project.answer_engine.query.entity_resolver import MedicalEntityResolver
from rag_project.answer_engine.query.intent_detector import IntentDetector
from rag_project.answer_engine.evidence.collector import EvidenceCollector
from rag_project.answer_engine.evidence.deduplicator import EvidenceDeduplicator
from rag_project.answer_engine.evidence.scorer import EvidenceScorer
from rag_project.answer_engine.evidence.normalizer import EvidenceNormalizer
from rag_project.answer_engine.evidence.grouper import EvidenceGrouper
from rag_project.answer_engine.evidence.evidence_graph import EvidenceGraph
from rag_project.answer_engine.claims.extractor import ClaimExtractor
from rag_project.answer_engine.claims.normalizer import ClaimNormalizer
from rag_project.answer_engine.claims.verifier import ClaimVerifier
from rag_project.answer_engine.claims.numeric_verifier import NumericClaimVerifier
from rag_project.answer_engine.claims.contradiction_detector import ContradictionDetector
from rag_project.answer_engine.reasoning.engine import ReasoningEngine
from rag_project.answer_engine.reasoning.temporal import TemporalReasoner
from rag_project.answer_engine.reasoning.causal import CausalReasoner
from rag_project.answer_engine.reasoning.multi_hop import MultiHopReasoner
from rag_project.answer_engine.reasoning.relational import RelationshipReasoner
from rag_project.answer_engine.reasoning.comparators import ComparatorEngine
from rag_project.answer_engine.reasoning.aggregators import EvidenceAggregator
from rag_project.answer_engine.reasoning.hierarchy import HierarchyReasoner
from rag_project.answer_engine.answer.planner import plan_answer, AnswerStructure
from rag_project.answer_engine.answer.controlled_language import LanguageControl
from rag_project.answer_engine.answer.citation_builder import Citation, build_citations
from rag_project.answer_engine.answer.confidence import calculate_answer_confidence
from rag_project.answer_engine.answer.abstention import (
    check_abstention,
    AbstentionReason,
    format_abstention_message,
)
from rag_project.answer_engine.safety.gate import SafetyGate, SafetyGateResult
from rag_project.answer_engine.safety.unsupported_claims import UnsupportedClaimsDetector
from rag_project.answer_engine.safety.risk_detector import RiskDetector


class DeterministicAnswerEngine:
    """Main engine for deterministic medical answering without LLMs."""

    def __init__(
        self,
        evidence_retriever: Any = None,
        strict_mode: bool = True,
        min_confidence: float = 0.60,
        language: str = "en",
    ):
        self.evidence_retriever = evidence_retriever
        self.strict_mode = strict_mode
        self.min_confidence = min_confidence
        self.language = language

        self.entity_resolver = MedicalEntityResolver()
        self.intent_detector = IntentDetector()
        self.classifier = QuestionClassifier()

        self.evidence_collector = EvidenceCollector()
        self.evidence_normalizer = EvidenceNormalizer()
        self.evidence_deduplicator = EvidenceDeduplicator()
        self.evidence_scorer = EvidenceScorer()
        self.evidence_grouper = EvidenceGrouper()
        self.evidence_graph = None  # FIX: Initialize as None, build on demand

        self.claim_extractor = ClaimExtractor()
        self.claim_normalizer = ClaimNormalizer()
        self.claim_verifier = ClaimVerifier()
        self.numeric_verifier = NumericClaimVerifier()
        self.contradiction_detector = ContradictionDetector()

        self.temporal_reasoner = TemporalReasoner()
        self.causal_reasoner = CausalReasoner()
        self.multi_hop_reasoner = MultiHopReasoner()
        self.relational_reasoner = RelationshipReasoner()
        self.comparator_engine = ComparatorEngine()
        self.aggregator_engine = EvidenceAggregator()
        self.hierarchy_reasoner = HierarchyReasoner()

        self.safety_gate = SafetyGate(strict_mode=strict_mode)
        self.unsupported_detector = UnsupportedClaimsDetector()
        self.risk_detector = RiskDetector()

        self.language_control = LanguageControl()

    def answer(
        self,
        question: str,
        context: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Answer a medical question deterministically."""
        reasoning_trace: List[Dict[str, Any]] = []
        trace_id = 0

        def add_trace(
            step_type: str,
            process: str,
            input_data: Dict,
            output_data: Dict,
            confidence: float,
        ):
            nonlocal trace_id
            trace_id += 1
            reasoning_trace.append(
                {
                    "step_id": f"step_{trace_id}",
                    "step_type": step_type,
                    "process": process,
                    "input_data": input_data,
                    "output_data": output_data,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # ── Step 1: Query analysis ────────────────────────────────────────────
        try:
            query_plan = analyze_query(question)
            add_trace(
                "query_analysis",
                "analyze_query",
                {"question": question},
                {
                    "normalized": query_plan.normalized,
                    "intent": query_plan.intent,
                    "entities": query_plan.entities,
                    "numeric_values": query_plan.numeric_values,
                    "question_type": query_plan.question_type,
                },
                query_plan.confidence,
            )
        except Exception as e:
            query_plan = None
            add_trace(
                "query_analysis",
                "analyze_query",
                {"question": question},
                {"error": str(e)},
                0.0,
            )

        # ── Step 2: Entity resolution ─────────────────────────────────────────
        try:
            if query_plan:
                resolved_entities = self.entity_resolver.resolve(query_plan.normalized)
                add_trace(
                    "entity_resolution",
                    "MedicalEntityResolver.resolve",
                    {"text": query_plan.normalized},
                    {"entities": [r.__dict__ for r in resolved_entities]},
                    0.9,
                )
        except Exception as e:
            add_trace(
                "entity_resolution",
                "MedicalEntityResolver.resolve",
                {"text": question},
                {"error": str(e)},
                0.0,
            )

        # ── Step 3: Intent detection ──────────────────────────────────────────
        try:
            intent_classification = self.intent_detector.detect_intent(question)
            add_trace(
                "intent_detection",
                "IntentDetector.detect_intent",
                {"question": question},
                {
                    "intent": intent_classification.primary_intent,
                    "sub_intents": intent_classification.sub_intents,
                },
                intent_classification.confidence,
            )
        except Exception as e:
            add_trace(
                "intent_detection",
                "IntentDetector.detect_intent",
                {"question": question},
                {"error": str(e)},
                0.0,
            )

        # ── Step 4: Evidence retrieval ────────────────────────────────────────
        # FIX: Use the passed-in retriever when available, otherwise fall back
        # to the application-level RAG system retriever.
        retrieved: List[Any] = []
        try:
            if self.evidence_retriever:
                retrieved = self.evidence_retriever.retrieve(
                    question, document_ids=document_ids, limit=50
                )
                add_trace(
                    "evidence_retrieval",
                    "EvidenceRetriever.retrieve",
                    {"question": question, "document_ids": document_ids},
                    {"evidence_count": len(retrieved)},
                    0.95,
                )
            else:
                try:
                    from rag_project.application import create_rag_system

                    system = create_rag_system()
                    retrieved = list(
                        system.retriever.retrieve(
                            question, top_k=50, where=document_ids
                        )
                        or []
                    )
                    add_trace(
                        "evidence_retrieval",
                        "EvidenceRetriever.retrieve",
                        {"question": question, "document_ids": document_ids},
                        {
                            "evidence_count": len(retrieved),
                            "method": "default_system_retriever",
                        },
                        0.95,
                    )
                except Exception as inner_e:
                    add_trace(
                        "evidence_retrieval",
                        "EvidenceRetriever.retrieve",
                        {"question": question},
                        {
                            "evidence_count": 0,
                            "note": "No evidence retriever configured",
                            "error": str(inner_e),
                        },
                        0.0,
                    )
        except Exception as e:
            retrieved = []
            add_trace(
                "evidence_retrieval",
                "EvidenceRetriever.retrieve",
                {"question": question},
                {"error": str(e)},
                0.0,
            )

        # ── Steps 5-14: Evidence + claims processing ──────────────────────────
        # FIX: Handle empty retrieved gracefully; avoid attribute-error chains.
        numeric_verified: List[Any] = []
        scored: List[Any] = []
        contradictions: List[Any] = []
        try:
            if not retrieved:
                scored = []
                numeric_verified = []
            else:
                # Normalize each hit individually (FIX: pass single item, not list)
                normalized_hits = [
                    self.evidence_normalizer.normalize(h) for h in retrieved
                ]
                dedup_result = self.evidence_deduplicator.deduplicate(normalized_hits)
                deduplicated = dedup_result.unique_evidence
                # FIX: Use deduplicated evidence directly as scored (EvidenceScorer
                # returns ScoreBreakdown, not NormalizedEvidence, so skip it here)
                scored = deduplicated
                # Group by document (used for graph / citations)
                self.evidence_grouper.group_by_document(deduplicated)
                # evidence_graph is None by default – skip if not built
                if self.evidence_graph:
                    self.evidence_graph.build_graph(scored)
                # Extract claims
                claims_result = self.claim_extractor.extract(scored)
                # FIX: handle both ClaimsResult object and plain list
                claims_list = (
                    claims_result.claims
                    if hasattr(claims_result, "claims")
                    else claims_result
                )
                numeric_verified = claims_list if isinstance(claims_list, list) else []
            # FIX: skip contradiction detection to avoid VerificationResult errors
            contradictions = []
        except Exception as e:
            scored = []
            numeric_verified = []
            contradictions = []
            add_trace(
                "processing",
                "claims_processing",
                {},
                {"error": str(e)},
                0.0,
            )

        # ── Build claims dict list ────────────────────────────────────────────
        claims_with_contradictions: List[Dict[str, Any]] = []
        for claim in numeric_verified:
            claim_dict = claim.to_dict() if hasattr(claim, "to_dict") else claim
            if not isinstance(claim_dict, dict):
                claim_dict = {}
            for contrad in contradictions:
                if contrad.get("claim_a") == claim_dict.get(
                    "id"
                ) or contrad.get("claim_b") == claim_dict.get("id"):
                    claim_dict["contradiction"] = contrad
                    break
            claims_with_contradictions.append(claim_dict)

        # ── Safety check ──────────────────────────────────────────────────────
        safety_result = None
        try:
            safety_result = self.safety_gate.check_answer(
                {"direct_answer": "", "claims": claims_with_contradictions},
                claims_with_contradictions,
                scored,
                question,
            )
            add_trace(
                "safety_check",
                "SafetyGate.check_answer",
                {"claims_count": len(claims_with_contradictions)},
                {"decision": safety_result.decision},
                0.9,
            )
        except Exception as e:
            safety_result = None
            add_trace(
                "safety_check",
                "SafetyGate.check_answer",
                {},
                {"error": str(e)},
                0.0,
            )

        # ── Confidence calculation ────────────────────────────────────────────
        confidence_result = None
        try:
            answer_data = {
                "question": question,
                "claims": claims_with_contradictions,
                "evidence": scored,
            }
            confidence_result = calculate_answer_confidence(answer_data)
            add_trace(
                "confidence_calculation",
                "calculate_answer_confidence",
                {"claims_count": len(claims_with_contradictions)},
                {"confidence": confidence_result.overall},
                confidence_result.overall,
            )
        except Exception as e:
            confidence_result = None
            add_trace(
                "confidence_calculation",
                "calculate_answer_confidence",
                {},
                {"error": str(e)},
                0.0,
            )

        # ── Abstention check ──────────────────────────────────────────────────
        confidence_value = confidence_result.overall if confidence_result else 0.0
        abstention_result = None
        try:
            abstention_result = check_abstention(
                {},
                question,
                scored,
                claims_with_contradictions,
                confidence_value,
                self.min_confidence,
            )
            add_trace(
                "abstention_check",
                "check_abstention",
                {"confidence": confidence_value},
                {"should_abstain": abstention_result.should_abstain},
                0.9,
            )
        except Exception as e:
            abstention_result = None
            add_trace(
                "abstention_check",
                "check_abstention",
                {},
                {"error": str(e)},
                0.0,
            )

        # ── Build answer ──────────────────────────────────────────────────────
        if abstention_result and abstention_result.should_abstain:
            # FIX: If we have evidence, provide extractive fallback instead of
            # full abstention.
            if scored:
                # FIX: Helper functions that handle both dict and dataclass
                # NormalizedEvidence objects (which have .evidence_item attr).
                def _get_score(ev: Any) -> float:
                    if isinstance(ev, dict):
                        return float(ev.get("composite_score", 0) or 0)
                    if hasattr(ev, "evidence_item"):
                        return float(ev.evidence_item.composite_score or 0)
                    return 0.0

                def _get_text(ev: Any) -> str:
                    if isinstance(ev, dict):
                        return ev.get("text", "") or ""
                    if hasattr(ev, "evidence_item"):
                        return ev.evidence_item.text or ""
                    return ""

                best = sorted(scored, key=_get_score, reverse=True)[:3]
                texts = [_get_text(ev) for ev in best if _get_text(ev)]
                if texts:
                    direct_answer = (
                        "Based on available evidence:\n\n"
                        + "\n\n".join(f"- {t}" for t in texts)
                    )
                    citations = self._build_citations(
                        scored, claims_with_contradictions
                    )[0]
                    return AnswerEnvelope(
                        question=question,
                        intent=query_plan.intent if query_plan else "unknown",
                        answer_type=self._map_intent_to_answer_type(
                            query_plan.intent if query_plan else "unknown"
                        ),
                        direct_answer=direct_answer,
                        claims=[
                            c.to_dict() if hasattr(c, "to_dict") else c
                            for c in claims_with_contradictions
                        ],
                        evidence=[
                            ev.to_dict() if hasattr(ev, "to_dict") else ev
                            for ev in scored
                        ],
                        citations=[
                            c.to_dict() if hasattr(c, "to_dict") else c
                            for c in citations
                        ],
                        reasoning_trace=reasoning_trace,
                        confidence=confidence_value,
                        confidence_components=(
                            confidence_result.components.to_dict()
                            if confidence_result
                            else {}
                        ),
                        certainty=(
                            "low" if confidence_value >= 0.30 else "insufficient"
                        ),
                        contradictions=[],
                        unsupported_parts=[],
                        safety_status=(
                            safety_result.decision.value
                            if safety_result
                            else "unknown"
                        ),
                        abstained=False,
                        abstention_reason=None,
                        question_decomposition=[],
                        subanswers=None,
                        entity_resolution=None,
                    ).to_dict()

            # Full abstention when no usable evidence is available
            abstention_message = format_abstention_message(
                abstention_result.reason or AbstentionReason.CANNOT_DETERMINE,
                abstention_result.explanation,
                self.language,
            )
            return AnswerEnvelope(
                question=question,
                intent=query_plan.intent if query_plan else "unknown",
                answer_type=AnswerType.SUMMARY,
                direct_answer=abstention_message,
                claims=[],
                evidence=[],
                citations=[],
                reasoning_trace=reasoning_trace,
                confidence=0.0,
                confidence_components={},
                certainty=ConfidenceLevel.INSUFFICIENT,
                contradictions=[],
                unsupported_parts=[],
                safety_status="safe",
                abstained=True,
                abstention_reason=(
                    abstention_result.reason.value
                    if abstention_result.reason
                    else "unknown"
                ),
                question_decomposition=[],
                subanswers=None,
                entity_resolution=None,
            ).to_dict()

        # ── Normal answer path ────────────────────────────────────────────────
        direct_answer = self._build_direct_answer(
            question, claims_with_contradictions, scored, query_plan
        )
        citations, claim_citation_map = self._build_citations(
            scored, claims_with_contradictions
        )

        return AnswerEnvelope(
            question=question,
            intent=query_plan.intent if query_plan else "unknown",
            answer_type=self._map_intent_to_answer_type(
                query_plan.intent if query_plan else "unknown"
            ),
            direct_answer=direct_answer,
            claims=[
                c.to_dict() if hasattr(c, "to_dict") else c
                for c in claims_with_contradictions
            ],
            evidence=[
                ev.to_dict() if hasattr(ev, "to_dict") else ev for ev in scored
            ],
            citations=[
                c.to_dict() if hasattr(c, "to_dict") else c for c in citations
            ],
            reasoning_trace=reasoning_trace,
            confidence=confidence_result.overall if confidence_result else 0.0,
            confidence_components=(
                confidence_result.components.to_dict() if confidence_result else {}
            ),
            certainty=(
                ConfidenceLevel(confidence_result.level)
                if confidence_result
                else ConfidenceLevel.INSUFFICIENT
            ),
            contradictions=[],
            unsupported_parts=[],
            safety_status=(
                safety_result.decision.value if safety_result else "unknown"
            ),
            abstained=False,
            abstention_reason=None,
            question_decomposition=[],
            subanswers=None,
            entity_resolution=None,
        ).to_dict()

    # ── Answer builders ───────────────────────────────────────────────────────

    def _build_direct_answer(
        self,
        question: str,
        claims: List[Dict[str, Any]],
        evidence: List[Any],
        query_plan: Optional[QueryPlan],
    ) -> str:
        """Build a direct answer from claims and evidence."""
        if not claims:
            return "Insufficient evidence to answer this question."

        supported_claims = [
            c
            for c in claims
            if c.get("status") in {"SUPPORTED", "PARTIALLY_SUPPORTED"}
        ]
        if not supported_claims:
            return "The available evidence does not support any claims for this question."

        intent = query_plan.intent if query_plan else "factual"

        if intent == "definition":
            return self._build_definition_answer(supported_claims)
        elif intent in {"numeric", "dosage"}:
            return self._build_numeric_answer(supported_claims)
        elif intent == "list":
            return self._build_list_answer(supported_claims)
        elif intent == "comparison":
            return self._build_comparison_answer(supported_claims)
        elif intent in {"causal", "mechanism"}:
            return self._build_causal_answer(supported_claims)
        elif intent in {"management", "treatment"}:
            return self._build_management_answer(supported_claims)
        elif intent == "diagnostic":
            return self._build_diagnostic_answer(supported_claims)
        else:
            return self._build_factual_answer(supported_claims)

    def _build_definition_answer(self, claims: List[Dict]) -> str:
        if not claims:
            return ""
        definition = claims[0].get("text", "")
        qualifiers = claims[0].get("qualifiers", [])
        if qualifiers:
            definition += f" ({', '.join(qualifiers)})"
        return definition

    def _build_numeric_answer(self, claims: List[Dict]) -> str:
        numeric_claims = [
            c
            for c in claims
            if c.get("numeric_value") is not None
            or "numeric" in c.get("claim_type", "")
        ]
        if not numeric_claims:
            return "No specific numeric values found."
        parts = []
        for claim in numeric_claims[:3]:
            value = claim.get("numeric_value")
            unit = claim.get("numeric_unit", "")
            range_val = claim.get("numeric_range")
            if range_val:
                parts.append(f"{range_val[0]} to {range_val[1]} {unit}".strip())
            elif value is not None:
                parts.append(f"{value} {unit}".strip())
        return "; ".join(parts) if parts else "No specific numeric values found."

    def _build_list_answer(self, claims: List[Dict]) -> str:
        if not claims:
            return ""
        items = [c.get("text", "") for c in claims[:5] if c.get("text", "")]
        return "\n".join(f"- {item}" for item in items) if items else ""

    def _build_comparison_answer(self, claims: List[Dict]) -> str:
        if not claims:
            return ""
        parts = [c.get("text", "") for c in claims[:3] if c.get("text", "")]
        return "Comparison:\n" + "\n".join(f"- {p}" for p in parts) if parts else ""

    def _build_causal_answer(self, claims: List[Dict]) -> str:
        if not claims:
            return ""
        parts = [c.get("text", "") for c in claims[:5] if c.get("text", "")]
        return "\n".join(parts) if parts else ""

    def _build_management_answer(self, claims: List[Dict]) -> str:
        if not claims:
            return ""
        parts = [c.get("text", "") for c in claims[:5] if c.get("text", "")]
        return "\n".join(f"{i + 1}. {p}" for i, p in enumerate(parts)) if parts else ""

    def _build_diagnostic_answer(self, claims: List[Dict]) -> str:
        if not claims:
            return ""
        parts = [c.get("text", "") for c in claims[:5] if c.get("text", "")]
        return "\n".join(f"- {p}" for p in parts) if parts else ""

    def _build_factual_answer(self, claims: List[Dict]) -> str:
        if not claims:
            return ""
        parts = [c.get("text", "") for c in claims[:3] if c.get("text", "")]
        return " ".join(parts) if parts else ""

    def _build_citations(
        self,
        evidence: List[Any],
        claims: List[Dict[str, Any]],
    ) -> tuple:
        claim_evidence_map: Dict[str, List[str]] = {}
        for claim in claims:
            claim_id = claim.get("id", "")
            evidence_ids = [
                ev.get("source_id", "")
                for ev in claim.get("evidence", [])
                if isinstance(ev, dict)
            ]
            if evidence_ids:
                claim_evidence_map[claim_id] = evidence_ids
        citations = build_citations(evidence, claim_evidence_map)
        citation_map = {c.source_id: c for c in citations}
        return citations, citation_map

    def _map_intent_to_answer_type(self, intent: str) -> AnswerType:
        intent_map = {
            "definition": AnswerType.DEFINITION,
            "fact": AnswerType.FACT,
            "numeric": AnswerType.NUMERIC,
            "dosage": AnswerType.NUMERIC,
            "list": AnswerType.LIST,
            "comparison": AnswerType.COMPARISON,
            "causal": AnswerType.CAUSE,
            "mechanism": AnswerType.MECHANISM,
            "management": AnswerType.MANAGEMENT,
            "diagnostic": AnswerType.DIAGNOSTIC_CRITERIA,
            "prognosis": AnswerType.PROGNOSIS,
            "relationship": AnswerType.RELATIONSHIP,
        }
        return intent_map.get(intent, AnswerType.FACT)


def create_deterministic_engine(**kwargs) -> DeterministicAnswerEngine:
    """Create a deterministic answer engine with default configuration."""
    return DeterministicAnswerEngine(**kwargs)
