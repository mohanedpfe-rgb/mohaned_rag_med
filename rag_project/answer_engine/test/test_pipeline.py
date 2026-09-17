"""Test suite for answer_engine pipeline."""
from __future__ import annotations

import unittest
from typing import List, Dict, Any

from rag_project.answer_engine.schemas import AnswerType, ConfidenceLevel, ClaimStatus
from rag_project.answer_engine.pipeline import (
    AnswerPipeline,
    PipelineGate,
    GateResult,
    PipelineResult,
    EvidenceGate,
    ClaimGate,
    NumericGate,
    CitationGate,
    ContradictionGate,
    CompletenessGate,
    SafetyGate,
)


class TestPipelineGates(unittest.TestCase):
    """Test individual pipeline gates."""
    
    def test_evidence_gate_passes_with_sufficient_evidence(self):
        """EvidenceGate should pass when evidence count and quality are sufficient."""
        gate = EvidenceGate()
        
        evidence = [
            {"id": "e1", "composite_score": 0.8},
            {"id": "e2", "composite_score": 0.7},
        ]
        
        context = {"evidence": evidence}
        result = gate.check(context)
        
        self.assertTrue(result.passed)
        self.assertEqual(result.reason, "Evidence check passed")
    
    def test_evidence_gate_fails_with_insufficient_count(self):
        """EvidenceGate should fail when evidence count is below minimum."""
        gate = EvidenceGate()
        
        evidence = []  # Empty evidence
        
        context = {"evidence": evidence}
        result = gate.check(context)
        
        self.assertFalse(result.passed)
        self.assertIn("Insufficient evidence", result.reason)
    
    def test_evidence_gate_fails_with_low_quality(self):
        """EvidenceGate should fail when more than 50% is low quality."""
        gate = EvidenceGate()
        
        evidence = [
            {"id": "e1", "composite_score": 0.1},  # Low quality
            {"id": "e2", "composite_score": 0.1},
            {"id": "e3", "composite_score": 0.9},
        ]
        
        context = {"evidence": evidence}
        result = gate.check(context)
        
        self.assertFalse(result.passed)
        self.assertIn("low quality", result.reason)
    
    def test_claim_gate_passes_with_supported_claims(self):
        """ClaimGate should pass when claims are supported."""
        gate = ClaimGate()
        
        claims = [
            {"id": "c1", "status": "SUPPORTED", "support_ratio": 0.9},
            {"id": "c2", "status": "SUPPORTED", "support_ratio": 0.8},
        ]
        
        context = {"claims": claims}
        result = gate.check(context)
        
        self.assertTrue(result.passed)
    
    def test_claim_gate_fails_with_too_many_unsupported(self):
        """ClaimGate should fail when more than 50% unsupported."""
        gate = ClaimGate()
        
        claims = [
            {"id": "c1", "status": "UNSUPPORTED"},
            {"id": "c2", "status": "UNSUPPORTED"},
            {"id": "c3", "status": "SUPPORTED", "support_ratio": 0.9},
        ]
        
        context = {"claims": claims}
        result = gate.check(context)
        
        self.assertFalse(result.passed)
        self.assertIn("too many unsupported", result.reason)
    
    def test_claim_gate_fails_with_low_support_ratio(self):
        """ClaimGate should fail when support ratio is below threshold."""
        gate = ClaimGate()
        
        claims = [
            {"id": "c1", "status": "SUPPORTED", "support_ratio": 0.3},
        ]
        
        context = {"claims": claims}
        result = gate.check(context)
        
        self.assertFalse(result.passed)
        self.assertIn("support ratio below threshold", result.reason)
    
    def test_numeric_gate_passes_with_units(self):
        """NumericGate should pass when numeric claims have units."""
        gate = NumericGate()
        
        claims = [
            {"id": "c1", "numeric_value": 100, "numeric_unit": "mg"},
            {"id": "c2", "numeric_value": 50, "numeric_unit": "ml"},
        ]
        
        context = {"claims": claims}
        result = gate.check(context)
        
        self.assertTrue(result.passed)
    
    def test_numeric_gate_fails_without_units(self):
        """NumericGate should fail when numeric claims lack units."""
        gate = NumericGate()
        gate.require_units = True
        
        claims = [
            {"id": "c1", "numeric_value": 100},  # No unit
        ]
        
        context = {"claims": claims}
        result = gate.check(context)
        
        self.assertFalse(result.passed)
        self.assertIn("without units", result.reason)
    
    def test_citation_gate_passes_with_high_ratio(self):
        """CitationGate should pass when citation ratio is sufficient."""
        gate = CitationGate()
        
        claims = [
            {"id": "c1", "evidence": [{"source_id": "e1"}]},
            {"id": "c2", "evidence": [{"source_id": "e2"}]},
        ]
        
        context = {"claims": claims}
        result = gate.check(context)
        
        self.assertTrue(result.passed)
    
    def test_citation_gate_fails_with_low_ratio(self):
        """CitationGate should fail when citation ratio is below threshold."""
        gate = CitationGate()
        
        claims = [
            {"id": "c1", "evidence": []},  # No evidence
            {"id": "c2", "evidence": []},
            {"id": "c3", "evidence": [{"source_id": "e1"}]},
        ]
        
        context = {"claims": claims}
        result = gate.check(context)
        
        self.assertFalse(result.passed)
        self.assertIn("below threshold", result.reason)
    
    def test_contradiction_gate_passes_without_contradictions(self):
        """ContradictionGate should pass when no contradictions exist."""
        gate = ContradictionGate()
        
        claims = [
            {"id": "c1", "contradiction": None},
            {"id": "c2", "contradiction": None},
        ]
        
        context = {"claims": claims}
        result = gate.check(context)
        
        self.assertTrue(result.passed)
    
    def test_contradiction_gate_fails_with_contradictions(self):
        """ContradictionGate should fail when contradictions exist."""
        gate = ContradictionGate()
        
        claims = [
            {"id": "c1", "contradiction": {"id": "contradiction_1"}},
            {"id": "c2", "contradiction": None},
        ]
        
        context = {"claims": claims}
        result = gate.check(context)
        
        self.assertFalse(result.passed)
        self.assertIn("contradictory claims", result.reason)
    
    def test_completeness_gate_passes_with_sufficient_length(self):
        """CompletenessGate should pass when answer is sufficiently long."""
        gate = CompletenessGate()
        
        context = {
            "direct_answer": "This is a reasonably long answer that provides sufficient information.",
            "question": "What is diabetes?",
        }
        result = gate.check(context)
        
        self.assertTrue(result.passed)
    
    def test_completeness_gate_fails_with_too_short_answer(self):
        """CompletenessGate should fail when answer is too short."""
        gate = CompletenessGate()
        
        context = {
            "direct_answer": "Short.",
            "question": "What is diabetes?",
        }
        result = gate.check(context)
        
        self.assertFalse(result.passed)
        self.assertIn("too short", result.reason)
    
    def test_safety_gate_passes_when_approved(self):
        """SafetyGate should pass when safety status is approved."""
        gate = SafetyGate()
        
        context = {"safety_status": "approved"}
        result = gate.check(context)
        
        self.assertTrue(result.passed)
    
    def test_safety_gate_fails_when_rejected(self):
        """SafetyGate should fail when safety status is rejected."""
        gate = SafetyGate()
        
        context = {"safety_status": "rejected"}
        result = gate.check(context)
        
        self.assertFalse(result.passed)
        self.assertIn("rejected", result.reason)


class TestAnswerPipeline(unittest.TestCase):
    """Test complete answer pipeline."""
    
    def test_pipeline_runs_successfully(self):
        """Pipeline should run successfully when all gates pass."""
        pipeline = AnswerPipeline()
        
        evidence = [
            {"id": "e1", "composite_score": 0.8},
            {"id": "e2", "composite_score": 0.7},
        ]
        
        claims = [
            {"id": "c1", "status": "SUPPORTED", "support_ratio": 0.9, "evidence": [{"source_id": "e1"}]},
        ]
        
        result = pipeline.run(
            question="What is diabetes?",
            evidence=evidence,
            claims=claims,
            answer="Diabetes is a chronic condition.",
            safety_status="approved",
        )
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.gates_passed), 7)
        self.assertEqual(len(result.gates_failed), 0)
    
    def test_pipeline_fails_on_first_critical_gate(self):
        """Pipeline should stop on first critical gate failure."""
        pipeline = AnswerPipeline()
        
        evidence = []  # Will fail evidence gate
        
        result = pipeline.run(
            question="What is diabetes?",
            evidence=evidence,
            claims=[],
            safety_status="approved",
        )
        
        self.assertFalse(result.success)
        self.assertIn("evidence", result.gates_failed)


class TestAdversarialCases(unittest.TestCase):
    """Test adversarial and edge cases."""
    
    def test_pipeline_with_empty_inputs(self):
        """Pipeline should handle empty inputs gracefully."""
        pipeline = AnswerPipeline()
        
        result = pipeline.run(
            question="",
            evidence=[],
            claims=[],
            safety_status="approved",
        )
        
        self.assertFalse(result.success)
    
    def test_pipeline_with_high_confidence_low_quality(self):
        """Pipeline should catch high confidence but low quality evidence."""
        pipeline = AnswerPipeline()
        
        evidence = [
            {"id": "e1", "composite_score": 0.1},  # Low quality
        ]
        
        result = pipeline.run(
            question="What is diabetes?",
            evidence=evidence,
            claims=[],
            safety_status="approved",
        )
        
        self.assertFalse(result.success)
        self.assertIn("evidence", result.gates_failed)
    
    def test_pipeline_with_contradictory_claims(self):
        """Pipeline should catch contradictory claims."""
        pipeline = AnswerPipeline()
        
        evidence = [
            {"id": "e1", "composite_score": 0.9},
        ]
        
        claims = [
            {"id": "c1", "status": "SUPPORTED", "support_ratio": 0.9, "evidence": [{"source_id": "e1"}], "contradiction": {"id": "c2"}},
        ]
        
        result = pipeline.run(
            question="What is diabetes?",
            evidence=evidence,
            claims=claims,
            safety_status="approved",
        )
        
        self.assertFalse(result.success)
        self.assertIn("contradiction", result.gates_failed)
    
    def test_pipeline_with_unsupported_claims(self):
        """Pipeline should catch unsupported claims."""
        pipeline = AnswerPipeline()
        
        evidence = [
            {"id": "e1", "composite_score": 0.9},
        ]
        
        claims = [
            {"id": "c1", "status": "UNSUPPORTED", "support_ratio": 0.0, "evidence": [{"source_id": "e1"}]},
        ]
        
        result = pipeline.run(
            question="What is diabetes?",
            evidence=evidence,
            claims=claims,
            safety_status="approved",
        )
        
        self.assertFalse(result.success)
        self.assertIn("claim", result.gates_failed)
    
    def test_pipeline_with_missing_citations(self):
        """Pipeline should catch claims without citations."""
        pipeline = AnswerPipeline()
        
        evidence = [
            {"id": "e1", "composite_score": 0.9},
        ]
        
        claims = [
            {"id": "c1", "status": "SUPPORTED", "support_ratio": 0.9, "evidence": []},  # No evidence
        ]
        
        result = pipeline.run(
            question="What is diabetes?",
            evidence=evidence,
            claims=claims,
            safety_status="approved",
        )
        
        self.assertFalse(result.success)
        self.assertIn("citation", result.gates_failed)
    
    def test_pipeline_with_safety_rejection(self):
        """Pipeline should catch safety rejections."""
        pipeline = AnswerPipeline()
        
        evidence = [
            {"id": "e1", "composite_score": 0.9},
        ]
        
        claims = [
            {"id": "c1", "status": "SUPPORTED", "support_ratio": 0.9, "evidence": [{"source_id": "e1"}]},
        ]
        
        result = pipeline.run(
            question="What is diabetes?",
            evidence=evidence,
            claims=claims,
            safety_status="rejected",
        )
        
        self.assertFalse(result.success)
        self.assertIn("safety", result.gates_failed)


class TestIntegration(unittest.TestCase):
    """Integration tests for the pipeline."""
    
    def test_full_valid_answer_flow(self):
        """Test a complete valid answer flow."""
        pipeline = AnswerPipeline()
        
        evidence = [
            {"id": "e1", "composite_score": 0.9, "semantic_relevance": 0.9},
            {"id": "e2", "composite_score": 0.85, "semantic_relevance": 0.8},
        ]
        
        claims = [
            {
                "id": "c1",
                "status": "SUPPORTED",
                "support_ratio": 0.95,
                "text": "Diabetes is a chronic metabolic disorder.",
                "evidence": [{"source_id": "e1"}],
                "claim_type": "definition",
            },
        ]
        
        answer = "Diabetes is a chronic metabolic disorder characterized by high blood sugar levels."
        
        result = pipeline.run(
            question="What is diabetes?",
            evidence=evidence,
            claims=claims,
            answer=answer,
            safety_status="approved",
        )
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.gates_passed), 7)
    
    def test_invalid_answer_with_multiple_gate_failures(self):
        """Test answer that fails multiple gates."""
        pipeline = AnswerPipeline()
        
        # Low quality, insufficient evidence
        evidence = [
            {"id": "e1", "composite_score": 0.1},
        ]
        
        # Mostly unsupported claims, low support ratio, no citations
        claims = [
            {
                "id": "c1",
                "status": "UNSUPPORTED",
                "support_ratio": 0.2,
                "evidence": [],
            },
        ]
        
        result = pipeline.run(
            question="What is diabetes?",
            evidence=evidence,
            claims=claims,
            answer="",
            safety_status="approved",
        )
        
        self.assertFalse(result.success)
        self.assertIn("evidence", result.gates_failed)


if __name__ == "__main__":
    unittest.main()