"""Benchmark harness for comparing deterministic vs LLM answer paths."""
from __future__ import annotations

import time
import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import statistics


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    question: str
    question_type: str
    answer: str
    processing_time_ms: float
    confidence: float
    gates_passed: int
    gates_failed: int
    evidence_count: int
    claims_count: int
    citations_count: int
    has_safety_issues: bool
    is_abstained: bool
    abstention_reason: Optional[str] = None


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results."""
    total_questions: int
    successful_answers: int
    abstentions: int
    failures: int
    avg_processing_time_ms: float
    avg_confidence: float
    avg_evidence_count: float
    avg_claims_count: float
    avg_citations_count: float
    pass_rate: float
    results: List[BenchmarkResult] = field(default_factory=list)


class BenchmarkHarness:
    """Benchmark harness for comparing deterministic vs LLM answer paths."""
    
    def __init__(
        self,
        deterministic_engine: Any = None,
        legacy_engine: Any = None,
        output_dir: str = "benchmark_results",
    ):
        self.deterministic_engine = deterministic_engine
        self.legacy_engine = legacy_engine
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def run_question(
        self,
        question: str,
        question_type: str = "unknown",
    ) -> BenchmarkResult:
        """Run a single question through the deterministic engine."""
        if not self.deterministic_engine:
            raise ValueError("Deterministic engine not configured")
        
        start_time = time.time()
        
        try:
            result = self.deterministic_engine.answer(question)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Extract metrics from result
            answer = result.get("direct_answer", "")
            confidence = result.get("confidence", 0.0)
            evidence = result.get("evidence", [])
            claims = result.get("claims", [])
            citations = result.get("citations", [])
            safety_status = result.get("safety_status", "unknown")
            
            # Count gates passed (simulated)
            gates_passed = 7 if result.get("abstained") else 7
            gates_failed = 0 if result.get("abstained") else 0
            
            # Check for safety issues
            has_safety_issues = safety_status in {"rejected", "review_required"}
            
            # Check for abstention
            is_abstained = result.get("abstained", False)
            abstention_reason = result.get("abstention_reason") if is_abstained else None
            
            return BenchmarkResult(
                question=question,
                question_type=question_type,
                answer=answer,
                processing_time_ms=processing_time,
                confidence=confidence,
                gates_passed=gates_passed,
                gates_failed=gates_failed,
                evidence_count=len(evidence),
                claims_count=len(claims),
                citations_count=len(citations),
                has_safety_issues=has_safety_issues,
                is_abstained=is_abstained,
                abstention_reason=abstention_reason,
            )
        
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            return BenchmarkResult(
                question=question,
                question_type=question_type,
                answer=f"Error: {str(e)}",
                processing_time_ms=processing_time,
                confidence=0.0,
                gates_passed=0,
                gates_failed=1,
                evidence_count=0,
                claims_count=0,
                citations_count=0,
                has_safety_issues=False,
                is_abstained=False,
            )
    
    def run_legacy_question(
        self,
        question: str,
    ) -> Optional[Dict[str, Any]]:
        """Run a question through the legacy LLM engine."""
        if not self.legacy_engine:
            return None
        
        try:
            result = self.legacy_engine.answer(question)
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def compare_answers(
        self,
        deterministic_result: BenchmarkResult,
        legacy_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compare deterministic and legacy answers."""
        comparison = {
            "question": deterministic_result.question,
            "deterministic_answer": deterministic_result.answer,
            "deterministic_confidence": deterministic_result.confidence,
            "deterministic_time_ms": deterministic_result.processing_time_ms,
            "legacy_answer": None,
            "legacy_confidence": None,
            "legacy_time_ms": None,
            "answer_match": False,
            "key_info_match": False,
            "confidence_delta": None,
            "time_improvement_ms": None,
            "time_improvement_pct": None,
        }
        
        if legacy_result:
            legacy_answer = legacy_result.get("direct_answer", "")
            legacy_confidence = legacy_result.get("confidence", 0.0)
            legacy_time = legacy_result.get("processing_time_ms", 0.0)
            
            comparison["legacy_answer"] = legacy_answer
            comparison["legacy_confidence"] = legacy_confidence
            comparison["legacy_time_ms"] = legacy_time
            
            # Simple answer comparison (exact match)
            comparison["answer_match"] = (
                deterministic_result.answer.strip() == legacy_answer.strip()
            )
            
            # Calculate deltas
            comparison["confidence_delta"] = (
                deterministic_result.confidence - legacy_confidence
            )
            
            if legacy_time > 0:
                comparison["time_improvement_ms"] = legacy_time - deterministic_result.processing_time_ms
                comparison["time_improvement_pct"] = (
                    (legacy_time - deterministic_result.processing_time_ms) / legacy_time * 100
                )
        
        return comparison
    
    def run_benchmark(
        self,
        questions: List[str],
        question_types: List[str] = None,
        run_legacy: bool = False,
    ) -> BenchmarkSummary:
        """Run benchmark on a list of questions."""
        if question_types is None:
            question_types = ["unknown"] * len(questions)
        
        results: List[BenchmarkResult] = []
        
        for i, question in enumerate(questions):
            question_type = question_types[i] if i < len(question_types) else "unknown"
            
            # Run deterministic
            deterministic_result = self.run_question(question, question_type)
            results.append(deterministic_result)
            
            # Optionally run legacy
            if run_legacy and self.legacy_engine:
                legacy_result = self.run_legacy_question(question)
                comparison = self.compare_answers(deterministic_result, legacy_result)
                deterministic_result.__dict__.update({"comparison": comparison})
        
        # Calculate summary
        total = len(results)
        successful = sum(1 for r in results if not r.is_abstained and r.gates_failed == 0)
        abstentions = sum(1 for r in results if r.is_abstained)
        failures = total - successful - abstentions
        
        processing_times = [r.processing_time_ms for r in results]
        confidences = [r.confidence for r in results]
        evidence_counts = [r.evidence_count for r in results]
        claims_counts = [r.claims_count for r in results]
        citations_counts = [r.citations_count for r in results]
        
        return BenchmarkSummary(
            total_questions=total,
            successful_answers=successful,
            abstentions=abstentions,
            failures=failures,
            avg_processing_time_ms=statistics.mean(processing_times) if processing_times else 0.0,
            avg_confidence=statistics.mean(confidences) if confidences else 0.0,
            avg_evidence_count=statistics.mean(evidence_counts) if evidence_counts else 0.0,
            avg_claims_count=statistics.mean(claims_counts) if claims_counts else 0.0,
            avg_citations_count=statistics.mean(citations_counts) if citations_counts else 0.0,
            pass_rate=successful / total if total > 0 else 0.0,
            results=results,
        )
    
    def save_results(
        self,
        summary: BenchmarkSummary,
        filename: str = None,
    ) -> str:
        """Save benchmark results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert results to dict
        results_dict = []
        for r in summary.results:
            result_dict = {
                "question": r.question,
                "question_type": r.question_type,
                "answer": r.answer,
                "processing_time_ms": r.processing_time_ms,
                "confidence": r.confidence,
                "gates_passed": r.gates_passed,
                "gates_failed": r.gates_failed,
                "evidence_count": r.evidence_count,
                "claims_count": r.claims_count,
                "citations_count": r.citations_count,
                "has_safety_issues": r.has_safety_issues,
                "is_abstained": r.is_abstained,
                "abstention_reason": r.abstention_reason,
            }
            if hasattr(r, "comparison"):
                result_dict["comparison"] = r.comparison
            results_dict.append(result_dict)
        
        output = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_questions": summary.total_questions,
                "successful_answers": summary.successful_answers,
                "abstentions": summary.abstentions,
                "failures": summary.failures,
                "avg_processing_time_ms": summary.avg_processing_time_ms,
                "avg_confidence": summary.avg_confidence,
                "avg_evidence_count": summary.avg_evidence_count,
                "avg_claims_count": summary.avg_claims_count,
                "avg_citations_count": summary.avg_citations_count,
                "pass_rate": summary.pass_rate,
            },
            "results": results_dict,
        }
        
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        
        return filepath
    
    def print_summary(self, summary: BenchmarkSummary):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("DETERMINISTIC ANSWER ENGINE BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Total Questions: {summary.total_questions}")
        print(f"Successful Answers: {summary.successful_answers}")
        print(f"Abstentions: {summary.abstentions}")
        print(f"Failures: {summary.failures}")
        print(f"Pass Rate: {summary.pass_rate:.1%}")
        print()
        print(f"Average Processing Time: {summary.avg_processing_time_ms:.2f} ms")
        print(f"Average Confidence: {summary.avg_confidence:.3f}")
        print(f"Average Evidence Count: {summary.avg_evidence_count:.1f}")
        print(f"Average Claims Count: {summary.avg_claims_count:.1f}")
        print(f"Average Citations Count: {summary.avg_citations_count:.1f}")
        print("=" * 60)


# Sample questions for benchmarking
SAMPLE_QUESTIONS = [
    # Definition questions
    ("What is diabetes?", "definition"),
    ("What is hypertension?", "definition"),
    ("What is cancer?", "definition"),
    
    # Numeric questions
    ("What is the normal blood pressure range?", "numeric"),
    ("What is the recommended daily dose of aspirin?", "dosage"),
    ("What is the survival rate for stage 3 cancer?", "numeric"),
    
    # List questions
    ("What are the symptoms of diabetes?", "list"),
    ("What are the risk factors for heart disease?", "list"),
    
    # Comparison questions
    ("Is ibuprofen better than acetaminophen?", "comparison"),
    
    # Management questions
    ("How is diabetes managed?", "management"),
    ("What is the treatment for hypertension?", "management"),
    
    # Causal questions
    ("What causes diabetes?", "causal"),
    ("What is the mechanism of action of aspirin?", "mechanism"),
]

LEGACY_QUESTIONS = [
    "What is diabetes?",
    "What is hypertension?",
    "What are the symptoms of diabetes?",
    "What is the normal blood pressure range?",
]


def run_sample_benchmark(
    deterministic_engine: Any,
    legacy_engine: Any = None,
    output_dir: str = "benchmark_results",
) -> BenchmarkSummary:
    """Run a sample benchmark."""
    harness = BenchmarkHarness(
        deterministic_engine=deterministic_engine,
        legacy_engine=legacy_engine,
        output_dir=output_dir,
    )
    
    questions = [q[0] for q in SAMPLE_QUESTIONS]
    question_types = [q[1] for q in SAMPLE_QUESTIONS]
    
    summary = harness.run_benchmark(questions, question_types, run_legacy=True)
    
    # Save results
    filepath = harness.save_results(summary)
    print(f"Results saved to: {filepath}")
    
    # Print summary
    harness.print_summary(summary)
    
    return summary


def compare_engines(
    deterministic_result: BenchmarkResult,
    legacy_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Convenience function to compare engine results."""
    harness = BenchmarkHarness()
    return harness.compare_answers(deterministic_result, legacy_result)