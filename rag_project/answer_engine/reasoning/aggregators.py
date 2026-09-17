"""Evidence aggregators for deterministic medical answers."""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from rag_project.answer_engine.evidence.normalizer import NormalizedEvidence


@dataclass
class AggregationResult:
    """Result of evidence aggregation."""
    aggregated_evidence: List[NormalizedEvidence]
    agreement: float  # 0-1
    summary: str
    confidence: float


class EvidenceAggregator:
    """Aggregates multiple pieces of evidence."""
    
    def __init__(self):
        self._aggregation_strategies = {
            "average": self._aggregate_average,
            "max": self._aggregate_max,
            "min": self._aggregate_min,
            "majority": self._aggregate_majority,
            "consensus": self._aggregate_consensus,
        }
    
    def aggregate(self, evidence: List[NormalizedEvidence], strategy: str = "average") -> AggregationResult:
        """Aggregate evidence using specified strategy."""
        if not evidence:
            return AggregationResult(
                aggregated_evidence=[],
                agreement=0.0,
                summary="No evidence to aggregate",
                confidence=0.0,
            )
        
        # Get scores
        scores = [e.evidence_item.composite_score for e in evidence]
        
        # Apply aggregation strategy
        strategy_func = self._aggregation_strategies.get(strategy, self._aggregate_average)
        aggregated_score = strategy_func(scores)
        
        # Calculate agreement
        agreement = self._calculate_agreement(scores)
        
        # Generate summary
        summary = self._generate_summary(evidence, aggregated_score, strategy)
        
        return AggregationResult(
            aggregated_evidence=evidence,
            agreement=agreement,
            summary=summary,
            confidence=aggregated_score,
        )
    
    def _aggregate_average(self, scores: List[float]) -> float:
        """Aggregate using average."""
        return sum(scores) / len(scores) if scores else 0.0
    
    def _aggregate_max(self, scores: List[float]) -> float:
        """Aggregate using maximum."""
        return max(scores) if scores else 0.0
    
    def _aggregate_min(self, scores: List[float]) -> float:
        """Aggregate using minimum."""
        return min(scores) if scores else 0.0
    
    def _aggregate_majority(self, scores: List[float]) -> float:
        """Aggregate using majority vote."""
        # Simplified - would classify scores as supported/unsupported
        supported = sum(1 for s in scores if s >= 0.5)
        return supported / len(scores) if scores else 0.0
    
    def _aggregate_consensus(self, scores: List[float]) -> float:
        """Aggregate using consensus (high agreement required)."""
        if not scores:
            return 0.0
        
        avg = sum(scores) / len(scores)
        variance = sum((s - avg) ** 2 for s in scores) / len(scores)
        
        # Low variance = high agreement
        agreement_factor = 1.0 - min(1.0, variance)
        
        return avg * agreement_factor
    
    def _calculate_agreement(self, scores: List[float]) -> float:
        """Calculate agreement between scores."""
        if len(scores) < 2:
            return 1.0
        
        avg = sum(scores) / len(scores)
        variance = sum((s - avg) ** 2 for s in scores) / len(scores)
        
        # Lower variance = higher agreement
        return 1.0 - min(1.0, variance)
    
    def _generate_summary(self, evidence: List[NormalizedEvidence], 
                          score: float, strategy: str) -> str:
        """Generate summary of aggregated evidence."""
        if not evidence:
            return "No evidence"
        
        if score >= 0.7:
            support_level = "strongly supported"
        elif score >= 0.5:
            support_level = "moderately supported"
        else:
            support_level = "weakly supported"
        
        return f"Aggregated {len(evidence)} pieces of evidence using {strategy}: {support_level}"
    
    def aggregate_numeric(self, evidence: List[NormalizedEvidence], 
                          numeric_field: str = "numeric_value") -> Dict[str, Any]:
        """Aggregate numeric values from evidence."""
        values = []
        units = set()
        
        for item in evidence:
            # Simplified - would extract from actual numeric values
            pass
        
        return {
            "count": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "avg": sum(values) / len(values) if values else None,
            "units": list(units),
        }


def aggregate_evidence(evidence: List[NormalizedEvidence], 
                       strategy: str = "average") -> AggregationResult:
    """Convenience function to aggregate evidence."""
    aggregator = EvidenceAggregator()
    return aggregator.aggregate(evidence, strategy)
