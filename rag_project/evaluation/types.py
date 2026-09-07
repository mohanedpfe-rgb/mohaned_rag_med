from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class EvalCategory(str, Enum):
    FACTUAL_RECALL = "factual_recall"
    MULTI_HOP_REASONING = "multi_hop_reasoning"
    TABLE_LOOKUP = "table_lookup"
    CROSS_SECTION_COMPARISON = "cross_section_comparison"
    CONTRAINDICATION = "contraindication"
    DOSAGE_CALCULATION = "dosage_calculation"
    NEGATIVE_CONTROL = "negative_control"


class EvalDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EvalSource(str, Enum):
    HUMAN_CURATED = "human_curated"
    LLM_SILVER_DRAFT = "llm_silver_draft"
    LLM_SILVER_REVIEWED = "llm_silver_reviewed"
    SYNTHETIC_NEGATIVE = "synthetic_negative"
    ADVERSARIAL_MANUAL = "adversarial_manual"


@dataclass
class EvalQuestion:
    id: str
    category: EvalCategory
    difficulty: EvalDifficulty
    question: str
    relevant_chunk_ids: List[str]
    relevant_page_numbers: List[int]
    acceptable_answers: List[str]
    required_keyphrases: List[str]
    forbidden_keyphrases: List[str]
    source_type: EvalSource
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["category"] = self.category.value
        data["difficulty"] = self.difficulty.value
        data["source_type"] = self.source_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalQuestion":
        return cls(
            id=data["id"],
            category=EvalCategory(data["category"]),
            difficulty=EvalDifficulty(data["difficulty"]),
            question=data["question"],
            relevant_chunk_ids=list(data.get("relevant_chunk_ids", [])),
            relevant_page_numbers=list(data.get("relevant_page_numbers", [])),
            acceptable_answers=list(data.get("acceptable_answers", [])),
            required_keyphrases=list(data.get("required_keyphrases", [])),
            forbidden_keyphrases=list(data.get("forbidden_keyphrases", [])),
            source_type=EvalSource(data.get("source_type", "llm_silver_draft")),
            notes=data.get("notes", ""),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class RetrievalMetrics:
    recall_at_3: float = 0.0
    recall_at_6: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_6: float = 0.0
    chunk_id_exact_match: bool = False
    retrieved_page_union: List[int] = field(default_factory=list)
    rank_of_first_relevant: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalMetrics":
        return cls(
            recall_at_3=float(data.get("recall_at_3", 0.0)),
            recall_at_6=float(data.get("recall_at_6", 0.0)),
            recall_at_10=float(data.get("recall_at_10", 0.0)),
            mrr=float(data.get("mrr", 0.0)),
            ndcg_at_6=float(data.get("ndcg_at_6", 0.0)),
            chunk_id_exact_match=bool(data.get("chunk_id_exact_match", False)),
            retrieved_page_union=[int(v) for v in data.get("retrieved_page_union", [])],
            rank_of_first_relevant=(
                int(data["rank_of_first_relevant"])
                if data.get("rank_of_first_relevant") is not None
                else None
            ),
        )


@dataclass
class GenerationMetrics:
    faithfulness: float = 0.0
    keyphrase_precision: float = 0.0
    forbidden_penalty: float = 0.0
    answer_acceptability: float = 0.0
    rouge_l_f1_best: float = 0.0
    citation_validity: float = 0.0
    negative_refusal: Optional[bool] = None
    llm_judge_score: Optional[float] = None
    llm_judge_explanation: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationMetrics":
        negative_refusal = data.get("negative_refusal")
        if negative_refusal is not None:
            negative_refusal = bool(negative_refusal)
        llm_judge_score = data.get("llm_judge_score")
        if llm_judge_score is not None:
            llm_judge_score = float(llm_judge_score)
        return cls(
            faithfulness=float(data.get("faithfulness", 0.0)),
            keyphrase_precision=float(data.get("keyphrase_precision", 0.0)),
            forbidden_penalty=float(data.get("forbidden_penalty", 0.0)),
            answer_acceptability=float(data.get("answer_acceptability", 0.0)),
            rouge_l_f1_best=float(data.get("rouge_l_f1_best", 0.0)),
            citation_validity=float(data.get("citation_validity", 0.0)),
            negative_refusal=negative_refusal,
            llm_judge_score=llm_judge_score,
            llm_judge_explanation=(
                str(data["llm_judge_explanation"])
                if data.get("llm_judge_explanation") is not None
                else None
            ),
        )


@dataclass
class StageTiming:
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageTiming":
        return cls(
            retrieval_ms=float(data.get("retrieval_ms", 0.0)),
            rerank_ms=float(data.get("rerank_ms", 0.0)),
            generation_ms=float(data.get("generation_ms", 0.0)),
            total_ms=float(data.get("total_ms", 0.0)),
        )


@dataclass
class EvalResult:
    question_id: str
    category: EvalCategory
    difficulty: EvalDifficulty
    retrieval: RetrievalMetrics
    generation: GenerationMetrics
    timing: StageTiming
    raw_answer: str = ""
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    retrieved_scores: List[float] = field(default_factory=list)
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalResult":
        return cls(
            question_id=str(data["question_id"]),
            category=EvalCategory(data.get("category", "factual_recall")),
            difficulty=EvalDifficulty(data.get("difficulty", "medium")),
            retrieval=RetrievalMetrics.from_dict(data.get("retrieval", {})),
            generation=GenerationMetrics.from_dict(data.get("generation", {})),
            timing=StageTiming.from_dict(data.get("timing", {})),
            raw_answer=str(data.get("raw_answer", "")),
            retrieved_chunk_ids=[str(v) for v in data.get("retrieved_chunk_ids", [])],
            retrieved_scores=[float(v) for v in data.get("retrieved_scores", [])],
            error=(str(data["error"]) if data.get("error") is not None else None),
        )


@dataclass
class AggregatedMetrics:
    mean: Dict[str, float] = field(default_factory=dict)
    p50: Dict[str, float] = field(default_factory=dict)
    p95: Dict[str, float] = field(default_factory=dict)
    by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_difficulty: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AggregatedMetrics":
        def _float_dict(raw: Any) -> Dict[str, float]:
            if not isinstance(raw, dict):
                return {}
            return {str(k): float(v) for k, v in raw.items()}

        def _nested_float_dict(raw: Any) -> Dict[str, Dict[str, float]]:
            if not isinstance(raw, dict):
                return {}
            return {str(k): _float_dict(v) for k, v in raw.items()}

        return cls(
            mean=_float_dict(data.get("mean", {})),
            p50=_float_dict(data.get("p50", {})),
            p95=_float_dict(data.get("p95", {})),
            by_category=_nested_float_dict(data.get("by_category", {})),
            by_difficulty=_nested_float_dict(data.get("by_difficulty", {})),
        )


@dataclass
class EvalReport:
    label: str
    dataset_version: str
    dataset_size: int
    retrieval: AggregatedMetrics
    generation: AggregatedMetrics
    latency_ms: AggregatedMetrics
    individual_results: List[EvalResult]
    created_at: str
    configuration: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "dataset_version": self.dataset_version,
            "dataset_size": self.dataset_size,
            "retrieval": asdict(self.retrieval),
            "generation": asdict(self.generation),
            "latency_ms": asdict(self.latency_ms),
            "individual_results": [asdict(r) for r in self.individual_results],
            "created_at": self.created_at,
            "configuration": self.configuration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalReport":
        return cls(
            label=str(data.get("label", "unknown")),
            dataset_version=str(data.get("dataset_version", "v1")),
            dataset_size=int(data.get("dataset_size", 0)),
            retrieval=AggregatedMetrics.from_dict(data.get("retrieval", {})),
            generation=AggregatedMetrics.from_dict(data.get("generation", {})),
            latency_ms=AggregatedMetrics.from_dict(data.get("latency_ms", {})),
            individual_results=[
                EvalResult.from_dict(item)
                for item in data.get("individual_results", [])
            ],
            created_at=str(data.get("created_at", "")),
            configuration=dict(data.get("configuration", {})),
        )


def default_eval_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "eval"
