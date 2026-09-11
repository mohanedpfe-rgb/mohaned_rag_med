"""Black-box benchmark for the production BookRAG service.

This evaluates the real ``system.answer`` contract against curated questions and
makes retrieval failures visible separately from generation/verification failures.
It does not require an external LLM judge.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from rag_project.evaluation.dataset import load_dataset


@dataclass
class BenchmarkCase:
    question_id: str
    category: str
    status: str
    recall_at_5: float
    recall_at_10: float
    page_hit: bool
    grounded: bool
    citation_count: int
    confidence: float
    latency_ms: float
    failure_stage: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hit_chunk_ids(result: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for hit in result.get("hits") or ():
        meta = getattr(hit, "metadata", {}) or {}
        value = meta.get("chunk_id") or getattr(hit, "chunk_id", None)
        if value:
            values.add(str(value))
    return values


def _hit_pages(result: dict[str, Any]) -> set[int]:
    values: set[int] = set()
    for hit in result.get("hits") or ():
        meta = getattr(hit, "metadata", {}) or {}
        pages = meta.get("page_numbers") or meta.get("page_number") or meta.get("page") or ()
        if not isinstance(pages, (list, tuple)):
            pages = (pages,)
        for value in pages:
            try:
                values.add(int(value))
            except (TypeError, ValueError):
                continue
    return values


def _failure_stage(result: dict[str, Any]) -> str:
    status = str(result.get("status") or "").upper()
    if status in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
        return ""
    trace = result.get("query_trace") or {}
    verification = trace.get("verification") or {}
    generation = trace.get("generation") or {}
    retrieval = trace.get("retrieval") or {}
    if not retrieval or int(retrieval.get("final_hits", 0) or 0) == 0:
        return "retrieval"
    if str(generation.get("status", "")).lower() not in {"completed", "extractive"}:
        return "generation"
    if str(verification.get("status", "")).lower() in {"failed", "blocked"}:
        return "verification"
    if status in {"ANSWER_UNAVAILABLE", "NOT_SUPPORTED"}:
        return "grounding"
    return "contract_or_runtime"


def _confidence(result: dict[str, Any]) -> float:
    confidence = result.get("confidence") or {}
    try:
        return max(0.0, min(1.0, float(confidence.get("evidence_confidence", 0.0) or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def evaluate(system: Any, questions: Iterable[Any]) -> dict[str, Any]:
    cases: list[BenchmarkCase] = []
    for item in questions:
        question = item if isinstance(item, dict) else item.to_dict()
        expected_chunks = {str(value) for value in question.get("relevant_chunk_ids", []) if str(value)}
        expected_pages = {int(value) for value in question.get("relevant_page_numbers", []) if str(value).lstrip("-").isdigit()}
        started = time.perf_counter()
        try:
            result = system.answer(str(question.get("question", "")))
        except Exception as exc:
            result = {"status": "RUNTIME_ERROR", "answer": "", "hits": [], "confidence": {}, "error": type(exc).__name__}
        elapsed = (time.perf_counter() - started) * 1000
        chunks = list(_hit_chunk_ids(result))
        pages = _hit_pages(result)
        recall5 = len(expected_chunks & set(chunks[:5])) / max(1, len(expected_chunks)) if expected_chunks else float(bool(expected_pages & set(list(pages)[:5])))
        recall10 = len(expected_chunks & set(chunks[:10])) / max(1, len(expected_chunks)) if expected_chunks else float(bool(expected_pages & set(list(pages)[:10])))
        grounded = str(result.get("status", "")).upper() in {"SUCCESS", "SUCCESS_WITH_WARNINGS"} and bool((result.get("grounding") or {}).get("allow", True))
        cases.append(BenchmarkCase(
            question_id=str(question.get("id", "unknown")),
            category=str(question.get("category", "unknown")),
            status=str(result.get("status", "UNKNOWN")),
            recall_at_5=max(0.0, min(1.0, recall5)),
            recall_at_10=max(0.0, min(1.0, recall10)),
            page_hit=bool(expected_pages & pages),
            grounded=grounded,
            citation_count=len(result.get("citations") or ()),
            confidence=_confidence(result),
            latency_ms=round(elapsed, 2),
            failure_stage=_failure_stage(result),
        ))

    def mean(values: list[float]) -> float:
        return sum(values) / max(1, len(values))

    failure_counts = Counter(case.failure_stage for case in cases if case.failure_stage)
    by_category: dict[str, dict[str, float]] = {}
    categories = sorted({case.category for case in cases})
    for category in categories:
        rows = [case for case in cases if case.category == category]
        by_category[category] = {
            "count": float(len(rows)),
            "recall_at_5": mean([row.recall_at_5 for row in rows]),
            "recall_at_10": mean([row.recall_at_10 for row in rows]),
            "grounded_rate": mean([float(row.grounded) for row in rows]),
            "mean_confidence": mean([row.confidence for row in rows]),
            "mean_latency_ms": mean([row.latency_ms for row in rows]),
        }

    return {
        "dataset_size": len(cases),
        "retrieval": {"recall_at_5": mean([case.recall_at_5 for case in cases]), "recall_at_10": mean([case.recall_at_10 for case in cases]), "page_hit_rate": mean([float(case.page_hit) for case in cases])},
        "answer_quality": {"grounded_rate": mean([float(case.grounded) for case in cases]), "citation_rate": mean([float(case.citation_count > 0) for case in cases]), "mean_confidence": mean([case.confidence for case in cases])},
        "latency": {"mean_ms": mean([case.latency_ms for case in cases]), "p95_ms": sorted([case.latency_ms for case in cases])[max(0, math.ceil(len(cases) * 0.95) - 1)] if cases else 0.0},
        "failure_clusters": dict(failure_counts),
        "by_category": by_category,
        "cases": [case.to_dict() for case in cases],
    }


def run_dataset(system: Any, dataset_path: str | Path) -> dict[str, Any]:
    dataset = load_dataset(Path(dataset_path))
    return evaluate(system, dataset)


def save_report(report: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return destination


__all__ = ["BenchmarkCase", "evaluate", "run_dataset", "save_report"]
