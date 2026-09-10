from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    category: str
    question: str
    expected_entities: tuple[str, ...] = ()
    expected_intent: str = ""
    gold_evidence_ids: tuple[str, ...] = ()
    answerable: bool = True


@dataclass(frozen=True)
class BenchmarkResult:
    case_id: str
    category: str
    entity_hit: bool
    intent_hit: bool
    evidence_hit: bool
    grounded: bool
    abstained_correctly: bool
    citation_precision: float
    support_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkReport:
    total: int
    entity_recall: float
    intent_accuracy: float
    evidence_recall: float
    grounding_rate: float
    abstention_accuracy: float
    mean_citation_precision: float
    mean_support_ratio: float
    by_category: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_cases(path: str | Path) -> tuple[BenchmarkCase, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Benchmark file must contain a JSON array.")
    cases: list[BenchmarkCase] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        cases.append(BenchmarkCase(
            case_id=str(item.get("case_id", "")),
            category=str(item.get("category", "general")),
            question=str(item.get("question", "")),
            expected_entities=tuple(str(x) for x in item.get("expected_entities", ()) or ()),
            expected_intent=str(item.get("expected_intent", "")),
            gold_evidence_ids=tuple(str(x) for x in item.get("gold_evidence_ids", ()) or ()),
            answerable=bool(item.get("answerable", True)),
        ))
    return tuple(cases)


def _status_abstained(result: dict[str, Any]) -> bool:
    return str(result.get("status", "")).upper() in {
        "NOT_SUPPORTED", "REASONING_ABSTAIN", "LOW_QUALITY_QUERY", "ABSTAINED", "INTERNAL_ERROR"
    }


def evaluate_case(case: BenchmarkCase, result: dict[str, Any]) -> BenchmarkResult:
    analysis = result.get("query_analysis") or {}
    entities = {str(x).casefold() for x in analysis.get("entities", ())}
    expected_entities = {str(x).casefold() for x in case.expected_entities}
    entity_hit = not expected_entities or bool(expected_entities & entities)
    primary = str(analysis.get("intent", ""))
    intent_hit = not case.expected_intent or primary == case.expected_intent
    hits = result.get("hits") or []
    hit_ids = set()
    for hit in hits:
        meta = getattr(hit, "metadata", {}) or {}
        hit_ids.add(str(meta.get("chunk_id") or getattr(hit, "doc_id", "")))
    evidence_hit = not case.gold_evidence_ids or bool(hit_ids & set(case.gold_evidence_ids))
    grounding = result.get("grounding") or {}
    grounded = bool(grounding.get("allow")) or str(result.get("status", "")).upper() == "SUCCESS"
    abstained = _status_abstained(result)
    abstention_correct = (not case.answerable and abstained) or (case.answerable and not abstained)
    claims = result.get("claims") or []
    sources = set()
    support = []
    for claim in claims:
        sources.update(str(x) for x in claim.get("sources", ()) if x)
        support.append(float(claim.get("support", 0.0)))
    citation_precision = sum(1 for source in sources if source.startswith("S")) / max(len(sources), 1) if sources else 0.0
    support_ratio = sum(support) / max(len(support), 1)
    return BenchmarkResult(case.case_id, case.category, entity_hit, intent_hit, evidence_hit, grounded, abstention_correct, round(citation_precision, 4), round(support_ratio, 4))


def run_benchmark(cases: Iterable[BenchmarkCase], runner: Callable[[str], dict[str, Any]]) -> BenchmarkReport:
    results = [evaluate_case(case, runner(case.question)) for case in cases]
    total = len(results)
    if not total:
        return BenchmarkReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {})
    categories: dict[str, list[BenchmarkResult]] = {}
    for item in results:
        categories.setdefault(item.category, []).append(item)
    by_category: dict[str, dict[str, float]] = {}
    for category, items in categories.items():
        by_category[category] = {
            "count": float(len(items)),
            "entity_recall": sum(i.entity_hit for i in items) / len(items),
            "intent_accuracy": sum(i.intent_hit for i in items) / len(items),
            "evidence_recall": sum(i.evidence_hit for i in items) / len(items),
            "grounding_rate": sum(i.grounded for i in items) / len(items),
            "abstention_accuracy": sum(i.abstained_correctly for i in items) / len(items),
            "citation_precision": sum(i.citation_precision for i in items) / len(items),
            "support_ratio": sum(i.support_ratio for i in items) / len(items),
        }
    return BenchmarkReport(
        total,
        sum(i.entity_hit for i in results) / total,
        sum(i.intent_hit for i in results) / total,
        sum(i.evidence_hit for i in results) / total,
        sum(i.grounded for i in results) / total,
        sum(i.abstained_correctly for i in results) / total,
        sum(i.citation_precision for i in results) / total,
        sum(i.support_ratio for i in results) / total,
        by_category,
    )
