from __future__ import annotations

import json
from pathlib import Path
from typing import List

from rag_project.evaluation.types import EvalQuestion, default_eval_root


EVAL_VERSION = "v1"


def dataset_path(version: str = EVAL_VERSION, filename: str = "dataset.jsonl") -> Path:
    path = default_eval_root() / version / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_dataset(
    version: str = EVAL_VERSION,
    filename: str = "dataset.jsonl",
    only_reviewed: bool = False,
) -> List[EvalQuestion]:
    path = dataset_path(version, filename)
    if not path.exists():
        return []
    questions: List[EvalQuestion] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                q = EvalQuestion.from_dict(json.loads(line))
                if only_reviewed and q.source_type.value not in {
                    "human_curated",
                    "llm_silver_reviewed",
                }:
                    continue
                questions.append(q)
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid eval dataset line {lineno} in {path}: {exc}"
                ) from exc
    return questions


def save_dataset(
    questions: List[EvalQuestion],
    version: str = EVAL_VERSION,
    filename: str = "dataset.jsonl",
) -> Path:
    path = dataset_path(version, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    with path.open("w", encoding="utf-8") as fh:
        for q in questions:
            if q.id in seen_ids:
                raise ValueError(f"Duplicate question id in dataset: {q.id}")
            seen_ids.add(q.id)
            fh.write(json.dumps(q.to_dict(), ensure_ascii=False))
            fh.write("\n")
    return path


def append_questions(
    questions: List[EvalQuestion],
    version: str = EVAL_VERSION,
    filename: str = "dataset.jsonl",
) -> Path:
    existing = load_dataset(version, filename)
    existing_ids = {q.id for q in existing}
    merged = list(existing)
    for q in questions:
        if q.id in existing_ids:
            continue
        merged.append(q)
    return save_dataset(merged, version, filename)


def dataset_summary(questions: List[EvalQuestion]) -> dict:
    by_category: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for q in questions:
        by_category[q.category.value] = by_category.get(q.category.value, 0) + 1
        by_difficulty[q.difficulty.value] = by_difficulty.get(q.difficulty.value, 0) + 1
        by_source[q.source_type.value] = by_source.get(q.source_type.value, 0) + 1
    return {
        "total": len(questions),
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "by_source": by_source,
    }
