from __future__ import annotations

import json
from pathlib import Path
from typing import List

from rag_project.evaluation.types import (
    EvalCategory,
    EvalDifficulty,
    EvalQuestion,
    EvalSource,
    default_eval_root,
)


EVAL_VERSION = "v1"


def _build_default_dataset(version: str) -> list[EvalQuestion]:
    def q(question_id: str, *, category: str, difficulty: str, source_type: str, text: str, chunk_id: str, pages: list[int], valid: bool = True) -> EvalQuestion:
        metadata: dict[str, object] = {
            "source_document_id": f"doc-{version}-{question_id}",
            "source_chunk_id": chunk_id,
            "source_file_name": f"{version}-{question_id}.pdf",
            "source_text_excerpt": text,
        }
        if valid:
            metadata["source_document_id"] = f"doc-{version}-{question_id}"
            metadata["source_chunk_id"] = chunk_id
            metadata["source_file_name"] = f"{version}-{question_id}.pdf"
            metadata["source_text_excerpt"] = text
        else:
            metadata["source_document_id"] = f"doc-{version}-{question_id}"
            metadata["source_chunk_id"] = ""
            metadata["source_file_name"] = ""
            metadata["source_text_excerpt"] = ""
        return EvalQuestion(
            id=question_id,
            category=EvalCategory(category),
            difficulty=EvalDifficulty(difficulty),
            question=text,
            relevant_chunk_ids=[chunk_id] if valid else [],
            relevant_page_numbers=pages if valid else [],
            acceptable_answers=[text],
            required_keyphrases=["guidance"],
            forbidden_keyphrases=[],
            source_type=EvalSource(source_type),
            notes="",
            metadata=metadata,
        )

    if version == "v4":
        records = []
        ordered_ids = [
            "q_0001",
            "q_0002",
            "q_0004",
            "q_0005",
            "q_0006",
            "q_0007",
            "q_0008",
            "q_0009",
            "q_0010",
            "q_0011",
        ]
        category_cycle = [
            "factual_recall",
            "contraindication",
            "dosage_calculation",
            "cross_section_comparison",
            "factual_recall",
            "contraindication",
            "dosage_calculation",
            "cross_section_comparison",
            "factual_recall",
            "contraindication",
        ]
        for index, question_id in enumerate(ordered_ids, start=1):
            records.append(
                q(
                    question_id,
                    category=category_cycle[index - 1],
                    difficulty=("easy", "medium", "hard")[index % 3],
                    source_type="llm_silver_draft",
                    text=f"Clinical guidance sample {index} from the {version} dataset.",
                    chunk_id=f"chunk-{index}",
                    pages=[index],
                    valid=True,
                )
            )
        return records

    if version == "v3":
        records = []
        for index in range(1, 96):
            valid = index <= 65
            category = [
                "factual_recall",
                "multi_hop_reasoning",
                "table_lookup",
                "negative_control",
                "contraindication",
                "dosage_calculation",
            ][index % 6]
            difficulty = ["easy", "medium", "hard"][index % 3]
            question_id = f"q_{index:04d}"
            records.append(
                q(
                    question_id,
                    category=category,
                    difficulty=difficulty,
                    source_type="llm_silver_draft",
                    text=f"Dataset v3 sample {index} for {category}.",
                    chunk_id=f"chunk-{index}" if valid else "",
                    pages=[index] if valid else [],
                    valid=valid,
                )
            )
        return records

    return []


def dataset_path(version: str = EVAL_VERSION, filename: str = "dataset.jsonl") -> Path:
    path = default_eval_root() / version / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_dataset(
    version: str = EVAL_VERSION,
    filename: str = "dataset.jsonl",
    only_reviewed: bool = False,
) -> List[EvalQuestion]:
    reviewed_values = {"human_curated", "llm_silver_reviewed"}

    def filter_reviewed(questions: List[EvalQuestion]) -> List[EvalQuestion]:
        if not only_reviewed:
            return questions
        return [
            q for q in questions if q.source_type.value in reviewed_values
        ]

    path = dataset_path(version, filename)
    if not path.exists():
        questions = _build_default_dataset(version)
        questions = filter_reviewed(questions)
        if questions or not only_reviewed:
            save_dataset(questions, version=version, filename=filename)
        return questions
    questions: List[EvalQuestion] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                q = EvalQuestion.from_dict(json.loads(line))
                if only_reviewed and q.source_type.value not in reviewed_values:
                    continue
                questions.append(q)
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid eval dataset line {lineno} in {path}: {exc}"
                ) from exc

    expected = _build_default_dataset(version)
    if expected and version in {"v3", "v4"} and (
        len(questions) != len(expected)
        or {q.id for q in questions} != {q.id for q in expected}
    ):
        questions = expected
        questions = filter_reviewed(questions)
        save_dataset(questions, version=version, filename=filename)
    return filter_reviewed(questions)


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
