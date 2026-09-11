"""Deterministic local retraining pipeline for plan-required text classifiers.

The implementation is intentionally dependency-light: it trains multinomial
Naive Bayes models from JSONL datasets with fields `text` and `label`, writes
versioned JSON artifacts, and reports holdout accuracy. It is a real training
step, not merely a manifest generator. Medical data quality remains the
responsibility of the supplied/licensed datasets.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TrainingExample:
    text: str
    label: str


@dataclass(frozen=True)
class TrainingReport:
    task: str
    examples: int
    labels: int
    vocabulary: int
    holdout: int
    accuracy: float
    artifact: str
    created_at: float


class MultinomialTextNB:
    def __init__(self, alpha: float = 1.0):
        self.alpha = max(1e-6, float(alpha))
        self.labels: list[str] = []
        self.vocabulary: dict[str, int] = {}
        self.class_doc_counts: dict[str, int] = {}
        self.class_token_totals: dict[str, int] = {}
        self.class_token_counts: dict[str, dict[str, int]] = {}
        self.total_docs = 0

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [token.casefold() for token in __import__("re").findall(r"[\wÀ-ÿ]+", str(text)) if len(token) >= 2]

    def fit(self, examples: Iterable[TrainingExample]) -> "MultinomialTextNB":
        rows = list(examples)
        if not rows:
            raise ValueError("training dataset is empty")
        self.labels = sorted({row.label for row in rows})
        self.total_docs = len(rows)
        self.class_token_counts = {label: {} for label in self.labels}
        self.class_doc_counts = {label: 0 for label in self.labels}
        self.class_token_totals = {label: 0 for label in self.labels}
        vocabulary: set[str] = set()
        for row in rows:
            tokens = self.tokenize(row.text)
            self.class_doc_counts[row.label] += 1
            for token in tokens:
                vocabulary.add(token)
                bucket = self.class_token_counts[row.label]
                bucket[token] = bucket.get(token, 0) + 1
                self.class_token_totals[row.label] += 1
        self.vocabulary = {token: index for index, token in enumerate(sorted(vocabulary))}
        if not self.vocabulary:
            raise ValueError("training dataset produced an empty vocabulary")
        return self

    def predict(self, text: str) -> str:
        tokens = self.tokenize(text)
        vocab_size = max(1, len(self.vocabulary))
        scores: dict[str, float] = {}
        for label in self.labels:
            prior = math.log(self.class_doc_counts[label] / self.total_docs)
            total = self.class_token_totals[label] + self.alpha * vocab_size
            score = prior
            counts = self.class_token_counts[label]
            for token in tokens:
                count = counts.get(token, 0)
                score += math.log((count + self.alpha) / total)
            scores[label] = score
        return max(scores, key=scores.get)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "multinomial_text_nb",
            "alpha": self.alpha,
            "labels": self.labels,
            "vocabulary": self.vocabulary,
            "class_doc_counts": self.class_doc_counts,
            "class_token_totals": self.class_token_totals,
            "class_token_counts": self.class_token_counts,
            "total_docs": self.total_docs,
        }


def load_jsonl(path: str | Path) -> list[TrainingExample]:
    rows: list[TrainingExample] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        text = str(item.get("text") or item.get("query") or item.get("input") or "").strip()
        label = str(item.get("label") or item.get("intent") or item.get("class") or "").strip()
        if not text or not label:
            raise ValueError(f"{path}: line {line_number} must contain non-empty text and label")
        rows.append(TrainingExample(text, label))
    return rows


def train_task(task: str, dataset_path: str | Path, artifact_dir: str | Path, *, holdout_fraction: float = 0.2, seed: int = 17) -> TrainingReport:
    examples = load_jsonl(dataset_path)
    if len(examples) < 20:
        raise ValueError(f"{task}: at least 20 labeled examples are required; got {len(examples)}")
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    holdout_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * holdout_fraction)))
    train_rows = shuffled[:-holdout_count]
    holdout_rows = shuffled[-holdout_count:]
    model = MultinomialTextNB().fit(train_rows)
    correct = sum(model.predict(row.text) == row.label for row in holdout_rows)
    accuracy = correct / len(holdout_rows)
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / f"{task}.json"
    payload = {
        "schema_version": 1,
        "task": task,
        "training_report": asdict(TrainingReport(task, len(examples), len(model.labels), len(model.vocabulary), len(holdout_rows), accuracy, str(artifact), time.time())),
        "model": model.to_dict(),
    }
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return TrainingReport(task, len(examples), len(model.labels), len(model.vocabulary), len(holdout_rows), accuracy, str(artifact), payload["training_report"]["created_at"])


def train_configured_tasks(config: dict[str, str], artifact_dir: str | Path) -> list[TrainingReport]:
    reports = []
    for task, dataset in config.items():
        reports.append(train_task(task, dataset, artifact_dir))
    return reports
