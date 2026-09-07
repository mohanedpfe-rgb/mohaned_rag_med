from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_project.evaluation.dataset import dataset_path, load_dataset
from rag_project.evaluation.dataset import save_dataset
from rag_project.evaluation.types import EvalQuestion, EvalSource
from rag_project.app.rag_system import RAGSystem
from rag_project.configuration.settings import Settings
from rag_project.utils.text_utils import detect_language


VALID_CATEGORIES = {
    "factual_recall",
    "multi_hop_reasoning",
    "table_lookup",
    "cross_section_comparison",
    "contraindication",
    "dosage_calculation",
    "negative_control",
}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def provenance_fingerprint(entry: dict[str, Any]) -> str:
    provenance = entry.get("provenance", {})
    payload = {
        "source_document_id": provenance.get("source_document_id"),
        "source_chunk_id": provenance.get("source_chunk_id"),
        "source_file_name": provenance.get("source_file_name"),
        "source_text_excerpt": provenance.get("source_text_excerpt"),
        "relevant_chunk_ids": entry.get("relevant_chunk_ids", []),
        "relevant_page_numbers": entry.get("relevant_page_numbers", []),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def question_quality_flag(question: str) -> tuple[str, str]:
    words = [word for word in question.split() if word.strip()]
    lowered = question.casefold()
    vague_terms = {"what", "how", "related", "text", "stated", "discuss"}
    if len(words) < 4:
        return "FLAG_REVIEW", "question is too short for reliable answer-quality evaluation"
    if len(words) < 7 and len(set(words) & vague_terms) >= 2:
        return "FLAG_REVIEW", "question uses vague wording and needs human clarification"
    if "?" not in question and not lowered.endswith(("؟", "？")):
        return "FLAG_REVIEW", "question has no explicit interrogative punctuation"
    return "OK", ""


def validate_review_entry(entry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    entry_id = entry.get("id", "<missing-id>")
    if not entry.get("id"):
        errors.append("missing id")
    if entry.get("review_status") not in {"pending_human_review", "human_reviewed"}:
        errors.append("invalid review_status")
    if entry.get("review_status") == "human_reviewed" and entry.get("source_type") not in {
        "human_curated",
        "llm_silver_reviewed",
    }:
        errors.append("human_reviewed requires human_curated or llm_silver_reviewed source_type")
    if entry.get("review_status") == "human_reviewed":
        for field in ("reviewer", "reviewed_at", "review_decision"):
            if not entry.get(field):
                errors.append(f"human_reviewed requires {field}")
        if entry.get("reviewed_at"):
            try:
                datetime.fromisoformat(str(entry["reviewed_at"]))
            except ValueError:
                errors.append("reviewed_at must be ISO-8601")
        if entry.get("review_decision") not in {"approved", "rejected"}:
            errors.append("human_reviewed requires review_decision approved or rejected")
        if entry.get("review_decision") == "rejected" and not entry.get("review_notes"):
            errors.append("rejected entries require review_notes")
    if entry.get("language") is not None and entry.get("language") not in {"en", "fr", "ar"}:
        errors.append("language must be one of en, fr, ar, or null")
    if not entry.get("question"):
        errors.append("missing question")
    if entry.get("category") not in VALID_CATEGORIES:
        errors.append("category must be a valid evaluation category")
    if entry.get("difficulty") not in VALID_DIFFICULTIES:
        errors.append("difficulty must be easy, medium, or hard")
    if not entry.get("relevant_chunk_ids"):
        errors.append("missing relevant_chunk_ids")
    if not entry.get("relevant_page_numbers"):
        errors.append("missing relevant_page_numbers")
    provenance = entry.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("missing provenance")
    else:
        for field in ("source_document_id", "source_chunk_id", "source_text_excerpt"):
            if not provenance.get(field):
                errors.append(f"missing provenance.{field}")
    expected_fingerprint = provenance_fingerprint(entry)
    if entry.get("provenance_fingerprint") != expected_fingerprint:
        errors.append("provenance_fingerprint does not match immutable source fields")
    return [f"{entry_id}: {error}" for error in errors]


def validate_review_manifest(
    entries: list[dict[str, Any]],
    require_reviewed: bool = False,
) -> list[str]:
    errors: list[str] = []
    if not entries:
        errors.append("manifest is empty")
    seen_ids: set[str] = set()
    for entry in entries:
        entry_id = entry.get("id")
        if entry_id in seen_ids:
            errors.append(f"{entry_id}: duplicate id")
        if entry_id:
            seen_ids.add(entry_id)
        errors.extend(validate_review_entry(entry))
        if require_reviewed and entry.get("review_status") != "human_reviewed":
            errors.append(f"{entry_id}: entry is not human_reviewed")
    return errors


def review_manifest_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}
    for entry in entries:
        status = str(entry.get("review_status", "invalid"))
        status_counts[status] = status_counts.get(status, 0) + 1
        source = str(entry.get("source_type", "missing"))
        source_counts[source] = source_counts.get(source, 0) + 1
        category = str(entry.get("category", "missing"))
        category_counts[category] = category_counts.get(category, 0) + 1
        language = str(entry.get("language") or "unverified")
        language_counts[language] = language_counts.get(language, 0) + 1
    reviewed = status_counts.get("human_reviewed", 0)
    approved = sum(
        1
        for entry in entries
        if entry.get("review_status") == "human_reviewed"
        and entry.get("review_decision") == "approved"
    )
    rejected = sum(
        1
        for entry in entries
        if entry.get("review_status") == "human_reviewed"
        and entry.get("review_decision") == "rejected"
    )
    total = len(entries)
    return {
        "total": total,
        "reviewed": reviewed,
        "approved": approved,
        "rejected": rejected,
        "pending": status_counts.get("pending_human_review", 0),
        "review_completion_ratio": reviewed / total if total else 0.0,
        "by_status": status_counts,
        "by_source": source_counts,
        "by_category": category_counts,
        "by_language": language_counts,
    }


def certification_readiness(
    entries: list[dict[str, Any]],
    minimum_questions: int = 30,
    maximum_questions: int = 50,
    required_languages: tuple[str, ...] = (),
    required_categories: tuple[str, ...] = (),
    required_difficulties: tuple[str, ...] = (),
) -> dict[str, Any]:
    summary = review_manifest_summary(entries)
    approved = summary["approved"]
    size_ok = minimum_questions <= approved <= maximum_questions
    validation_errors = validate_review_manifest(entries)
    ready = (
        not validation_errors
        and size_ok
        and summary["pending"] == 0
        and summary["rejected"] == 0
    )
    reasons: list[str] = []
    if validation_errors:
        reasons.append(f"manifest validation failed with {len(validation_errors)} error(s)")
    language_counts = summary["by_language"]
    missing_languages = [
        language for language in required_languages if language_counts.get(language, 0) == 0
    ]
    category_counts = summary["by_category"]
    missing_categories = [
        category for category in required_categories if category_counts.get(category, 0) == 0
    ]
    difficulty_counts: dict[str, int] = {}
    for entry in entries:
        if entry.get("review_status") == "human_reviewed" and entry.get("review_decision") == "approved":
            difficulty = str(entry.get("difficulty", "missing"))
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
    missing_difficulties = [
        difficulty for difficulty in required_difficulties if difficulty_counts.get(difficulty, 0) == 0
    ]
    unverified_language_count = language_counts.get("unverified", 0)
    if not size_ok:
        reasons.append(
            f"reviewed question count must be between {minimum_questions} and "
            f"{maximum_questions}; found {approved}"
        )
    if summary["pending"]:
        reasons.append(f"{summary['pending']} entries remain pending human review")
    if summary["rejected"]:
        reasons.append(f"{summary['rejected']} entries were rejected during human review")
    if missing_languages:
        reasons.append("missing required languages: " + ", ".join(missing_languages))
    if required_languages and unverified_language_count:
        reasons.append(
            f"{unverified_language_count} entries have unverified language metadata"
        )
    if missing_categories:
        reasons.append("missing required categories: " + ", ".join(missing_categories))
    if missing_difficulties:
        reasons.append("missing required difficulties: " + ", ".join(missing_difficulties))
    return {
        "ready": ready,
        "reviewed_question_count": approved,
        "minimum_questions": minimum_questions,
        "maximum_questions": maximum_questions,
        "required_languages": list(required_languages),
        "missing_languages": missing_languages,
        "unverified_language_count": unverified_language_count,
        "required_categories": list(required_categories),
        "missing_categories": missing_categories,
        "required_difficulties": list(required_difficulties),
        "missing_difficulties": missing_difficulties,
        "approved_by_difficulty": difficulty_counts,
        "reasons": reasons,
        "validation_errors": validation_errors,
    }


def _question_to_manifest_entry(question: EvalQuestion) -> dict[str, Any]:
    metadata = dict(question.metadata)
    return {
        "id": question.id,
        "review_status": "pending_human_review",
        "reviewer": None,
        "reviewed_at": None,
        "review_decision": None,
        "review_notes": None,
        "provenance_fingerprint": provenance_fingerprint(
            {
                "provenance": {
                    "source_document_id": metadata.get("source_document_id"),
                    "source_chunk_id": metadata.get("source_chunk_id"),
                    "source_file_name": metadata.get("source_file_name"),
                    "source_text_excerpt": metadata.get("source_text_excerpt"),
                },
                "relevant_chunk_ids": question.relevant_chunk_ids,
                "relevant_page_numbers": question.relevant_page_numbers,
            }
        ),
        "source_type": question.source_type.value,
        "category": question.category.value,
        "difficulty": question.difficulty.value,
        "language": metadata.get("language"),
        "question": question.question,
        "relevant_chunk_ids": question.relevant_chunk_ids,
        "relevant_page_numbers": question.relevant_page_numbers,
        "acceptable_answers": question.acceptable_answers,
        "required_keyphrases": question.required_keyphrases,
        "forbidden_keyphrases": question.forbidden_keyphrases,
        "notes": question.notes,
        "provenance": {
            "source_document_id": metadata.get("source_document_id"),
            "source_file_name": metadata.get("source_file_name"),
            "source_chunk_id": metadata.get("source_chunk_id"),
            "source_text_excerpt": metadata.get("source_text_excerpt"),
            "generated_by": metadata.get("generated_by"),
        },
        "review_checklist": [
            "verify_question_is_answerable_from_source",
            "verify_chunk_and_page_provenance",
            "correct_ocr_or_transcription_artifacts",
            "replace_ambiguous_or_overlong_answers",
            "confirm_language_and_category",
        ],
    }


def build_review_manifest(version: str = "v4", filename: str = "dataset.jsonl") -> list[dict[str, Any]]:
    """Create a provenance-preserving manifest without claiming human review."""
    return [_question_to_manifest_entry(q) for q in load_dataset(version=version, filename=filename)]


def build_candidate_review_manifest(
    versions: tuple[str, ...] = ("v3", "v4"),
    per_category: int = 10,
    target_count: int | None = None,
) -> list[dict[str, Any]]:
    """Build a deterministic pending queue balanced across available categories."""
    candidates: list[EvalQuestion] = []
    for version in versions:
        candidates.extend(load_dataset(version=version))
    candidates = [
        question
        for question in candidates
        if question.relevant_chunk_ids
        and question.relevant_page_numbers
        and question.metadata.get("source_document_id")
        and question.metadata.get("source_chunk_id")
        and question.metadata.get("source_text_excerpt")
    ]
    selected: list[EvalQuestion] = []
    seen_ids: set[str] = set()
    categories = sorted({question.category.value for question in candidates})
    for category in categories:
        count = 0
        for question in candidates:
            if question.id in seen_ids or question.category.value != category:
                continue
            selected.append(question)
            seen_ids.add(question.id)
            count += 1
            if count >= per_category:
                break
    if target_count is not None and len(selected) < target_count:
        for question in candidates:
            if question.id in seen_ids:
                continue
            selected.append(question)
            seen_ids.add(question.id)
            if len(selected) >= target_count:
                break
    return [_question_to_manifest_entry(q) for q in selected]


def candidate_review_audit(
    versions: tuple[str, ...] = ("v3", "v4"),
) -> dict[str, Any]:
    total = 0
    eligible = 0
    excluded: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    for version in versions:
        for question in load_dataset(version=version):
            total += 1
            reasons: list[str] = []
            if not question.relevant_chunk_ids:
                reasons.append("missing_chunk_ids")
            if not question.relevant_page_numbers:
                reasons.append("missing_page_numbers")
            for field in ("source_document_id", "source_chunk_id", "source_text_excerpt"):
                if not question.metadata.get(field):
                    reasons.append(f"missing_{field}")
            if reasons:
                for reason in reasons:
                    excluded[reason] = excluded.get(reason, 0) + 1
                continue
            eligible += 1
            category = question.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
    return {
        "versions": list(versions),
        "total_candidates": total,
        "provenance_valid_candidates": eligible,
        "excluded_candidates": total - eligible,
        "exclusion_reasons": excluded,
        "eligible_by_category": category_counts,
    }


def save_review_session_audit(
    version: str,
    source_versions: tuple[str, ...],
    packet_size: int,
    output_filename: str = "review_session.json",
) -> Path:
    manifest = load_review_manifest(version)
    audit = candidate_review_audit(source_versions)
    output = dataset_path(version=version, filename=output_filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_versions": list(source_versions),
        "manifest_filename": "review_manifest.jsonl",
        "manifest_count": len(manifest),
        "manifest_ids": [entry.get("id") for entry in manifest],
        "pending_count": sum(
            1 for entry in manifest if entry.get("review_status") == "pending_human_review"
        ),
        "packet_size": packet_size,
        "packet_count": (len(manifest) + packet_size - 1) // packet_size if packet_size else 0,
        "candidate_audit": audit,
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return output


def validate_review_packets(
    version: str,
    packet_filenames: tuple[str, ...],
    manifest_filename: str = "review_manifest.jsonl",
) -> list[str]:
    manifest = load_review_manifest(version, manifest_filename)
    expected_ids = [entry.get("id") for entry in manifest]
    expected = set(expected_ids)
    errors: list[str] = []
    packet_ids: list[str] = []
    for filename in packet_filenames:
        path = dataset_path(version=version, filename=filename)
        if not path.exists():
            errors.append(f"missing packet: {filename}")
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                entry_id = (row.get("item_id") or row.get("id") or "").strip()
                if entry_id in packet_ids:
                    errors.append(f"duplicate packet id: {entry_id}")
                packet_ids.append(entry_id)
    actual = set(packet_ids)
    errors.extend(f"missing packet id: {entry_id}" for entry_id in expected - actual)
    errors.extend(f"unexpected packet id: {entry_id}" for entry_id in actual - expected)
    if len(packet_ids) != len(expected_ids):
        errors.append(
            f"packet row count {len(packet_ids)} does not match manifest count {len(expected_ids)}"
        )
    return errors


def finalize_review_session(
    version: str,
    packet_filenames: tuple[str, ...],
    manifest_filename: str = "review_manifest.jsonl",
    output_filename: str = "review_manifest_final.jsonl",
    required_languages: tuple[str, ...] = ("en", "fr", "ar"),
    required_categories: tuple[str, ...] = (
        "factual_recall",
        "table_lookup",
        "multi_hop_reasoning",
        "negative_control",
    ),
    required_difficulties: tuple[str, ...] = ("easy", "medium", "hard"),
) -> dict[str, Any]:
    packet_errors = validate_review_packets(version, packet_filenames, manifest_filename)
    if packet_errors:
        raise ValueError("Packet validation failed:\n" + "\n".join(packet_errors))
    output = import_reviewer_worksheet(
        version=version,
        manifest_filename=manifest_filename,
        worksheet_filenames=packet_filenames,
        output_filename=output_filename,
    )
    entries = load_review_manifest(version, output_filename)
    manifest_errors = validate_review_manifest(entries)
    if manifest_errors:
        raise ValueError("Final manifest validation failed:\n" + "\n".join(manifest_errors))
    return {
        "output": str(output),
        "entry_count": len(entries),
        "summary": review_manifest_summary(entries),
        "readiness": certification_readiness(
            entries,
            required_languages=required_languages,
            required_categories=required_categories,
            required_difficulties=required_difficulties,
        ),
    }


def save_review_manifest(
    version: str = "v4",
    filename: str = "dataset.jsonl",
    output_filename: str = "review_manifest.jsonl",
) -> Path:
    output = dataset_path(version=version, filename=output_filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_review_manifest(version=version, filename=filename)
    with output.open("w", encoding="utf-8") as handle:
        for entry in manifest:
            handle.write(json.dumps(entry, ensure_ascii=False))
            handle.write("\n")
    return output


def load_review_manifest(version: str = "v4", filename: str = "review_manifest.jsonl") -> list[dict[str, Any]]:
    path = dataset_path(version=version, filename=filename)
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid review manifest line {lineno} in {path}: {exc}") from exc
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid review manifest line {lineno} in {path}: expected object")
            entries.append(entry)
    return entries


def save_reviewer_worksheet(
    version: str = "v4",
    manifest_filename: str = "review_manifest.jsonl",
    output_filename: str = "reviewer_worksheet.csv",
    batch_number: int | None = None,
    batch_size: int = 10,
) -> Path:
    entries = load_review_manifest(version, manifest_filename)
    if batch_number is not None:
        if batch_number < 1 or batch_size < 1:
            raise ValueError("batch_number and batch_size must be positive")
        start = (batch_number - 1) * batch_size
        entries = entries[start : start + batch_size]
        if not entries:
            raise ValueError(f"batch_number {batch_number} is outside the manifest")
    output = dataset_path(version=version, filename=output_filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "review_status",
        "source_type",
        "category",
        "difficulty",
        "language",
        "question",
        "acceptable_answers",
        "required_keyphrases",
        "relevant_chunk_ids",
        "relevant_page_numbers",
        "source_file_name",
        "source_text_excerpt",
        "provenance_fingerprint",
        "reviewer",
        "reviewed_at",
        "review_decision",
        "review_notes",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            provenance = entry.get("provenance", {})
            writer.writerow(
                {
                    "id": entry.get("id"),
                    "review_status": entry.get("review_status"),
                    "source_type": entry.get("source_type"),
                    "category": entry.get("category"),
                    "difficulty": entry.get("difficulty"),
                    "language": entry.get("language") or "",
                    "question": entry.get("question"),
                    "acceptable_answers": " | ".join(entry.get("acceptable_answers", [])),
                    "required_keyphrases": " | ".join(entry.get("required_keyphrases", [])),
                    "relevant_chunk_ids": " | ".join(entry.get("relevant_chunk_ids", [])),
                    "relevant_page_numbers": " | ".join(
                        str(page) for page in entry.get("relevant_page_numbers", [])
                    ),
                    "source_file_name": provenance.get("source_file_name"),
                    "source_text_excerpt": provenance.get("source_text_excerpt"),
                    "provenance_fingerprint": entry.get("provenance_fingerprint"),
                    "reviewer": entry.get("reviewer") or "",
                    "reviewed_at": entry.get("reviewed_at") or "",
                    "review_decision": entry.get("review_decision") or "",
                    "review_notes": entry.get("review_notes") or "",
                }
            )
    return output


def save_review_packet(
    version: str = "v6",
    manifest_filename: str = "review_manifest.jsonl",
    output_filename: str = "review_packet.csv",
    batch_number: int | None = None,
    batch_size: int = 10,
) -> Path:
    entries = load_review_manifest(version, manifest_filename)
    if batch_number is not None:
        if batch_number < 1 or batch_size < 1:
            raise ValueError("batch_number and batch_size must be positive")
        start = (batch_number - 1) * batch_size
        entries = entries[start : start + batch_size]
        if not entries:
            raise ValueError(f"batch_number {batch_number} is outside the manifest")
    fields = [
        "item_id", "query", "query_language", "expected_language",
        "candidate_answer", "reference_answer", "evidence_ids", "gold_evidence_ids",
        "accepted_evidence_ids", "evidence_text", "query_intent", "query_quality",
        "retrieval_relevance", "evidence_sufficiency", "answer_groundedness",
        "answer_correctness", "answer_completeness", "answer_language_correct",
        "citation_correctness", "contradiction_present", "abstention_expected",
        "failure_category", "severity", "review_decision", "corrected_answer",
        "reviewer_id", "review_timestamp", "review_notes", "review_version",
        "review_status",
    ]
    output = dataset_path(version=version, filename=output_filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            provenance = entry.get("provenance", {})
            writer.writerow({
                "item_id": entry.get("id"),
                "query": entry.get("question"),
                "query_language": entry.get("language") or detect_language(entry["question"]),
                "expected_language": entry.get("language") or detect_language(entry["question"]),
                "candidate_answer": "",
                "reference_answer": " | ".join(entry.get("acceptable_answers", [])),
                "evidence_ids": " | ".join(entry.get("relevant_chunk_ids", [])),
                "gold_evidence_ids": " | ".join(entry.get("relevant_chunk_ids", [])),
                "accepted_evidence_ids": "",
                "evidence_text": provenance.get("source_text_excerpt") or "",
                "query_intent": "",
                "query_quality": "",
                "retrieval_relevance": "",
                "evidence_sufficiency": "",
                "answer_groundedness": "",
                "answer_correctness": "",
                "answer_completeness": "",
                "answer_language_correct": "",
                "citation_correctness": "",
                "contradiction_present": "",
                "abstention_expected": "",
                "failure_category": "",
                "severity": "",
                "review_decision": "",
                "corrected_answer": "",
                "reviewer_id": "",
                "review_timestamp": "",
                "review_notes": "",
                "review_version": version,
                "review_status": "PENDING_HUMAN_REVIEW",
            })
    return output


def generate_review_packet(
    version: str = "v6",
    manifest_filename: str = "review_manifest.jsonl",
    output_filename: str = "review_packet_with_candidates.csv",
    batch_number: int | None = None,
    batch_size: int = 10,
    settings: Settings | None = None,
    rag_system: RAGSystem | None = None,
) -> Path:
    entries = load_review_manifest(version, manifest_filename)
    if batch_number is not None:
        if batch_number < 1 or batch_size < 1:
            raise ValueError("batch_number and batch_size must be positive")
        start = (batch_number - 1) * batch_size
        entries = entries[start : start + batch_size]
        if not entries:
            raise ValueError(f"batch_number {batch_number} is outside the manifest")
    system = rag_system or RAGSystem(settings or Settings.from_env())
    fields = [
        "item_id", "query", "query_language", "expected_language",
        "candidate_answer", "reference_answer", "evidence_ids", "gold_evidence_ids",
        "accepted_evidence_ids", "evidence_text", "query_intent", "query_quality",
        "retrieval_relevance", "evidence_sufficiency", "answer_groundedness",
        "answer_correctness", "answer_completeness", "answer_language_correct",
        "citation_correctness", "contradiction_present", "abstention_expected",
        "failure_category", "severity", "question_quality_flag", "question_quality_notes",
        "review_decision", "corrected_answer", "reviewer_id", "review_timestamp",
        "review_notes", "review_version", "review_status",
    ]
    output = dataset_path(version=version, filename=output_filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            result = system.answer(entry["question"])
            candidate_answer = str(result.get("answer", "")).strip()
            if not candidate_answer:
                raise RuntimeError(f"{entry['id']}: RAG returned an empty candidate answer")
            hits = result.get("hits", [])
            evidence_ids = [
                str(hit.metadata.get("chunk_id", hit.doc_id)) for hit in hits
            ]
            evidence_text = "\n\n".join(str(hit.text) for hit in hits)
            quality_flag, quality_notes = question_quality_flag(entry["question"])
            query_analysis = result.get("query_analysis") or {}
            evidence_alignment = result.get("evidence_alignment") or {}
            writer.writerow({
                "item_id": entry["id"],
                "query": entry["question"],
                "query_language": entry.get("language") or detect_language(entry["question"]),
                "expected_language": entry.get("language") or detect_language(entry["question"]),
                "candidate_answer": candidate_answer,
                "reference_answer": " | ".join(entry.get("acceptable_answers", [])),
                "evidence_ids": " | ".join(evidence_ids),
                "gold_evidence_ids": " | ".join(entry.get("relevant_chunk_ids", [])),
                "accepted_evidence_ids": "",
                "evidence_text": evidence_text,
                "query_intent": query_analysis.get("query_intent", ""),
                "query_quality": query_analysis.get("query_quality", ""),
                "retrieval_relevance": evidence_alignment.get("query_relevance", ""),
                "evidence_sufficiency": evidence_alignment.get("decision", ""),
                "answer_groundedness": evidence_alignment.get("answerability", ""),
                "answer_correctness": "",
                "answer_completeness": "",
                "answer_language_correct": "",
                "citation_correctness": "",
                "contradiction_present": evidence_alignment.get("contradiction", ""),
                "abstention_expected": str(query_analysis.get("should_abstain", "")).lower(),
                "failure_category": "",
                "severity": "",
                "question_quality_flag": quality_flag,
                "question_quality_notes": quality_notes,
                "review_decision": "",
                "corrected_answer": "",
                "reviewer_id": "",
                "review_timestamp": "",
                "review_notes": "",
                "review_version": version,
                "review_status": "PENDING_HUMAN_REVIEW",
            })
    return output


def import_reviewer_worksheet(
    version: str = "v4",
    manifest_filename: str = "review_manifest.jsonl",
    worksheet_filename: str = "reviewer_worksheet.csv",
    output_filename: str = "review_manifest_reviewed.jsonl",
    worksheet_filenames: tuple[str, ...] | None = None,
) -> Path:
    manifest = load_review_manifest(version, manifest_filename)
    by_id = {entry.get("id"): entry for entry in manifest}
    filenames = worksheet_filenames or (worksheet_filename,)
    updates: dict[str, dict[str, str]] = {}
    for filename in filenames:
        worksheet_path = dataset_path(version=version, filename=filename)
        if not worksheet_path.exists():
            raise ValueError(f"Reviewer worksheet does not exist: {worksheet_path}")
        with worksheet_path.open("r", encoding="utf-8", newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                entry_id = (row.get("id") or "").strip()
                if not entry_id or entry_id not in by_id:
                    raise ValueError(f"Worksheet line {row_number} contains unknown id: {entry_id}")
                if entry_id in updates:
                    raise ValueError(f"Worksheet contains duplicate id: {entry_id}")
                provenance = by_id[entry_id].get("provenance", {})
                if row.get("source_file_name") != provenance.get("source_file_name"):
                    raise ValueError(f"{entry_id}: source_file_name does not match manifest")
                worksheet_excerpt = (row.get("source_text_excerpt") or "").replace("\r\n", "\n")
                manifest_excerpt = (provenance.get("source_text_excerpt") or "").replace("\r\n", "\n")
                if worksheet_excerpt != manifest_excerpt:
                    raise ValueError(f"{entry_id}: source_text_excerpt does not match manifest")
                if row.get("provenance_fingerprint") != by_id[entry_id].get("provenance_fingerprint"):
                    raise ValueError(f"{entry_id}: provenance_fingerprint does not match manifest")
                updates[entry_id] = row
    if set(updates) != set(by_id):
        missing = sorted(set(by_id) - set(updates))
        raise ValueError(f"Worksheet is incomplete; missing ids: {', '.join(missing)}")

    updated_entries: list[dict[str, Any]] = []
    for entry in manifest:
        row = updates[entry["id"]]
        updated = dict(entry)
        updated["review_status"] = row.get("review_status") or entry["review_status"]
        updated["category"] = row.get("category") or entry["category"]
        updated["difficulty"] = row.get("difficulty") or entry["difficulty"]
        updated["language"] = row.get("language") or None
        updated["question"] = row.get("question") or entry["question"]
        updated["acceptable_answers"] = [
            answer.strip() for answer in (row.get("acceptable_answers") or "").split("|") if answer.strip()
        ]
        updated["required_keyphrases"] = [
            phrase.strip()
            for phrase in (row.get("required_keyphrases") or "").split("|")
            if phrase.strip()
        ]
        updated["reviewer"] = row.get("reviewer") or None
        updated["reviewed_at"] = row.get("reviewed_at") or None
        updated["review_decision"] = row.get("review_decision") or None
        updated["review_notes"] = row.get("review_notes") or None
        updated_entries.append(updated)
    errors = validate_review_manifest(updated_entries)
    if errors:
        raise ValueError("Imported worksheet failed validation:\n" + "\n".join(errors))
    output = dataset_path(version=version, filename=output_filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        for entry in updated_entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    try:
        os.replace(temporary_path, output)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise
    return output


def export_reviewed_dataset(
    version: str = "v4",
    manifest_filename: str = "review_manifest.jsonl",
    output_filename: str = "gold_dataset.jsonl",
) -> Path:
    entries = load_review_manifest(version, manifest_filename)
    errors = validate_review_manifest(entries, require_reviewed=True)
    if errors:
        raise ValueError("Cannot export reviewed dataset:\n" + "\n".join(errors))

    questions: list[EvalQuestion] = []
    for entry in entries:
        if entry.get("review_decision") != "approved":
            continue
        if entry["source_type"] not in {"human_curated", "llm_silver_reviewed"}:
            raise ValueError(
                f"{entry['id']}: reviewed entries must use a reviewed source type"
            )
        metadata = dict(entry.get("provenance", {}))
        metadata["review_status"] = entry["review_status"]
        questions.append(
            EvalQuestion.from_dict(
                {
                    "id": entry["id"],
                    "category": entry["category"],
                    "difficulty": entry["difficulty"],
                    "question": entry["question"],
                    "relevant_chunk_ids": entry["relevant_chunk_ids"],
                    "relevant_page_numbers": entry["relevant_page_numbers"],
                    "acceptable_answers": entry["acceptable_answers"],
                    "required_keyphrases": entry["required_keyphrases"],
                    "forbidden_keyphrases": entry["forbidden_keyphrases"],
                    "source_type": entry["source_type"],
                    "notes": entry.get("notes", ""),
                    "metadata": metadata,
                }
            )
        )
    return save_dataset(questions, version=version, filename=output_filename)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a human-review manifest for an eval dataset.")
    parser.add_argument("--dataset-version", default="v4")
    parser.add_argument("--dataset-filename", default="dataset.jsonl")
    parser.add_argument("--output-filename", default="review_manifest.jsonl")
    parser.add_argument("--gold-filename", default="gold_dataset.jsonl")
    parser.add_argument("--validate", action="store_true",
                        help="Validate an existing review manifest and exit")
    parser.add_argument("--require-reviewed", action="store_true",
                        help="Require every manifest entry to be human_reviewed")
    parser.add_argument("--export-reviewed", action="store_true",
                        help="Export validated human-reviewed entries as a dataset")
    parser.add_argument("--summary", action="store_true",
                        help="Print review progress summary and exit")
    parser.add_argument("--readiness", action="store_true",
                        help="Print certification readiness and exit")
    parser.add_argument("--require-languages", default="",
                        help="Comma-separated language metadata required for readiness")
    parser.add_argument("--require-categories", default="",
                        help="Comma-separated evaluation categories required for readiness")
    parser.add_argument("--require-difficulties", default="",
                        help="Comma-separated difficulties required for readiness")
    parser.add_argument("--candidate-versions", default="",
                        help="Comma-separated dataset versions for a balanced review queue")
    parser.add_argument("--per-category", type=int, default=10,
                        help="Maximum candidates per category when building a review queue")
    parser.add_argument("--target-count", type=int, default=None,
                        help="Target pending-candidate count, capped by provenance-valid candidates")
    parser.add_argument("--candidate-audit", action="store_true",
                        help="Report candidate provenance and category availability")
    parser.add_argument("--session-audit", action="store_true",
                        help="Write a reproducible review-session audit artifact")
    parser.add_argument("--session-audit-filename", default="review_session.json")
    parser.add_argument("--worksheet", action="store_true",
                        help="Export a human-review CSV worksheet")
    parser.add_argument("--worksheet-filename", default="reviewer_worksheet.csv")
    parser.add_argument("--review-packet", action="store_true",
                        help="Export the required review-ready packet schema")
    parser.add_argument("--review-packet-filename", default="review_packet.csv")
    parser.add_argument("--generate-candidates", action="store_true",
                        help="Run the actual RAG system and populate candidate_answer")
    parser.add_argument("--candidate-packet-filename", default="review_packet_candidates.csv")
    parser.add_argument("--batch-number", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--import-worksheet", action="store_true",
                        help="Import a completed reviewer worksheet into a new manifest")
    parser.add_argument("--import-output-filename", default="review_manifest_reviewed.jsonl")
    parser.add_argument("--import-worksheets", default="",
                        help="Comma-separated worksheet packets to merge during import")
    parser.add_argument("--validate-packets", action="store_true",
                        help="Validate that worksheet packets partition the manifest")
    parser.add_argument("--packet-files", default="",
                        help="Comma-separated packet filenames for partition validation")
    parser.add_argument("--finalize-session", action="store_true",
                        help="Validate, merge, and assess reviewer packets in one operation")
    parser.add_argument("--final-output-filename", default="review_manifest_final.jsonl")
    parser.add_argument("--final-required-languages", default="en,fr,ar")
    parser.add_argument(
        "--final-required-categories",
        default="factual_recall,table_lookup,multi_hop_reasoning,negative_control",
    )
    parser.add_argument("--final-required-difficulties", default="easy,medium,hard")
    args = parser.parse_args()
    if args.candidate_audit:
        versions = tuple(
            version.strip() for version in args.candidate_versions.split(",") if version.strip()
        ) or (args.dataset_version,)
        print(json.dumps(candidate_review_audit(versions), indent=2, ensure_ascii=False))
        return 0
    if args.session_audit:
        versions = tuple(
            version.strip() for version in args.candidate_versions.split(",") if version.strip()
        ) or (args.dataset_version,)
        output = save_review_session_audit(
            version=args.dataset_version,
            source_versions=versions,
            packet_size=args.batch_size,
            output_filename=args.session_audit_filename,
        )
        print(output)
        return 0
    if args.worksheet:
        output = save_reviewer_worksheet(
            version=args.dataset_version,
            manifest_filename=args.output_filename,
            output_filename=args.worksheet_filename,
            batch_number=args.batch_number,
            batch_size=args.batch_size,
        )
        print(output)
        return 0
    if args.review_packet:
        output = save_review_packet(
            version=args.dataset_version,
            manifest_filename=args.output_filename,
            output_filename=args.review_packet_filename,
            batch_number=args.batch_number,
            batch_size=args.batch_size,
        )
        print(output)
        return 0
    if args.generate_candidates:
        output = generate_review_packet(
            version=args.dataset_version,
            manifest_filename=args.output_filename,
            output_filename=args.candidate_packet_filename,
            batch_number=args.batch_number,
            batch_size=args.batch_size,
        )
        print(output)
        return 0
    if args.import_worksheet:
        try:
            output = import_reviewer_worksheet(
                version=args.dataset_version,
                manifest_filename=args.output_filename,
                worksheet_filename=args.worksheet_filename,
                output_filename=args.import_output_filename,
                worksheet_filenames=tuple(
                    filename.strip()
                    for filename in args.import_worksheets.split(",")
                    if filename.strip()
                ) or None,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 2
        print(output)
        return 0
    if args.validate_packets:
        packet_files = tuple(
            filename.strip() for filename in args.packet_files.split(",") if filename.strip()
        )
        if not packet_files:
            print("ERROR: --packet-files is required")
            return 2
        errors = validate_review_packets(args.dataset_version, packet_files, args.output_filename)
        if errors:
            print("\n".join(errors))
            return 1
        print(f"valid: {len(packet_files)} packets partition {args.output_filename}")
        return 0
    if args.finalize_session:
        packet_files = tuple(
            filename.strip() for filename in args.packet_files.split(",") if filename.strip()
        )
        if not packet_files:
            print("ERROR: --packet-files is required")
            return 2
        try:
            result = finalize_review_session(
                version=args.dataset_version,
                packet_filenames=packet_files,
                manifest_filename=args.output_filename,
                output_filename=args.final_output_filename,
                required_languages=tuple(
                    value.strip()
                    for value in args.final_required_languages.split(",")
                    if value.strip()
                ),
                required_categories=tuple(
                    value.strip()
                    for value in args.final_required_categories.split(",")
                    if value.strip()
                ),
                required_difficulties=tuple(
                    value.strip()
                    for value in args.final_required_difficulties.split(",")
                    if value.strip()
                ),
            )
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 2
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.validate:
        entries = load_review_manifest(args.dataset_version, args.output_filename)
        errors = validate_review_manifest(entries, require_reviewed=args.require_reviewed)
        if errors:
            print("\n".join(errors))
            return 1
        print(f"valid: {len(entries)} entries")
        return 0
    if args.summary:
        print(json.dumps(
            review_manifest_summary(load_review_manifest(args.dataset_version, args.output_filename)),
            indent=2,
            ensure_ascii=False,
        ))
        return 0
    if args.readiness:
        readiness = certification_readiness(
            load_review_manifest(args.dataset_version, args.output_filename),
            required_languages=tuple(
                language.strip() for language in args.require_languages.split(",") if language.strip()
            ),
            required_categories=tuple(
                category.strip() for category in args.require_categories.split(",") if category.strip()
            ),
            required_difficulties=tuple(
                difficulty.strip()
                for difficulty in args.require_difficulties.split(",")
                if difficulty.strip()
            ),
        )
        print(json.dumps(readiness, indent=2, ensure_ascii=False))
        return 0 if readiness["ready"] else 2
    if args.export_reviewed:
        try:
            output = export_reviewed_dataset(
                version=args.dataset_version,
                manifest_filename=args.output_filename,
                output_filename=args.gold_filename,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 2
        print(output)
        return 0
    manifest = (
        build_candidate_review_manifest(
            tuple(version.strip() for version in args.candidate_versions.split(",") if version.strip()),
            per_category=args.per_category,
            target_count=args.target_count,
        )
        if args.candidate_versions
        else build_review_manifest(args.dataset_version, args.dataset_filename)
    )
    output = dataset_path(version=args.dataset_version, filename=args.output_filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for entry in manifest:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
