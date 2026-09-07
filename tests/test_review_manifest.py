import json
import csv
from io import StringIO

import rag_project.evaluation.dataset as dataset_module
import rag_project.evaluation.review_manifest as review_manifest_module
from rag_project.evaluation.review_manifest import (
    build_review_manifest,
    build_candidate_review_manifest,
    candidate_review_audit,
    certification_readiness,
    export_reviewed_dataset,
    review_manifest_summary,
    save_review_session_audit,
    validate_review_packets,
    import_reviewer_worksheet,
    finalize_review_session,
    save_reviewer_worksheet,
    save_review_packet,
    validate_review_manifest,
    provenance_fingerprint,
)


def test_review_manifest_preserves_provenance_and_pending_status():
    manifest = build_review_manifest("v4")

    assert len(manifest) == 10
    assert all(entry["review_status"] == "pending_human_review" for entry in manifest)
    assert all(entry["provenance"]["source_chunk_id"] for entry in manifest)
    assert all(entry["relevant_page_numbers"] for entry in manifest)
    assert all("verify_chunk_and_page_provenance" in entry["review_checklist"] for entry in manifest)
    assert validate_review_manifest(manifest) == []
    assert manifest[0]["provenance_fingerprint"] == provenance_fingerprint(manifest[0])


def test_candidate_review_manifest_is_balanced_and_pending():
    manifest = build_candidate_review_manifest(("v3", "v4"), per_category=2)

    assert len(manifest) >= 4
    assert all(entry["review_status"] == "pending_human_review" for entry in manifest)
    assert len({entry["id"] for entry in manifest}) == len(manifest)
    assert len({entry["category"] for entry in manifest}) >= 2


def test_candidate_review_manifest_can_fill_target_count():
    manifest = build_candidate_review_manifest(("v3", "v4"), per_category=2, target_count=20)

    assert len(manifest) == 20
    assert all(entry["review_status"] == "pending_human_review" for entry in manifest)
    assert validate_review_manifest(manifest) == []


def test_candidate_review_audit_explains_available_provenance():
    audit = candidate_review_audit(("v3", "v4"))

    assert audit["total_candidates"] == 105
    assert audit["provenance_valid_candidates"] == 75
    assert audit["excluded_candidates"] == 30
    assert "missing_chunk_ids" in audit["exclusion_reasons"]


def test_review_session_audit_records_queue_handoff(tmp_path, monkeypatch):
    monkeypatch.setattr(
        review_manifest_module,
        "dataset_path",
        lambda version, filename: tmp_path / version / filename,
    )
    manifest = build_review_manifest("v4")
    manifest_path = tmp_path / "v4" / "review_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in manifest),
        encoding="utf-8",
    )

    output = save_review_session_audit("v4", ("v3", "v4"), packet_size=4)
    audit = json.loads(output.read_text(encoding="utf-8"))

    assert audit["manifest_count"] == 10
    assert audit["pending_count"] == 10
    assert audit["packet_count"] == 3
    assert len(audit["manifest_ids"]) == 10


def test_review_packets_must_partition_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        review_manifest_module,
        "dataset_path",
        lambda version, filename: tmp_path / version / filename,
    )
    manifest = build_review_manifest("v4")
    manifest_path = tmp_path / "v4" / "review_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in manifest),
        encoding="utf-8",
    )
    save_reviewer_worksheet("v4", batch_number=1, batch_size=5, output_filename="a.csv")
    save_reviewer_worksheet("v4", batch_number=2, batch_size=5, output_filename="b.csv")

    assert validate_review_packets("v4", ("a.csv", "b.csv")) == []
    errors = validate_review_packets("v4", ("a.csv",))
    assert any(error.startswith("missing packet id:") for error in errors)
    assert any(error.startswith("packet row count") for error in errors)


def test_reviewer_worksheet_preserves_pending_status_and_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(
        review_manifest_module,
        "dataset_path",
        lambda version, filename: tmp_path / version / filename,
    )
    manifest = build_review_manifest("v4")[:1]
    manifest_path = tmp_path / "v4" / "review_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest[0], ensure_ascii=False) + "\n", encoding="utf-8")

    output = save_reviewer_worksheet("v4")
    text = output.read_text(encoding="utf-8")

    assert "review_status" in text
    assert "pending_human_review" in text
    assert manifest[0]["provenance"]["source_chunk_id"] in text
    assert manifest[0]["provenance"]["source_file_name"] in text


def test_reviewer_worksheet_import_requires_complete_matching_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(
        review_manifest_module,
        "dataset_path",
        lambda version, filename: tmp_path / version / filename,
    )
    manifest = build_review_manifest("v4")[:1]
    manifest_path = tmp_path / "v4" / "review_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest[0], ensure_ascii=False) + "\n", encoding="utf-8")
    worksheet = save_reviewer_worksheet("v4")
    worksheet.write_text(worksheet.read_text(encoding="utf-8"), encoding="utf-8")

    output = import_reviewer_worksheet("v4")

    imported = json.loads(output.read_text(encoding="utf-8").strip())
    assert imported["id"] == manifest[0]["id"]
    assert imported["review_status"] == "pending_human_review"


def test_reviewer_worksheet_supports_deterministic_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(
        review_manifest_module,
        "dataset_path",
        lambda version, filename: tmp_path / version / filename,
    )
    manifest = build_review_manifest("v4")
    manifest_path = tmp_path / "v4" / "review_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in manifest),
        encoding="utf-8",
    )

    output = save_reviewer_worksheet("v4", batch_number=1, batch_size=3)
    rows = list(csv.DictReader(StringIO(output.read_text(encoding="utf-8"))))

    assert len(rows) == 3
    assert [row["id"] for row in rows] == ["q_0001", "q_0002", "q_0004"]


def test_review_packet_contains_required_human_review_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(
        review_manifest_module,
        "dataset_path",
        lambda version, filename: tmp_path / version / filename,
    )
    manifest = build_review_manifest("v4")[:1]
    manifest_path = tmp_path / "v4" / "review_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest[0], ensure_ascii=False) + "\n", encoding="utf-8")

    output = save_review_packet("v4")
    row = next(csv.DictReader(StringIO(output.read_text(encoding="utf-8"))))

    assert row["item_id"] == "q_0001"
    assert row["review_status"] == "PENDING_HUMAN_REVIEW"
    assert row["review_decision"] == ""
    assert row["corrected_answer"] == ""
    assert row["reviewer_id"] == ""
    assert row["review_timestamp"] == ""
    assert row["review_notes"] == ""


def test_reviewer_packet_import_merges_all_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(
        review_manifest_module,
        "dataset_path",
        lambda version, filename: tmp_path / version / filename,
    )
    manifest = build_review_manifest("v4")
    manifest_path = tmp_path / "v4" / "review_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in manifest),
        encoding="utf-8",
    )
    save_reviewer_worksheet("v4", batch_number=1, batch_size=5, output_filename="batch_a.csv")
    save_reviewer_worksheet("v4", batch_number=2, batch_size=5, output_filename="batch_b.csv")

    output = import_reviewer_worksheet(
        "v4",
        worksheet_filenames=("batch_a.csv", "batch_b.csv"),
    )

    imported = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(imported) == 10


def test_reviewer_packet_import_replaces_output_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(
        review_manifest_module,
        "dataset_path",
        lambda version, filename: tmp_path / version / filename,
    )
    manifest = build_review_manifest("v4")[:1]
    manifest_path = tmp_path / "v4" / "review_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest[0], ensure_ascii=False) + "\n", encoding="utf-8")
    save_reviewer_worksheet("v4", batch_number=1, batch_size=1)

    output = import_reviewer_worksheet("v4")

    assert output.exists()
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_finalize_review_session_validates_merges_and_reports_readiness(tmp_path, monkeypatch):
    monkeypatch.setattr(
        review_manifest_module,
        "dataset_path",
        lambda version, filename: tmp_path / version / filename,
    )
    manifest = build_review_manifest("v4")
    manifest_path = tmp_path / "v4" / "review_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in manifest),
        encoding="utf-8",
    )
    save_reviewer_worksheet("v4", batch_number=1, batch_size=5, output_filename="a.csv")
    save_reviewer_worksheet("v4", batch_number=2, batch_size=5, output_filename="b.csv")

    result = finalize_review_session("v4", ("a.csv", "b.csv"))

    assert result["entry_count"] == 10
    assert result["summary"]["pending"] == 10
    assert result["readiness"]["ready"] is False
    assert result["readiness"]["missing_languages"] == ["en", "fr", "ar"]
    assert "table_lookup" in result["readiness"]["missing_categories"]
    assert result["readiness"]["missing_difficulties"] == ["easy", "medium", "hard"]


def test_review_manifest_rejects_unproven_human_promotion():
    manifest = build_review_manifest("v4")
    manifest[0]["review_status"] = "human_reviewed"
    errors = validate_review_manifest(manifest)
    assert "q_0001: human_reviewed requires human_curated or llm_silver_reviewed source_type" in errors
    assert "q_0001: human_reviewed requires reviewer" in errors
    assert "q_0001: human_reviewed requires reviewed_at" in errors
    assert "q_0001: human_reviewed requires review_decision" in errors


def test_review_manifest_rejects_invalid_language_and_decision():
    manifest = build_review_manifest("v4")
    manifest[0]["language"] = "de"
    manifest[0]["review_status"] = "human_reviewed"
    manifest[0]["source_type"] = "llm_silver_reviewed"
    manifest[0]["reviewer"] = "reviewer@example.test"
    manifest[0]["reviewed_at"] = "2026-09-06T19:15:00+01:00"
    manifest[0]["review_decision"] = "maybe"

    errors = validate_review_manifest(manifest)

    assert "q_0001: language must be one of en, fr, ar, or null" in errors
    assert "q_0001: human_reviewed requires review_decision approved or rejected" in errors


def test_review_manifest_rejects_changed_provenance():
    manifest = build_review_manifest("v4")
    manifest[0]["provenance"]["source_text_excerpt"] += " changed"

    assert "q_0001: provenance_fingerprint does not match immutable source fields" in (
        validate_review_manifest(manifest)
    )


def test_reviewer_packet_import_rejects_changed_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setattr(
        review_manifest_module,
        "dataset_path",
        lambda version, filename: tmp_path / version / filename,
    )
    manifest = build_review_manifest("v4")[:1]
    manifest_path = tmp_path / "v4" / "review_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest[0], ensure_ascii=False) + "\n", encoding="utf-8")
    worksheet = save_reviewer_worksheet("v4")
    text = worksheet.read_text(encoding="utf-8")
    worksheet.write_text(
        text.replace(manifest[0]["provenance_fingerprint"], "tampered"),
        encoding="utf-8",
    )

    try:
        import_reviewer_worksheet("v4")
    except ValueError as exc:
        assert "provenance_fingerprint does not match manifest" in str(exc)
    else:
        raise AssertionError("tampered fingerprint must be rejected")


def test_rejected_review_requires_reason():
    manifest = build_review_manifest("v4")
    manifest[0].update(
        {
            "review_status": "human_reviewed",
            "source_type": "llm_silver_reviewed",
            "reviewer": "reviewer@example.test",
            "reviewed_at": "2026-09-06T19:15:00+01:00",
            "review_decision": "rejected",
        }
    )

    assert "q_0001: rejected entries require review_notes" in validate_review_manifest(manifest)


def test_human_review_requires_iso8601_timestamp():
    manifest = build_review_manifest("v4")
    manifest[0].update(
        {
            "review_status": "human_reviewed",
            "source_type": "llm_silver_reviewed",
            "reviewer": "reviewer@example.test",
            "reviewed_at": "not-a-timestamp",
            "review_decision": "approved",
        }
    )

    assert "q_0001: reviewed_at must be ISO-8601" in validate_review_manifest(manifest)


def test_review_manifest_rejects_unknown_category_and_difficulty():
    manifest = build_review_manifest("v4")
    manifest[0]["category"] = "table-loookup"
    manifest[0]["difficulty"] = "hardd"

    errors = validate_review_manifest(manifest)

    assert "q_0001: category must be a valid evaluation category" in errors
    assert "q_0001: difficulty must be easy, medium, or hard" in errors


def test_certification_readiness_fails_on_manifest_validation_errors():
    manifest = build_review_manifest("v4")
    manifest[0]["relevant_chunk_ids"] = []

    readiness = certification_readiness(manifest)

    assert readiness["ready"] is False
    assert readiness["validation_errors"] == [
        "q_0001: missing relevant_chunk_ids",
        "q_0001: provenance_fingerprint does not match immutable source fields",
    ]
    assert "manifest validation failed" in readiness["reasons"][0]


def test_review_manifest_requires_explicit_human_review_for_certification():
    manifest = build_review_manifest("v4")
    errors = validate_review_manifest(manifest, require_reviewed=True)
    assert errors[0] == "q_0001: entry is not human_reviewed"


def test_reviewed_dataset_export_fails_closed_for_pending_manifest():
    try:
        export_reviewed_dataset("v4")
    except ValueError as exc:
        assert "Cannot export reviewed dataset" in str(exc)
    else:
        raise AssertionError("pending manifest must not be exported")


def test_reviewed_dataset_export_writes_only_reviewed_entries(tmp_path, monkeypatch):
    entries = build_review_manifest("v4")[:1]
    monkeypatch.setattr(
        review_manifest_module,
        "dataset_path",
        lambda version, filename: tmp_path / version / filename,
    )
    monkeypatch.setattr(
        dataset_module,
        "default_eval_root",
        lambda: tmp_path,
    )
    entries[0]["review_status"] = "human_reviewed"
    entries[0]["source_type"] = "llm_silver_reviewed"
    entries[0]["reviewer"] = "reviewer@example.test"
    entries[0]["reviewed_at"] = "2026-09-06T19:15:00+01:00"
    entries[0]["review_decision"] = "approved"
    manifest_path = tmp_path / "v4" / "review_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(entries[0], ensure_ascii=False) + "\n")

    output = export_reviewed_dataset("v4")

    assert output == tmp_path / "v4" / "gold_dataset.jsonl"
    exported = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(exported) == 1
    assert json.loads(exported[0])["source_type"] == "llm_silver_reviewed"


def test_review_manifest_summary_reports_pending_progress():
    summary = review_manifest_summary(build_review_manifest("v4"))

    assert summary["total"] == 10
    assert summary["reviewed"] == 0
    assert summary["pending"] == 10
    assert summary["review_completion_ratio"] == 0.0
    assert summary["by_source"]["llm_silver_draft"] == 10
    assert summary["by_language"]["unverified"] == 10


def test_certification_readiness_fails_for_small_pending_manifest():
    readiness = certification_readiness(build_review_manifest("v4"))

    assert readiness["ready"] is False
    assert readiness["reviewed_question_count"] == 0
    assert any("between 30 and 50" in reason for reason in readiness["reasons"])
    assert any("10 entries remain pending" in reason for reason in readiness["reasons"])


def test_certification_readiness_reports_missing_required_languages():
    readiness = certification_readiness(
        build_review_manifest("v4"),
        required_languages=("en", "fr", "ar"),
    )

    assert readiness["missing_languages"] == ["en", "fr", "ar"]
    assert readiness["unverified_language_count"] == 10
    assert any("missing required languages" in reason for reason in readiness["reasons"])
    assert any("unverified language metadata" in reason for reason in readiness["reasons"])


def test_certification_readiness_reports_missing_required_categories():
    readiness = certification_readiness(
        build_review_manifest("v4"),
        required_categories=("table_lookup", "multi_hop_reasoning", "negative_control"),
    )

    assert readiness["missing_categories"] == [
        "table_lookup",
        "multi_hop_reasoning",
        "negative_control",
    ]
    assert any("missing required categories" in reason for reason in readiness["reasons"])


def test_certification_readiness_reports_missing_required_difficulties():
    readiness = certification_readiness(
        build_review_manifest("v4"),
        required_difficulties=("easy", "medium", "hard"),
    )

    assert readiness["missing_difficulties"] == ["easy", "medium", "hard"]
    assert any("missing required difficulties" in reason for reason in readiness["reasons"])


def test_rejected_review_is_not_certification_ready_or_exported():
    manifest = build_review_manifest("v4")[:1]
    manifest[0].update(
        {
            "review_status": "human_reviewed",
            "source_type": "llm_silver_reviewed",
            "reviewer": "reviewer@example.test",
            "reviewed_at": "2026-09-06T19:15:00+01:00",
            "review_decision": "rejected",
        }
    )

    readiness = certification_readiness(manifest, minimum_questions=1, maximum_questions=1)

    assert readiness["ready"] is False
    assert readiness["reviewed_question_count"] == 0
    assert any("1 entries were rejected" in reason for reason in readiness["reasons"])
