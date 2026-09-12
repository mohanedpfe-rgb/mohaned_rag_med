from rag_project.app.ingestion_job_contract import summarize_ingestion_results


def test_ready_results_count_as_completed():
    completed, failed, status = summarize_ingestion_results(
        [{"status": "READY"}, {"status": "COMPLETED"}, {"status": "success"}, {"status": "skipped"}]
    )

    assert (completed, failed, status) == (4, 0, "COMPLETED")


def test_failed_results_are_reported_when_no_success_exists():
    completed, failed, status = summarize_ingestion_results(
        [{"status": "FAILED"}, {"status": "failed_indexing"}]
    )

    assert (completed, failed, status) == (0, 2, "FAILED")


def test_mixed_success_and_failure_is_not_reported_as_fully_successful():
    completed, failed, status = summarize_ingestion_results(
        [{"status": "READY"}, {"status": "FAILED"}]
    )

    assert (completed, failed, status) == (1, 1, "COMPLETED_WITH_FAILURES")


def test_unknown_status_is_not_silently_marked_successful():
    completed, failed, status = summarize_ingestion_results([{"status": "RUNNING"}])

    assert (completed, failed, status) == (0, 0, "UNKNOWN")
