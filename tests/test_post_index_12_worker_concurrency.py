from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz

from rag_project.configuration.settings import Settings
from rag_project.ingestion import robust_ingestor


WORKERS = 12


def _make_pdf(path: Path, marker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    try:
        page = document.new_page()
        page.insert_text((72, 72), f"{marker} stable post-index worker regression source")
        document.save(str(path))
    finally:
        document.close()


def _settings(root: Path) -> Settings:
    return Settings(
        project_root=root,
        incoming_dir=root / "incoming",
        processed_dir=root / "processed",
        failed_dir=root / "failed",
        archive_dir=root / "archive",
        vector_db_dir=root / "vectors",
        log_dir=root / "logs",
        ingestion_db_path=root / "ingestion.sqlite3",
        embedding_test_mode=True,
        embedding_model="test",
        ocr_enabled=False,
        ingestion_lease_seconds=60,
    )


def _worker(root: Path, index: int):
    from rag_project.app.rag_system import RAGSystem

    settings = _settings(root)
    source = settings.incoming_dir / f"worker-{index:02d}.pdf"
    _make_pdf(source, f"worker {index}")
    system = RAGSystem(settings)
    return robust_ingestor.robust_ingest_file(system, source), system


def test_twelve_workers_complete_real_post_index_publication_in_parallel(tmp_path):
    roots = [tmp_path / f"worker-root-{index:02d}" for index in range(WORKERS)]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        outcomes = list(pool.map(lambda pair: _worker(*pair), zip(roots, range(WORKERS))))

    assert len(outcomes) == WORKERS
    for result, system in outcomes:
        assert str(result["status"]).lower() in {"success", "ready", "completed"}, result
        row = system.state_store.get_document(result["document_id"])
        assert row is not None
        assert row["status"] == "READY"
        assert row["current_stage"] == "READY"
        assert row["index_state"] == "READY"
        assert not row["lease_owner"]
        assert Path(row["file_path"]).is_file()
        validation = system.vector_store.validate_document_index(
            result["document_id"], row["version_id"]
        )
        assert validation["valid"] is True, validation
        assert validation["semantic_count"] == validation["lexical_count"] == result["embedding_count"]
