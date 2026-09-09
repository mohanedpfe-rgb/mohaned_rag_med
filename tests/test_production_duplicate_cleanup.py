from pathlib import Path
from types import SimpleNamespace

from rag_project.app.production_rag import ProductionRAGSystem


def _system(tmp_path: Path) -> ProductionRAGSystem:
    system = object.__new__(ProductionRAGSystem)
    system.settings = SimpleNamespace(
        incoming_dir=tmp_path / "incoming",
        archive_dir=tmp_path / "archive",
    )
    system.settings.incoming_dir.mkdir()
    system.settings.archive_dir.mkdir()
    return system


def test_duplicate_upload_is_archived_after_successful_skip(tmp_path: Path, monkeypatch):
    system = _system(tmp_path)
    duplicate = system.settings.incoming_dir / "study.pdf"
    duplicate.write_bytes(b"%PDF-test")
    monkeypatch.setattr(
        "rag_project.app.production_rag.robust_ingestor.robust_ingest_file",
        lambda _system, _path: {"status": "skipped", "document_id": "doc-123"},
    )
    monkeypatch.setattr(system, "_hash_file", lambda _path: "a" * 64, raising=False)

    result = system.ingest_file(duplicate)

    assert result["status"] == "skipped"
    assert result["archived_duplicate"]
    assert not duplicate.exists()
    assert Path(result["archived_duplicate"]).exists()


def test_duplicate_archive_failure_does_not_turn_skip_into_failure(tmp_path: Path, monkeypatch):
    system = _system(tmp_path)
    duplicate = system.settings.incoming_dir / "study.pdf"
    duplicate.write_bytes(b"%PDF-test")
    monkeypatch.setattr(
        "rag_project.app.production_rag.robust_ingestor.robust_ingest_file",
        lambda _system, _path: {"status": "skipped", "document_id": "doc-123"},
    )
    monkeypatch.setattr(system, "_hash_file", lambda _path: "b" * 64, raising=False)
    monkeypatch.setattr(Path, "replace", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")))

    result = system.ingest_file(duplicate)

    assert result["status"] == "skipped"
    assert "archive_warning" in result
    assert duplicate.exists()


def test_duplicate_outside_incoming_is_not_moved(tmp_path: Path, monkeypatch):
    system = _system(tmp_path)
    external = tmp_path / "external.pdf"
    external.write_bytes(b"%PDF-test")
    monkeypatch.setattr(
        "rag_project.app.production_rag.robust_ingestor.robust_ingest_file",
        lambda _system, _path: {"status": "skipped", "document_id": "doc-123"},
    )

    result = system.ingest_file(external)

    assert result["status"] == "skipped"
    assert external.exists()
    assert "archived_duplicate" not in result
