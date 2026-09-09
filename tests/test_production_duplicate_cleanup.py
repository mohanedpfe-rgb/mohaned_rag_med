from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from rag_project.app.production_rag import ProductionRAGSystem


class FakeState:
    pass


def test_duplicate_upload_is_archived_after_successful_skip(tmp_path: Path, monkeypatch):
    incoming = tmp_path / "incoming"
    archive = tmp_path / "archive"
    incoming.mkdir()
    duplicate = incoming / "study.pdf"
    duplicate.write_bytes(b"%PDF-test")

    system = object.__new__(ProductionRAGSystem)
    system.settings = SimpleNamespace(archive_dir=archive)
    monkeypatch.setattr(
        "rag_project.app.production_rag.robust_ingestor.robust_ingest_file",
        lambda _system, _path: {"status": "skipped", "document_id": "doc-123"},
    )
    monkeypatch.setattr(
        system,
        "_hash_file",
        lambda _path: "a" * 64,
        raising=False,
    )

    result = system.ingest_file(duplicate)

    assert result["status"] == "skipped"
    assert "archived_duplicate" in result
    assert not duplicate.exists()
    archived = Path(result["archived_duplicate"])
    assert archived.exists()


def test_duplicate_archive_failure_does_not_turn_skip_into_failure(tmp_path: Path, monkeypatch):
    incoming = tmp_path / "incoming"
    archive = tmp_path / "archive"
    incoming.mkdir()
    duplicate = incoming / "study.pdf"
    duplicate.write_bytes(b"%PDF-test")

    system = object.__new__(ProductionRAGSystem)
    system.settings = SimpleNamespace(archive_dir=archive)
    monkeypatch.setattr(
        "rag_project.app.production_rag.robust_ingestor.robust_ingest_file",
        lambda _system, _path: {"status": "skipped", "document_id": "doc-123"},
    )
    monkeypatch.setattr(system, "_hash_file", lambda _path: "b" * 64, raising=False)
    monkeypatch.setattr(Path, "replace", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")))

    result = system.ingest_file(duplicate)

    assert result["status"] == "skipped"
    assert "archive_warning" in result
    assert "disk full" not in result["archive_warning"]
