from __future__ import annotations

import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

from rag_project.ingestion import auto_supervisor


class FakeStore:
    def __init__(self, document=None):
        self.document = document
        self.recovered = 0

    def get_by_path(self, file_path: str):
        return self.document

    def recover_stale_documents(self):
        value = self.recovered
        self.recovered = 0
        return value


class FakeSystem:
    def __init__(self, root: Path, document=None):
        self.settings = SimpleNamespace(
            incoming_dir=root / "incoming",
            log_dir=root / "logs",
        )
        self.state_store = FakeStore(document)
        self.ingested: list[Path] = []

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def ingest_file(self, path: Path):
        self.ingested.append(path)
        return {"status": "success", "file": path.name}


def _mark_stable(path: Path) -> None:
    signature = auto_supervisor._file_signature(path)
    auto_supervisor._STABILITY[str(path.resolve())] = (signature, time.monotonic() - 2.0)


def test_new_pdf_is_an_automatic_candidate(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    path = incoming / "study.pdf"
    path.write_bytes(b"%PDF-fake")
    _mark_stable(path)
    system = FakeSystem(tmp_path)

    assert auto_supervisor._candidate(system, path) is True


def test_completed_same_content_is_not_reprocessed(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    path = incoming / "study.pdf"
    payload = b"%PDF-fake"
    path.write_bytes(payload)
    _mark_stable(path)
    document = {
        "status": "READY",
        "content_hash": hashlib.sha256(payload).hexdigest(),
    }
    system = FakeSystem(tmp_path, document)

    assert auto_supervisor._candidate(system, path) is False


def test_changed_content_reopens_terminal_record(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    path = incoming / "study.pdf"
    path.write_bytes(b"%PDF-new-revision")
    _mark_stable(path)
    old = b"%PDF-old-revision"
    document = {
        "status": "READY",
        "content_hash": hashlib.sha256(old).hexdigest(),
    }
    system = FakeSystem(tmp_path, document)

    assert auto_supervisor._candidate(system, path) is True


def test_unstable_pdf_is_not_ingested_mid_copy(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    path = incoming / "study.pdf"
    path.write_bytes(b"%PDF-partial")
    system = FakeSystem(tmp_path)

    assert auto_supervisor._candidate(system, path) is False
    assert system.ingested == []


def test_scan_once_calls_canonical_ingestion_without_ui_job(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    path = incoming / "study.pdf"
    path.write_bytes(b"%PDF-fake")
    _mark_stable(path)
    system = FakeSystem(tmp_path)

    auto_supervisor._scan_once(system)

    assert system.ingested == [path]
    snapshot = auto_supervisor.snapshot(system)
    assert snapshot["auto_started"] >= 1
    assert snapshot["completed"] >= 1


def test_recovered_jobs_are_counted(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    system = FakeSystem(tmp_path)
    system.state_store.recovered = 2

    auto_supervisor._scan_once(system)

    assert auto_supervisor.snapshot(system)["recovered"] >= 2
