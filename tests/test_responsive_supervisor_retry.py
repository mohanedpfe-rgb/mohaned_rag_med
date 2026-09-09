from __future__ import annotations

import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

from rag_project.ingestion import responsive_supervisor


class FakeStore:
    def __init__(self, document):
        self.document = document

    def get_by_path(self, _path: str):
        return self.document


class FakeSystem:
    def __init__(self, root: Path, document):
        self.settings = SimpleNamespace(incoming_dir=root / "incoming", log_dir=root / "logs")
        self.state_store = FakeStore(document)

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def test_failed_same_content_is_retryable_after_backoff(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    path = incoming / "study.pdf"
    payload = b"%PDF-failed"
    path.write_bytes(payload)
    signature = responsive_supervisor._file_signature(path)
    responsive_supervisor._STABILITY[str(path.resolve())] = (signature, time.monotonic() - 2.0)
    key = str(path.resolve())
    responsive_supervisor._RETRY_ATTEMPTS[key] = 1
    responsive_supervisor._RETRY_NOT_BEFORE[key] = time.monotonic() - 0.1

    document = {
        "status": "FAILED_EMBEDDING",
        "content_hash": hashlib.sha256(payload).hexdigest(),
    }
    system = FakeSystem(tmp_path, document)

    assert responsive_supervisor._candidate(system, path) is True


def test_failed_same_content_stops_after_retry_budget(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    path = incoming / "study.pdf"
    payload = b"%PDF-failed"
    path.write_bytes(payload)
    signature = responsive_supervisor._file_signature(path)
    responsive_supervisor._STABILITY[str(path.resolve())] = (signature, time.monotonic() - 2.0)
    key = str(path.resolve())
    responsive_supervisor._RETRY_ATTEMPTS[key] = responsive_supervisor._MAX_AUTO_RETRIES
    responsive_supervisor._RETRY_NOT_BEFORE[key] = time.monotonic() - 0.1

    document = {
        "status": "FAILED_INDEXING",
        "content_hash": hashlib.sha256(payload).hexdigest(),
    }
    system = FakeSystem(tmp_path, document)

    assert responsive_supervisor._candidate(system, path) is False


def test_content_change_resets_failed_retry_budget(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    path = incoming / "study.pdf"
    path.write_bytes(b"%PDF-new")
    signature = responsive_supervisor._file_signature(path)
    responsive_supervisor._STABILITY[str(path.resolve())] = (signature, time.monotonic() - 2.0)
    key = str(path.resolve())
    responsive_supervisor._RETRY_ATTEMPTS[key] = responsive_supervisor._MAX_AUTO_RETRIES
    responsive_supervisor._RETRY_NOT_BEFORE[key] = time.monotonic() + 60

    document = {
        "status": "FAILED_EMBEDDING",
        "content_hash": hashlib.sha256(b"%PDF-old").hexdigest(),
    }
    system = FakeSystem(tmp_path, document)

    assert responsive_supervisor._candidate(system, path) is True
    assert key not in responsive_supervisor._RETRY_ATTEMPTS
    assert key not in responsive_supervisor._RETRY_NOT_BEFORE
