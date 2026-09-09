from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from rag_project.ingestion import responsive_supervisor as supervisor


class _StateStore:
    def get_by_path(self, _path: str):
        return None

    def recover_stale_documents(self):
        return 0


class _System:
    def __init__(self, incoming: Path):
        self.settings = SimpleNamespace(incoming_dir=incoming, log_dir=incoming)
        self.state_store = _StateStore()
        self.started = threading.Event()
        self.finished = threading.Event()

    def _hash_file(self, path: Path):
        return str(path.stat().st_size)

    def ingest_file(self, _path: Path):
        self.started.set()
        time.sleep(0.35)
        self.finished.set()
        return {"status": "success"}


def test_scan_dispatches_ingestion_without_blocking(tmp_path, monkeypatch):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    pdf = incoming / "book.pdf"
    pdf.write_bytes(b"%PDF-test")
    system = _System(incoming)

    # Bypass the 0.75s stable-write wait so the test isolates dispatch behavior.
    monkeypatch.setattr(supervisor, "_stable_enough", lambda *_args: True)
    supervisor._WORKING.clear()

    started = time.perf_counter()
    supervisor._scan_once(system)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.15
    assert system.started.wait(timeout=0.5)
    assert system.finished.wait(timeout=1.0)
    supervisor._WORKING.clear()


def test_supervisor_state_exposes_in_flight_worker(tmp_path):
    supervisor._WORKING.clear()
    key = str((tmp_path / "book.pdf").resolve())
    supervisor._WORKING.add(key)
    try:
        state = supervisor.snapshot()
        assert state["in_flight"] == 1
    finally:
        supervisor._WORKING.clear()
