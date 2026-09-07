from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Observation:
    timestamp: str
    document_id: str
    file_name: str
    status: str
    stage: str
    current_page: int
    total_pages: int
    details: dict[str, Any] = field(default_factory=dict)


class IngestionObserver:
    """Poll the durable ingestion state without touching Streamlit session state."""

    def __init__(self, state_store: Any, interval_seconds: float = 0.75, max_events: int = 5000):
        self.state_store = state_store
        self.interval_seconds = max(0.25, float(interval_seconds))
        self.max_events = max(100, int(max_events))
        self._lock = threading.Lock()
        self._events: list[Observation] = []
        self._last_snapshot: dict[str, tuple[Any, ...]] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="rag-ingestion-observer", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.poll_once()
            self._stop.wait(self.interval_seconds)

    def poll_once(self) -> None:
        try:
            documents = self.state_store.get_all_documents()
        except Exception:
            return
        for document in documents:
            document_id = str(document.get("document_id") or "")
            snapshot = (
                document.get("status"),
                document.get("current_stage"),
                document.get("current_page"),
                document.get("total_pages"),
                document.get("error"),
                document.get("embedding_dimension"),
                document.get("version_id"),
            )
            with self._lock:
                previous = self._last_snapshot.get(document_id)
                self._last_snapshot[document_id] = snapshot
                if previous == snapshot:
                    continue
                self._events.append(
                    Observation(
                        timestamp=self._now(),
                        document_id=document_id,
                        file_name=str(document.get("file_name") or document_id),
                        status=str(document.get("status") or "UNKNOWN"),
                        stage=str(document.get("current_stage") or "UNKNOWN"),
                        current_page=int(document.get("current_page") or 0),
                        total_pages=int(document.get("total_pages") or 0),
                        details={
                            "error": document.get("error"),
                            "embedding_dimension": document.get("embedding_dimension"),
                            "version_id": document.get("version_id"),
                        },
                    )
                )
                if len(self._events) > self.max_events:
                    self._events = self._events[-self.max_events:]

    def events(self, document_id: str | None = None, limit: int = 250) -> list[Observation]:
        with self._lock:
            selected = [event for event in self._events if not document_id or event.document_id == document_id]
            return list(selected[-max(1, int(limit)):])

    def latest_by_document(self) -> dict[str, Observation]:
        with self._lock:
            result: dict[str, Observation] = {}
            for event in self._events:
                result[event.document_id] = event
            return dict(result)
