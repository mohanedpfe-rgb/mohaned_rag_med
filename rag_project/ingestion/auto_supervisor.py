"""Compatibility facade for the event-driven ingestion supervisor."""
from __future__ import annotations

from watchdog.observers import Observer

from rag_project.ingestion.responsive_supervisor import (
    _acquire_process_lock,
    _stable_enough,
    _trim_caches,
    snapshot,
    start,
    stop,
)

# The underlying start() creates the watchdog Observer() only when enabled.

__all__ = ["Observer", "_acquire_process_lock", "_stable_enough", "_trim_caches", "snapshot", "start", "stop"]
