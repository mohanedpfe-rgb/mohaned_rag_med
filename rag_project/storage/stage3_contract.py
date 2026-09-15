"""Stage 3 storage invariants used by lifecycle/concurrency verification."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def index_lock_path(store: Any) -> Path:
    """Return the single lock path used for a persistent index."""
    return Path(store.persist_directory) / ".semantic_lexical_index.lock"
