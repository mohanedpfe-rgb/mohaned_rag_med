"""Persistent, bounded replay traces for diagnosing real retrieval failures."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def trace_path(root: str | Path) -> Path:
    path = Path(root).expanduser().resolve() / "logs" / "retrieval_replay.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def record(root: str | Path, payload: dict[str, Any], *, max_bytes: int = 10_000_000) -> Path | None:
    """Append a sanitized trace; rotate instead of allowing logs to grow forever."""
    try:
        path = trace_path(root)
        item = dict(payload)
        item.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        text = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        if path.exists() and path.stat().st_size + len(text.encode("utf-8")) > max_bytes:
            rotated = path.with_suffix(".jsonl.1")
            if rotated.exists():
                rotated.unlink()
            path.replace(rotated)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
        return path
    except OSError:
        return None


def load(root: str | Path, *, limit: int = 500) -> list[dict[str, Any]]:
    path = trace_path(root)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in list(handle)[-max(1, int(limit)):]:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError:
        return []
    return rows


def failure_summary(root: str | Path, *, limit: int = 500) -> dict[str, Any]:
    rows = load(root, limit=limit)
    failures = [row for row in rows if not bool(row.get("success", False))]
    stages = Counter(str(row.get("failure_stage", "unknown")) for row in failures)
    routes = Counter(str((row.get("routing") or {}).get("kind", "unknown")) for row in failures)
    return {"trace_count": len(rows), "failure_count": len(failures), "failure_rate": len(failures) / max(1, len(rows)), "by_stage": dict(stages), "by_route": dict(routes)}


__all__ = ["trace_path", "record", "load", "failure_summary"]
