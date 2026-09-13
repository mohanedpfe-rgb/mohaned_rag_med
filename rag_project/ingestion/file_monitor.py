"""Lightweight content-aware PDF directory monitor used by diagnostics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class FileMonitor:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / ".file_index.json"

    @staticmethod
    def _digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _load(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {}
        try:
            value = json.loads(self.index_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, json.JSONDecodeError):
            return {}

    def _save(self, value: dict[str, Any]) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.index_path)

    def scan(self) -> list[dict[str, Any]]:
        previous = self._load()
        current: dict[str, Any] = {}
        candidates = sorted(
            (path for path in self.directory.iterdir() if path.is_file() and path.suffix.casefold() == ".pdf" and path.name != self.index_path.name),
            key=lambda item: item.name.casefold(),
        )
        digests: dict[str, list[str]] = {}
        for path in candidates:
            digest = self._digest(path)
            key = str(path)
            digests.setdefault(digest, []).append(key)
            current[key] = {"hash": digest, "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}

        results: list[dict[str, Any]] = []
        for path in candidates:
            key = str(path)
            digest = current[key]["hash"]
            old = previous.get(key) if isinstance(previous.get(key), dict) else None
            unchanged = bool(old and old.get("hash") == digest and old.get("size") == current[key]["size"])
            duplicate = len(digests[digest]) > 1
            results.append({
                "path": key,
                "file_name": path.name,
                "hash": digest,
                "size": current[key]["size"],
                "mtime_ns": current[key]["mtime_ns"],
                "status": "unchanged" if unchanged else "new_or_changed",
                "duplicate": duplicate,
            })
        self._save(current)
        return results


__all__ = ["FileMonitor"]
