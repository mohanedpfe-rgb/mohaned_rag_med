from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List


class FileMonitor:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def scan(self) -> List[Dict[str, str | bool]]:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        results: List[Dict[str, str | bool]] = []
        marker_file = self.base_dir / ".file_index.json"
        existing: dict[str, str] = {}
        if marker_file.exists():
            try:
                loaded = json.loads(marker_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = {str(key): str(value) for key, value in loaded.items()}
            except json.JSONDecodeError:
                existing = {}
        for file_path in sorted(self.base_dir.glob("**/*.*")):
            if file_path.suffix.lower() != ".pdf":
                continue
            digest = self._hash_file(file_path)
            status = "new"
            key = str(file_path.resolve())
            previous = existing.get(key)
            if previous == digest:
                status = "unchanged"
            elif previous is not None:
                status = "modified"
            results.append({"path": str(file_path), "name": file_path.name, "hash": digest, "status": status})

        duplicate_hashes = {}
        for item in results:
            duplicate_hashes.setdefault(item["hash"], []).append(item)
        for group in duplicate_hashes.values():
            if len(group) > 1:
                for item in group:
                    item["duplicate"] = True
        marker_file.write_text(
            json.dumps({str(Path(item["path"]).resolve()): item["hash"] for item in results}, indent=2),
            encoding="utf-8",
        )
        return results

    def detect_duplicates(self, pdf_paths: List[str | Path]) -> List[List[str]]:
        grouped: Dict[str, List[str]] = {}
        for pdf_path in pdf_paths:
            digest = self._hash_file(Path(pdf_path))
            grouped.setdefault(digest, []).append(str(pdf_path))
        return [paths for paths in grouped.values() if len(paths) > 1]
