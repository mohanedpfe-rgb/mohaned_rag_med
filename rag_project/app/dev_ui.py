"""Small compatibility helpers for legacy upload-focused tests."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path


def _ascii_stem(name: str) -> str:
    stem = unicodedata.normalize("NFKD", Path(name).stem).encode("ascii", "ignore").decode("ascii")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "upload"
    return stem[:80]


def _safe_upload_path(directory: str | Path, filename: str, content: bytes) -> Path:
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(bytes(content)).hexdigest()[:16]
    return root / f"{_ascii_stem(filename)}_{digest}.pdf"


def _save_uploaded_pdf(directory: str | Path, filename: str, content: bytes) -> Path:
    target = _safe_upload_path(directory, filename, content)
    target.write_bytes(bytes(content))
    return target


__all__ = ["_safe_upload_path", "_save_uploaded_pdf"]
