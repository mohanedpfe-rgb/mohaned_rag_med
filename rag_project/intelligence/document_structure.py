from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


STRUCTURE_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class HeadingCandidate:
    title: str
    level: int
    line_index: int
    numbering: str = ""


@dataclass(frozen=True)
class StructureSnapshot:
    document_id: str
    page_number: int
    chapter: str | None
    chapter_id: str | None
    section: str | None
    section_id: str | None
    parent_id: str
    hierarchy_path: tuple[str, ...]
    headings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "structure_schema_version": STRUCTURE_SCHEMA_VERSION,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "chapter": self.chapter,
            "chapter_id": self.chapter_id,
            "section": self.section,
            "section_id": self.section_id,
            "parent_id": self.parent_id,
            "hierarchy_path": list(self.hierarchy_path),
            "headings": list(self.headings),
        }


def normalize_heading(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).strip(" -—:;")


def stable_id(*parts: object, prefix: str = "id") -> str:
    payload = "|".join(str(part or "").strip().casefold() for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _numbered_heading(line: str) -> tuple[int, str] | None:
    match = re.match(r"^\s*((?:\d+(?:\.\d+)*)|(?:[IVXLC]+))[.)]?\s+(.+?)\s*$", line, re.I)
    if not match:
        return None
    numbering = match.group(1)
    title = normalize_heading(match.group(2))
    depth = numbering.count(".") + 1 if numbering[0].isdigit() else 1
    return max(1, min(depth, 6)), title


def extract_heading_candidates(text: str) -> list[HeadingCandidate]:
    candidates: list[HeadingCandidate] = []
    lines = [line.strip() for line in str(text or "").splitlines()]
    for index, raw in enumerate(lines):
        if not raw:
            continue
        markdown = re.match(r"^(#{1,6})\s+(.+?)\s*$", raw)
        if markdown:
            candidates.append(HeadingCandidate(normalize_heading(markdown.group(2)), len(markdown.group(1)), index))
            continue
        numbered = _numbered_heading(raw)
        if numbered:
            level, title = numbered
            numbering_match = re.match(r"^\s*((?:\d+(?:\.\d+)*)|(?:[IVXLC]+))", raw, re.I)
            numbering = numbering_match.group(1) if numbering_match else ""
            candidates.append(HeadingCandidate(title, level, index, numbering))
            continue
        chapter_match = re.match(r"^\s*(chapter|chapitre|part|partie|section|section)\s+([\w.-]+)\s*[:.-]?\s*(.*?)\s*$", raw, re.I)
        if chapter_match and len(raw) <= 180:
            kind = chapter_match.group(1).casefold()
            level = 1 if kind in {"chapter", "chapitre", "part", "partie"} else 2
            suffix = normalize_heading(chapter_match.group(3))
            title = normalize_heading(" ".join(x for x in (chapter_match.group(2), suffix) if x))
            candidates.append(HeadingCandidate(title, level, index, chapter_match.group(2)))
            continue
        if len(raw) <= 140 and len(raw.split()) <= 14 and raw.upper() == raw and any(ch.isalpha() for ch in raw):
            # Conservative typography fallback for textbook headings. Avoid full-sentence lines.
            if not raw.endswith((".", ";", ",", ":")) and not re.search(r"\d{2,}", raw):
                candidates.append(HeadingCandidate(normalize_heading(raw), 2, index))
    return candidates


def split_by_headings(text: str, initial_path: tuple[str, ...]) -> list[tuple[str, tuple[str, ...], bool]]:
    lines = str(text or "").splitlines()
    candidates_by_line = {candidate.line_index: candidate for candidate in extract_heading_candidates(text)}
    path = list(initial_path)
    segments: list[tuple[list[str], tuple[str, ...], bool]] = []
    current_lines: list[str] = []
    current_path = tuple(path)
    current_is_heading = False

    def flush() -> None:
        nonlocal current_lines
        value = "\n".join(current_lines).strip()
        if value:
            segments.append((current_lines[:], current_path, current_is_heading))
        current_lines = []

    for index, line in enumerate(lines):
        candidate = candidates_by_line.get(index)
        if candidate:
            if current_lines:
                flush()
            level = max(1, min(candidate.level, 6))
            path = path[: level - 1]
            path.append(candidate.title)
            current_path = tuple(path)
            current_is_heading = True
            current_lines = [candidate.title]
            continue
        current_lines.append(line)
        current_is_heading = False
    flush()

    if not segments and str(text or "").strip():
        segments.append(([str(text).strip()], tuple(path), False))
    return [("\n".join(lines).strip(), path, is_heading) for lines, path, is_heading in segments if "\n".join(lines).strip()]


class DocumentStructureTracker:
    """Streaming document-global hierarchy tracker.

    State survives page/chunk batches, so a section spanning many pages retains
    one stable section/parent identity.
    """

    def __init__(self, document_id: str):
        self.document_id = str(document_id)
        self.path: list[str] = []

    def analyze_page(self, page_number: int, text: str) -> StructureSnapshot:
        candidates = extract_heading_candidates(text)
        for candidate in candidates:
            level = max(1, min(candidate.level, 6))
            self.path = self.path[: level - 1]
            self.path.append(candidate.title)
        return self.snapshot(page_number, tuple(c.title for c in candidates))

    def snapshot(self, page_number: int, headings: Iterable[str] = ()) -> StructureSnapshot:
        path = tuple(self.path)
        chapter = path[0] if path else None
        section = path[-1] if len(path) >= 2 else chapter
        chapter_id = stable_id(self.document_id, "chapter", chapter, prefix="chapter") if chapter else None
        section_id = stable_id(self.document_id, "section", *path, prefix="section") if section else None
        parent_id = section_id or chapter_id or stable_id(self.document_id, "document", prefix="document")
        return StructureSnapshot(
            self.document_id,
            int(page_number),
            chapter,
            chapter_id,
            section,
            section_id,
            parent_id,
            path,
            tuple(headings),
        )

    def split_page(self, page_number: int, text: str) -> list[tuple[str, StructureSnapshot, bool]]:
        # Apply headings sequentially while retaining the path between segments.
        lines = str(text or "").splitlines()
        candidates_by_line = {candidate.line_index: candidate for candidate in extract_heading_candidates(text)}
        if not candidates_by_line:
            return [(str(text or "").strip(), self.snapshot(page_number), False)] if str(text or "").strip() else []
        segments: list[tuple[str, StructureSnapshot, bool]] = []
        current: list[str] = []
        current_heading_lines: list[str] = []
        current_heading = False

        def flush() -> None:
            nonlocal current, current_heading_lines, current_heading
            value = "\n".join(current).strip()
            if value:
                snapshot = self.snapshot(page_number, tuple(current_heading_lines))
                segments.append((value, snapshot, current_heading))
            current = []
            current_heading_lines = []
            current_heading = False

        for line_index, line in enumerate(lines):
            candidate = candidates_by_line.get(line_index)
            if candidate:
                if current:
                    flush()
                level = max(1, min(candidate.level, 6))
                self.path = self.path[: level - 1]
                self.path.append(candidate.title)
                current = [candidate.title]
                current_heading_lines = [candidate.title]
                current_heading = True
            else:
                current.append(line)
        if current:
            flush()
        return segments


class DocumentStructureStore:
    """Durable document/page structure sidecar used for rebuilds and audits."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS page_structure (
                    document_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    schema_version INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(document_id, page_number)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_page_structure_document ON page_structure(document_id, page_number)"
            )

    def put_page(self, document_id: str, page_number: int, payload: dict[str, Any]) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO page_structure(document_id,page_number,schema_version,payload)
                VALUES(?,?,?,?)
                ON CONFLICT(document_id,page_number) DO UPDATE SET
                    schema_version=excluded.schema_version,
                    payload=excluded.payload,
                    created_at=CURRENT_TIMESTAMP
                """,
                (str(document_id), int(page_number), STRUCTURE_SCHEMA_VERSION, json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)),
            )

    def get_page(self, document_id: str, page_number: int) -> dict[str, Any] | None:
        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT payload FROM page_structure WHERE document_id=? AND page_number=?",
                (str(document_id), int(page_number)),
            ).fetchone()
        if not row:
            return None
        try:
            value = json.loads(row[0])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def delete_document(self, document_id: str) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("DELETE FROM page_structure WHERE document_id=?", (str(document_id),))


__all__ = [
    "STRUCTURE_SCHEMA_VERSION",
    "HeadingCandidate",
    "StructureSnapshot",
    "DocumentStructureTracker",
    "DocumentStructureStore",
    "extract_heading_candidates",
    "split_by_headings",
    "normalize_heading",
    "stable_id",
]
