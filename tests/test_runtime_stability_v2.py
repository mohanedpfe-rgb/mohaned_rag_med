from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from rag_project.runtime_stability_v2 import _guarded_extract_tables


def test_table_extraction_is_skipped_without_table_signal() -> None:
    calls: list[str] = []

    class Page:
        def get_text(self, mode: str) -> str:
            return "A normal medical paragraph with no tabular cue."

        def get_images(self, full: bool = True):
            return []

        def find_tables(self):
            calls.append("called")
            return SimpleNamespace(tables=[])

    assert _guarded_extract_tables(Page()) == ""
    assert calls == []


def test_table_extraction_runs_when_signal_exists() -> None:
    calls: list[str] = []

    class Page:
        def get_text(self, mode: str) -> str:
            return "Table 1: laboratory results"

        def get_images(self, full: bool = True):
            return []

        def find_tables(self):
            calls.append("called")
            row = ["Test", "42"]
            return SimpleNamespace(tables=[SimpleNamespace(extract=lambda: [row])])

    result = _guarded_extract_tables(Page())
    assert "Test | 42" in result
    assert calls == ["called"]
