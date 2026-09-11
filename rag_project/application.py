from __future__ import annotations

from typing import Any

from rag_project.canonical_runtime import build_system
from rag_project.generation.answer_engine import AnswerEngine


class RAGApplication:
    def __init__(self, system: Any | None = None):
        self.system = system or build_system()
        self.answer_engine = AnswerEngine(self.system)

    def ingest(self, pdf_path: str) -> dict[str, Any]:
        return self.system.ingest_file(pdf_path)

    def answer(self, question: str, **kwargs: Any) -> dict[str, Any]:
        return self.answer_engine.answer(question, **kwargs)


__all__ = ["RAGApplication"]
