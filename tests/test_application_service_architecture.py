from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPLICATION = ROOT / "rag_project" / "application.py"
ANSWER_SERVICE = ROOT / "rag_project" / "application_answer_service.py"
CANONICAL = ROOT / "rag_project" / "canonical_runtime.py"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_application_service_is_composition_focused():
    source = APPLICATION.read_text(encoding="utf-8")
    imports = _imports(APPLICATION)
    assert "rag_project.composition" not in imports
    assert "rag_project.application_answer_service" in imports
    assert "def create_rag_system(" in source
    assert "def runtime_contract(" in source
    assert "class MedEvidenceProductionRAGSystem" in source
    assert "_certified_god_answer = staticmethod(_med_evidence_answer)" in source


def test_answer_service_is_presentation_independent():
    imports = _imports(ANSWER_SERVICE)
    assert "streamlit" not in imports
    assert not any(name.startswith("rag_project.app") for name in imports)
    source = ANSWER_SERVICE.read_text(encoding="utf-8")
    for symbol in (
        "def answer(",
        "def detect_answer_language(",
        "def normalize_public_answer_path(",
        "def install_runtime_adapters(",
    ):
        assert symbol in source


def test_canonical_runtime_keeps_single_callable_identity_seam():
    canonical = CANONICAL.read_text(encoding="utf-8")
    application = APPLICATION.read_text(encoding="utf-8")
    assert "getattr(application, \"_med_evidence_answer\", None)" in canonical
    assert "callable(certified) and callable(expected) and certified is expected" in canonical
    assert "_certified_god_answer = staticmethod(_med_evidence_answer)" in application
