from __future__ import annotations

import ast
from pathlib import Path

from rag_project import composition
from rag_project.canonical_runtime import ANSWER_AUTHORITY, CANONICAL_SERVICE

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
COMPOSITION = ROOT / "rag_project" / "composition.py"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_app_is_a_thin_presentation_entrypoint():
    source = APP.read_text(encoding="utf-8")
    assert "prepare_runtime(" in source
    assert "install_pipeline_integrity" not in source
    assert "install_production_contract" not in source
    assert "install_ingestion_contract" not in source
    assert "install_canonical_runtime" not in source


def test_composition_module_does_not_depend_on_ui():
    imports = _imports(COMPOSITION)
    assert not any(name == "streamlit" or name.startswith("streamlit.") for name in imports)
    assert not any(name.startswith("rag_project.app") for name in imports)


def test_composition_boundary_exports_stable_contract():
    assert composition.RUNTIME_COMPOSITION_VERSION.startswith("2026-09-13-")
    assert callable(composition.prepare_runtime)
    assert callable(composition.install_production_contracts)
    assert callable(composition.normalize_runtime_environment)


def test_canonical_runtime_remains_single_authority():
    source = (ROOT / "rag_project" / "canonical_runtime.py").read_text(encoding="utf-8")
    assert ANSWER_AUTHORITY in source
    assert CANONICAL_SERVICE in source
    assert '"monkey_patch": False' in source


def test_runtime_environment_clamps_to_supported_bounds(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "999")
    monkeypatch.setenv("EMBEDDING_RETRIES", "0")
    monkeypatch.setenv("EMBEDDING_TIMEOUT_SECONDS", "2")
    result = composition.normalize_runtime_environment()
    assert result == {
        "EMBEDDING_BATCH_SIZE": 32,
        "EMBEDDING_RETRIES": 1,
        "EMBEDDING_TIMEOUT_SECONDS": 30.0,
    }


def test_architecture_document_names_the_composition_boundary():
    architecture = (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "rag_project.composition.prepare_runtime" in architecture
    assert "app.py" in architecture
    assert "presentation entrypoint" in architecture.lower()
