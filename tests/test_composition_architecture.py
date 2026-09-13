from __future__ import annotations

import ast
from pathlib import Path

from rag_project import composition
from rag_project.canonical_runtime import ANSWER_AUTHORITY, CANONICAL_SERVICE
from scripts.architecture_gate import inspect as inspect_architecture_gate

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
COMPOSITION = ROOT / "rag_project" / "composition.py"
UI_SECURITY = ROOT / "rag_project" / "app" / "ui_security_boundary.py"
APPLICATION = ROOT / "rag_project" / "application.py"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_app_is_a_thin_orchestration_entrypoint():
    source = APP.read_text(encoding="utf-8")
    assert "prepare_runtime(" in source
    assert "install_ui_security(" in source
    assert "validate_storage_path" not in source
    assert "validate_pdf_payload" not in source
    assert "render_intelligence_panel" not in source
    assert len(source.splitlines()) <= 55


def test_composition_module_does_not_depend_on_ui():
    imports = _imports(COMPOSITION)
    assert not any(name == "streamlit" or name.startswith("streamlit.") for name in imports)
    assert not any(name.startswith("rag_project.app") for name in imports)


def test_application_uses_composition_for_contract_ordering():
    imports = _imports(APPLICATION)
    assert "rag_project.composition" in imports
    assert "rag_project.intelligence.pipeline_integrity" not in imports
    assert "rag_project.intelligence.production_contract_v2" not in imports
    assert "rag_project.ingestion.ingestion_contract" not in imports
    assert "rag_project.canonical_runtime" in imports


def test_ui_security_boundary_owns_presentation_security_capture():
    source = UI_SECURITY.read_text(encoding="utf-8")
    assert "validate_storage_path" in source
    assert "validate_pdf_payload" in source
    assert "register_session_upload" in source
    assert "render_intelligence_panel" in source
    assert "def install()" in source


def test_composition_boundary_exports_stable_contract():
    assert composition.RUNTIME_COMPOSITION_VERSION.startswith("2026-09-13-")
    assert composition.RUNTIME_PREPARED_ENV == "BOOKRAG_RUNTIME_PREPARED_VERSION"
    assert callable(composition.prepare_runtime)
    assert callable(composition.install_production_contracts)
    assert callable(composition.normalize_runtime_environment)
    assert callable(composition.runtime_is_prepared)


def test_runtime_prepared_marker_tracks_exact_composition_version(monkeypatch):
    monkeypatch.delenv(composition.RUNTIME_PREPARED_ENV, raising=False)
    assert composition.runtime_is_prepared() is False
    monkeypatch.setenv(composition.RUNTIME_PREPARED_ENV, "wrong-version")
    assert composition.runtime_is_prepared() is False
    monkeypatch.setenv(composition.RUNTIME_PREPARED_ENV, composition.RUNTIME_COMPOSITION_VERSION)
    assert composition.runtime_is_prepared() is True


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


def test_executable_architecture_gate_is_clean():
    report = inspect_architecture_gate()
    assert report["ready"], report
    assert report["violations"] == []


def test_architecture_document_names_the_composition_boundary_and_gate():
    architecture = (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "rag_project.composition.prepare_runtime" in architecture
    assert "rag_project.app.ui_security_boundary" in architecture
    assert "scripts/architecture_gate.py" in architecture
    assert "app.py" in architecture
    assert "presentation entrypoint" in architecture.lower()
