from __future__ import annotations

from pathlib import Path


def test_repository_has_single_application_composition_root() -> None:
    root = Path(__file__).resolve().parents[1]
    application = root / "rag_project" / "application.py"
    runtime = root / "rag_project" / "runtime.py"
    architecture = root / "ARCHITECTURE.md"

    assert application.is_file()
    assert runtime.is_file()
    assert architecture.is_file()
    source = application.read_text(encoding="utf-8")
    assert "def create_rag_system" in source
    assert "install()" in source


def test_source_control_policy_keeps_binary_and_local_state_out() -> None:
    root = Path(__file__).resolve().parents[1]
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert "__pycache__/" in gitignore
    assert "*.py[cod]" in gitignore
    assert "data/vector_db/" in gitignore
    assert "!requirements.lock" in gitignore
