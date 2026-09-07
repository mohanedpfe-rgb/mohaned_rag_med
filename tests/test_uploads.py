from pathlib import Path

from rag_project.app.dev_ui import _safe_upload_path, _save_uploaded_pdf


def test_safe_upload_path_handles_unicode_windows_filename(tmp_path: Path):
    content = b"%PDF-1.7\n"
    original = "1-iKB Endocrinologie, Diab_\u0631tologie -nutrition 9e _\u0631d.pdf"

    target = _save_uploaded_pdf(tmp_path, original, content)

    assert target.exists()
    assert target.suffix == ".pdf"
    assert target.name.isascii()
    assert target.read_bytes() == content


def test_safe_upload_path_is_stable_for_duplicate_uploads(tmp_path: Path):
    content = b"same pdf"
    first = _safe_upload_path(tmp_path, "report.pdf", content)
    second = _safe_upload_path(tmp_path, "report.pdf", content)

    assert first == second
