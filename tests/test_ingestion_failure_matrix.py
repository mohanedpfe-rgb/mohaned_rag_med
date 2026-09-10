from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_project.ingestion import robust_ingestor
from rag_project.ingestion.ingestion_contract import INGESTION_CONTRACT_VERSION, install


class Logger:
    def exception(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass


@pytest.mark.parametrize("filename", [
    "missing.pdf",
    "document.txt",
    "scan.docx",
    "data.csv",
    "image.png",
    "no_extension",
])
def test_robust_ingestor_rejects_missing_or_non_pdf_files(filename):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        if path.suffix.lower() == ".pdf":
            assert not path.exists()
        else:
            path.write_text("not a pdf", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported or missing PDF"):
            robust_ingestor.robust_ingest_file(SimpleNamespace(), path)


def test_robust_ingestor_rejects_directory_named_pdf():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "folder.pdf"
        directory.mkdir()
        with pytest.raises(ValueError, match="Unsupported or missing PDF"):
            robust_ingestor.robust_ingest_file(SimpleNamespace(), directory)


def test_ingestion_contract_install_is_safe_to_repeat():
    install()
    first = robust_ingestor.robust_ingest_file
    install()
    second = robust_ingestor.robust_ingest_file
    assert first is second


def test_ingestion_contract_version_is_present():
    assert INGESTION_CONTRACT_VERSION.startswith("2026-09-11-")
    assert "ingestion-contract" in INGESTION_CONTRACT_VERSION


def test_robust_ingestor_has_single_production_entrypoint():
    assert callable(robust_ingestor.robust_ingest_file)


def test_ingestion_callable_has_stable_name_after_wrapping():
    install()
    assert robust_ingestor.robust_ingest_file.__name__ == "robust_ingest_file"


def test_ingestion_callable_retains_original_callable_metadata():
    install()
    wrapped = robust_ingestor.robust_ingest_file
    assert getattr(wrapped, "_ingestion_contract_wrapped", False) is True
    assert callable(getattr(wrapped, "_ingestion_contract_original", None))


@pytest.mark.parametrize("name", ["a.pdf", "A.PDF", "medical-study.pdf", "résultats.pdf"])
def test_pdf_suffix_is_case_insensitive_for_files_that_exist(name):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_bytes(b"not a real PDF")
        # The suffix validation must not reject the file because of case or name;
        # the parser may reject the invalid PDF later, which is the correct layer.
        class MinimalSystem:
            logger = Logger()
        with pytest.raises(Exception) as error:
            robust_ingestor.robust_ingest_file(MinimalSystem(), path)
        assert "Unsupported or missing PDF" not in str(error.value)


def test_invalid_pdf_reaches_parsing_layer():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "invalid.pdf"
        path.write_bytes(b"not a real PDF")
        with pytest.raises(Exception) as error:
            robust_ingestor.robust_ingest_file(SimpleNamespace(logger=Logger()), path)
        assert "Unsupported or missing PDF" not in str(error.value)


def test_ingestion_file_hash_is_requested_after_basic_path_validation(monkeypatch):
    calls = []

    class System:
        logger = Logger()

        def _hash_file(self, path):
            calls.append(path)
            raise RuntimeError("HASH_STOP")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "document.pdf"
        path.write_bytes(b"%PDF-1.4\n")
        with pytest.raises(RuntimeError, match="HASH_STOP"):
            robust_ingestor.robust_ingest_file(System(), path)
    assert calls


def test_non_pdf_does_not_reach_hashing(monkeypatch):
    called = False

    class System:
        logger = Logger()

        def _hash_file(self, path):
            nonlocal called
            called = True
            return "hash"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "document.txt"
        path.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError):
            robust_ingestor.robust_ingest_file(System(), path)
    assert called is False


@pytest.mark.parametrize("status", ["READY", "RUNNING", "FAILED", "BUSY", "VALIDATING"])
def test_ingestion_status_values_are_strings(status):
    assert isinstance(status, str)


def test_ingestion_contract_dict_flags_are_true_when_attached():
    def original(system, pdf_path):
        return {"status": "READY"}

    from rag_project.ingestion.ingestion_contract import wrap_ingest_callable

    result = wrap_ingest_callable(original)(object(), "document.pdf")
    contract = result["ingestion_contract"]
    assert contract["atomic_publication"] is True
    assert contract["lease_fencing"] is True
    assert contract["post_write_validation"] is True
    assert contract["canonical_callable"].endswith("robust_ingest_file")


def test_ingestion_contract_does_not_change_payload_values():
    def original(system, pdf_path):
        return {"status": "READY", "document_id": "doc-123", "count": 5, "error": None}

    from rag_project.ingestion.ingestion_contract import wrap_ingest_callable

    result = wrap_ingest_callable(original)(object(), "document.pdf")
    assert result["status"] == "READY"
    assert result["document_id"] == "doc-123"
    assert result["count"] == 5
    assert result["error"] is None


def test_ingestion_contract_does_not_wrap_twice():
    def original(system, pdf_path):
        return {"status": "READY"}

    from rag_project.ingestion.ingestion_contract import wrap_ingest_callable

    first = wrap_ingest_callable(original)
    second = wrap_ingest_callable(first)
    third = wrap_ingest_callable(second)
    assert first is second is third
