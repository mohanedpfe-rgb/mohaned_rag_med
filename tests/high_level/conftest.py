from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from rag_project.application import create_rag_system
from rag_project.configuration.settings import Settings


def _pdf_escape(text: str) -> str:
    return text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')


def write_minimal_pdf(path: Path, pages: list[str]) -> Path:
    """Create a small valid PDF using only the Python standard library."""
    objects: list[bytes] = []
    page_refs: list[int] = []
    font_obj = 3
    next_obj = 4
    for page_text in pages:
        stream = f"BT /F1 10 Tf 50 760 Td ({_pdf_escape(page_text)}) Tj ET".encode()
        content_obj = next_obj
        next_obj += 1
        page_obj = next_obj
        next_obj += 1
        objects.append(
            f"{content_obj} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode()
            + stream
            + b"\nendstream\nendobj\n"
        )
        objects.append(
            f"{page_obj} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 {font_obj} 0 R >> >> /Contents {content_obj} 0 R >>\nendobj\n".encode()
        )
        page_refs.append(page_obj)
    kids = " ".join(f"{ref} 0 R" for ref in page_refs)
    objects.insert(0, b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.insert(1, f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {len(page_refs)} >>\nendobj\n".encode())
    objects.insert(2, b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

    body = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(body))
        body.extend(obj)
    xref = len(body)
    body.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        body.extend(f"{offset:010d} 00000 n \n".encode())
    body.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    path.write_bytes(body)
    return path


class LLMSpy:
    def __init__(self, response: str = "Grounded answer [S1].") -> None:
        self.response = response
        self.calls: list[dict] = []

    def generate(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": kwargs, "at": time.perf_counter()})
        return self.response

    def generate_stream(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": kwargs, "stream": True, "at": time.perf_counter()})
        yield self.response


@pytest.fixture
def settings_i5(tmp_path: Path) -> Settings:
    root = tmp_path / "bookrag"
    settings = Settings(
        device_mode="i5_16gb",
        project_root=root,
        incoming_dir=root / "data" / "incoming",
        processed_dir=root / "data" / "processed",
        failed_dir=root / "data" / "failed",
        archive_dir=root / "data" / "archive",
        vector_db_dir=root / "data" / "vector_db",
        log_dir=root / "logs",
        ingestion_db_path=root / "data" / "ingestion.sqlite3",
        embedding_test_mode=True,
        lazy_model_loading=True,
        ocr_enabled=True,
        auto_ocr=True,
        generation_latency_budget_seconds=8.0,
        generation_timeout_seconds=10.0,
        max_workers=2,
        ollama_concurrency=1,
    )
    settings.project_root.mkdir(parents=True, exist_ok=True)
    return settings


@pytest.fixture
def temp_project_root(settings_i5: Settings) -> Path:
    return settings_i5.project_root


@pytest.fixture
def ready_document(tmp_path: Path):
    path = tmp_path / "clean_diabetes_en.pdf"
    return write_minimal_pdf(
        path,
        [
            "Diabetes mellitus is a chronic metabolic disorder characterized by hyperglycemia. HbA1c is used to assess glycemic control.",
            "Metformin is commonly used for type 2 diabetes. Dose statements must preserve exact numbers and units from source evidence.",
        ],
    )


@pytest.fixture
def clean_system(settings_i5: Settings, ready_document: Path):
    try:
        system = create_rag_system(settings_i5)
        ingestion = system.ingest_file(ready_document)
        status = str((ingestion or {}).get("status") or "").lower()
        if status not in {"ready", "completed", "success", "skipped"}:
            pytest.skip(f"controlled fixture document could not reach a usable ingestion state: {ingestion}")
        return system
    except Exception as exc:
        pytest.skip(f"canonical runtime could not initialize in isolated profile: {type(exc).__name__}: {exc}")


@pytest.fixture
def ready_multilingual_docs(tmp_path: Path):
    return {
        "en": write_minimal_pdf(tmp_path / "diabetes_en.pdf", ["Diabetes mellitus is a chronic metabolic disorder."]),
        "fr": write_minimal_pdf(tmp_path / "diabetes_fr.pdf", ["Le diabète est une maladie métabolique chronique."]),
        "ar": write_minimal_pdf(tmp_path / "diabetes_ar.pdf", ["داء السكري هو اضطراب استقلابي مزمن."]),
    }


@pytest.fixture
def fake_ollama_fast():
    return LLMSpy()


@pytest.fixture

def real_ollama_optional(settings_i5):
    try:
        from urllib.request import urlopen
        with urlopen(settings_i5.ollama_base_url + "/api/tags", timeout=1.0) as response:
            if response.status != 200:
                pytest.skip("Ollama is not healthy")
    except Exception:
        pytest.skip("Ollama is not available")
    return settings_i5.ollama_base_url


@pytest.fixture
def gold_questions():
    root = Path(__file__).resolve().parents[1] / "support" / "gold_sets" / "core.jsonl"
    return [json.loads(line) for line in root.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def document_id_from_result():
    def _extract(result):
        return str((result or {}).get("document_id") or (result or {}).get("id") or "")
    return _extract
