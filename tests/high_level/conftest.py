from __future__ import annotations

import json
import time
from io import BytesIO
from pathlib import Path

import pytest

from rag_project.application import create_rag_system
from rag_project.configuration.settings import Settings


def _pdf_escape(text: str) -> str:
    return text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')


def _build_pdf(objects: list[bytes], root_object: int = 1) -> bytes:
    body = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(body))
        body.extend(obj)
    xref = len(body)
    body.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        body.extend(f"{offset:010d} 00000 n \n".encode())
    body.extend(f"trailer\n<< /Size {len(objects) + 1} /Root {root_object} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(body)


def write_minimal_pdf(path: Path, pages: list[str]) -> Path:
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
        objects.append(f"{content_obj} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream\nendobj\n")
        objects.append(f"{page_obj} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 {font_obj} 0 R >> >> /Contents {content_obj} 0 R >>\nendobj\n".encode())
        page_refs.append(page_obj)
    kids = " ".join(f"{ref} 0 R" for ref in page_refs)
    objects.insert(0, b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.insert(1, f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {len(page_refs)} >>\nendobj\n".encode())
    objects.insert(2, b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    path.write_bytes(_build_pdf(objects))
    return path


def write_scanned_pdf(path: Path, pages: int = 1) -> Path:
    """Create a real image-only PDF with large, controlled text for OCR."""
    import fitz
    from PIL import Image, ImageDraw, ImageFont

    document = fitz.open()
    font = ImageFont.load_default(size=54)
    for page_number in range(1, max(1, pages) + 1):
        image = Image.new("RGB", (1700, 2200), "white")
        draw = ImageDraw.Draw(image)
        lines = [
            "Diabetes mellitus is a chronic metabolic disorder.",
            "OCR_CONTROLLED_MARKER: hyperglycemia.",
            f"Scanned medical page {page_number}.",
        ]
        y = 220
        for line in lines:
            draw.text((140, y), line, font=font, fill="black")
            y += 100
        payload = BytesIO()
        image.save(payload, format="PNG")
        page = document.new_page(width=612, height=792)
        page.insert_image(page.rect, stream=payload.getvalue())
    document.save(path)
    document.close()
    return path


def write_empty_pdf(path: Path) -> Path:
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n",
    ]
    path.write_bytes(_build_pdf(objects))
    return path


def write_large_pdf(path: Path, pages: int = 100) -> Path:
    return write_minimal_pdf(path, [f"Controlled large-document page {i}: clinical evidence marker." for i in range(1, pages + 1)])


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


@pytest.fixture(scope="session")
def settings_i5(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    root = tmp_path_factory.mktemp("bookrag-high-level")
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


@pytest.fixture(scope="session")
def temp_project_root(settings_i5: Settings) -> Path:
    return settings_i5.project_root


@pytest.fixture(scope="session")
def ready_document(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("ready-document") / "clean_diabetes_en.pdf"
    return write_minimal_pdf(path, [
        "Diabetes mellitus is a chronic metabolic disorder characterized by hyperglycemia. HbA1c is used to assess glycemic control.",
        "Metformin is commonly used for type 2 diabetes. The controlled treatment dose in this fixture is 500 mg twice daily.",
        "Table 1: HbA1c target 7%. Figure 1: glycemic control trend. Numeric evidence marker: 500 mg.",
    ])


@pytest.fixture(scope="session")
def scanned_document(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_scanned_pdf(tmp_path_factory.mktemp("scanned-document") / "scanned_image_only.pdf", pages=2)


@pytest.fixture(scope="session")
def large_document(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_large_pdf(tmp_path_factory.mktemp("large-document") / "large_100_page.pdf", pages=100)


@pytest.fixture(scope="session")
def empty_document(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_empty_pdf(tmp_path_factory.mktemp("empty-document") / "empty_content.pdf")


@pytest.fixture(scope="session")
def clean_system(settings_i5: Settings, ready_document: Path):
    system = create_rag_system(settings_i5)
    ingestion = system.ingest_file(ready_document)
    status = str((ingestion or {}).get("status") or "").upper()
    assert status in {"READY", "COMPLETED", "SUCCESS", "SKIPPED"}, (
        "controlled fixture document failed to reach a usable ingestion state: "
        f"{ingestion!r}"
    )
    system._high_level_default_llm = getattr(system, "llm", None)
    return system


@pytest.fixture(scope="session")
def ready_multilingual_docs(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("multilingual-docs")
    return {
        "en": write_minimal_pdf(root / "diabetes_en.pdf", ["Diabetes mellitus is a chronic metabolic disorder."]),
        "fr": write_minimal_pdf(root / "diabetes_fr.pdf", ["Le diabète est une maladie métabolique chronique."]),
        "ar": write_minimal_pdf(root / "diabetes_ar.pdf", ["داء السكري هو اضطراب استقلابي مزمن."]),
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


@pytest.fixture(autouse=True)
def reset_shared_high_level_state(request: pytest.FixtureRequest):
    """Keep the session system cheap to share without leaking conversational/LLM state."""
    system = None
    if "clean_system" in request.fixturenames:
        system = request.getfixturevalue("clean_system")
        memory = getattr(system, "conversation_memory", None)
        history = getattr(memory, "history", None)
        if hasattr(history, "clear"):
            history.clear()
        default_llm = getattr(system, "_high_level_default_llm", None)
        system.llm = default_llm
        system._med_selected_hits = []
        system._high_level_test_marker = None
    yield


@pytest.fixture
def gold_questions():
    root = Path(__file__).resolve().parents[1] / "support" / "gold_sets" / "core.jsonl"
    return [json.loads(line) for line in root.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def document_id_from_result():
    def _extract(result):
        return str((result or {}).get("document_id") or (result or {}).get("id") or "")
    return _extract
