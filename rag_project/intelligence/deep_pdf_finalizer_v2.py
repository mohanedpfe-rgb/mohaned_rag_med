from __future__ import annotations

import copy
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

import fitz

from rag_project.chunking.semantic_chunker import SemanticChunker
from rag_project.ingestion.document_classifier import DocumentClassifier
from rag_project.ocr.ocr_service import OCRService
from rag_project.parsing.pdf_extractor import PDFExtractor
from rag_project.retrieval.context_builder import ContextBuilder
from rag_project.storage.vector_store import VectorStore
from rag_project.utils.text_utils import detect_language, meaningful_tokens

_LOCK = threading.RLock()
_INSTALLED = False


def _normalized_boxes(raw_result: Any, width: float, height: float) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for item in raw_result or []:
        if not item or len(item) < 2:
            continue
        raw_box = item[0]
        text = str(item[1] or "").strip()
        if not text:
            continue
        try:
            points = [tuple(map(float, point)) for point in raw_box]
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            x0, x1 = min(xs) / max(width, 1.0), max(xs) / max(width, 1.0)
            y0, y1 = min(ys) / max(height, 1.0), max(ys) / max(height, 1.0)
        except Exception:
            continue
        boxes.append({"bbox": [max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)], "text": text, "confidence": float(item[2]) if len(item) > 2 else None})
    return boxes


def _ocr_ordered_text(regions: list[dict[str, Any]]) -> str:
    if not regions:
        return ""
    centers = sorted(((float(item["bbox"][0] + item["bbox"][2]) / 2.0, item) for item in regions), key=lambda value: value[0])
    columns: list[list[dict[str, Any]]] = []
    for center, item in centers:
        if not columns:
            columns.append([item])
            continue
        last = sum((x["bbox"][0] + x["bbox"][2]) / 2.0 for x in columns[-1]) / len(columns[-1])
        if abs(center - last) > 0.16 and len(columns) < 4:
            columns.append([item])
        else:
            columns[-1].append(item)
    for column in columns:
        column.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return "\n".join(item["text"] for column in columns for item in column if item.get("text"))


def _token_overlap(a: str, b: str) -> float:
    aa = set(meaningful_tokens(a or ""))
    bb = set(meaningful_tokens(b or ""))
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / max(len(aa), 1)


def _decorate_headings(text: str, candidates: list[dict[str, Any]]) -> str:
    if not text or not candidates:
        return text
    by_line: dict[str, int] = {}
    for candidate in candidates:
        title = str(candidate.get("text") or "").strip()
        if title:
            by_line[title] = max(1, min(6, int(candidate.get("level") or 2)))
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        level = by_line.get(stripped)
        if level and not stripped.startswith("#"):
            lines.append("#" * level + " " + stripped)
        else:
            lines.append(line)
    return "\n".join(lines)


def _classify_pdf(pdf_path: str | Path) -> dict[str, Any]:
    pdf = fitz.open(str(pdf_path))
    try:
        page_count = pdf.page_count
        text_pages = image_pages = table_pages = mixed_pages = 0
        total_chars = 0
        for index in range(page_count):
            page = pdf[index]
            text = page.get_text("text") or ""
            images = len(page.get_images(full=True) or [])
            tables = 0
            find_tables = getattr(page, "find_tables", None)
            if find_tables is not None:
                try:
                    tables = len(getattr(find_tables(), "tables", []) or [])
                except Exception:
                    tables = 0
            if len(text.strip()) >= 50:
                text_pages += 1
                total_chars += len(text)
            if images:
                image_pages += 1
            if tables:
                table_pages += 1
            if images and len(text.strip()) >= 50:
                mixed_pages += 1
        ratio = text_pages / max(page_count, 1)
        document_type = "empty" if page_count == 0 else "text_based" if ratio > 0.75 else "mixed" if ratio > 0.35 else "scanned_or_ocr_required"
        return {
            "file_name": Path(pdf_path).name,
            "page_count": page_count,
            "document_type": document_type,
            "text_pages": text_pages,
            "image_heavy_pages": image_pages,
            "table_heavy_pages": table_pages,
            "approx_text_chars": total_chars,
            "ocr_required": document_type in {"scanned_or_ocr_required", "mixed"},
            "sample_pages": list(range(page_count)),
            "classification_scope": "all_pages",
            "mixed_pages": mixed_pages,
            "layout_aware": True,
        }
    finally:
        pdf.close()


def _patch_classifier() -> None:
    if getattr(DocumentClassifier, "_final_pdf_v2_patched", False):
        return
    DocumentClassifier.classify = staticmethod(_classify_pdf)
    DocumentClassifier._final_pdf_v2_patched = True


def _patch_ocr() -> None:
    if getattr(OCRService, "_final_pdf_v2_patched", False):
        return

    def recognize(self, image_path):
        if self._rapidocr is None:
            raise RuntimeError(self._init_error or "No OCR engine available; install rapidocr-onnxruntime.")
        result, _ = self._rapidocr(str(image_path))
        if not result:
            from PIL import Image
            from PIL import ImageEnhance, ImageFilter, ImageOps

            with Image.open(image_path) as image:
                enhanced = ImageOps.autocontrast(ImageOps.grayscale(image))
                enhanced = ImageEnhance.Contrast(enhanced).enhance(self.enhance_contrast)
                enhanced = enhanced.filter(ImageFilter.SHARPEN)
                enhanced.save(image_path)
            result, _ = self._rapidocr(str(image_path))
        if not result:
            self._last_ocr_regions = []
            raise RuntimeError("OCR produced no text for this page.")
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
        self._last_ocr_regions = _normalized_boxes(result, width, height)
        text = "\n".join(item[1] for item in result if item and len(item) > 1).strip()
        confidence = sum(float(item[2]) for item in result if len(item) > 2) / max(len(result), 1)
        return text, float(confidence)

    OCRService._recognize = recognize
    OCRService._final_pdf_v2_patched = True


def _patch_extractor() -> None:
    if getattr(PDFExtractor, "_final_pdf_v2_patched", False):
        return
    original = PDFExtractor.extract_iter

    def extract_iter(self: PDFExtractor, pdf_path, document_id=None):
        pdf = fitz.open(str(pdf_path))
        try:
            for page in original(self, pdf_path, document_id):
                physical = int(page.page_number or page.page_index + 1)
                try:
                    layout = page.metadata.get("layout") or {}
                    regions = list(getattr(self.ocr_service, "_last_ocr_regions", []) or []) if page.ocr_status == "success" else []
                    page.metadata["ocr_regions"] = regions
                    if regions:
                        ordered_ocr = _ocr_ordered_text(regions)
                        if ordered_ocr and _token_overlap(page.text, ordered_ocr) >= 0.70:
                            page.metadata["ocr_reading_order"] = "column_aware"
                            page.metadata["ocr_ordered_text"] = ordered_ocr
                    if page.extraction_method in {"pdf_text", "cached"} and layout.get("text_regions"):
                        ordered = []
                        blocks = {int(region.get("block_index")): str(region.get("text") or "") for region in layout.get("text_regions") or []}
                        for block_index in layout.get("reading_order") or []:
                            value = blocks.get(int(block_index), "").strip()
                            if value:
                                ordered.append(value)
                        candidate = "\n\n".join(ordered)
                        if candidate and _token_overlap(page.text, candidate) >= 0.78 and len(candidate) >= max(40, int(len(page.text) * 0.55)):
                            page.metadata["native_reading_order"] = "layout_blocks"
                            page.metadata["native_ordered_text"] = candidate
                    if page.quality_score < 0.35:
                        try:
                            fallback = pdf[physical - 1].get_text("text", sort=True)
                            if _token_overlap(page.text, fallback) >= 0.50 and len(fallback) > len(page.text):
                                page.text = fallback.strip()
                                page.extraction_method = "alternate_native_text"
                                page.routing_decision = "alternate_extractor"
                        except Exception:
                            pass
                    base_quality = float(page.quality_score or 0.0)
                    structure_bonus = 0.0
                    if page.headings:
                        structure_bonus += 0.04
                    if page.table_texts:
                        structure_bonus += 0.03
                    if page.figure_captions:
                        structure_bonus += 0.03
                    page.quality_score = min(1.0, base_quality + structure_bonus)
                    page.metadata["structure_quality_score"] = page.quality_score
                except Exception:
                    pass
                yield page
        finally:
            pdf.close()

    PDFExtractor.extract_iter = extract_iter
    PDFExtractor._final_pdf_v2_patched = True


def _patch_chunker() -> None:
    if getattr(SemanticChunker, "_final_pdf_v2_patched", False):
        return
    original = SemanticChunker.chunk_pages

    def chunk_pages(self: SemanticChunker, pages):
        source_pages = list(pages or [])
        transformed = []
        for source in source_pages:
            page = copy.deepcopy(source)
            layout = page.metadata.get("layout") or {}
            ordered_text = str(page.metadata.get("native_ordered_text") or "").strip()
            if page.ocr_status == "success":
                ordered_text = str(page.metadata.get("ocr_ordered_text") or ordered_text).strip()
            if ordered_text and _token_overlap(page.text, ordered_text) >= 0.70:
                page.text = ordered_text
            page.text = _decorate_headings(page.text, list(layout.get("heading_candidates") or []))
            transformed.append(page)
        chunks = original(self, transformed)
        for chunk in chunks:
            language = detect_language(chunk.text or "")
            chunk.metadata["chunk_language"] = language
            chunk.metadata["language_scope"] = "chunk"
            chunk.metadata["structure_schema_version"] = 3
        return chunks

    SemanticChunker.chunk_pages = chunk_pages
    SemanticChunker._final_pdf_v2_patched = True


def _patch_vector_store() -> None:
    if getattr(VectorStore, "_final_pdf_v2_patched", False):
        return
    original = VectorStore.add_documents

    def add_documents(self, documents, metadatas, embeddings, ids):
        adjusted = []
        for metadata in list(metadatas or []):
            item = dict(metadata or {})
            if item.get("chunk_language"):
                item["language"] = item["chunk_language"]
            adjusted.append(item)
        return original(self, documents, adjusted, embeddings, ids)

    VectorStore.add_documents = add_documents
    VectorStore._final_pdf_v2_patched = True


def _patch_context() -> None:
    if getattr(ContextBuilder, "_final_pdf_v2_patched", False):
        return

    def build(self, hits):
        ordered = [item for item in list(hits or []) if item is not None]
        if self.neighbor_expansion and self.neighbor_resolver:
            ordered = self._expand_neighbors(ordered)
        priority = {"figure_visual": 0, "table": 1, "figure_caption": 2, "section_anchor": 3, "chapter_anchor": 4, "book_anchor": 5, "canonical": 6}
        ordered.sort(key=lambda hit: (priority.get(str((hit.metadata or {}).get("representation_type") or "canonical"), 6), -float(getattr(hit, "score", 0.0))))
        selected = []
        seen = set()
        per_document: dict[str, int] = defaultdict(int)
        representations: dict[str, set[str]] = defaultdict(set)
        used_tokens = 0
        for index, hit in enumerate(ordered):
            metadata = hit.metadata or {}
            chunk_id = str(metadata.get("chunk_id") or f"{hit.doc_id}:{index}")
            document_id = str(metadata.get("document_id") or hit.doc_id)
            representation = str(metadata.get("representation_type") or "canonical")
            cap = self.max_per_document
            if representation in {"table", "figure_caption", "figure_visual", "section_anchor", "chapter_anchor", "book_anchor"} and per_document[document_id] < self.max_per_document:
                cap = max(cap, self.max_per_document + 1)
            if chunk_id in seen or per_document[document_id] >= cap:
                continue
            estimated_tokens = max(1, len(str(hit.text or "").split()) * 4 // 3)
            if selected and used_tokens + estimated_tokens > self.token_budget:
                continue
            if representation == "canonical" and per_document[document_id] >= max(1, cap - 1) and not representations[document_id]:
                continue
            seen.add(chunk_id)
            per_document[document_id] += 1
            representations[document_id].add(representation)
            used_tokens += estimated_tokens
            selected.append(hit)
        context = "\n\n".join(
            f'<evidence id="S{index + 1}" chunk_id="{str((hit.metadata or {}).get("chunk_id") or f"{hit.doc_id}:{index}") }">'
            f'[{(hit.metadata or {}).get("file_name", "unknown")} pages {(hit.metadata or {}).get("page_numbers", [])}]\n{hit.text}\n</evidence>'
            for index, hit in enumerate(selected)
        )
        return context, selected

    ContextBuilder.build = build
    ContextBuilder._final_pdf_v2_patched = True


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        _patch_classifier()
        _patch_ocr()
        _patch_extractor()
        _patch_chunker()
        _patch_vector_store()
        _patch_context()
        _INSTALLED = True


__all__ = ["install"]
