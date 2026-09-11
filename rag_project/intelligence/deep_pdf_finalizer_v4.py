from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

import fitz

from rag_project.intelligence.document_structure import DocumentStructureStore
from rag_project.intelligence.pdf_intelligence import score_page_quality
from rag_project.parsing.pdf_extractor import PDFExtractor

_LOCK = threading.RLock()
_INSTALLED = False

_STRUCTURE_HEADER_RE = re.compile(r"\[RAG-STRUCTURE\s+(?:schema=(?P<schema>\d+);\s*)?chapter_id=(?P<chapter_id>[^;\]]*);\s*chapter=(?P<chapter>[^;\]]*);\s*section_id=(?P<section_id>[^;\]]*);\s*section=(?P<section>[^;\]]*);\s*parent_id=(?P<parent_id>[^;\]]*);(?:\s*path=(?P<path>[^;\]]*);)?\s*(?:quality=(?P<quality>[^;\]]*);\s*)?page_type=(?P<page_type>[^;\]]*);\s*(?:ocr_status|ocr)=(?P<ocr_status>[^;\]]*)[^\]]*\]", re.I)


def _parse_structure_header(text: str) -> dict[str, Any]:
    match = _STRUCTURE_HEADER_RE.search(str(text or ""))
    if not match:
        return {}
    values = {key: value for key, value in match.groupdict().items() if value not in {None, ""}}
    try:
        values["structure_version"] = int(values.get("schema") or 1)
    except (TypeError, ValueError):
        values["structure_version"] = 1
    try:
        if "quality" in values:
            values["quality_score"] = float(values["quality"])
    except (TypeError, ValueError):
        pass
    values.pop("schema", None)
    return values


def _patch_structure_parser() -> None:
    from rag_project.storage.enhanced_vector_store import EnhancedVectorStore

    if getattr(EnhancedVectorStore, "_final_pdf_v4_patched", False):
        return
    original = EnhancedVectorStore._enrich_metadata

    @staticmethod
    def enrich(metadata: dict[str, Any], document: str, item_id: str = "") -> dict[str, Any]:
        enriched = dict(metadata or {})
        parsed = _parse_structure_header(document)
        for field in (
            "chapter_id", "chapter", "section_id", "section", "parent_id",
            "page_type", "ocr_status", "quality_score", "structure_version",
        ):
            if field in parsed and parsed[field] not in {None, ""}:
                enriched[field] = parsed[field]
        return original(enriched, document, item_id)

    EnhancedVectorStore._enrich_metadata = enrich
    EnhancedVectorStore._final_pdf_v4_patched = True


def _patch_figure_roles() -> None:
    if getattr(PDFExtractor, "_final_pdf_v4_patched", False):
        return
    original = PDFExtractor.extract_iter

    def extract_iter(self: PDFExtractor, pdf_path, document_id=None):
        pdf = fitz.open(str(pdf_path))
        try:
            for page in original(self, pdf_path, document_id):
                physical = int(page.page_number or page.page_index + 1)
                try:
                    pdf_page = pdf[physical - 1]
                    layout = page.metadata.get("layout") or {}
                    page_bbox = layout.get("page_bbox") or [0, 0, pdf_page.rect.width, pdf_page.rect.height]
                    page_width = max(1.0, float(page_bbox[2]) - float(page_bbox[0]))
                    page_height = max(1.0, float(page_bbox[3]) - float(page_bbox[1]))
                    figure_regions = []
                    for region in layout.get("figure_regions") or []:
                        bbox = list(region.get("bbox") or [])
                        if len(bbox) != 4:
                            continue
                        area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
                        full_page = area >= page_width * page_height * 0.90
                        caption = str(region.get("caption") or "").strip()
                        if full_page and not caption:
                            continue
                        figure_regions.append(region)
                    page.metadata["figure_regions"] = figure_regions
                    # An image resource is not automatically a semantic figure. Keep
                    # image_count for OCR routing, but expose figure_ids only for
                    # semantically plausible figure regions.
                    semantic_ids = [str(region.get("figure_id")) for region in figure_regions if region.get("figure_id")]
                    if semantic_ids:
                        page.figure_ids = semantic_ids
                    elif not page.figure_captions:
                        page.figure_ids = []
                    page.metadata["semantic_figure_count"] = len(page.figure_ids)
                    # Use physical page geometry rather than a fixed 50,000-point area.
                    quality = score_page_quality(
                        page.text,
                        page_number=physical,
                        image_count=int(page.image_count or 0),
                        page_area=page_width * page_height,
                    )
                    page.quality_score = quality.quality
                    page.metadata["quality_score"] = quality.quality
                    page.metadata["quality_route"] = quality.route
                    page.metadata["quality_warnings"] = list(quality.warnings)
                    if page.ocr_status == "success":
                        page.routing_decision = "ocr"
                    elif page.extraction_method == "alternate_native_text":
                        page.routing_decision = "alternate_extractor"
                    else:
                        page.routing_decision = quality.route
                    page.metadata["routing_decision"] = page.routing_decision

                    if self.state_store is not None:
                        store = DocumentStructureStore(Path(self.state_store.database_path).parent / "structure.sqlite3")
                        payload = store.get_page(page.document_id, physical) or {}
                        payload.update(
                            {
                                "quality_score": page.quality_score,
                                "routing_decision": page.routing_decision,
                                "quality_route": quality.route,
                                "quality_warnings": list(quality.warnings),
                                "figure_regions": figure_regions,
                                "semantic_figure_count": len(page.figure_ids),
                            }
                        )
                        store.put_page(page.document_id, physical, payload)
                except Exception:
                    pass
                yield page
        finally:
            pdf.close()

    PDFExtractor.extract_iter = extract_iter
    PDFExtractor._final_pdf_v4_patched = True


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        _patch_structure_parser()
        _patch_figure_roles()
        _INSTALLED = True


__all__ = ["install"]
