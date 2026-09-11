from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

import fitz
import requests

from rag_project.chunking.semantic_chunker import SemanticChunker
from rag_project.ingestion.document_models import Chunk
from rag_project.intelligence.document_structure import DocumentStructureStore, STRUCTURE_SCHEMA_VERSION
from rag_project.parsing.pdf_extractor import PDFExtractor
from rag_project.retrieval.context_builder import ContextBuilder
from rag_project.retrieval.hybrid_retriever import HybridRetriever
from rag_project.storage.enhanced_vector_store import EnhancedVectorStore
from rag_project.storage.vector_store import VectorStore

_LOCK = threading.RLock()
_INSTALLED = False

_LAYOUT_SCHEMA_VERSION = 1
_VISION_REPRESENTATION = "figure_visual"

_CAPTION_RE = re.compile(
    r"^\s*(?:figure|fig\.?|illustration|image|photo|diagram|scheme|schéma|plate|panel)\b",
    re.I,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if result == result else default
    except (TypeError, ValueError):
        return default


def _caption_line(text: str) -> bool:
    return bool(_CAPTION_RE.match(str(text or "").strip()))


def _page_layout(page: fitz.Page, document_id: str, page_number: int) -> dict[str, Any]:
    """Build a geometry-backed page representation without retaining PDF objects."""
    try:
        raw = page.get_text("dict", sort=False)
        blocks = list(raw.get("blocks") or [])
    except Exception:
        blocks = []

    text_regions: list[dict[str, Any]] = []
    sizes: list[float] = []
    for block_index, block in enumerate(blocks):
        if int(block.get("type", 0) or 0) != 0:
            continue
        bbox = list(block.get("bbox") or [])
        text = " ".join(
            str(span.get("text") or "").strip()
            for line in block.get("lines") or []
            for span in line.get("spans") or []
            if str(span.get("text") or "").strip()
        ).strip()
        if not text or len(bbox) != 4:
            continue
        span_sizes = [
            _safe_float(span.get("size"), 0.0)
            for line in block.get("lines") or []
            for span in line.get("spans") or []
            if span.get("text")
        ]
        font_size = max(span_sizes or [0.0])
        bold = any("bold" in str(span.get("font") or "").casefold() for line in block.get("lines") or [] for span in line.get("spans") or [])
        sizes.append(font_size)
        text_regions.append(
            {
                "type": "text",
                "block_index": block_index,
                "bbox": [round(_safe_float(v), 2) for v in bbox],
                "text": text[:4000],
                "font_size": round(font_size, 2),
                "bold": bool(bold),
            }
        )

    median_size = sorted(sizes)[len(sizes) // 2] if sizes else 0.0
    heading_candidates: list[dict[str, Any]] = []
    for region in text_regions:
        text = region["text"]
        words = text.split()
        numbered = bool(re.match(r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s+", text, re.I))
        typographic = bool(region["font_size"] >= median_size * 1.18 and len(words) <= 18)
        uppercase = bool(text.upper() == text and any(ch.isalpha() for ch in text) and len(words) <= 14)
        caption = _caption_line(text)
        if caption:
            continue
        if numbered or typographic or (uppercase and region["font_size"] >= median_size):
            if len(text) <= 180:
                level = 1 if numbered and re.match(r"^\s*\d+\.?(?:\s|$)", text) else 2
                if numbered:
                    numbering = re.match(r"^\s*((?:\d+(?:\.\d+)*)|(?:[IVXLC]+))", text, re.I)
                    if numbering:
                        level = min(6, numbering.group(1).count(".") + 1 if numbering.group(1)[0].isdigit() else 1)
                elif region["font_size"] >= median_size * 1.45:
                    level = 1
                heading_candidates.append({"text": text, "level": int(level), "bbox": region["bbox"], "block_index": region["block_index"]})

    # Approximate column structure from text-block x centers. This is deliberately
    # conservative: it only splits columns when there is a substantial horizontal gap.
    centers = sorted(((r["bbox"][0] + r["bbox"][2]) / 2.0, r) for r in text_regions)
    columns: list[list[dict[str, Any]]] = []
    gap_threshold = max(45.0, float(page.rect.width) * 0.08)
    for center, region in centers:
        if not columns:
            columns.append([region])
            continue
        last_center = sum((item["bbox"][0] + item["bbox"][2]) / 2.0 for item in columns[-1]) / len(columns[-1])
        if abs(center - last_center) > gap_threshold and len(columns) < 4:
            columns.append([region])
        else:
            columns[-1].append(region)
    for column in columns:
        column.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

    image_regions: list[dict[str, Any]] = []
    for block_index, block in enumerate(blocks):
        if int(block.get("type", 0) or 0) != 1:
            continue
        bbox = list(block.get("bbox") or [])
        if len(bbox) == 4:
            image_regions.append({"bbox": [round(_safe_float(v), 2) for v in bbox], "block_index": block_index})
    if not image_regions:
        try:
            for image_number, img in enumerate(page.get_images(full=True), start=1):
                for rect in page.get_image_rects(img[0]):
                    image_regions.append({"bbox": [round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2)], "block_index": image_number})
        except Exception:
            pass

    table_regions: list[dict[str, Any]] = []
    find_tables = getattr(page, "find_tables", None)
    if find_tables is not None:
        try:
            for table_index, table in enumerate(getattr(find_tables(), "tables", []) or [], start=1):
                bbox = list(getattr(table, "bbox", ()) or ())
                rows = table.extract() or []
                if len(bbox) == 4:
                    table_regions.append(
                        {
                            "table_index": table_index,
                            "bbox": [round(_safe_float(v), 2) for v in bbox],
                            "rows": [[str(cell if cell is not None else "") for cell in row] for row in rows[:200]],
                        }
                    )
        except Exception:
            pass

    figure_regions: list[dict[str, Any]] = []
    for figure_index, image in enumerate(image_regions, start=1):
        x0, y0, x1, y1 = image["bbox"]
        nearest: tuple[float, str] | None = None
        for region in text_regions:
            text = region["text"]
            if not _caption_line(text):
                continue
            rx0, ry0, rx1, ry1 = region["bbox"]
            distance = abs(ry0 - y1) + abs((rx0 + rx1) / 2.0 - (x0 + x1) / 2.0) * 0.2
            if nearest is None or distance < nearest[0]:
                nearest = (distance, text)
        figure_regions.append(
            {
                "figure_id": f"{document_id}:p{page_number}:figure:{figure_index}",
                "bbox": image["bbox"],
                "caption": nearest[1] if nearest and nearest[0] <= max(120.0, page.rect.height * 0.12) else None,
            }
        )

    ordered_regions: list[dict[str, Any]] = []
    for column_index, column in enumerate(columns):
        for region in column:
            item = dict(region)
            item["column"] = column_index
            ordered_regions.append(item)

    return {
        "layout_schema_version": _LAYOUT_SCHEMA_VERSION,
        "page_bbox": [round(page.rect.x0, 2), round(page.rect.y0, 2), round(page.rect.x1, 2), round(page.rect.y1, 2)],
        "column_count": len(columns),
        "columns": [
            {
                "column_index": index,
                "bbox": [
                    min(item["bbox"][0] for item in column),
                    min(item["bbox"][1] for item in column),
                    max(item["bbox"][2] for item in column),
                    max(item["bbox"][3] for item in column),
                ],
                "block_indices": [item["block_index"] for item in column],
            }
            for index, column in enumerate(columns)
            if column
        ],
        "reading_order": [item["block_index"] for item in ordered_regions],
        "text_regions": text_regions,
        "heading_candidates": heading_candidates,
        "table_regions": table_regions,
        "figure_regions": figure_regions,
    }


class VisionFigureAdapter:
    """Optional Ollama multimodal adapter for true figure semantic descriptions.

    It is disabled unless VISION_MODEL is explicitly configured, preventing an
    11B-class vision model from unexpectedly consuming a 16GB CPU laptop.
    """

    def __init__(self, model: str | None = None, base_url: str | None = None, timeout: float = 90.0):
        self.model = (model or os.getenv("VISION_MODEL", "")).strip()
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.timeout = max(10.0, min(float(timeout), 180.0))

    @property
    def enabled(self) -> bool:
        return bool(self.model)

    def describe(self, image_bytes: bytes) -> str | None:
        if not self.enabled or not image_bytes:
            return None
        payload = {
            "model": self.model,
            "prompt": "Describe this textbook figure for semantic retrieval. Mention the object/system shown, labeled structures, diagram type, arrows/relationships, and important visible terms. Do not invent details that are not visible.",
            "images": [base64.b64encode(image_bytes).decode("ascii")],
            "stream": False,
            "options": {"temperature": 0},
        }
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout, allow_redirects=False)
            response.raise_for_status()
            value = response.json().get("response")
            return str(value).strip()[:4000] if value else None
        except (requests.RequestException, ValueError, TypeError):
            return None


def _render_figure(page: fitz.Page, bbox: list[float]) -> bytes | None:
    try:
        rect = fitz.Rect(*bbox)
        width = max(1.0, rect.width)
        height = max(1.0, rect.height)
        scale = min(2.0, (1_500_000.0 / max(width * height, 1.0)) ** 0.5)
        pix = page.get_pixmap(matrix=fitz.Matrix(max(0.5, scale), max(0.5, scale)), clip=rect, alpha=False)
        return pix.tobytes("png")
    except Exception:
        return None


def _patch_extractor() -> None:
    if getattr(PDFExtractor, "_final_pdf_patched", False):
        return
    original = PDFExtractor.extract_iter

    def extract_iter(self: PDFExtractor, pdf_path: str | Path, document_id: str | None = None):
        pdf = fitz.open(str(pdf_path))
        structure_store = None
        if self.state_store is not None:
            structure_store = DocumentStructureStore(Path(self.state_store.database_path).parent / "structure.sqlite3")
        vision = VisionFigureAdapter()
        try:
            for page in original(self, pdf_path, document_id):
                physical = int(page.page_number or page.page_index + 1)
                stored = structure_store.get_page(page.document_id, physical) if structure_store else None
                layout = stored.get("layout") if isinstance(stored, dict) else None
                if not isinstance(layout, dict):
                    try:
                        layout = _page_layout(pdf[physical - 1], page.document_id, physical)
                    except Exception:
                        layout = {"layout_schema_version": _LAYOUT_SCHEMA_VERSION, "column_count": 1, "columns": [], "reading_order": [], "text_regions": [], "heading_candidates": [], "table_regions": [], "figure_regions": []}

                page.metadata["layout"] = layout
                page.metadata["layout_schema_version"] = _LAYOUT_SCHEMA_VERSION
                page.metadata["heading_candidates"] = list(layout.get("heading_candidates") or [])
                page.metadata["table_regions"] = list(layout.get("table_regions") or [])
                page.metadata["figure_regions"] = list(layout.get("figure_regions") or [])
                page.metadata["column_count"] = int(layout.get("column_count") or 1)

                # Typography-derived headings supplement regex extraction and are
                # encoded only in chunking metadata, keeping the original page text intact.
                heading_texts = [str(item.get("text") or "").strip() for item in layout.get("heading_candidates") or [] if item.get("text")]
                page.headings = list(dict.fromkeys([*page.headings, *heading_texts]))[:100]

                figure_visuals: list[dict[str, Any]] = []
                if vision.enabled and page.has_images:
                    try:
                        pdf_page = pdf[physical - 1]
                        for figure in layout.get("figure_regions") or []:
                            image_bytes = _render_figure(pdf_page, list(figure.get("bbox") or []))
                            description = vision.describe(image_bytes or b"")
                            if description:
                                figure_visuals.append({"figure_id": figure.get("figure_id"), "bbox": figure.get("bbox"), "description": description})
                    except Exception:
                        figure_visuals = []
                page.metadata["figure_visuals"] = figure_visuals

                if structure_store:
                    payload = {
                        "document_id": page.document_id,
                        "page_number": physical,
                        "page_type": page.page_type,
                        "quality_score": page.quality_score,
                        "routing_decision": page.routing_decision,
                        "ocr_status": page.ocr_status,
                        "ocr_confidence": page.ocr_confidence,
                        "image_count": page.image_count,
                        "has_images": page.has_images,
                        "table_ids": list(page.table_ids),
                        "table_texts": list(page.table_texts),
                        "figure_ids": list(page.figure_ids),
                        "figure_captions": list(page.figure_captions),
                        "figure_visuals": figure_visuals,
                        "headings": list(page.headings),
                        "layout": layout,
                        "text_checksum": hashlib.sha256((page.text or "").encode("utf-8")).hexdigest(),
                    }
                    structure_store.put_page(page.document_id, physical, payload)
                yield page
        finally:
            pdf.close()

    PDFExtractor.extract_iter = extract_iter
    PDFExtractor._final_pdf_patched = True


def _patch_chunker() -> None:
    if getattr(SemanticChunker, "_final_pdf_patched", False):
        return
    original = SemanticChunker.chunk_pages

    def chunk_pages(self: SemanticChunker, pages):
        pages = list(pages or [])
        result = original(self, pages)
        by_page: dict[tuple[str, int], Any] = {}
        for page in pages:
            by_page[(str(page.document_id), int(page.page_number or page.page_index + 1))] = page

        # Attach geometry-aware structure metadata and add optional true visual
        # representations. The text/caption path remains the CPU-safe fallback.
        augmented: list[Chunk] = []
        for chunk in result:
            key_pages = chunk.page_numbers or [1]
            page = by_page.get((str(chunk.doc_id), int(key_pages[0])))
            if page is None:
                augmented.append(chunk)
                continue
            metadata = dict(chunk.metadata or {})
            metadata["layout_schema_version"] = _LAYOUT_SCHEMA_VERSION
            metadata["column_count"] = int((page.metadata.get("column_count") or 1))
            metadata["heading_candidates"] = list(page.metadata.get("heading_candidates") or [])
            metadata["table_regions"] = list(page.metadata.get("table_regions") or [])
            metadata["figure_regions"] = list(page.metadata.get("figure_regions") or [])
            if chunk.representation_type in {"canonical", "table", "figure_caption"}:
                metadata["representation_type"] = chunk.representation_type
            chunk.metadata = metadata
            augmented.append(chunk)

        for page in pages:
            snapshot = None
            tracker = getattr(self, "_deep_trackers", {}).get(str(page.document_id))
            if tracker is not None:
                snapshot = tracker.snapshot(page.page_number or page.page_index + 1)
            if snapshot is None:
                continue
            visuals = list(page.metadata.get("figure_visuals") or [])
            for index, visual in enumerate(visuals):
                description = str(visual.get("description") or "").strip()
                figure_id = str(visual.get("figure_id") or f"{page.document_id}:p{page.page_number}:figure:{index + 1}")
                if not description:
                    continue
                enriched = __import__("rag_project.intelligence.pdf_intelligence", fromlist=["enrich_text"]).enrich_text(description)
                augmented.append(
                    Chunk(
                        doc_id=str(page.document_id),
                        file_name=page.file_name,
                        chunk_index=len(augmented),
                        text=f"[FIGURE VISUAL]\n{description}",
                        page_numbers=[page.page_number or 1],
                        metadata={
                            "source_pages": [page.page_number or 1],
                            "page_numbers": [page.page_number or 1],
                            "evidence_types": ["figure", "visual"],
                            "chapter": snapshot.chapter,
                            "chapter_id": snapshot.chapter_id,
                            "section": snapshot.section,
                            "section_id": snapshot.section_id or snapshot.parent_id,
                            "global_section_id": snapshot.section_id or snapshot.parent_id,
                            "parent_id": snapshot.parent_id,
                            "hierarchy_path": list(snapshot.hierarchy_path),
                            "child_index": index,
                            "normalized_text": enriched["normalized_text"],
                            "entities": enriched["entities"],
                            "headings": enriched["headings"],
                            "number_forms": enriched["number_forms"],
                            "table_id": None,
                            "figure_id": figure_id,
                            "document_id": page.document_id,
                            "file_name": page.file_name,
                            "page_type": page.page_type,
                            "quality_score": page.quality_score,
                            "ocr_status": page.ocr_status,
                            "ocr_confidence": page.ocr_confidence,
                            "routing_decision": "vision",
                            "visual_model": os.getenv("VISION_MODEL", ""),
                            "visual_bbox": visual.get("bbox"),
                        },
                        representation_type=_VISION_REPRESENTATION,
                        parent_id=snapshot.parent_id,
                        section_id=snapshot.section_id or snapshot.parent_id,
                        table_id=None,
                        figure_id=figure_id,
                        normalized_text=enriched["normalized_text"],
                    )
                )
        for index, chunk in enumerate(augmented):
            chunk.chunk_index = index
        return augmented

    SemanticChunker.chunk_pages = chunk_pages
    SemanticChunker._final_pdf_patched = True


def _patch_vector_contract() -> None:
    if getattr(VectorStore, "_final_pdf_patched", False):
        return
    original_add = VectorStore.add_documents
    original_validate = VectorStore.validate_document_index

    def add_documents(self, documents, metadatas, embeddings, ids):
        for metadata in list(metadatas or []):
            representation = str((metadata or {}).get("representation_type") or "canonical")
            if representation == _VISION_REPRESENTATION and not (metadata or {}).get("figure_id"):
                raise ValueError("figure_visual representation requires figure_id")
        return original_add(self, documents, metadatas, embeddings, ids)

    def validate_document_index(self, document_id: str, version_id: str | None = None):
        result = original_validate(self, document_id, version_id)
        if result.get("valid"):
            records = self.collection.get(where={"document_id": document_id}, include=["metadatas"])
            allowed = {"canonical", "table", "figure_caption", "figure_visual", "section_anchor", "chapter_anchor", "book_anchor"}
            for raw in records.get("metadatas", []) or []:
                meta = self._coerce_metadata(raw)
                representation = str(meta.get("representation_type") or "")
                if representation not in allowed:
                    result.setdefault("issues", []).append(f"invalid representation_type: {meta.get('chunk_id')}")
            result["valid"] = not result.get("issues")
        return result

    VectorStore.add_documents = add_documents
    VectorStore.validate_document_index = validate_document_index
    VectorStore._final_pdf_patched = True


def _patch_enhanced_store() -> None:
    if getattr(EnhancedVectorStore, "_final_pdf_patched", False):
        return
    original_publish = EnhancedVectorStore._publish_hierarchy
    original_search = getattr(EnhancedVectorStore, "search", None)

    def publish(self, document_id: str, version_id: str):
        # Deep ingestion passes the content hash to set_version_index_state while
        # the vector metadata uses the canonical configuration-aware version ID.
        # Accept either identity so hierarchy anchors are always rebuilt.
        records = self.collection.get(where={"document_id": document_id}, include=["metadatas"])
        canonical_versions = {
            str((meta or {}).get("version_id") or "")
            for meta in records.get("metadatas", []) or []
            if str((meta or {}).get("content_hash") or "") == str(version_id)
        }
        if canonical_versions:
            chosen = next(iter(canonical_versions))
            return original_publish(self, document_id, chosen)
        return original_publish(self, document_id, version_id)

    def search(self, embedding, n_results=5, where=None):
        base = original_search(self, embedding, n_results, where) if original_search else VectorStore.search(self, embedding, n_results, where)
        try:
            result = self.hierarchy_collection.query(
                query_embeddings=[list(map(float, embedding))],
                n_results=max(1, int(n_results)),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            ids = list((result.get("ids") or [[]])[0])
            docs = list((result.get("documents") or [[]])[0])
            metas = list((result.get("metadatas") or [[]])[0])
            distances = list((result.get("distances") or [[]])[0])
            existing = set((base.get("ids") or [[]])[0])
            for item_id, doc, meta, distance in zip(ids, docs, metas, distances, strict=False):
                if item_id in existing:
                    continue
                base.setdefault("ids", [[]])[0].append(str(item_id))
                base.setdefault("documents", [[]])[0].append(str(doc))
                base.setdefault("metadatas", [[]])[0].append(dict(meta or {}))
                base.setdefault("distances", [[]])[0].append(float(distance))
        except Exception:
            pass
        return base

    EnhancedVectorStore._publish_hierarchy = publish
    EnhancedVectorStore.search = search
    EnhancedVectorStore._final_pdf_patched = True


def _patch_retrieval() -> None:
    if getattr(HybridRetriever, "_final_pdf_patched", False):
        return
    original = HybridRetriever.retrieve

    def retrieve(self, query: str, top_k: int = 6, where=None):
        hits = list(original(self, query, top_k=max(int(top_k) * 4, int(top_k)), where=where) or [])
        lowered = str(query or "").casefold()
        for hit in hits:
            meta = hit.metadata or {}
            rep = str(meta.get("representation_type") or "canonical")
            bonus = 0.0
            if rep == "figure_visual":
                bonus += 0.20 if any(token in lowered for token in ("figure", "diagram", "image", "shown", "illustrates", "visual", "anatomy")) else 0.03
            elif rep == "figure_caption":
                bonus += 0.12 if any(token in lowered for token in ("figure", "diagram", "image", "caption")) else 0.02
            elif rep == "table":
                bonus += 0.18 if any(token in lowered for token in ("table", "range", "values", "dose", "laboratory", "lab")) else 0.02
            elif rep in {"section_anchor", "chapter_anchor", "book_anchor"}:
                bonus += 0.15 if any(token in lowered for token in ("chapter", "section", "where in the book", "overview")) else 0.04
            bonus += 0.05 * max(0.0, min(1.0, _safe_float(meta.get("quality_score"), 0.0)))
            if meta.get("ocr_status") == "success":
                bonus += 0.02
            hit.score = float(hit.score) + bonus
            hit.metadata["structural_bonus"] = round(bonus, 6)
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[: max(1, int(top_k))]

    HybridRetriever.retrieve = retrieve
    HybridRetriever._final_pdf_patched = True


def _patch_context() -> None:
    if getattr(ContextBuilder, "_final_pdf_patched", False):
        return
    original = ContextBuilder.build

    def build(self, hits):
        incoming = [item for item in list(hits or []) if item is not None]
        # Greedy structural diversity: never let a burst of ordinary prose consume
        # the whole per-document budget before a relevant table/figure/anchor is seen.
        priority = {"figure_visual": 0, "table": 1, "figure_caption": 2, "section_anchor": 3, "chapter_anchor": 4, "book_anchor": 5, "canonical": 6}
        incoming.sort(key=lambda hit: (priority.get(str((hit.metadata or {}).get("representation_type") or "canonical"), 6), -float(getattr(hit, "score", 0.0))))
        return original(self, incoming)

    ContextBuilder.build = build
    ContextBuilder._final_pdf_patched = True


def _clear_structure_before_rebuild() -> None:
    from rag_project.ingestion import robust_ingestor

    if getattr(robust_ingestor.robust_ingest_file, "_final_structure_patched", False):
        return
    original = robust_ingestor.robust_ingest_file

    def wrapped(system, pdf_path):
        try:
            content_hash = system._hash_file(Path(pdf_path))
            existing = system.state_store.get_by_hash(content_hash)
            previous = system.state_store.get_by_path(str(Path(pdf_path).resolve()))
            document_id = existing.get("document_id") if existing else (previous.get("document_id") if previous else content_hash)
            current_chunking = json.dumps({"size": system.settings.chunk_size, "overlap": system.settings.chunk_overlap}, sort_keys=True)
            current_ocr = json.dumps({"engine": "rapidocr", "scale": 2, "structure_schema": STRUCTURE_SCHEMA_VERSION}, sort_keys=True)
            expected = system._ingestion_version_id(
                content_hash=content_hash,
                parser_version="pdf-extractor-v3",
                ocr_config=current_ocr,
                chunking_config=current_chunking,
                embedding_model=system.settings.embedding_model,
                embedding_profile=None,
                embedding_dimension=None,
            )
            ready_same = bool(existing and system.state_store.is_ready_status(existing.get("status")) and existing.get("version_id") == expected)
            if not ready_same:
                DocumentStructureStore(Path(system.state_store.database_path).parent / "structure.sqlite3").delete_document(str(document_id))
        except Exception:
            pass
        return original(system, pdf_path)

    wrapped._final_structure_patched = True
    wrapped._final_structure_original = original
    robust_ingestor.robust_ingest_file = wrapped


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        _patch_extractor()
        _patch_chunker()
        _patch_vector_contract()
        _patch_enhanced_store()
        _patch_retrieval()
        _patch_context()
        _clear_structure_before_rebuild()
        _INSTALLED = True


__all__ = ["install", "VisionFigureAdapter"]
