from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import fitz

logger = logging.getLogger(__name__)

from rag_project.ingestion.document_models import PageExtraction
from rag_project.ingestion.state_store import IngestionStateStore
from rag_project.intelligence.pdf_intelligence import enrich_text, score_page_quality
from rag_project.ocr.ocr_service import OCRService
from rag_project.security import validate_pdf_page_count
from rag_project.utils.text_utils import clean_text, extract_page_number, split_paragraphs


class PDFExtractor:
    """PDF extractor using bounded native text/table parsing and isolated OCR services."""
    def __init__(self, state_store: IngestionStateStore | None = None, *, ocr_enabled: bool = True, ocr_confidence_threshold: float = 0.55, ocr_min_char_density: float = 0.001, ocr_image_coverage_threshold: float = 0.55, ocr_service: OCRService | None = None):
        self.ocr_enabled = bool(ocr_enabled)
        self.ocr_confidence_threshold = max(0.0, float(ocr_confidence_threshold))
        self.ocr_min_char_density = max(0.0, float(ocr_min_char_density))
        self.ocr_image_coverage_threshold = max(0.0, float(ocr_image_coverage_threshold))
        self.ocr_service = ocr_service or OCRService(use_rapidocr=True, lazy_init=True)
        self.state_store = state_store

    @staticmethod
    def _normalize_for_dedupe(value: str) -> str:
        return re.sub(r"\s+", " ", (value or "").casefold()).strip()

    @classmethod
    def _looks_duplicate(cls, existing: str, candidate: str) -> bool:
        a = cls._normalize_for_dedupe(existing); b = cls._normalize_for_dedupe(candidate)
        if not a or not b: return False
        if a == b or a in b or b in a: return True
        at, bt = set(re.findall(r"\w+", a)), set(re.findall(r"\w+", b))
        return bool(at and bt) and len(at & bt) / max(len(at | bt), 1) >= 0.8

    @classmethod
    def _merge_native_and_ocr_text(cls, native_text: str, ocr_text: str) -> str:
        parts=[p.strip() for p in split_paragraphs(native_text or "") if p.strip()]
        parts.extend(p.strip() for p in split_paragraphs(ocr_text or "") if p.strip())
        merged=[]
        for part in parts:
            if not any(cls._looks_duplicate(existing, part) for existing in merged): merged.append(part)
        return clean_text("\n\n".join(merged)) if merged else clean_text(native_text or ocr_text or "")

    @staticmethod
    def _safe_image_stats(page: fitz.Page) -> tuple[int, float]:
        try: images=page.get_images(full=True)
        except (ValueError,RuntimeError,AttributeError): return 0,0.0
        coverage=0.0
        for img in images:
            try:
                for box in page.get_image_rects(img[0]): coverage += box.width*box.height
            except Exception: continue
        return len(images),coverage

    @classmethod
    def _assess_page_ocr_need(cls,page:fitz.Page,text:str,*,min_char_count:int=120,min_alpha_count:int=40,min_word_count:int=25,image_coverage_threshold:float=0.55,image_heavy_word_limit:int=30,min_char_density:float=0.0008)->dict[str,Any]:
        word_count=len((text or "").split()); char_count=len(text or ""); alpha_count=sum(1 for c in text if unicodedata.category(c).startswith(("L","N")))
        image_count,image_coverage_pixels=cls._safe_image_stats(page); area=page.rect.width*page.rect.height if page.rect.width and page.rect.height else 1.0
        coverage=image_coverage_pixels/area; density=char_count/area; reasons=[]
        if char_count<min_char_count: reasons.append(f"few_chars ({char_count})")
        if alpha_count<min_alpha_count and word_count<min_word_count: reasons.append(f"sparse_text (alpha={alpha_count}, words={word_count})")
        if coverage>image_coverage_threshold and word_count<image_heavy_word_limit: reasons.append(f"image_heavy (coverage={coverage:.2f})")
        if char_count>=min_char_count:
            printable=sum(1 for c in text if c.isprintable() or c in "\n\r\t")/max(char_count,1)
            if printable<0.85: reasons.append("garbled_text_layer")
        if density<min_char_density and word_count<min_word_count: reasons.append(f"low_char_density ({density:.5f})")
        ocr_required=bool(reasons)
        page_type="image_heavy" if image_count and word_count<60 else ("scanned_or_ocr_required" if ocr_required else ("sparse_text" if word_count<100 else "text_based"))
        return {"ocr_required":ocr_required,"reasons":reasons,"page_type":page_type,"word_count":word_count,"char_count":char_count,"alpha_count":alpha_count,"image_count":image_count,"has_images":image_count>0,"image_coverage":float(coverage),"char_density":float(density)}

    @staticmethod
    def _load_page(pdf: fitz.Document,index:int)->fitz.Page:
        try:return pdf.load_page(index)
        except Exception:return pdf[index]

    def extract(self,pdf_path:str|Path,document_id:str|None=None)->list[PageExtraction]: return list(self.extract_iter(pdf_path,document_id))

    def extract_iter(self,pdf_path:str|Path,document_id:str|None=None)->Iterator[PageExtraction]:
        pdf_file=Path(pdf_path); document_id=document_id or str(uuid.uuid4()); pdf=None
        try:
            try: pdf=fitz.open(str(pdf_file))
            except RuntimeError as exc: raise RuntimeError(f"{pdf_file.name}: failed to open PDF") from exc
            page_count=validate_pdf_page_count(pdf.page_count)
            cached_pages={x["page_number"]:x for x in self.state_store.get_pages(document_id)} if self.state_store else {}; last_progress=0
            for index in range(page_count):
                physical_page=index+1; cached=cached_pages.get(physical_page) if self.state_store else None
                if cached and cached["extraction_status"]=="COMPLETED" and cached.get("text"):
                    text=cached["text"]; quality=score_page_quality(text,page_number=physical_page); meta=dict(cached.get("metadata") or {}); meta.update({"physical_page":physical_page,"cached":True,"quality_score":quality.quality,"routing_decision":quality.route,**enrich_text(text)})
                    yield PageExtraction(document_id,pdf_file.name,index,physical_page,text,cached.get("extraction_method") or "cached",ocr_status=cached.get("ocr_status","not_required"),page_type=meta.get("page_type","unknown"),image_count=int(meta.get("image_count") or 0),table_count=int(meta.get("table_count") or 0),has_images=bool(meta.get("has_images")),blocks=[p.strip() for p in split_paragraphs(text) if p.strip()],metadata=meta,source_path=str(pdf_file),quality_score=quality.quality,routing_decision=quality.route,normalized_text=meta.get("normalized_text",""),entities=meta.get("entities",{}),headings=meta.get("headings",[]))
                    if self.state_store and (physical_page%4==0 or physical_page==page_count) and physical_page!=last_progress:
                        self.state_store.update_document(document_id,current_stage="EXTRACTING",current_page=physical_page); last_progress=physical_page
                    continue
                try: page=self._load_page(pdf,index); raw_text=self._extract_page_text(page)
                except Exception: page=self._load_page(pdf,index); raw_text=""
                text=clean_text(raw_text); assessment=self._assess_page_ocr_need(page,text,image_coverage_threshold=self.ocr_image_coverage_threshold,min_char_density=self.ocr_min_char_density); tables_text=self._extract_tables(page)
                if tables_text:text=clean_text(f"{text}\n\n[TABLE]\n{tables_text}")
                enrichment=enrich_text(text); quality=score_page_quality(text,page_number=physical_page,image_count=assessment["image_count"]); printed_page_number=extract_page_number(text)
                figure_ids=[f"{document_id}:p{physical_page}:figure:{i+1}" for i in range(assessment["image_count"])]
                table_blocks=[x for x in tables_text.split("\n\n") if x.strip()] if tables_text else []; table_ids=[f"{document_id}:p{physical_page}:table:{i+1}" for i in range(len(table_blocks))]
                extraction=PageExtraction(document_id=document_id,file_name=pdf_file.name,page_index=index,page_number=physical_page,text=text,extraction_method="pdf_text",ocr_required=assessment["ocr_required"],ocr_status="not_required",page_type=assessment["page_type"],image_count=assessment["image_count"],table_count=len(table_blocks),has_images=assessment["has_images"],blocks=[p.strip() for p in split_paragraphs(text) if p.strip()],metadata={"word_count":assessment["word_count"],"char_count":assessment["char_count"],"alpha_count":assessment["alpha_count"],"image_coverage":round(assessment["image_coverage"],4),"char_density":round(assessment["char_density"],6),"physical_page":physical_page,"printed_page_number":printed_page_number,"ocr_reasons":assessment["reasons"],"table_ids":table_ids,"figure_ids":figure_ids,"evidence_types":(["text"] if text else [])+(["table"] if table_blocks else [])+(["figure"] if figure_ids else []),"quality_score":quality.quality,"routing_decision":quality.route,**enrichment},source_path=str(pdf_file),quality_score=quality.quality,routing_decision=quality.route,table_ids=table_ids,figure_ids=figure_ids,normalized_text=enrichment["normalized_text"],entities=enrichment["entities"],headings=enrichment["headings"])
                if extraction.ocr_required:
                    if not self.ocr_enabled: extraction.ocr_status="skipped_disabled"; extraction.metadata["ocr_disabled"]=True
                    else:
                        try:
                            ocr_text,confidence=self.ocr_service.ocr_page_object(page,index,force=True)
                            if not ocr_text: extraction.ocr_status="skipped_no_result"
                            elif confidence is not None and confidence<self.ocr_confidence_threshold: extraction.ocr_status="skipped_low_confidence"; extraction.metadata["ocr_confidence"]=float(confidence)
                            else:
                                ocr_text=clean_text(ocr_text); native=extraction.text.strip(); final=self._merge_native_and_ocr_text(native,ocr_text)
                                if final!=native: extraction.text=final; extraction.extraction_method="ocr"; extraction.blocks=[p.strip() for p in split_paragraphs(final) if p.strip()]; extraction.metadata.update(enrich_text(final))
                                extraction.ocr_status="success"; extraction.ocr_confidence=confidence; extraction.ocr_required=True; extraction.metadata["ocr_char_count"]=len(ocr_text)
                        except Exception as exc: extraction.ocr_status="failed"; extraction.metadata["ocr_error"]=type(exc).__name__; extraction.metadata["quality_warning"]="Page requires OCR but OCR failed."
                if self.state_store and self.state_store.get_document(document_id):
                    self.state_store.upsert_page(document_id,physical_page,extraction_status="COMPLETED" if extraction.text else "FAILED",ocr_status=extraction.ocr_status,extraction_method=extraction.extraction_method,text=extraction.text,processing_error=extraction.metadata.get("ocr_error"),checksum=hashlib.sha256(extraction.text.encode("utf-8")).hexdigest())
                    if physical_page%4==0 or physical_page==page_count:
                        self.state_store.update_document(document_id,current_stage="EXTRACTING",current_page=physical_page); last_progress=physical_page
                yield extraction
        finally:
            if pdf is not None: pdf.close()

    @staticmethod
    def _extract_page_text(page:fitz.Page)->str:
        blocks=page.get_text("blocks",sort=True)
        if blocks:return "\n\n".join(str(block[4]).strip() for block in blocks if len(block)>4 and str(block[4]).strip())
        return page.get_text("text",sort=True)

    @staticmethod
    def _extract_tables(page:fitz.Page)->str:
        find_tables=getattr(page,"find_tables",None)
        if find_tables is None:return ""
        try:
            rendered=[]
            for table in getattr(find_tables(),"tables",[]) or []:
                rows=table.extract(); lines=[" | ".join(clean_text(str(cell) if cell is not None else "") for cell in row) for row in rows or []]; rendered.append("\n".join(line for line in lines if line.strip()))
            return "\n\n".join(x for x in rendered if x.strip())
        except (RuntimeError,ValueError,AttributeError): return ""

    def extract_markdown(self,pdf_path:str|Path)->str:
        """Return safe native text; intentionally avoids the unbounded pymupdf4llm helper."""
        return "\n\n".join(page.text for page in self.extract_iter(pdf_path) if page.text)
