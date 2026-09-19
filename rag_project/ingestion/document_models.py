from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PageExtraction:
    document_id: str
    file_name: str
    page_index: int
    page_number: int | None
    text: str
    extraction_method: str
    ocr_required: bool = False
    ocr_status: str = "not_required"
    ocr_confidence: float | None = None
    page_type: str = "unknown"
    image_count: int = 0
    table_count: int = 0
    has_images: bool = False
    blocks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None
    quality_score: float = 0.0
    routing_decision: str = "native"
    table_ids: List[str] = field(default_factory=list)
    figure_ids: List[str] = field(default_factory=list)
    table_texts: List[str] = field(default_factory=list)
    figure_captions: List[str] = field(default_factory=list)
    normalized_text: str = ""
    entities: Dict[str, List[str]] = field(default_factory=dict)
    headings: List[str] = field(default_factory=list)
    representation_ids: List[str] = field(default_factory=list)
    # Phase 15 fix: Add footnote extraction fields
    footnotes: List[str] = field(default_factory=list)
    footnote_ids: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    # Phase 14 fix: Add multi-page table tracking
    table_continuation: bool = False
    table_continuation_id: str | None = None
    multi_page_table_groups: Dict[str, List[str]] = field(default_factory=dict)
    # Phase 12 fix: Add figure-caption association
    figure_caption_associations: Dict[str, str] = field(default_factory=dict)  # figure_id -> caption_id
    # Phase 39 fix: Add document edition/version tracking
    document_edition: str | None = None
    document_version: str | None = None
    publication_year: int | None = None
    isbn: str | None = None
    doi: str | None = None


@dataclass
class Chunk:
    doc_id: str
    file_name: str
    chunk_index: int
    text: str
    page_numbers: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)
    representation_type: str = "canonical"
    parent_id: str | None = None
    section_id: str | None = None
    table_id: str | None = None
    figure_id: str | None = None
    normalized_text: str = ""
