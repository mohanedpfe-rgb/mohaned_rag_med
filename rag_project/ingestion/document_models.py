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
    normalized_text: str = ""
    entities: Dict[str, List[str]] = field(default_factory=dict)
    headings: List[str] = field(default_factory=list)
    representation_ids: List[str] = field(default_factory=list)


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
