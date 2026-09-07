from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from typing import List

from rag_project.ingestion.document_models import Chunk, PageExtraction
from rag_project.utils.text_utils import split_paragraphs


class SemanticChunker:
    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 120):
        self.chunk_size = max(1, int(chunk_size))
        self.chunk_overlap = max(0, int(chunk_overlap))

    def _flush_chunk(
        self,
        chunks: List[Chunk],
        doc_id: str,
        file_name: str,
        text: str,
        page_numbers: set[int],
        evidence_types: set[str],
    ) -> None:
        if not text or not text.strip():
            return
        chunks.append(
            Chunk(
                doc_id=doc_id,
                file_name=file_name,
                chunk_index=len(chunks),
                text=text.strip(),
                page_numbers=sorted(page_numbers),
                metadata={
                    "source_pages": sorted(page_numbers),
                    "evidence_types": sorted(evidence_types),
                },
            )
        )

    def _split_paragraph(self, paragraph: str) -> list[str]:
        text = paragraph.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?。！？])\s+", text)
            if sentence.strip()
        ]
        tokens = (
            sentences
            if len(sentences) > 1 and max(map(len, sentences), default=0) <= self.chunk_size
            else text.split()
        )
        segments: list[str] = []
        current_tokens: list[str] = []

        for token in tokens:
            candidate = "".join(current_tokens) if not current_tokens else " ".join(current_tokens + [token])
            if len(candidate) <= self.chunk_size or not current_tokens:
                current_tokens.append(token)
                continue
            segments.append(" ".join(current_tokens))
            current_tokens = [token]

        if current_tokens:
            segments.append(" ".join(current_tokens))

        return segments

    def chunk_pages(self, pages: List[PageExtraction]) -> List[Chunk]:
        if not pages:
            return []
        
        # Process each page individually to preserve per-page evidence types
        chunks: List[Chunk] = []
        chunk_idx = 0
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
        headers = [("#", "chapter"), ("##", "section")]
        parent_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        for page in pages:
            # Split the page text into parent sections (e.g., chapters/sections)
            lc_chunks = parent_splitter.split_text(page.text or "")
            for lc in lc_chunks:
                metadata = lc.metadata
                parent_text = lc.page_content

                # Build context prefix from chapter/section metadata
                context_parts = []
                if metadata.get("chapter"):
                    context_parts.append(f"Chapter: {metadata['chapter']}")
                if metadata.get("section"):
                    context_parts.append(f"Section: {metadata['section']}")
                context_prefix = " - ".join(context_parts)
                if context_prefix:
                    contextualized_parent = f"[{context_prefix}]\n{parent_text}"
                else:
                    contextualized_parent = parent_text

                # Determine evidence types for this page
                evidence: set[str] = {"text"}
                if page.has_images:
                    evidence.add("figure")
                if page.table_count > 0:
                    evidence.add("table")

                # Split parent section into child chunks
                child_texts = child_splitter.split_text(parent_text)
                for child_text in child_texts:
                    if context_prefix:
                        child_search_text = f"[{context_prefix}]\n{child_text}"
                    else:
                        child_search_text = child_text

                    chunks.append(
                        Chunk(
                            doc_id=page.document_id,
                            file_name=page.file_name,
                            chunk_index=chunk_idx,
                            text=child_search_text,
                            page_numbers=[page.page_number or 1],
                            metadata={
                                "source_pages": [page.page_number or 1],
                                "evidence_types": sorted(evidence),
                                "chapter": metadata.get("chapter"),
                                "section": metadata.get("section"),
                                "parent_text": contextualized_parent,
                            },
                        )
                    )
                    chunk_idx += 1
        return chunks

    def chunk_page_batches(
        self,
        pages: Iterable[PageExtraction],
        batch_size: int = 16,
    ) -> Iterator[List[Chunk]]:
        """Yield one batch per page while maintaining sequential chunk indices."""
        chunk_index_offset = 0
        for page in pages:
            page_chunks = self.chunk_pages([page])
            if not page_chunks:
                continue
            for chunk in page_chunks:
                chunk.chunk_index = chunk_index_offset
                chunk_index_offset += 1
            yield page_chunks
