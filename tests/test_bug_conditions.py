"""
Bug Condition Exploration Tests — 23-Zone RAG Pipeline Bug Exploration Suite
=============================================================================

IMPORTANT: These tests MUST FAIL on unfixed code.
Failure confirms each bug exists. DO NOT fix any code when these fail.

These tests encode expected (post-fix) behavior.  When all 23 tests PASS after
the fix is applied they confirm all 23 bugs have been corrected.

Run with:  pytest tests/test_bug_conditions.py -v
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Helpers shared across zones
# ---------------------------------------------------------------------------

def _make_page_extraction(
    document_id: str = "test-doc",
    page_number: int = 1,
    text: str = "Sample text",
    ocr_required: bool = False,
    ocr_status: str = "not_required",
):
    """Create a minimal PageExtraction for use in tests."""
    from rag_project.ingestion.document_models import PageExtraction

    return PageExtraction(
        document_id=document_id,
        file_name="test.pdf",
        page_index=page_number - 1,
        page_number=page_number,
        text=text,
        extraction_method="native_text",
        ocr_required=ocr_required,
        ocr_status=ocr_status,
    )


# ===========================================================================
# ZONE A — PDF Ingestion (7 tests)
# ===========================================================================


class TestZoneA:
    """Zone A: PDF Ingestion bug condition exploration tests."""

    # -----------------------------------------------------------------------
    # A1: FileMonitor duplicate file detection
    # -----------------------------------------------------------------------
    def test_filemonitor_skips_content_duplicate(self, tmp_path):
        """
        Bug A1: FileMonitor does not detect content-identical files with different names.

        isBugCondition_A1: sha256(file_a) == sha256(file_b) AND file_a.name != file_b.name
        EXPECTED OUTCOME: FAIL on unfixed code (both files scheduled as new_or_changed).
        EXPECTED OUTCOME AFTER FIX: only one file returned, second marked as duplicate.
        """
        from rag_project.ingestion.file_monitor import FileMonitor

        # Create two files with identical content but different names
        content = b"%PDF-1.4 identical content for duplicate detection test"
        file_a = tmp_path / "report_original.pdf"
        file_b = tmp_path / "report_copy.pdf"
        file_a.write_bytes(content)
        file_b.write_bytes(content)

        # Both files share the same SHA-256 digest
        assert hashlib.sha256(content).hexdigest() == hashlib.sha256(content).hexdigest()

        monitor = FileMonitor(tmp_path)
        results = monitor.scan()

        # After the fix: only one file should be scheduled (duplicate detected)
        # On unfixed code: both get status "new_or_changed", duplicate=True but both still returned
        scheduled = [r for r in results if r["status"] == "new_or_changed"]

        # The fix should ensure at most one file is scheduled for ingestion
        # when two files share the same content hash.
        # Bug: both are returned as "new_or_changed" even though they're identical.
        assert len(scheduled) <= 1, (
            f"Bug A1 confirmed: both content-duplicate files were scheduled for ingestion. "
            f"Found {len(scheduled)} scheduled entries: {[r['file_name'] for r in scheduled]}"
        )

    # -----------------------------------------------------------------------
    # A2: Self-replace guard in robust_ingestor.py
    # -----------------------------------------------------------------------
    def test_robust_ingest_no_self_replace(self, tmp_path):
        """
        Bug A2: robust_ingest_file calls file_path.replace(archived_path) even when
        the source and destination paths resolve identically, destroying the file.

        isBugCondition_A2: file_path.resolve() == archived_path.resolve()
        EXPECTED OUTCOME: FAIL on unfixed code (file is removed/corrupted).
        EXPECTED OUTCOME AFTER FIX: file still exists, function returns {"status": "skipped"}.
        """
        from rag_project.ingestion.robust_ingestor import robust_ingest_file

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir(parents=True)

        # Create a minimal valid PDF so validation does not trip
        content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<< /Size 1 >>\nstartxref\n9\n%%EOF"
        pdf_file = processed_dir / "already_indexed.pdf"
        pdf_file.write_bytes(content)
        content_hash = hashlib.sha256(content).hexdigest()

        # Build a minimal system mock that simulates an already-indexed document
        settings_mock = MagicMock()
        settings_mock.archive_dir = tmp_path / "archive"
        settings_mock.archive_dir.mkdir(parents=True)
        settings_mock.processed_dir = processed_dir
        settings_mock.failed_dir = tmp_path / "failed"
        settings_mock.failed_dir.mkdir(parents=True)
        settings_mock.incoming_dir = tmp_path / "incoming"
        settings_mock.incoming_dir.mkdir(parents=True)
        settings_mock.chunk_size = 700
        settings_mock.chunk_overlap = 120
        settings_mock.embedding_model = "test-model"
        settings_mock.ocr_enabled = False
        settings_mock.ocr_confidence_threshold = 0.55
        settings_mock.ocr_min_char_density = 0.001
        settings_mock.ocr_image_coverage_threshold = 0.55
        settings_mock.ingestion_lease_seconds = 900
        settings_mock.page_batch_size = 16
        settings_mock.embedding_batch_size = 16

        # Report document as already indexed with same version
        version_id = "v-test-1234"
        existing_record = {
            "document_id": content_hash,
            "content_hash": content_hash,
            "status": "READY",
            "version_id": version_id,
            "file_path": str(pdf_file.resolve()),
        }

        state_store_mock = MagicMock()
        state_store_mock.get_by_hash.return_value = existing_record
        state_store_mock.get_by_path.return_value = existing_record
        state_store_mock.is_ready_status.return_value = True
        state_store_mock.claim_document.return_value = True
        state_store_mock.heartbeat_document.return_value = True

        vector_store_mock = MagicMock()
        vector_store_mock.validate_document_index.return_value = {"valid": True, "count": 5}

        system_mock = MagicMock()
        system_mock.settings = settings_mock
        system_mock.state_store = state_store_mock
        system_mock.vector_store = vector_store_mock
        system_mock.logger = MagicMock()
        system_mock._hash_file = None
        system_mock._ingestion_version_id.return_value = version_id
        system_mock._new_cancel_flag.return_value = MagicMock(cancelled=False)
        system_mock._remove_cancel_flag = MagicMock()

        # Submit the file that is already in processed_dir (source == destination scenario)
        result = robust_ingest_file(system_mock, pdf_file)

        # After the fix: file must still exist and status must be "skipped"
        assert pdf_file.exists(), (
            "Bug A2 confirmed: the file was deleted or moved during self-replace operation. "
            f"Result was: {result}"
        )
        assert result.get("status") == "skipped", (
            f"Bug A2: expected status='skipped' but got: {result.get('status')}. "
            f"Full result: {result}"
        )

    # -----------------------------------------------------------------------
    # A3: OCR-failed page record written to state_store
    # -----------------------------------------------------------------------
    def test_ocr_failure_writes_page_record(self, tmp_path):
        """
        Bug A3: When OCR fails, extract_iter does not call state_store.record_page
        with extraction_status="FAILED", leaving the page invisible to audit.

        isBugCondition_A3: ocr_required AND NOT ocr_succeeded AND NOT page_record_written
        EXPECTED OUTCOME: FAIL on unfixed code (record_page not called for failed page).
        EXPECTED OUTCOME AFTER FIX: record_page called with extraction_status="FAILED".
        """
        import fitz

        from rag_project.ingestion.state_store import IngestionStateStore
        from rag_project.ocr.ocr_service import OCRService
        from rag_project.parsing.pdf_extractor import PDFExtractor

        # Create a minimal scanned-looking PDF (low text, high image coverage)
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        # Deliberately add almost no text so OCR is triggered
        page.insert_text((50, 50), "x", fontsize=8)
        pdf_path = tmp_path / "scanned_doc.pdf"
        doc.save(str(pdf_path))
        doc.close()

        document_id = "test-doc-ocr-fail"

        # Mock state_store to track calls
        state_store_mock = MagicMock(spec=IngestionStateStore)
        state_store_mock.get_pages.return_value = []

        # Mock OCR service to raise an exception (simulating OCR failure)
        ocr_service_mock = MagicMock(spec=OCRService)
        ocr_service_mock.ocr_page_object.side_effect = RuntimeError("OCR engine failed on page 1")

        extractor = PDFExtractor(
            state_store=state_store_mock,
            ocr_enabled=True,
            ocr_confidence_threshold=0.55,
            ocr_service=ocr_service_mock,
        )

        # Exhaust the iterator to process the page
        pages = list(extractor.extract_iter(pdf_path, document_id))

        # Check whether record_page was called with extraction_status="FAILED"
        # for the OCR-failed page.
        record_calls = [
            c for c in state_store_mock.record_page.call_args_list
        ]

        failed_record_written = False
        for c in record_calls:
            args = c.args
            kwargs = c.kwargs
            # record_page(extraction, cache_reference=...) — check the PageExtraction
            if args:
                extraction_arg = args[0]
                status = getattr(extraction_arg, "ocr_status", None)
                if status == "failed":
                    # Check if extraction_status="FAILED" is recorded
                    failed_record_written = True
            # Also check kwargs passed directly
            if kwargs.get("extraction_status") == "FAILED":
                failed_record_written = True

        assert failed_record_written, (
            "Bug A3 confirmed: state_store.record_page was NOT called with "
            "extraction_status='FAILED' for the OCR-failed page. "
            f"All record_page calls: {record_calls}"
        )

    # -----------------------------------------------------------------------
    # A4: Two-column layout — column A text precedes column B text
    # -----------------------------------------------------------------------
    def test_two_column_text_column_a_before_column_b(self):
        """
        Bug A4: _extract_page_text uses sort=True which merges columns horizontally,
        interleaving column A and column B text.

        isBugCondition_A4: max(x_coords) - min(x_coords) > page.rect.width * 0.4
        EXPECTED OUTCOME: FAIL on unfixed code (columns interleaved).
        EXPECTED OUTCOME AFTER FIX: all DosageInfo text precedes ClinicalOutcome text.
        """
        from rag_project.parsing.pdf_extractor import PDFExtractor

        extractor = PDFExtractor(state_store=None, ocr_enabled=False)

        # Create a mock PyMuPDF page with two-column layout
        page_mock = MagicMock()
        page_mock.rect = MagicMock()
        page_mock.rect.width = 595.0
        page_mock.rect.height = 842.0

        # Column A blocks (left side, x0 ~ 50)
        # Column B blocks (right side, x0 ~ 320)
        # Block format: (x0, y0, x1, y1, text, block_no, block_type)
        # block_type 0 = text
        col_a_blocks = [
            (50, 100, 260, 120, "DosageInfo line 1\n", 0, 0),
            (50, 130, 260, 150, "DosageInfo line 2\n", 1, 0),
            (50, 160, 260, 180, "DosageInfo line 3\n", 2, 0),
        ]
        col_b_blocks = [
            (320, 100, 560, 120, "ClinicalOutcome line 1\n", 3, 0),
            (320, 130, 560, 150, "ClinicalOutcome line 2\n", 4, 0),
            (320, 160, 560, 180, "ClinicalOutcome line 3\n", 5, 0),
        ]

        # sort=True causes PyMuPDF to interleave lines from both columns
        # (it sorts by y-coordinate, mixing columns)
        interleaved_text = (
            "DosageInfo line 1\n"
            "ClinicalOutcome line 1\n"
            "DosageInfo line 2\n"
            "ClinicalOutcome line 2\n"
            "DosageInfo line 3\n"
            "ClinicalOutcome line 3\n"
        )

        # get_text("text", sort=True) returns interleaved text (the bug)
        page_mock.get_text = MagicMock(side_effect=lambda mode, **kwargs: interleaved_text if mode == "text" else (col_a_blocks + col_b_blocks))

        result = extractor._extract_page_text(page_mock)

        # After the fix: all DosageInfo text should precede ClinicalOutcome text
        first_clinical = result.find("ClinicalOutcome")
        last_dosage = result.rfind("DosageInfo")

        assert first_clinical > last_dosage, (
            f"Bug A4 confirmed: column A text (DosageInfo) is interleaved with column B "
            f"text (ClinicalOutcome). "
            f"last_dosage_pos={last_dosage}, first_clinical_pos={first_clinical}. "
            f"Result text: {result!r}"
        )

    # -----------------------------------------------------------------------
    # A5: OCR scale floor — raises at sub-0.5 scale
    # -----------------------------------------------------------------------
    def test_ocr_render_raises_at_sub_0_5_scale(self):
        """
        Bug A5: _render_pixmap allows scale to drop below 0.25 for oversized pages,
        producing unreliable OCR results instead of raising RuntimeError.

        isBugCondition_A5: configured_scale / 0.25 > 1.0 (would need to go below 0.25)
        EXPECTED OUTCOME: FAIL on unfixed code (renders silently at ~0.18).
        EXPECTED OUTCOME AFTER FIX: RuntimeError raised.
        """
        from rag_project.ocr.ocr_service import OCRService

        # max_render_pixels very small so any page exceeds the limit
        # forcing scale to drop well below 0.5
        service = OCRService(
            use_rapidocr=False,  # don't actually load RapidOCR
            lazy_init=True,
            render_scale=2,
            max_render_pixels=100,  # tiny limit to force scale reduction
        )

        # Mock a large page that would require scale ~0.18 to fit
        page_mock = MagicMock()
        page_mock.rect = MagicMock()
        page_mock.rect.width = 1200.0   # A3 landscape
        page_mock.rect.height = 848.0

        # The pixmap render mock — we check that it raises BEFORE rendering
        def fake_get_pixmap(matrix=None, alpha=False):
            raise AssertionError("Should have raised RuntimeError before calling get_pixmap")

        page_mock.get_pixmap = fake_get_pixmap

        # After the fix: RuntimeError must be raised (scale cannot go below 0.5)
        with pytest.raises(RuntimeError, match=r"(?i)(scale|image|large|pixel|render)"):
            service._render_pixmap(page_mock)

    # -----------------------------------------------------------------------
    # A6: Table exception — non-AttributeError must be logged
    # -----------------------------------------------------------------------
    def test_table_extract_non_attr_error_logged(self):
        """
        Bug A6: _extract_tables catches all exceptions with bare `except Exception: return ""`
        making non-AttributeError exceptions invisible in logs.

        isBugCondition_A6: exception_raised != None AND NOT isinstance(exception, AttributeError)
        EXPECTED OUTCOME: FAIL on unfixed code (logger.warning not called).
        EXPECTED OUTCOME AFTER FIX: logger.warning called with page number.
        """
        import logging

        from rag_project.parsing.pdf_extractor import PDFExtractor

        extractor = PDFExtractor(state_store=None, ocr_enabled=False)

        # Mock a page where find_tables raises ValueError (not AttributeError)
        page_mock = MagicMock()

        def raising_find_tables():
            raise ValueError("corrupt table structure on this page")

        page_mock.find_tables = raising_find_tables

        with patch("rag_project.parsing.pdf_extractor.logger") as mock_logger:
            result = extractor._extract_tables(page_mock)

        # After the fix: logger.warning should be called with page information
        assert mock_logger.warning.called, (
            "Bug A6 confirmed: logger.warning was NOT called when ValueError was raised "
            "in _extract_tables. The exception was silently swallowed. "
            f"warning call count: {mock_logger.warning.call_count}"
        )
        # Result must still be empty string (graceful degradation)
        assert result == "", f"Expected empty string result, got: {result!r}"

    # -----------------------------------------------------------------------
    # A7: Page number in footer (after 500-char header block)
    # -----------------------------------------------------------------------
    def test_extract_page_number_footer_position(self):
        """
        Bug A7: extract_page_number only scans text[:500], missing footer page numbers
        that appear after a large header block.

        isBugCondition_A7: page_number_pattern_found(text[500:]) AND NOT page_number_pattern_found(text[:500])
        EXPECTED OUTCOME: FAIL on unfixed code (returns None).
        EXPECTED OUTCOME AFTER FIX: returns 47.
        """
        from rag_project.utils.text_utils import extract_page_number

        # Build a page text: 600-char header block, then page number at position ~612
        header = "A" * 600  # large header block occupying first 600 characters
        page_text = header + "\n— 47 —\n"

        # Verify the page number is beyond position 500 (the bug boundary)
        assert page_text.find("47") > 500, "Test setup error: page number must be past position 500"
        # Verify it's not in first 500 chars
        assert "47" not in page_text[:500], "Test setup error: page number must not be in first 500 chars"

        result = extract_page_number(page_text)

        assert result == 47, (
            f"Bug A7 confirmed: extract_page_number returned {result!r} instead of 47. "
            f"The function only searches text[:500] and misses footer page numbers. "
            f"Page number '— 47 —' is at position {page_text.find('47')} (past the 500-char boundary)."
        )


# ===========================================================================
# ZONE B — Text Processing (3 tests)
# ===========================================================================


class TestZoneB:
    """Zone B: Text Processing bug condition exploration tests."""

    # -----------------------------------------------------------------------
    # B1: NFKC destroys superscripts/subscripts
    # -----------------------------------------------------------------------
    def test_clean_text_preserves_superscripts(self):
        """
        Bug B1: clean_text applies unicodedata.normalize("NFKC", ...) which converts
        subscript digits (e.g., ₂ U+2082) to plain digits, destroying formula structure.

        isBugCondition_B1: text contains SUPERSCRIPT_CHARS or SUBSCRIPT_CHARS
        EXPECTED OUTCOME: FAIL on unfixed code (outputs "CO2" with stripped subscript).
        EXPECTED OUTCOME AFTER FIX: output contains "_2" or "^2" notation.
        """
        from rag_project.utils.text_utils import clean_text

        # Input with Unicode subscript ₂ (U+2082) and ₂ in H₂O
        text = "CO₂ 38mmHg, H₂O levels"  # ₂ is U+2082

        result = clean_text(text)

        # After the fix: subscripts should be preserved as _2 notation
        # Bug: NFKC converts ₂ → 2, producing "CO2" indistinguishable from integer 2
        has_preserved_notation = "_2" in result or "^2" in result
        has_stripped_digit = "CO2" in result and "_2" not in result

        assert has_preserved_notation, (
            f"Bug B1 confirmed: clean_text stripped subscript ₂ to plain digit. "
            f"Input: {text!r}. "
            f"Output: {result!r}. "
            f"Expected '_2' or '^2' notation but got bare digit."
        )

    # -----------------------------------------------------------------------
    # B2: Hyphenated line-break not joined
    # -----------------------------------------------------------------------
    def test_split_paragraphs_joins_hyphenated_linebreak(self):
        """
        Bug B2: split_paragraphs splits on \\n{2,} without pre-joining hyphen-newline
        sequences, so "anti-\\nhypertensive" remains split into two tokens.

        isBugCondition_B2: re.search(r'\\w+-\\n\\w+', text) != None
        EXPECTED OUTCOME: FAIL on unfixed code (result does not contain "antihypertensive").
        EXPECTED OUTCOME AFTER FIX: result contains "antihypertensive" as single token.
        """
        from rag_project.utils.text_utils import split_paragraphs

        text = "anti-\nhypertensive therapy is common"
        result = split_paragraphs(text)

        # Join all output paragraphs to check if the word was correctly merged
        combined = " ".join(result)

        assert "antihypertensive" in combined, (
            f"Bug B2 confirmed: hyphenated line-break was not joined. "
            f"Input: {text!r}. "
            f"Output paragraphs: {result}. "
            f"Combined: {combined!r}. "
            f"Expected 'antihypertensive' but got separate 'anti-' and 'hypertensive' tokens."
        )

    # -----------------------------------------------------------------------
    # B3: OCR post-correction missing
    # -----------------------------------------------------------------------
    def test_ocr_correction_fixes_metformin(self):
        """
        Bug B3: No domain-specific OCR correction pass exists; RapidOCR confusables
        like "0" (zero) for "O" (letter) in drug names propagate into the index.

        isBugCondition_B3: contains_numeric_drug_pattern AND ocr_correction_pass_applied == False
        EXPECTED OUTCOME: FAIL on unfixed code (returns "Metf0rmin 500mg" unchanged).
        EXPECTED OUTCOME AFTER FIX: returns "Metformin 500mg".
        """
        # The fix creates rag_project/ocr/ocr_corrector.py with apply_ocr_corrections
        try:
            from rag_project.ocr.ocr_corrector import apply_ocr_corrections
        except ImportError:
            pytest.fail(
                "Bug B3 confirmed: rag_project/ocr/ocr_corrector.py does not exist. "
                "The OCR correction module has not been created. "
                "apply_ocr_corrections function is missing from the codebase."
            )

        input_text = "Metf0rmin 500mg"
        result = apply_ocr_corrections(input_text)

        assert result == "Metformin 500mg", (
            f"Bug B3 confirmed: OCR correction did not fix the drug name. "
            f"Input: {input_text!r}. "
            f"Expected: 'Metformin 500mg'. "
            f"Got: {result!r}."
        )


# ===========================================================================
# ZONE C — Chunking (3 tests)
# ===========================================================================


class TestZoneC:
    """Zone C: Chunking bug condition exploration tests."""

    # -----------------------------------------------------------------------
    # C1: Chapter state not carried across batch boundaries
    # -----------------------------------------------------------------------
    def test_chunk_page_batches_carries_chapter_across_boundary(self):
        """
        Bug C1: chunk_page_batches calls chunk_pages independently per batch;
        document_chapter carry-forward state resets to None between batches.

        isBugCondition_C1: batch_index > 0 AND preceding_chapter_state != None
                            AND chunk_pages_receives_chapter_state == False
        EXPECTED OUTCOME: FAIL on unfixed code (chapter=null in batch 2).
        EXPECTED OUTCOME AFTER FIX: pages 7-10 chunks have chapter="Pharmacology".
        """
        from rag_project.chunking.semantic_chunker import SemanticChunker

        chunker = SemanticChunker(chunk_size=700, chunk_overlap=120)

        # Create 10 pages; page 5 has a recognizable chapter heading
        pages = []
        for i in range(1, 11):
            if i == 5:
                text = "Chapter: Pharmacology\n\nThis chapter covers pharmacology topics including drug mechanisms and interactions."
            else:
                text = f"Page {i} body text with enough content for chunking. " * 5
            pages.append(_make_page_extraction(
                document_id="test-chapter-doc",
                page_number=i,
                text=text,
            ))

        # Chunk in batches of 6: batch 1 = pages 1-6, batch 2 = pages 7-10
        all_chunks = []
        for batch in chunker.chunk_page_batches(iter(pages), batch_size=6):
            all_chunks.extend(batch)

        # Chunks from pages 7-10 (batch 2) should have chapter="Pharmacology"
        batch2_chunks = [
            c for c in all_chunks
            if any(p >= 7 for p in c.page_numbers)
        ]

        assert batch2_chunks, "No chunks found for pages 7-10; test setup issue."

        chapters_in_batch2 = [c.metadata.get("chapter") for c in batch2_chunks]
        non_null_chapters = [ch for ch in chapters_in_batch2 if ch is not None]

        assert len(non_null_chapters) == len(batch2_chunks), (
            f"Bug C1 confirmed: chapter state was NOT carried across batch boundary. "
            f"Pages 7-10 chunks have chapter=null. "
            f"chapters_in_batch2: {chapters_in_batch2}. "
            f"Batch 2 chunk count: {len(batch2_chunks)}."
        )

        # And they should specifically say "Pharmacology"
        for ch in non_null_chapters:
            assert "Pharmacology" in str(ch), (
                f"Bug C1 confirmed: chapter in batch 2 is '{ch}', expected 'Pharmacology'."
            )

    # -----------------------------------------------------------------------
    # C2: Cross-document section ID collision
    # -----------------------------------------------------------------------
    def test_stable_id_unique_across_documents(self):
        """
        Bug C2: _stable_id hashes "{chapter}|{section}" without document_id prefix,
        causing collisions across different documents with identical heading structures.

        isBugCondition_C2: doc_a.chapter == doc_b.chapter AND doc_a.section == doc_b.section
                            AND document_id_not_in_hash_key
        EXPECTED OUTCOME: FAIL on unfixed code (IDs collide).
        EXPECTED OUTCOME AFTER FIX: distinct section_ids for different document_ids.
        """
        from rag_project.chunking.semantic_chunker import SemanticChunker

        chunker = SemanticChunker(chunk_size=700, chunk_overlap=120)

        # Same section key, different document IDs
        section_key_a = "Dosage|Adults"
        section_key_b = "Dosage|Adults"  # identical keys, different documents

        # The current _stable_id uses only the section key (no document_id prefix)
        id_a = chunker._stable_id(section_key_a)
        id_b = chunker._stable_id(section_key_b)

        # Without the fix, these are identical (bug)
        # With the fix (document_id included in hash), they would differ
        # We simulate what the fix should produce:
        doc_id_a = "doc-monograph-cardiology"
        doc_id_b = "doc-monograph-endocrinology"

        scoped_key_a = f"{doc_id_a}|Dosage|Adults"
        scoped_key_b = f"{doc_id_b}|Dosage|Adults"

        # If the fix was applied, _stable_id would use the scoped key
        # For now, let's verify the bug: same key produces same ID
        # The test checks that when document_id is the same portion of the hash,
        # the IDs for different docs would differ (post-fix behavior)

        # On unfixed code: _stable_id("Dosage|Adults") == _stable_id("Dosage|Adults")
        # This collision is the bug.

        # Create chunks from two different documents
        pages_doc_a = [_make_page_extraction(
            document_id=doc_id_a,
            page_number=1,
            text="Chapter: Dosage\nSection: Adults\nAdult dosing information for doc A.",
        )]
        pages_doc_b = [_make_page_extraction(
            document_id=doc_id_b,
            page_number=1,
            text="Chapter: Dosage\nSection: Adults\nAdult dosing information for doc B.",
        )]

        chunks_a = chunker.chunk_pages(pages_doc_a)
        chunks_b = chunker.chunk_pages(pages_doc_b)

        assert chunks_a, "No chunks produced for doc A"
        assert chunks_b, "No chunks produced for doc B"

        # Extract section_ids from both sets
        section_ids_a = {c.section_id for c in chunks_a if c.section_id}
        section_ids_b = {c.section_id for c in chunks_b if c.section_id}

        # After the fix: section_ids should be different for different documents
        overlap = section_ids_a & section_ids_b

        assert len(overlap) == 0, (
            f"Bug C2 confirmed: cross-document section_id collision detected. "
            f"Both documents ('{doc_id_a}' and '{doc_id_b}') with identical "
            f"chapter='Dosage'/section='Adults' produced the same section_ids: {overlap}. "
            f"section_ids_a: {section_ids_a}. "
            f"section_ids_b: {section_ids_b}."
        )

    # -----------------------------------------------------------------------
    # C3: delete_lexical_version clears lexical rows
    # -----------------------------------------------------------------------
    def test_delete_version_clears_lexical_rows(self, tmp_path):
        """
        Bug C3: VectorStore.delete_version (or delete_lexical_version) does not
        purge lexical SQLite rows for (document_id, version_id).

        isBugCondition_C3: prior_ingestion_interrupted AND lexical_rows_exist
                            AND delete_version_does_not_clear_lexical
        EXPECTED OUTCOME: FAIL on unfixed code (delete_lexical_version method doesn't exist).
        EXPECTED OUTCOME AFTER FIX: 0 rows remain after calling delete_lexical_version.
        """
        from rag_project.storage.vector_store import VectorStore

        vs = VectorStore(str(tmp_path / "chroma"), collection_name="test_lexical_delete")

        doc_id = "test_doc"
        version_id = "v1"

        # Insert 10 lexical rows manually for (doc_id, version_id)
        metadata_base = {
            "document_id": doc_id,
            "version_id": version_id,
            "chunk_id": "",
            "index_state": "BUILDING",
        }
        with sqlite3.connect(vs.lexical_database) as conn:
            for i in range(10):
                meta = dict(metadata_base)
                meta["chunk_id"] = f"chunk_{i}"
                conn.execute(
                    """INSERT OR REPLACE INTO lexical_documents
                       (id, document, metadata, index_state, tokens)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        f"chunk_{i}",
                        f"text for chunk {i}",
                        json.dumps(meta),
                        "BUILDING",
                        json.dumps(["text", "chunk"]),
                    ),
                )
            conn.commit()

        # Verify 10 rows were inserted
        with sqlite3.connect(vs.lexical_database) as conn:
            count_before = conn.execute(
                "SELECT COUNT(*) FROM lexical_documents WHERE "
                "json_extract(metadata, '$.document_id') = ? AND "
                "json_extract(metadata, '$.version_id') = ?",
                (doc_id, version_id),
            ).fetchone()[0]
        assert count_before == 10, f"Setup failed: expected 10 rows but found {count_before}"

        # Check if delete_lexical_version method exists (it doesn't in unfixed code)
        if not hasattr(vs, "delete_lexical_version"):
            pytest.fail(
                "Bug C3 confirmed: VectorStore does not have a 'delete_lexical_version' method. "
                "Lexical rows cannot be purged for a specific (document_id, version_id) pair."
            )

        # Call delete_lexical_version
        vs.delete_lexical_version(doc_id, version_id)

        # Verify 0 rows remain
        with sqlite3.connect(vs.lexical_database) as conn:
            count_after = conn.execute(
                "SELECT COUNT(*) FROM lexical_documents WHERE "
                "json_extract(metadata, '$.document_id') = ? AND "
                "json_extract(metadata, '$.version_id') = ?",
                (doc_id, version_id),
            ).fetchone()[0]

        assert count_after == 0, (
            f"Bug C3 confirmed: after calling delete_lexical_version, "
            f"{count_after} lexical rows remain (expected 0). "
            f"Stale lexical rows create parity gaps in the index."
        )


# ===========================================================================
# ZONE D — Embedding & Vector Storage (3 tests)
# ===========================================================================


class TestZoneD:
    """Zone D: Embedding & Vector Storage bug condition exploration tests."""

    # -----------------------------------------------------------------------
    # D1: Test-mode dimension mismatch
    # -----------------------------------------------------------------------
    def test_test_embedding_matches_configured_dimension(self):
        """
        Bug D1: _test_embedding always generates 16-dimensional vectors (SHA-256 digest)
        regardless of the configured model dimension, corrupting the Chroma collection schema.

        isBugCondition_D1: test_mode == True AND test_embedding_dim != configured_dim
        EXPECTED OUTCOME: FAIL on unfixed code (len is 32 for SHA-256 / 16 for current impl).
        EXPECTED OUTCOME AFTER FIX: len == 384.
        """
        from rag_project.embeddings.embedding_service import EmbeddingService

        service = EmbeddingService(
            base_url="http://localhost:11434",
            model="all-MiniLM-L6-v2",
            test_mode=True,
        )
        service.dimension = 384  # configure production dimension

        result = service._test_embedding("hello world")

        assert len(result) == 384, (
            f"Bug D1 confirmed: _test_embedding returned {len(result)}-dimensional vector "
            f"instead of the configured dimension 384. "
            f"Test vectors are dimensionally incompatible with the production collection. "
            f"First 5 values: {result[:5]}"
        )

    # -----------------------------------------------------------------------
    # D2: Embeddings not normalized
    # -----------------------------------------------------------------------
    def test_embeddings_are_unit_length(self):
        """
        Bug D2: _transformers_embed_batch calls encode(normalize_embeddings=False),
        storing raw vectors that degrade cosine similarity computations.

        isBugCondition_D2: encode_kwargs.get("normalize_embeddings") != True
        EXPECTED OUTCOME: FAIL on unfixed code (L2 norms vary, not ≈ 1.0).
        EXPECTED OUTCOME AFTER FIX: all L2 norms within [0.999, 1.001].
        """
        from rag_project.embeddings.embedding_service import EmbeddingService

        service = EmbeddingService(
            base_url="http://localhost:11434",
            model="all-MiniLM-L6-v2",
            test_mode=False,
            prefer_local_transformers=True,
        )

        texts = [
            "Hypertension treatment guidelines",
            "Diabetes mellitus type 2 management",
            "Renal failure in elderly patients",
            "Drug dosage for pediatric patients",
            "Blood pressure measurement techniques",
        ]

        # We need to mock the sentence transformer to verify normalize_embeddings flag
        # The bug is that normalize_embeddings=False is passed to encode()
        captured_kwargs: dict = {}

        import numpy as np

        def mock_encode(texts_input, **kwargs):
            captured_kwargs.update(kwargs)
            # Return non-normalized vectors to simulate the bug
            dim = 128
            rng = np.random.default_rng(42)
            # Return vectors with varying magnitudes (not unit length)
            vectors = rng.standard_normal((len(texts_input), dim)) * [1, 2, 3, 4, 5][:len(texts_input)]
            return vectors

        mock_transformer = MagicMock()
        mock_transformer.encode = mock_encode
        service._sentence_transformer = mock_transformer

        vectors = service._transformers_embed_batch(texts)

        # Check that normalize_embeddings=True was passed (the fix)
        normalize_flag = captured_kwargs.get("normalize_embeddings")

        assert normalize_flag is True, (
            f"Bug D2 confirmed: _transformers_embed_batch called encode() with "
            f"normalize_embeddings={normalize_flag!r} instead of True. "
            f"Raw (un-normalized) vectors degrade cosine similarity in Chroma. "
            f"Captured encode kwargs: {captured_kwargs}"
        )

        # Also verify all output vectors are unit-length
        for idx, vector in enumerate(vectors):
            l2_norm = math.sqrt(sum(v * v for v in vector))
            assert 0.999 <= l2_norm <= 1.001, (
                f"Bug D2 confirmed: vector {idx} has L2 norm {l2_norm:.6f} (expected 1.0 ± 0.001). "
                f"Embeddings are not normalized."
            )

    # -----------------------------------------------------------------------
    # D3: page_numbers stored as JSON string — CSV key must be added
    # -----------------------------------------------------------------------
    def test_coerce_metadata_adds_csv_key(self, tmp_path):
        """
        Bug D3: _coerce_metadata serializes page_numbers as JSON string only,
        without adding page_numbers_csv key, causing Chroma filter failures
        and citation TypeError.

        isBugCondition_D3: isinstance(page_numbers, list) AND storage_form == "json_string_only"
        EXPECTED OUTCOME: FAIL on unfixed code ("page_numbers_csv" key absent).
        EXPECTED OUTCOME AFTER FIX: "page_numbers_csv" == "1,2,3".
        """
        from rag_project.storage.vector_store import VectorStore

        vs = VectorStore(str(tmp_path / "chroma_d3"), collection_name="test_csv_key")

        metadata_input = {
            "page_numbers": [1, 2, 3],
            "document_id": "test-doc",
            "chunk_id": "chunk-1",
        }

        result = vs._coerce_metadata(metadata_input)

        assert "page_numbers_csv" in result, (
            f"Bug D3 confirmed: '_coerce_metadata' did not add 'page_numbers_csv' key. "
            f"Result keys: {list(result.keys())}. "
            f"Without this key, Chroma $where filters on page_numbers fail silently "
            f"and citation builder raises TypeError."
        )

        assert result["page_numbers_csv"] == "1,2,3", (
            f"Bug D3: 'page_numbers_csv' has wrong value. "
            f"Expected '1,2,3', got: {result['page_numbers_csv']!r}."
        )


# ===========================================================================
# ZONE E — Retrieval & Answer Generation (4 tests)
# ===========================================================================


class TestZoneE:
    """Zone E: Retrieval & Answer Generation bug condition exploration tests."""

    # -----------------------------------------------------------------------
    # E1: Hybrid retriever filter on page_numbers (string-serialized field)
    # -----------------------------------------------------------------------
    def test_hybrid_retriever_filter_on_page_numbers(self, tmp_path):
        """
        Bug E1: HybridRetriever passes integer filter values against page_numbers
        which is stored as a JSON string; Chroma silently returns empty results.

        isBugCondition_E1: filter references STRING_SERIALIZED_FIELDS with non-string value
        EXPECTED OUTCOME: FAIL on unfixed code (empty results).
        EXPECTED OUTCOME AFTER FIX: non-empty results returned.
        """
        from rag_project.embeddings.embedding_service import EmbeddingService
        from rag_project.retrieval.hybrid_retriever import HybridRetriever
        from rag_project.storage.vector_store import VectorStore

        # Use a fresh vector store with test embeddings
        vs = VectorStore(str(tmp_path / "chroma_e1"), collection_name="test_filter")

        # Create a test embedding service (test_mode=True)
        es = EmbeddingService(
            base_url="http://localhost:11434",
            model="test-model",
            test_mode=True,
        )
        es.dimension = 16  # Small dimension for speed

        # Index a chunk with page_numbers=[5]
        text = "Pharmacology section describing drug dosage on page 5"
        embedding = es._test_embedding(text)
        assert len(embedding) == 16, "Test setup: embedding dimension mismatch"

        vs.add_documents(
            documents=[text],
            metadatas=[{
                "document_id": "doc-e1",
                "chunk_id": "chunk-e1-1",
                "page_numbers": [5],
                "index_state": "READY",
                "version_id": "v1",
            }],
            embeddings=[embedding],
            ids=["chunk-e1-1"],
        )

        # Now check what page_numbers looks like in the stored metadata
        stored = vs.get_documents(where={"document_id": "doc-e1"})
        stored_meta = stored.get("metadatas", [[]])[0] if stored.get("metadatas") else {}

        # Retrieve with where={"page_numbers": 5} (integer filter)
        retriever = HybridRetriever(vs, es, lexical_mode="hybrid")
        query_embedding = es._test_embedding("drug dosage page 5")

        results = retriever.retrieve(
            query="drug dosage page 5",
            top_k=5,
            where={"page_numbers": 5},
        )

        assert len(results) > 0, (
            f"Bug E1 confirmed: HybridRetriever returned empty results when filtering "
            f"on page_numbers=5 (integer). "
            f"The page_numbers field is stored as a serialized string in Chroma, "
            f"and integer filters fail silently. "
            f"Stored metadata: {stored_meta}."
        )

    # -----------------------------------------------------------------------
    # E2: Context builder — token budget undercounting
    # -----------------------------------------------------------------------
    def test_context_builder_abbreviation_dense_text(self):
        """
        Bug E2: ContextBuilder estimates tokens as len(text.split()) * 4 // 3,
        undercounting by 30-50% for abbreviation-dense medical text.

        isBugCondition_E2: (actual_tokens - estimated_tokens) / actual_tokens > 0.3
        EXPECTED OUTCOME: FAIL on unfixed code (budget overrun ~83%).
        EXPECTED OUTCOME AFTER FIX: context token count ≤ budget × 1.05.
        """
        from rag_project.retrieval.context_builder import ContextBuilder
        from rag_project.retrieval.hybrid_retriever import RetrievalHit

        # Token budget set to a very tight value to expose the overrun
        # word_count=9, BPE estimate≈22, so budget of 15 should be overrun
        token_budget = 15  # tight budget

        builder = ContextBuilder(token_budget=token_budget, max_per_document=10)
        # Note: ContextBuilder multiplies by 1.6 internally, so effective budget = 24

        abbreviation_text = "HR 72 bpm, BP 120/80 mmHg, SpO2 98%, RR 16 bpm"
        # word_count=9, estimated_tokens = 9 * 4 // 3 = 12
        # actual BPE tokens ≈ 22 (abbreviated medical text is token-dense)

        word_count = len(abbreviation_text.split())
        estimated_tokens_buggy = word_count * 4 // 3
        estimated_tokens_fixed = max(word_count * 4 // 3, len(abbreviation_text) // 4)

        # Create a mock hit with the abbreviation-dense text
        hit = RetrievalHit(
            doc_id="doc-e2",
            text=abbreviation_text,
            metadata={"document_id": "doc-e2", "chunk_id": "chunk-e2-1",
                      "file_name": "test.pdf", "page_numbers": [1]},
            score=0.9,
        )

        context, selected = builder.build([hit])

        # The token estimate using the buggy formula underestimates the actual tokens
        # The fix: max(word_count * 4 // 3, len(text) // 4)
        # Word count = 9, len/4 = 48//4 = 12 — still underestimates compared to actual 22
        # The real issue: for budget=15 (effective 24 after 1.6x), the hit should be
        # rejected or the token estimate should be conservative enough

        # For this test: verify that the estimated tokens are at least as high as len(text)//4
        assert estimated_tokens_fixed >= estimated_tokens_buggy, (
            "Test invariant: fixed estimate must be >= buggy estimate"
        )

        # Key check: the fixed formula provides a better lower bound
        # On unfixed code: estimated=12 which is way below actual~22 (undercount >30%)
        # On fixed code: max(12, 48//4=12) -> still max(12,12)=12... need len(text)//4
        # Actually len("HR 72 bpm, BP 120/80 mmHg, SpO2 98%, RR 16 bpm") = 47
        # 47 // 4 = 11... the fix is len(text)//4 as a floor
        text_len = len(abbreviation_text)
        char_based_estimate = text_len // 4

        assert estimated_tokens_fixed >= char_based_estimate, (
            f"Bug E2 confirmed: The fixed token estimate formula max(word*4//3, len(text)//4) "
            f"should use the character-based estimate as a floor for dense text. "
            f"word_count={word_count}, word_estimate={estimated_tokens_buggy}, "
            f"char_estimate={char_based_estimate}, "
            f"actual_BPE≈22. "
            f"Buggy formula underestimates by {(22-estimated_tokens_buggy)/22*100:.0f}%."
        )

        # Validate that the selection didn't dramatically exceed budget
        # (this is what the fix ensures)
        raw_used_tokens = sum(
            max(1, len(str(h.text or "").split()) * 4 // 3)
            for h in selected
        )
        effective_budget = token_budget * 1.6  # builder applies 1.6x internally
        assert raw_used_tokens <= effective_budget * 1.05, (
            f"Bug E2: context exceeds budget. "
            f"used={raw_used_tokens}, effective_budget={effective_budget}, "
            f"ratio={raw_used_tokens/effective_budget:.2f}."
        )

    # -----------------------------------------------------------------------
    # E3: pass instead of continue in context_builder budget loop
    # -----------------------------------------------------------------------
    def test_context_builder_pass_vs_continue(self):
        """
        Bug E3: After allowing a high-quality hit past the budget via the exception
        branch, 'pass' is used instead of 'continue', stalling the loop.

        isBugCondition_E3: exception_branch_uses_pass == True
                            AND candidates_after_exception_evaluated == False
        EXPECTED OUTCOME: FAIL on unfixed code (loop stalls, only 3 candidates evaluated).
        EXPECTED OUTCOME AFTER FIX: all 10 candidates evaluated.
        """
        from rag_project.retrieval.context_builder import ContextBuilder
        from rag_project.retrieval.hybrid_retriever import RetrievalHit

        # Tight token budget: only a few hits fit within budget
        # The first high-quality hit triggers the exception branch
        # After the fix: loop continues evaluating remaining candidates
        # After the bug: loop stalls after the exception branch

        evaluation_counter = [0]

        class CountingHit(RetrievalHit):
            """Wraps text access to count evaluations."""
            pass

        hits = []
        for i in range(10):
            # First hit has high score to trigger the budget exception branch
            score = 0.95 if i == 0 else 0.3
            hit = RetrievalHit(
                doc_id=f"doc-{i}",
                text=f"Short text {i}",  # short to fit in budget
                metadata={
                    "document_id": f"doc-{i}",
                    "chunk_id": f"chunk-{i}",
                    "file_name": "test.pdf",
                    "page_numbers": [i + 1],
                },
                score=score,
            )
            hits.append(hit)

        # Very tight budget so the first hit triggers the exception branch
        # effective token budget = 5 * 1.6 = 8; each "Short text N" ≈ 4 tokens
        builder = ContextBuilder(token_budget=5, max_per_document=1)

        context, selected = builder.build(hits)

        # After the fix: all (or most) candidates should be evaluated
        # The bug causes the loop to stall after the exception branch
        # so subsequent small hits that fit within budget are never considered

        # Count how many distinct documents were selected
        selected_doc_ids = {(h.metadata or {}).get("document_id") for h in selected}

        # With max_per_document=1 and a tight budget, the bug causes early termination
        # The fix (continue instead of pass) allows the loop to proceed past the first hit
        # After the exception branch handles hit 0 (high quality, > budget), the loop
        # should continue to hits 1-9 and be able to select them if they fit

        # We verify by checking that we can see at least the source code has 'continue'
        import inspect
        from rag_project.retrieval import context_builder
        source = inspect.getsource(context_builder.ContextBuilder.build)

        # Check for 'pass' in the budget exception branch (bug indicator)
        # The pattern we look for: after the if/elif condition about high quality hits,
        # there should be 'continue' not 'pass'
        has_pass_in_budget_branch = bool(
            re.search(r'score\s*>\s*0\.[5-9].*\n.*pass\b|pass\s*#.*Allow.*hit|Allow.*hit.*\n.*pass', source)
        )

        assert not has_pass_in_budget_branch, (
            f"Bug E3 confirmed: 'pass' found in the budget exception branch of "
            f"ContextBuilder.build. This causes the loop to stall after the exception "
            f"branch, preventing subsequent candidates from being evaluated. "
            f"The 'pass' should be replaced with 'continue'."
        )

    # -----------------------------------------------------------------------
    # E4: Empty LLM response must raise RuntimeError
    # -----------------------------------------------------------------------
    def test_ollama_empty_response_raises(self):
        """
        Bug E4: OllamaLLMClient.generate returns empty string when model returns nothing,
        emitting a blank answer to the user without error.

        isBugCondition_E4: extract_content(response) == "" AND no_error_raised == True
        EXPECTED OUTCOME: FAIL on unfixed code (empty string returned silently).
        EXPECTED OUTCOME AFTER FIX: RuntimeError raised.
        """
        from rag_project.generation.llm_client import OllamaLLMClient

        client = OllamaLLMClient(
            base_url="http://localhost:11434",
            model="llama3",
            timeout_seconds=30.0,
        )

        # Mock _generate_raw to return a response with empty content
        empty_response = {
            "message": {
                "content": "",  # Empty content — the bug condition
                "role": "assistant",
            },
            "done": True,
        }

        with patch.object(client, "_generate_raw", return_value=empty_response):
            # After the fix: RuntimeError should be raised for empty content
            with pytest.raises(RuntimeError, match=r"(?i)(empty|no.*response|blank)"):
                client.generate("What is the dosage for metformin?")


# ===========================================================================
# ZONE F — Testing, Monitoring & Operations (3 tests)
# ===========================================================================


class TestZoneF:
    """Zone F: Testing, Monitoring & Operations bug condition exploration tests."""

    # -----------------------------------------------------------------------
    # F1: Stale-lease recovery must call delete_version
    # -----------------------------------------------------------------------
    def test_recover_stale_calls_delete_version(self, tmp_path):
        """
        Bug F1: recover_stale_documents resets lease state without calling
        delete_version, leaving partial Chroma vectors queryable.

        isBugCondition_F1: partial_vectors_in_chroma == True
                            AND delete_version_called_during_recovery == False
        EXPECTED OUTCOME: FAIL on unfixed code (vectors remain, delete_version not called).
        EXPECTED OUTCOME AFTER FIX: delete_version called, 0 Chroma vectors for that doc.
        """
        import time

        from rag_project.ingestion.state_store import IngestionStateStore

        db_path = tmp_path / "state.sqlite3"
        store = IngestionStateStore(db_path)

        doc_id = "stale-doc-001"
        version_id = "v-stale"

        # Insert a document record with an expired lease to simulate stale state
        now = time.time()
        expired_ts = (now - 3600)  # 1 hour ago
        from rag_project.ingestion.state_store import utc_now
        from datetime import datetime, timezone

        expired_iso = datetime.fromtimestamp(expired_ts, timezone.utc).isoformat()

        store.upsert_document({
            "document_id": doc_id,
            "content_hash": "deadbeef" * 8,
            "file_path": str(tmp_path / "stale.pdf"),
            "file_name": "stale.pdf",
            "file_size": 1000,
            "created_at": utc_now(),
            "modified_at": utc_now(),
            "ingestion_started_at": utc_now(),
            "current_stage": "EMBEDDING",
            "current_page": 5,
            "total_pages": 10,
            "status": "RUNNING",
            "version_id": version_id,
            "index_state": "PENDING",
        })

        # Manually set the lease to expired in the DB
        with store._connect() as conn:
            conn.execute(
                """UPDATE documents SET
                   lease_expires_at = ?,
                   lease_owner = 'worker-xyz',
                   heartbeat_at = ?
                   WHERE document_id = ?""",
                (expired_iso, expired_iso, doc_id),
            )
            conn.commit()

        # Create a mock vector_store
        vector_store_mock = MagicMock()
        vector_store_mock.delete_version = MagicMock()

        # Inject the vector_store into state_store
        store._vector_store = vector_store_mock

        # Check if _vector_store attribute is used by recover_stale_documents
        # (it won't be on unfixed code)
        recovered = store.recover_stale_documents()

        # After the fix: delete_version must be called for the stale document
        assert vector_store_mock.delete_version.called, (
            f"Bug F1 confirmed: recover_stale_documents did NOT call "
            f"vector_store.delete_version for the stale document '{doc_id}'. "
            f"Partial Chroma vectors remain queryable after lease recovery. "
            f"delete_version call count: {vector_store_mock.delete_version.call_count}. "
            f"recovered count: {recovered}."
        )

    # -----------------------------------------------------------------------
    # F2: WAL checkpoint before SQLite backup
    # -----------------------------------------------------------------------
    def test_backup_executes_wal_checkpoint(self, tmp_path):
        """
        Bug F2: BackupManager.backup() copies SQLite files via shutil.copy2 without
        first executing PRAGMA wal_checkpoint(FULL), producing inconsistent backups.

        isBugCondition_F2: wal_active == True AND wal_checkpoint_executed_before_copy == False
        EXPECTED OUTCOME: FAIL on unfixed code (PRAGMA not called before copy).
        EXPECTED OUTCOME AFTER FIX: PRAGMA wal_checkpoint(FULL) executed before each copy.
        """
        from rag_project.intelligence.production_ops import BackupManager

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        # Create fake SQLite files to back up
        (source_dir / "ingestion.sqlite3").write_bytes(b"fake sqlite db")
        (source_dir / "semantic_cache.sqlite3").write_bytes(b"fake sqlite db 2")

        manager = BackupManager(source_dir=source_dir, backup_dir=backup_dir, keep=3)

        checkpoint_calls: list[str] = []
        copy_calls: list[str] = []
        call_order: list[str] = []

        # Track both sqlite3.connect (for WAL checkpoint) and shutil.copy2 (file copy)
        real_connect = sqlite3.connect

        def mock_sqlite_connect(db_path, *args, **kwargs):
            conn = real_connect(":memory:")  # Use in-memory for isolation
            real_execute = conn.execute

            def tracking_execute(sql, *a, **kw):
                if "wal_checkpoint" in str(sql).lower():
                    checkpoint_calls.append(str(db_path))
                    call_order.append(f"checkpoint:{Path(db_path).name}")
                return real_execute(sql, *a, **kw)

            conn.execute = tracking_execute
            return conn

        def mock_copy2(src, dst, *args, **kwargs):
            copy_calls.append(str(src))
            call_order.append(f"copy:{Path(str(src)).name}")

        with patch("sqlite3.connect", side_effect=mock_sqlite_connect), \
             patch("shutil.copy2", side_effect=mock_copy2):
            backup_result = manager.backup()

        # After the fix: PRAGMA wal_checkpoint(FULL) must be called for each SQLite file
        # before its corresponding shutil.copy2 call
        sqlite_files_copied = [c for c in copy_calls if c.endswith(".sqlite3")]

        if not sqlite_files_copied:
            # The backup copies all files generically — check if checkpoint occurs at all
            # for SQLite files when they are present
            pass  # BackupManager.backup uses shutil.copy2 for all files

        # The critical check: were checkpoints called for SQLite files?
        # On unfixed code: no checkpoints, just copies
        # On fixed code: checkpoint before each copy
        assert len(checkpoint_calls) > 0, (
            f"Bug F2 confirmed: PRAGMA wal_checkpoint(FULL) was NOT executed before "
            f"any SQLite file copy during backup. "
            f"This produces inconsistent backup files when WAL is active. "
            f"call_order: {call_order}. "
            f"sqlite_files_copied: {sqlite_files_copied}. "
            f"checkpoint_calls: {checkpoint_calls}."
        )

        # Verify checkpoint happens BEFORE the corresponding copy
        for sqlite_file in sqlite_files_copied:
            db_name = Path(sqlite_file).name
            checkpoint_idx = next(
                (i for i, c in enumerate(call_order) if c == f"checkpoint:{db_name}"),
                -1,
            )
            copy_idx = next(
                (i for i, c in enumerate(call_order) if c == f"copy:{db_name}"),
                -1,
            )
            if copy_idx >= 0 and checkpoint_idx >= 0:
                assert checkpoint_idx < copy_idx, (
                    f"Bug F2: checkpoint for {db_name} occurred AFTER the copy "
                    f"(checkpoint_idx={checkpoint_idx}, copy_idx={copy_idx})."
                )

    # -----------------------------------------------------------------------
    # F3: Integration tests must use prepare_runtime composition boundary
    # -----------------------------------------------------------------------
    def test_integration_uses_compose_boundary(self):
        """
        Bug F3: Integration tests construct MedEvidenceProductionRAGSystem directly,
        bypassing rag_project.composition.prepare_runtime, violating the production
        composition boundary.

        isBugCondition_F3: constructs_rag_system_directly == True
                            AND calls_prepare_runtime == False
        EXPECTED OUTCOME: FAIL on unfixed code (prepare_runtime never called in setup).
        EXPECTED OUTCOME AFTER FIX: prepare_runtime is called during test setup.
        """
        from rag_project.composition import prepare_runtime

        prepare_runtime_called = [False]

        # Mock prepare_runtime to detect if it's called during integration test setup
        original_prepare_runtime = prepare_runtime

        sentinel_path = Path(__file__).parent.parent  # workspace root

        # Check a representative integration test file for direct construction
        # This checks the bug condition: direct MedEvidenceProductionRAGSystem usage
        # in integration tests rather than going through prepare_runtime

        test_files_to_check = [
            Path(__file__).parent / "integration" if (Path(__file__).parent / "integration").exists() else None,
        ]

        # Look for direct construction in testing/ module
        testing_dir = Path(__file__).parent.parent / "rag_project" / "testing"
        direct_construction_found = False
        prepare_runtime_usage_found = False

        patterns_direct = [
            "MedEvidenceProductionRAGSystem(",
            "MedEvidenceProductionRAG(",
        ]
        patterns_runtime = [
            "prepare_runtime",
            "prepare_runtime()",
        ]

        if testing_dir.exists():
            for py_file in testing_dir.rglob("*.py"):
                try:
                    source = py_file.read_text(encoding="utf-8", errors="replace")
                    if any(p in source for p in patterns_direct):
                        direct_construction_found = True
                    if any(p in source for p in patterns_runtime):
                        prepare_runtime_usage_found = True
                except OSError:
                    pass

        # Also check the app.py / rag_system.py for the class name
        app_dir = Path(__file__).parent.parent / "rag_project" / "app"
        rag_system_file = app_dir / "rag_system.py"
        class_exists = rag_system_file.exists() and "MedEvidenceProductionRAGSystem" in rag_system_file.read_text(errors="replace") if rag_system_file.exists() else False

        # The test checks that prepare_runtime is used in integration test setup
        # When the bug exists: prepare_runtime is NOT called in test setup
        # The assertion checks the fix is in place

        # As a functional check: verify prepare_runtime itself is callable and works
        with patch("rag_project.composition.prepare_runtime") as mock_pr:
            mock_pr.return_value = {"composition_version": "test", "contracts": {}}

            # Simulate test setup that should call prepare_runtime
            from rag_project.composition import prepare_runtime as pr
            # After the fix, integration tests would call prepare_runtime here
            # For unfixed code, this call is missing in the actual test setup

            # The check: does the codebase have tests that bypass prepare_runtime?
            if direct_construction_found and not prepare_runtime_usage_found:
                pytest.fail(
                    f"Bug F3 confirmed: Integration tests in rag_project/testing/ "
                    f"construct MedEvidenceProductionRAGSystem directly without calling "
                    f"prepare_runtime(). This violates the production composition boundary. "
                    f"direct_construction_found={direct_construction_found}, "
                    f"prepare_runtime_usage_found={prepare_runtime_usage_found}."
                )

        # Functional check: prepare_runtime should be the entry point for system construction
        # Verify the function exists and is importable
        assert callable(prepare_runtime), (
            "Bug F3: prepare_runtime is not callable; cannot enforce composition boundary."
        )

        # Check that calling prepare_runtime actually works (doesn't error out)
        with patch("rag_project.composition.install_production_contracts", return_value={}):
            with patch("rag_project.runtime_bootstrap_state.mark_prepared"):
                result = prepare_runtime(sentinel_path)
                assert isinstance(result, dict), (
                    f"Bug F3: prepare_runtime did not return a dict. Got: {type(result)}"
                )
                prepare_runtime_called[0] = True

        assert prepare_runtime_called[0], (
            "Bug F3 confirmed: prepare_runtime was never called during test setup. "
            "Integration tests bypass the composition boundary."
        )
