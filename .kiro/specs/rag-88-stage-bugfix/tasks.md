# Implementation Plan

## RAG 88-Stage Bugfix — 23 Defects Across 6 Zones

---

- [-] 1. Write bug condition exploration tests (BEFORE implementing any fix)
  - **Property 1: Bug Condition** - 23-Zone RAG Pipeline Bug Exploration Suite
  - **CRITICAL**: These tests MUST FAIL on unfixed code — failure confirms each bug exists
  - **DO NOT attempt to fix the tests or the code when they fail**
  - **NOTE**: These tests encode expected behavior — they validate each fix when they pass after implementation
  - **GOAL**: Surface counterexamples that demonstrate all 23 bugs exist
  - **Scoped PBT Approach**: For deterministic bugs, scope each property to the concrete failing case to ensure reproducibility

  **Zone A — PDF Ingestion (7 exploration tests):**

  - A1 (`test_filemonitor_skips_content_duplicate`): Create two files with identical bytes and different names; assert only one ingestion is scheduled. From design: `isBugCondition_A1` — `sha256(file_a) == sha256(file_b) AND file_a.name != file_b.name`. Run on UNFIXED code — **EXPECTED OUTCOME: FAIL** (both files scheduled).
  - A2 (`test_robust_ingest_no_self_replace`): Submit a PDF already in `processed_dir`; assert `file_path` still exists after the call. From design: `isBugCondition_A2` — `file_path.resolve() == archived_path.resolve()`. Run on UNFIXED code — **EXPECTED OUTCOME: FAIL** (file deleted or corrupted).
  - A3 (`test_ocr_failure_writes_page_record`): Mock OCR to raise an exception on page 7; assert `state_store.record_page` was called with `extraction_status="FAILED"`. From design: `isBugCondition_A3` — `ocr_required AND NOT ocr_succeeded AND NOT page_record_written`. Run on UNFIXED code — **EXPECTED OUTCOME: FAIL** (no record written).
  - A4 (`test_two_column_text_column_a_before_column_b`): Feed a synthetic two-column page where column-A text is "DosageInfo" and column-B text is "ClinicalOutcome"; assert all column-A text precedes first column-B token. From design: `isBugCondition_A4` — `max(x_coords) - min(x_coords) > page.rect.width * 0.4`. Run on UNFIXED code — **EXPECTED OUTCOME: FAIL** (columns interleaved).
  - A5 (`test_ocr_render_raises_at_sub_0_5_scale`): Provide an A3-format oversized page; assert `RuntimeError` is raised when scale must drop below 0.5. From design: `isBugCondition_A5` — `configured_scale / 0.25 > 1.0`. Run on UNFIXED code — **EXPECTED OUTCOME: FAIL** (renders silently at ~0.18).
  - A6 (`test_table_extract_non_attr_error_logged`): Mock `find_tables` to raise `ValueError`; assert `logger.warning` was called with the page number. From design: `isBugCondition_A6` — `exception_raised != None AND NOT isinstance(exception, AttributeError)`. Run on UNFIXED code — **EXPECTED OUTCOME: FAIL** (warning not called).
  - A7 (`test_extract_page_number_footer_position`): Provide page text with 600-character header block and page number "— 47 —" at position 612; assert function returns 47. From design: `isBugCondition_A7` — `page_number_pattern_found(text[500:]) AND NOT page_number_pattern_found(text[:500])`. Run on UNFIXED code — **EXPECTED OUTCOME: FAIL** (returns None).

  **Zone B — Text Processing (3 exploration tests):**

  - B1 (`test_clean_text_preserves_superscripts`): Input `"CO₂ 38mmHg, H₂O levels"` (Unicode subscripts U+2082); assert output contains `"CO_2"` or `"CO^2"` notation, not bare `"CO2"`. From design: `isBugCondition_B1` — text contains SUPERSCRIPT_CHARS or SUBSCRIPT_CHARS. Run on UNFIXED code — **EXPECTED OUTCOME: FAIL** (outputs `"CO2"` with stripped digit).
  - B2 (`test_split_paragraphs_joins_hyphenated_linebreak`): Input `"anti-\nhypertensive therapy is common"`, assert output contains `"antihypertensive"` as a single token. From design: `isBugCondition_B2` — `re.search(r'\w+-\n\w+', text) != None`. Run on UNFIXED code — **EXPECTED OUTCOME: FAIL** (split into `"anti-"` and `"hypertensive"`).
  - B3 (`test_ocr_correction_fixes_metformin`): Input `"Metf0rmin 500mg"` (zero for O); assert pipeline output is `"Metformin 500mg"`. From design: `isBugCondition_B3` — `contains_numeric_drug_pattern AND ocr_correction_pass_applied == False`. Run on UNFIXED code — **EXPECTED OUTCOME: FAIL** (confusable uncorrected).

  **Zone C — Chunking (3 exploration tests):**

  - C1 (`test_chunk_page_batches_carries_chapter_across_boundary`): Create 10 pages, chapter heading on page 5, chunk in batches of 6; assert pages 7–10 chunks show the chapter heading in metadata. From design: `isBugCondition_C1` — `batch_index > 0 AND preceding_chapter_state != None AND chunk_pages_receives_chapter_state == False`. Run on UNFIXED code — **EXPECTED OUTCOME: FAIL** (`chapter=null` in batch 2).
  - C2 (`test_stable_id_unique_across_documents`): Create chunks from two docs both with `chapter="Dosage", section="Adults"`; assert `section_id` values differ. From design: `isBugCondition_C2` — `doc_a.chapter == doc_b.chapter AND doc_a.section == doc_b.section AND document_id_not_in_hash_key`. Run on UNFIXED code — **EXPECTED OUTCOME: FAIL** (IDs collide).
  - C3 (`test_delete_version_clears_lexical_rows`): Insert 10 lexical rows for `(doc_id, version_id)`, call `delete_lexical_version`; assert 0 rows remain. From design: `isBugCondition_C3` — `prior_ingestion_interrupted AND lexical_rows_exist AND delete_version_does_not_clear_lexical`. Run on UNFIXED code — **EXPECTED OUTCOME: FAIL** (`delete_lexical_version` method does not exist).

  **Zone D — Embedding & Vector Storage (3 exploration tests):**

  - D1 (`test_test_embedding_matches_configured_dimension`): Instantiate `EmbeddingService(test_mode=True, dimension=384)`; assert `len(_test_embedding("x")) == 384`. From design: `isBugCondition_D1` — `test_mode == True AND test_embedding_dim != configured_dim`. Run on UNFIXED code — **EXPECTED OUTCOME: FAIL** (length is 16).
  - D2 (`test_embeddings_are_unit_length`): Embed a batch of 5 texts; assert each vector's L2 norm is within `[0.999, 1.001]`. From design: `isBugCondition_D2` — `encode_kwargs.get("normalize_embeddings") != True`. Run on UNFIXED code — **EXPECTED OUTCOME: FAIL** (norms vary).
  - D3 (`test_coerce_metadata_adds_csv_key`): Coerce `{"page_numbers": [1, 2, 3]}`; assert `"page_numbers_csv" == "1,2,3"`. From design: `isBugCondition_D3` — `isinstance(page_numbers, list) AND storage_form == "json_string_only"`. Run on UNFIXED code — **EXPECTED OUTCOME: FAIL** (key absent).

  **Zone E — Retrieval & Answer Generation (4 exploration tests):**

  - E1 (`test_hybrid_retriever_filter_on_page_numbers`): Index a chunk with `page_numbers=[5]`; retrieve with `where={"page_numbers": 5}`; assert non-empty results returned. From design: `isBugCondition_E1` — filter references STRING_SERIALIZED_FIELDS with non-string value. Run on UNFIXED code — **EXPECTED OUTCOME: FAIL** (empty results).
  - E2 (`test_context_builder_abbreviation_dense_text`): Feed `"HR 72 bpm, BP 120/80 mmHg, SpO2 98%, RR 16 bpm"` (word_count=9, actual BPE≈22); assert assembled context token count ≤ budget × 1.05. From design: `isBugCondition_E2` — `(actual_tokens - estimated_tokens) / actual_tokens > 0.3`. Run on UNFIXED code — **EXPECTED OUTCOME: FAIL** (budget overrun ~83%).
  - E3 (`test_context_builder_pass_vs_continue`): Provide 10 candidates with tight budget; assert all 10 were evaluated (counter reaches 10). From design: `isBugCondition_E3` — `exception_branch_uses_pass == True AND candidates_after_exception_evaluated == False`. Run on UNFIXED code — **EXPECTED OUTCOME: FAIL** (loop stalls, only 3 evaluated).
  - E4 (`test_ollama_empty_response_raises`): Mock Ollama API to return `""`; assert `RuntimeError` is raised. From design: `isBugCondition_E4` — `extract_content(response) == "" AND no_error_raised`. Run on UNFIXED code — **EXPECTED OUTCOME: FAIL** (empty string returned silently).

  **Zone F — Testing, Monitoring & Operations (3 exploration tests):**

  - F1 (`test_recover_stale_calls_delete_version`): Insert partial Chroma vectors for a document, mark it stale, call `recover_stale_documents`; assert `delete_version` called and Chroma has 0 vectors for that document. From design: `isBugCondition_F1` — `partial_vectors_in_chroma == True AND delete_version_called_during_recovery == False`. Run on UNFIXED code — **EXPECTED OUTCOME: FAIL** (vectors remain).
  - F2 (`test_backup_executes_wal_checkpoint`): Mock `sqlite3.connect`; assert `PRAGMA wal_checkpoint(FULL)` executed before each `shutil.copy`. From design: `isBugCondition_F2` — `wal_active == True AND wal_checkpoint_executed_before_copy == False`. Run on UNFIXED code — **EXPECTED OUTCOME: FAIL** (PRAGMA not called).
  - F3 (`test_integration_uses_compose_boundary`): Assert `prepare_runtime()` is called during test setup (via mock sentinel on `prepare_runtime`). From design: `isBugCondition_F3` — `constructs_rag_system_directly == True AND calls_prepare_runtime == False`. Run on UNFIXED code — **EXPECTED OUTCOME: FAIL** (`prepare_runtime` never called).

  - Document all counterexamples found to understand each root cause
  - Mark task complete when all 23 tests are written, run, and failures are documented
  - _Requirements: 1.1, 1.2, 1.5, 1.6, 1.8, 1.11, 1.12, 1.13, 1.15, 1.18, 1.22, 1.23, 1.24, 1.26, 1.27, 1.31, 1.34, 1.37, 1.38, 1.41, 1.43, 1.44, 1.45_

---

- [~] 2. Write preservation property tests (BEFORE implementing any fix)
  - **Property 2: Preservation** - Unchanged RAG Pipeline Behavior for Non-Buggy Inputs
  - **IMPORTANT**: Follow observation-first methodology — run UNFIXED code with non-buggy inputs first
  - Observe behavior on UNFIXED code for all inputs where NO bug condition holds (`¬C(X)`)
  - Write property-based tests capturing observed behavior patterns from Preservation Requirements (design section 3.x)
  - Property-based testing generates many test cases for stronger guarantees across the valid input domain

  **Observation targets (run on UNFIXED code):**

  - Observe: Normal single-copy PDF (no content duplicate, source ≠ destination) completes full pipeline
  - Observe: Already-indexed duplicate PDF returns `{"status": "skipped", ...}`
  - Observe: Well-formed single-column text PDF extracts without OCR
  - Observe: OCR page with confidence ≥ 0.55 persists via `state_store.record_page` with `ocr_status="completed"`
  - Observe: `clean_text` on ASCII prose (no superscripts/subscripts) normalizes whitespace, strips leading/trailing space
  - Observe: `detect_language("pure Arabic text")` returns `"ar"`
  - Observe: `tokenize("BP 120/80 mmHg")` keeps `"120/80"` as single unit
  - Observe: `chunk_pages` emits RAG-STRUCTURE markers with `chapter_id`, `section_id`, `parent_id` for pages with headings
  - Observe: `EmbeddingService.embed_texts` returns vectors in input order, cache hit returns identical vector
  - Observe: `VectorStore.add_documents` atomically inserts Chroma and lexical rows; rolls back on failure
  - Observe: `delete_version` removes both Chroma and lexical rows for targeted version
  - Observe: `HybridRetriever` runs RRF fusion in hybrid mode; falls back to lexical when vector fails
  - Observe: `ContextBuilder` enforces `max_per_document` limit
  - Observe: `DeterministicAnswerGenerator` produces `SynthesizedAnswer` with citations when LLM unavailable
  - Observe: `recover_stale_documents` resets only documents whose lease expiry is in the past

  **Property-based tests to write:**

  - `prop_normal_ingestion_unchanged`: For random single-copy PDFs (content-unique, source ≠ destination path) — assert `ingest'(X) == ingest(X)`. Non-bug-condition: `sha256(file_a) != sha256(file_b)` AND `file_path.resolve() != archived_path.resolve()`.
  - `prop_clean_text_ascii_unchanged`: For random ASCII prose strings with no Unicode superscripts/subscripts (U+2070–U+209F, U+00B2, U+00B3), no hyphen–newline — assert `clean_text'(X) == clean_text(X)`. Verify tests pass on UNFIXED code.
  - `prop_split_paragraphs_no_hyphen_unchanged`: For paragraph text with no `\w+-\n\w+` pattern — assert `split_paragraphs'(X) == split_paragraphs(X)`.
  - `prop_single_batch_chunking_unchanged`: For single-batch page sets (no cross-batch boundary) — assert chunking output identical.
  - `prop_production_embeddings_normalized_order`: For random text batches — assert `embed'(X)` vectors are in input order and LRU cache returns identical vector for same input.
  - `prop_retrieval_no_serialized_field_filter`: For queries with no filters on STRING_SERIALIZED_FIELDS — assert `retrieve'(X) == retrieve(X)` (same result set and order).
  - `prop_context_builder_non_abbreviation_text`: For non-abbreviation-dense text where `word_count_estimate ≈ actual_tokens` (ratio < 1.3) — assert token estimate is unchanged.
  - `prop_ocr_correction_clean_text_unchanged`: For text with no OCR confusable patterns — assert `apply_ocr_corrections(X) == X`.

  - Verify ALL property-based tests PASS on UNFIXED code before implementing any fix
  - **EXPECTED OUTCOME**: Tests PASS (confirms baseline behavior to preserve)
  - Mark task complete when all preservation tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.6, 3.7, 3.8, 3.10, 3.14, 3.15, 3.16, 3.18, 3.19, 3.20, 3.21, 3.22, 3.24_

---

- [ ] 3. Apply all 23 fixes across Zones A–F

  - [~] 3.1 Fix A1 — FileMonitor duplicate file detection (`rag_project/ingestion/file_monitor.py`)
    - Add `self._content_hash_registry: dict[str, Path] = {}` to `__init__`
    - After computing `sha256(file_path)`, check `if digest in self._content_hash_registry`; emit `"skipped_duplicate"` state store event and skip scheduling
    - Otherwise register `self._content_hash_registry[digest] = file_path` before scheduling
    - _Bug_Condition: `isBugCondition_A1` — `sha256(file_a) == sha256(file_b) AND file_a.name != file_b.name AND both_scheduled_for_ingestion`_
    - _Expected_Behavior: Property 1 — only first alphabetically-sorted duplicate processed; second records `"skipped_duplicate"` event_
    - _Preservation: Req 3.1, 3.2 — normal single-copy ingestion and version-matched duplicate skipping unchanged_
    - _Requirements: 2.1, 3.1, 3.2_

  - [~] 3.2 Fix A2 — Self-replace guard in `robust_ingestor.py` (`rag_project/ingestion/robust_ingestor.py`)
    - Before `file_path.replace(archived_path)`, add: `if file_path.resolve() == archived_path.resolve(): return {"status": "skipped", "reason": "source_is_destination"}`
    - _Bug_Condition: `isBugCondition_A2` — `is_already_indexed AND file_path.resolve() == target.resolve()`_
    - _Expected_Behavior: Property 1 — no file operation when source resolves identically to destination_
    - _Preservation: Req 3.2 — already-indexed duplicate with different path still archived and skipped_
    - _Requirements: 2.2, 3.2_

  - [~] 3.3 Fix A3 — OCR-failed page record in `pdf_extractor.py` (`rag_project/parsing/pdf_extractor.py`)
    - In OCR exception handler inside `extract_iter`, before `continue`, add: `state_store.record_page(document_id, page_no, extraction_status="FAILED", ocr_status="failed")`
    - _Bug_Condition: `isBugCondition_A3` — `ocr_required AND NOT ocr_succeeded AND NOT page_record_written(page_no)`_
    - _Expected_Behavior: Property 2 — `extraction_status="FAILED"` record written; page visible in audit queries_
    - _Preservation: Req 3.4 — OCR pages with confidence ≥ 0.55 continue to persist with `ocr_status="completed"`_
    - _Requirements: 2.5, 3.3, 3.4_

  - [~] 3.4 Fix A4 — Two-column layout detection in `pdf_extractor.py` (`rag_project/parsing/pdf_extractor.py`)
    - Add `_is_two_column(page) -> bool` helper: compute x-coordinates of all text blocks; return `True` if `max_x - min_x > page.rect.width * 0.4` and there are ≥ 4 text blocks
    - In `_extract_page_text`, when `_is_two_column(page)` is `True`, sort blocks by `(round(block.x0 / (page.rect.width * 0.5)), block.y0)` so left-column blocks precede right-column blocks
    - _Bug_Condition: `isBugCondition_A4` — `max(x_coords) - min(x_coords) > page.rect.width * 0.4 AND extraction_mode == "default_sort"`_
    - _Expected_Behavior: Property 3 — all column-A text (top-to-bottom) precedes all column-B text_
    - _Preservation: Req 3.3 — single-column PDFs continue to extract text without invoking this heuristic_
    - _Requirements: 2.6, 3.3_

  - [~] 3.5 Fix A5 — OCR scale floor 0.25→0.5 in `ocr_service.py` (`rag_project/ocr/ocr_service.py`)
    - Change `MIN_RENDER_SCALE = 0.25` to `MIN_RENDER_SCALE = 0.5`
    - After scale reduction, add: `if scale < MIN_RENDER_SCALE: raise RuntimeError(f"Page {page_no}: image too large even at scale {MIN_RENDER_SCALE}; treating as OCR failure")`
    - _Bug_Condition: `isBugCondition_A5` — `(page.width * page.height * configured_scale^2) > max_render_pixels AND required_reduction > 1.0`_
    - _Expected_Behavior: Property 4 — `RuntimeError` raised rather than rendering at sub-0.5 scale_
    - _Preservation: Req 3.4 — OCR pages that fit within pixel budget at ≥ 0.5 scale continue to succeed_
    - _Requirements: 2.7, 3.4_

  - [~] 3.6 Fix A6 — Table exception swallowing in `pdf_extractor.py` (`rag_project/parsing/pdf_extractor.py`)
    - Replace `except Exception: return ""` with: `except AttributeError: return ""` (version compatibility) and `except Exception as e: logger.warning("Table extraction failed on page %d: %s", page_no, e); return ""`
    - _Bug_Condition: `isBugCondition_A6` — `exception_raised != None AND NOT isinstance(exception, AttributeError) AND exception_silently_swallowed`_
    - _Expected_Behavior: Property 5 — non-AttributeError exceptions logged at WARNING with page number_
    - _Preservation: Req 3.3 — `AttributeError` (old PyMuPDF) still silently ignored_
    - _Requirements: 2.10, 3.3_

  - [~] 3.7 Fix A7 — Page number search limited to 500 chars in `pdf_extractor.py` (`rag_project/parsing/pdf_extractor.py`)
    - Replace `text[:500]` with `text` in the primary regex search
    - Add fallback scan: `re.search(PAGE_NUMBER_PATTERN, text[-200:])` when primary scan returns `None`
    - _Bug_Condition: `isBugCondition_A7` — `page_number_pattern_found(text[500:]) AND NOT page_number_pattern_found(text[:500])`_
    - _Expected_Behavior: Property 6 — footer-position page numbers found from full text including last 200 chars_
    - _Preservation: Req 3.3 — pages with page numbers in first 500 chars continue to work_
    - _Requirements: 2.11, 3.3_

  - [~] 3.8 Fix B1 — NFKC destroys superscripts/subscripts in `text_utils.py` (`rag_project/utils/text_utils.py`)
    - Implement `_protect_superscripts(text)` using `str.translate` over U+2070–U+2079, U+00B2, U+00B3, U+00B9 → `^N` notation
    - Implement `_protect_subscripts(text)` using `str.translate` over U+2080–U+2089 → `_N` notation
    - Before `unicodedata.normalize("NFKC", text)`, call both protection functions
    - _Bug_Condition: `isBugCondition_B1` — `any(char in text for char in SUPERSCRIPT_CHARS + SUBSCRIPT_CHARS) AND normalize_mode == "NFKC_unrestricted"`_
    - _Expected_Behavior: Property 7 — `"CO₂"` → `"CO_2"` (not `"CO2"`); formula semantics preserved_
    - _Preservation: Req 3.6 — ASCII prose normalization, whitespace collapse, strip unchanged_
    - _Requirements: 2.12, 3.6_

  - [~] 3.9 Fix B2 — Hyphenated line-break not joined in `text_utils.py` (`rag_project/utils/text_utils.py`)
    - At the top of `split_paragraphs`, add: `text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)`
    - _Bug_Condition: `isBugCondition_B2` — `re.search(r'\w+-\n\w+', text) != None AND hyphen_newline_joining == False`_
    - _Expected_Behavior: Property 8 — `"anti-\nhypertensive"` → `"antihypertensive"` as single searchable token_
    - _Preservation: Req 3.6 — paragraph splitting on `\n{2,}` and other text_utils behavior unchanged_
    - _Requirements: 2.14, 3.6_

  - [~] 3.10 Fix B3 — OCR post-correction missing — create `ocr_corrector.py` and wire up (`rag_project/ocr/ocr_corrector.py`, `rag_project/intelligence/pdf_intelligence.py`)
    - Create new file `rag_project/ocr/ocr_corrector.py` with `OCR_CORRECTIONS` dict:
      - `r'(?<=\d)O(?=\d)'` → `'0'` (letter O between digits = zero)
      - `r'(?<=\d)l(?=\d)'` → `'1'` (lowercase L between digits = one)
      - `r'\brn\b'` → `'m'` for known medical abbreviation false-splits
      - Drug-name specific patterns: `Metf0rmin` → `Metformin`, `Am0xicillin` → `Amoxicillin`
    - Expose `apply_ocr_corrections(text: str) -> str`
    - In `rag_project/intelligence/pdf_intelligence.py`, inside `enrich_text`, after OCR text is available, call `apply_ocr_corrections(text)`
    - _Bug_Condition: `isBugCondition_B3` — `contains_numeric_drug_pattern AND ocr_correction_pass_applied == False`_
    - _Expected_Behavior: Property 9 — `"Metf0rmin 500mg"` → `"Metformin 500mg"` before entering index_
    - _Preservation: Req 3.6 — clean text without confusable patterns passes through correction unchanged_
    - _Requirements: 2.16, 3.6_

  - [~] 3.11 Fix C1 — Chapter state not carried across batch boundaries in `semantic_chunker.py` (`rag_project/chunking/semantic_chunker.py`)
    - Add `initial_chapter: Optional[str] = None` parameter to `chunk_pages`
    - Initialize `document_chapter = initial_chapter` instead of `None`
    - Update `chunk_pages` to return `(chunks, final_chapter)` tuple
    - In `chunk_page_batches`, thread running chapter state: `current_chapter = None; for batch in batches: chunks, current_chapter = chunk_pages(batch, initial_chapter=current_chapter)`
    - _Bug_Condition: `isBugCondition_C1` — `batch_index > 0 AND preceding_chapter_state != None AND chunk_pages_receives_chapter_state == False`_
    - _Expected_Behavior: Property 10 — all chunks from batch 2+ inherit correct chapter from preceding batch_
    - _Preservation: Req 3.10 — RAG-STRUCTURE markers with chapter_id/section_id/parent_id still emitted for pages with headings_
    - _Requirements: 2.19, 3.10_

  - [~] 3.12 Fix C2 — Cross-document section ID collision in `semantic_chunker.py` (`rag_project/chunking/semantic_chunker.py`)
    - In `_stable_id`, change hash key from `f"{chapter}|{section}"` to `f"{document_id}|{chapter}|{section}"`
    - Apply same change to `parent_id` key
    - _Bug_Condition: `isBugCondition_C2` — `doc_a.chapter == doc_b.chapter AND doc_a.section == doc_b.section AND document_id_not_in_hash_key`_
    - _Expected_Behavior: Property 11 — distinct `section_id` values for same chapter/section titles across different documents_
    - _Preservation: Req 3.10 — stable IDs remain deterministic for same document/chapter/section inputs_
    - _Requirements: 2.20, 3.10_

  - [~] 3.13 Fix C3 — Lexical rows not purged before re-ingestion (`rag_project/ingestion/robust_ingestor.py`, `rag_project/knowledge/vector_store.py`)
    - In `vector_store.py`, implement `delete_lexical_version(document_id, version_id)`: `DELETE FROM lexical_index WHERE document_id = ? AND version_id = ?`
    - In `robust_ingestor.py`, after `vector_store.delete_version(document_id, version_id)`, add: `vector_store.delete_lexical_version(document_id, version_id)`
    - _Bug_Condition: `isBugCondition_C3` — `prior_ingestion_interrupted AND lexical_rows_exist(doc_id, version_id) AND delete_version_does_not_clear_lexical`_
    - _Expected_Behavior: Property 12 — zero stale lexical rows before new ingestion begins; parity restored_
    - _Preservation: Req 3.16, 3.18 — `add_documents` atomic insert and `delete_version` Chroma removal unchanged_
    - _Requirements: 2.21, 3.16, 3.18_

  - [~] 3.14 Fix D1 — Test-mode dimension mismatch in `embedding_service.py` (`rag_project/embeddings/embedding_service.py`)
    - In `_test_embedding`, replace hardcoded 16-element list with: `seed = int(hashlib.md5(text.encode()).hexdigest(), 16); rng = random.Random(seed); vec = [rng.uniform(-1, 1) for _ in range(self.dimension)]; magnitude = sum(v**2 for v in vec) ** 0.5; return [v / magnitude for v in vec]`
    - _Bug_Condition: `isBugCondition_D1` — `test_mode == True AND test_embedding_dim != configured_dim`_
    - _Expected_Behavior: Property 13 — test vectors have length `self.dimension`; no Chroma schema corruption_
    - _Preservation: Req 3.14, 3.15 — embed_texts returns vectors in input order; LRU cache behavior unchanged_
    - _Requirements: 2.23, 3.14, 3.15_

  - [~] 3.15 Fix D2 — Embeddings not normalized in `embedding_service.py` (`rag_project/embeddings/embedding_service.py`)
    - In `_transformers_embed_batch`, change `encode(...)` to `encode(..., normalize_embeddings=True)`
    - _Bug_Condition: `isBugCondition_D2` — `encode_kwargs.get("normalize_embeddings") != True`_
    - _Expected_Behavior: Property 14 — all stored vectors are unit-length (L2 norm ≈ 1.0); cosine similarity correct_
    - _Preservation: Req 3.14 — embed_texts continues to return vectors in input order_
    - _Requirements: 2.24, 3.14_

  - [~] 3.16 Fix D3 — page_numbers stored as JSON string in `vector_store.py` (`rag_project/knowledge/vector_store.py`)
    - In `_coerce_metadata`, for `page_numbers` lists: `metadata["page_numbers_csv"] = ",".join(str(p) for p in value); metadata["page_numbers"] = json.dumps(value)`
    - Add `_parse_page_numbers(meta)` helper: try `page_numbers_csv` first (split on comma), then JSON parse `page_numbers`, then return as-is if already a list
    - Update all retrieval code paths that read `page_numbers` to use `_parse_page_numbers(meta)` (see also E1 and citation builder)
    - _Bug_Condition: `isBugCondition_D3` — `isinstance(page_numbers, list) AND storage_form == "json_string_only"`_
    - _Expected_Behavior: Property 15 — `page_numbers_csv` key present and parseable; citation builder receives list not string_
    - _Preservation: Req 3.16 — `add_documents` atomic insert unchanged; backward-compatible JSON form retained_
    - _Requirements: 2.28, 3.16_

  - [~] 3.17 Fix E1 — Chroma filter on JSON-string field in `hybrid_retriever.py` (`rag_project/retrieval/hybrid_retriever.py`)
    - Add `_adapt_filter(where_filter: dict) -> dict` helper: for fields in `STRING_SERIALIZED_FIELDS = ["page_numbers", "chapter_id", "section_id"]`, convert integer `$eq` values to check against `page_numbers_csv` using CSV-membership pattern
    - Call `_adapt_filter` on every `where_filter` before passing to vector and lexical stores
    - _Bug_Condition: `isBugCondition_E1` — `any(field in STRING_SERIALIZED_FIELDS for field in where_filter.keys()) AND filter_not_adapted`_
    - _Expected_Behavior: Property 16 — filter on `page_numbers=5` returns correct non-empty results_
    - _Preservation: Req 3.19, 3.20 — RRF fusion in hybrid mode and lexical fallback unchanged_
    - _Requirements: 2.31, 3.19, 3.20_

  - [~] 3.18 Fix E2 — Token budget undercounting in `context_builder.py` (`rag_project/retrieval/context_builder.py`)
    - Replace `estimated = len(text.split()) * 4 // 3` with `estimated = max(len(text.split()) * 4 // 3, len(text) // 4)`
    - _Bug_Condition: `isBugCondition_E2` — `(actual_tokens - estimated_tokens) / actual_tokens > 0.3 AND text_contains_abbreviation_density`_
    - _Expected_Behavior: Property 17 — context token count ≤ budget × 1.05 for abbreviation-dense medical text_
    - _Preservation: Req 3.21 — `max_per_document` limit enforcement unchanged_
    - _Requirements: 2.34, 3.21_

  - [~] 3.19 Fix E3 — `pass` instead of `continue` in `context_builder.py` (`rag_project/retrieval/context_builder.py`)
    - In the budget exception branch, change `pass` to `continue`
    - _Bug_Condition: `isBugCondition_E3` — `exception_branch_uses_pass == True AND candidates_after_exception_evaluated == False`_
    - _Expected_Behavior: Property 18 — loop advances after exception branch; all remaining candidates evaluated_
    - _Preservation: Req 3.21 — `max_per_document` limit and token budget enforcement still apply to all candidates_
    - _Requirements: 2.35, 3.21_

  - [~] 3.20 Fix E4 — Empty LLM response no fallback in `ollama_llm_client.py` (`rag_project/generation/ollama_llm_client.py`)
    - After `_content = extract_content(response)`, add: `if not _content: raise RuntimeError("LLM returned empty response")`
    - _Bug_Condition: `isBugCondition_E4` — `extract_content(response) == "" AND no_error_raised == True`_
    - _Expected_Behavior: Property 19 — `RuntimeError` raised; deterministic fallback path triggered_
    - _Preservation: Req 3.22 — `DeterministicAnswerGenerator` continues to produce `SynthesizedAnswer` with citations_
    - _Requirements: 2.37, 3.22_

  - [~] 3.21 Fix F1 — Stale-lease recovery missing vector cleanup in `state_store.py` (`rag_project/ingestion/state_store.py`)
    - In `recover_stale_documents`, before resetting each stale document's lease, add: `self._vector_store.delete_version(doc_id, version_id); self._vector_store.delete_lexical_version(doc_id, version_id)`
    - _Bug_Condition: `isBugCondition_F1` — `partial_vectors_in_chroma == True AND delete_version_called_during_recovery == False`_
    - _Expected_Behavior: Property 20 — `delete_version` called; 0 Chroma vectors for stale document before reset_
    - _Preservation: Req 3.24 — only documents with expired leases are reset; non-expired documents unaffected_
    - _Requirements: 2.39, 3.24_

  - [~] 3.22 Fix F2 — SQLite backup without WAL checkpoint (operational backup workflow/scripts)
    - In the backup routine, before each `shutil.copy(db_path, backup_path)`, add: `with sqlite3.connect(db_path) as conn: conn.execute("PRAGMA wal_checkpoint(FULL)")`
    - Apply to all SQLite databases: `ingestion.sqlite3`, `semantic_cache.sqlite3`, `med_evidence_cache.sqlite3`, `med_evidence_feedback.sqlite3`, `med_evidence_ops.sqlite3`, `lexical.sqlite3`
    - _Bug_Condition: `isBugCondition_F2` — `wal_active == True AND wal_checkpoint_executed_before_copy == False`_
    - _Expected_Behavior: Property 21 — WAL pages merged into main file; backup openable without WAL recovery_
    - _Preservation: Non-WAL databases (if any) unaffected by the checkpoint call_
    - _Requirements: 2.40_

  - [~] 3.23 Fix F3 — Integration tests bypass `prepare_runtime` (all affected files under `rag_project/testing/`)
    - Replace all `MedEvidenceProductionRAGSystem(...)` direct constructions in integration tests with: `from rag_project.composition import prepare_runtime; system = prepare_runtime(config=test_config)`
    - _Bug_Condition: `isBugCondition_F3` — `constructs_rag_system_directly == True AND calls_prepare_runtime == False`_
    - _Expected_Behavior: Property 22 — `prepare_runtime()` called first; tests validate production-equivalent behavior_
    - _Preservation: All existing test assertions about system behavior remain valid_
    - _Requirements: 2.41_

  - [~] 3.24 Verify bug condition exploration tests now pass (all 23)
    - **Property 1: Expected Behavior** - 23-Zone RAG Pipeline Bug Exploration Suite
    - **IMPORTANT**: Re-run the SAME tests from task 1 — do NOT write new tests
    - The tests from task 1 encode the expected behavior for each of the 23 bugs
    - When all 23 tests pass, it confirms all expected behaviors are satisfied
    - Run all exploration tests: A1–A7, B1–B3, C1–C3, D1–D3, E1–E4, F1–F3
    - **EXPECTED OUTCOME**: All 23 tests PASS (confirms all 23 bugs are fixed)
    - _Requirements: 2.1, 2.2, 2.5, 2.6, 2.7, 2.10, 2.11, 2.12, 2.14, 2.16, 2.19, 2.20, 2.21, 2.23, 2.24, 2.28, 2.31, 2.34, 2.35, 2.37, 2.39, 2.40, 2.41_

  - [~] 3.25 Verify preservation property tests still pass
    - **Property 2: Preservation** - Unchanged RAG Pipeline Behavior for Non-Buggy Inputs
    - **IMPORTANT**: Re-run the SAME tests from task 2 — do NOT write new tests
    - Run all preservation property-based tests: `prop_normal_ingestion_unchanged`, `prop_clean_text_ascii_unchanged`, `prop_split_paragraphs_no_hyphen_unchanged`, `prop_single_batch_chunking_unchanged`, `prop_production_embeddings_normalized_order`, `prop_retrieval_no_serialized_field_filter`, `prop_context_builder_non_abbreviation_text`, `prop_ocr_correction_clean_text_unchanged`
    - **EXPECTED OUTCOME**: All preservation tests PASS (confirms no regressions introduced)
    - Confirm all preservation tests still pass after all 23 fixes (no regressions)
    - _Requirements: 3.1–3.27_

---

- [~] 4. Checkpoint — Ensure all tests pass
  - Run the full test suite: all 23 exploration tests (now expected to pass) and all preservation property-based tests
  - Verify zero test failures across all 6 zones (A through F)
  - Confirm that the 88-stage pipeline can process `data/processed/DC_endocrino_version_2024__portrait_sans_astuces_76ac07d1c09e.pdf` end-to-end with no errors
  - Ensure all chunks have non-null chapter metadata (validates fix C1)
  - Ensure all `page_numbers_csv` fields are present in Chroma metadata (validates fix D3)
  - Ensure lexical/vector parity holds across all documents (validates fix C3 and F1)
  - Ask the user if any questions arise during final verification
