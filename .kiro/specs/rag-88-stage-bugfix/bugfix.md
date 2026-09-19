# Bugfix Requirements Document

## Introduction

This document covers systematic verification and remediation of all 88 pipeline stages of the RAG medical document ingestion and retrieval system at `mohaned_rag_med-master`. The pipeline spans PDF ingestion (stages 1–17), text processing (stages 18–31), chunking (stages 32–43), embedding and vector storage (stages 44–60), retrieval and answer generation (stages 61–78), and testing, monitoring, and operations (stages 79–88).

A thorough code review of the live codebase was conducted before authoring this document. Each section below documents: what currently breaks (current behavior), what correct behavior looks like (expected behavior), and what existing correct behavior must be preserved (regression prevention).

---

## Bug Analysis

### Current Behavior (Defect)

**Stages 1–17: PDF Ingestion**

1.1 WHEN the `FileMonitor` scans a directory and two PDF files share identical content but different filenames THEN the system records both as separate new entries and passes both to the ingestion pipeline, potentially creating duplicate document records in the state store.

1.2 WHEN `robust_ingest_file` receives a PDF that is already indexed with the same `version_id` and the vector store validation passes THEN the system correctly archives the duplicate — however WHEN the source file is already inside the `processed_dir` (i.e. `file_path.resolve() == target.resolve()`) THEN the `file_path.replace(archived_path)` call operates on the processed file itself, potentially removing the only copy from `processed_dir`.

1.3 WHEN `DocumentClassifier.classify` opens a PDF and iterates all pages to collect `sample_pages = list(range(page_count))` THEN the returned `"sample_pages"` key contains the full page list even for 500-page books, serializing a large integer list into every database event record unnecessarily.

1.4 WHEN `DocumentClassifier.classify` evaluates `doc_type` using the threshold `text_pages / max(page_count, 1) > 0.75` THEN a PDF that is 74% native text and 26% scanned pages is classified as `"scanned_or_ocr_required"` and every page has OCR attempted even when most pages already have adequate text, wasting CPU and time.

1.5 WHEN `PDFExtractor.extract_iter` encounters a page whose OCR is required but fails THEN the page is silently skipped with a `logger.warning` and no page record is written to the state store, making the page invisible to audit and leaving a gap in page-number coverage that citation code cannot detect.

1.6 WHEN `PDFExtractor._extract_page_text` calls `page.get_text("text", sort=True)` on a two-column medical layout THEN PyMuPDF's default text sorting merges column A and column B horizontally, producing interleaved text that mixes clinical content from separate columns into a single undifferentiated block.

1.7 WHEN `PDFExtractor._extract_page_text` falls back to iterating `"blocks"` mode and joining with `"\n"` THEN block order is determined by PyMuPDF's internal block ordering rather than by reading direction, producing incorrect sentence sequencing for right-to-left or vertical scripts.

1.8 WHEN `OCRService._render_pixmap` applies the configured `render_scale` and the resulting pixel count exceeds `max_render_pixels` THEN the scale is reduced to a value that may fall below `0.25`, and the resulting low-resolution pixmap can cause RapidOCR to misread numeric values such as drug dosages.

1.9 WHEN `PDFExtractor.extract_iter` processes a page and OCR succeeds but `_merge_native_and_ocr_text` determines the merged text equals the original native text THEN `ocr_status` is still set to `"completed"` with `extraction_method = "OCR"` even though the text is purely native, causing downstream routing logic to incorrectly apply OCR-quality penalties to the chunk's quality score.

1.10 WHEN `PDFExtractor._extract_figure_captions` iterates page blocks and a caption line matches the pattern but exceeds 1000 characters (e.g., a run-on caption concatenated with body text) THEN the caption is truncated to 1000 characters mid-sentence, destroying the semantic meaning of the caption and breaking figure–caption association.

1.11 WHEN `PDFExtractor._extract_tables` calls `page.find_tables()` on a page with no table-like structure THEN it catches all exceptions broadly with `except Exception: return ""` without distinguishing between expected "no tables found" and a genuine library error, making table-extraction failures invisible in logs.

1.12 WHEN `extract_page_number` searches the first 500 characters of page text for printed page-number patterns THEN on pages where the printed page number appears after the first 500 characters (e.g., bottom-of-page footers after a large title block) THEN the function returns `None`, and the `printed_page_number` metadata field is left null even though a valid label exists.

**Stages 18–31: Text Processing**

1.13 WHEN `clean_text` applies `unicodedata.normalize("NFKC", ...)` to text that contains Unicode superscripts used in chemical formulas (e.g., `²`, `³`, `⁴`) or subscripts used in molecular notation (e.g., `₂`, `H₂O`) THEN NFKC normalization replaces them with plain digits or letters, destroying formula structure and making dosage expressions ambiguous.

1.14 WHEN `clean_text` replaces `\xa0` (non-breaking space) with a regular space before other whitespace normalization THEN it also collapses meaningful en-dash separators and preserves soft-hyphen characters (`\u00ad`) that are invisible in display but disrupt tokenization of hyphenated drug names.

1.15 WHEN `split_paragraphs` splits on `\n{2,}` and a hyphenated word straddles a line break (e.g., `anti-\nhypertensive`) THEN the hyphen–newline sequence is preserved as two tokens `"anti-"` and `"hypertensive"` in separate paragraphs or blocks, causing lexical search to miss the full term `"antihypertensive"`.

1.16 WHEN `tokenize` uses `re.findall(r"[\w]+(?:[/.-][\w]+)*", ...)` to tokenize text containing chemical notation like `NaCl`, `H₂O` or ranges like `120/80 mmHg` THEN the slash-separated token `120/80` is kept as a single unit, which is correct, but the subscript digits previously normalized by NFKC are already gone, making chemical formulae unsearchable.

1.17 WHEN `detect_language` applies a heuristic regex looking for French function words (`le`, `la`, `les`, `des`, etc.) THEN Arabic-Latin mixed text (e.g., a French medical textbook with Arabic patient notes) may be misclassified as French or English, causing the wrong tokenization and synonym-expansion path to be used.

1.18 WHEN `enrich_text` (called from `pdf_intelligence`) processes OCR output that contains common RapidOCR misrecognitions (e.g., `"0"` for `"O"`, `"l"` for `"1"`) in drug names or dosages THEN no correction is applied because the OCR cleanup stage has no domain-specific post-processing dictionary, so numeric dosage errors silently propagate into the index.

1.19 WHEN `score_page_quality` (called from `pdf_intelligence`) assigns a quality score THEN pages where OCR produced garbled text that passed the confidence threshold (≥0.55) still receive a normal quality score, and the low-quality OCR text enters retrieval without any quality penalty.

**Stages 32–43: Chunking**

1.20 WHEN `SemanticChunker.chunk_pages` calls `_parent_sections` and the page text contains no detectable headings THEN `rows` defaults to `[(prose, {})]` and the entire page prose is passed as a single parent block to `_child_splitter`, which may produce a chunk exceeding 700 characters when a page has dense body text with no double-newline paragraph breaks.

1.21 WHEN `SemanticChunker.chunk_pages` produces the RAG-STRUCTURE marker for an unstructured page it emits `"[RAG-STRUCTURE schema=3; page={page_no}]"` without chapter, section, or quality metadata, so downstream retrieval has no hierarchy context and cannot apply section-level scoring or boosting for these chunks.

1.22 WHEN `SemanticChunker.chunk_page_batches` resets the `offset` counter for each batch correctly BUT `chunk_pages` is called independently per batch THEN the `document_chapter` carry-forward state (which tracks the most recent chapter heading for cross-page inheritance) is reset to `None` between batches, breaking chapter context inheritance for chapters that span multiple page batches.

1.23 WHEN `_stable_id` generates a chunk ID from a section hierarchy key like `"{chapter}|{section}"` THEN two sections in different documents with identical chapter and section titles produce the same `section_id` and `parent_id`, causing cross-document collision in the vector store metadata and incorrect hierarchy linking.

1.24 WHEN a duplicate PDF is re-ingested after the first ingestion was interrupted mid-way THEN `robust_ingest_file` calls `system.vector_store.delete_version(document_id, current_version_id)` before extraction, but if the previous partial ingestion did not commit all chunks atomically THEN orphaned chunk records may remain in the lexical index (SQLite) while the Chroma vector index is cleared, creating a parity gap.

1.25 WHEN `SemanticChunker._compact_children` de-duplicates child fragments that are too short THEN very short chunks (e.g., a standalone heading like "Chapter 3") are kept in the output list because they pass the minimum character check, producing junk single-word vectors that pollute the embedding space.

**Stages 44–60: Embedding and Vector Storage**

1.26 WHEN `EmbeddingService.__init__` is called with `test_mode=True` THEN `_test_embedding` generates a deterministic vector from `hashlib.md5` regardless of the configured embedding model's actual dimension, so if the production model produces 768-dimensional vectors but test mode produces 16-dimensional vectors THEN the test vectors are silently inserted into the same Chroma collection, causing a dimension mismatch that crashes all subsequent production queries.

1.27 WHEN `EmbeddingService._transformers_embed_batch` calls `encode` with `normalize_embeddings=False` THEN raw (un-normalized) vectors are stored in Chroma, but cosine-similarity search in Chroma expects unit-length vectors; without normalization, similarity scores degrade and dissimilar chunks may rank above similar ones.

1.28 WHEN `EmbeddingService.embed_texts` is called with a batch containing a very long text (> 12000 characters after `sanitize_model_text` truncation) THEN the text is truncated, but the truncation point may split in the middle of a sentence or medical term, creating an incomplete embedding that does not represent the full semantic content.

1.29 WHEN `VectorStore.add_documents` is called and the Chroma collection does not yet exist THEN `_resolve_dimension` creates the collection with the first batch's embedding dimension; if the first batch is from test mode and later batches are from production mode THEN the stored dimension is wrong and subsequent inserts raise a `DimensionMismatch` error.

1.30 WHEN `VectorStore._valid_vector` checks `expected_dimension` THEN it accepts `expected_dimension=0` as "no check", so vectors with incorrect dimensions can be inserted without triggering validation when the dimension has not yet been resolved from the first batch.

1.31 WHEN `VectorStore.add_documents` calls `_coerce_metadata` to sanitize metadata THEN `list` and `dict` fields are JSON-serialized to strings, but `page_numbers` which is a `list[int]` is serialized to a JSON string like `"[1, 2]"` — when retrieval later reads back `page_numbers` from Chroma metadata it receives a string instead of a list, causing citation generation to fail or produce malformed page references.

1.32 WHEN `VectorStore.reconcile_index` checks parity between Chroma vector count and lexical index row count THEN it counts total rows across all documents; if one document has extra lexical rows from a failed partial ingestion these are counted globally and the reconciliation reports "OK" as long as the sums match, even though individual documents have mismatched sub-counts.

1.33 WHEN `robust_ingest_file` validates the index after ingestion with `validate_document_index` and the count matches `embedding_count` THEN it considers the ingestion successful; however `embedding_count` only tracks vectors committed in the current run — if a previous partial run left stale vectors for the same `document_id` and `version_id` THEN `validate_document_index` may count both new and stale vectors, exceeding `embedding_count` and causing a false failure, or the counts may accidentally match while stale data is present.

**Stages 61–78: Retrieval and Answer Generation**

1.34 WHEN `HybridRetriever.retrieve` applies the `where` filter to both vector and lexical queries THEN the filter is passed as a Chroma `$where` clause; if the filter references a metadata field that was stored as a JSON-serialized string (e.g., `page_numbers`) THEN the Chroma filter silently fails to match any document, returning an empty result set rather than raising an error.

1.35 WHEN `HybridRetriever.retrieve` runs lexical and vector searches concurrently using `ThreadPoolExecutor` and the lexical search raises an exception THEN `lexical_error` is set but the error is never surfaced to the caller — the fallback to simplified query silently ignores the original error message, making it impossible to diagnose why retrieval returned partial results.

1.36 WHEN `HybridRetriever._intent_bonus` calculates a relevance bonus THEN the bonus is applied only when `effective_mode != "lexical"`, but the bonus calculation itself is not bounded per document — a single document with many matching intent tokens can receive a cumulative bonus that pushes its score above 2.0 (the stated ceiling of `min(2.0, hit.score + bonus)`), inadvertently suppressing diverse multi-document evidence.

1.37 WHEN `ContextBuilder.build` estimates token count as `len(text.split()) * 4 // 3` THEN for dense medical abbreviation text like `"HR 72 bpm, BP 120/80 mmHg, SpO2 98%, RR 16 bpm"` the word-based estimate undercounts actual tokens by 30–50%, causing the context to significantly exceed the intended `token_budget` and risk LLM context-window overflow.

1.38 WHEN `ContextBuilder.build` allows a high-quality hit (score > 0.7) to exceed the budget by up to 20% THEN this exception is applied inside the main selection loop after `break` would normally stop iteration — because the exception uses `pass` rather than `continue`, the loop does not advance to the next candidate, effectively preventing any further candidates from being considered even if a lower-quality candidate would fit within budget.

1.39 WHEN `DeterministicAnswerGenerator.generate_answer` assembles evidence from retrieval hits THEN it uses `AdvancedEvidenceSynthesizer.synthesize` which deduplicates sentences using a text-similarity threshold — but this deduplication does not account for medical paraphrases (e.g., `"renal failure"` vs `"kidney failure"`), so semantically identical evidence from two sources may both be included while superficially similar evidence is incorrectly removed.

1.40 WHEN `DeterministicAnswerGenerator._build_comprehensive_citations` creates citation strings THEN page numbers are read from `hit.metadata.get("page_numbers", [])` — because `page_numbers` was stored as a JSON string (see 1.31) THEN the citation builder receives `"[1, 2]"` (string) instead of `[1, 2]` (list) and the citation either raises a `TypeError` or emits a malformed string citation like `"page '[1, 2]'"`.

1.41 WHEN the LLM client `OllamaLLMClient.generate` calls the Ollama API and the response is empty THEN `_content` extracts an empty string and `generate` returns it without error, causing the answer service to emit an empty answer to the user with no indication that the model returned nothing.

1.42 WHEN the answer service processes a question whose retrieved evidence comes from multiple document versions (old READY + new READY) THEN neither the retrieval filter nor the context builder enforces single-version isolation, so evidence from the superseded version may mix with evidence from the current version, producing contradictory context.

**Stages 79–88: Testing, Monitoring, and Operations**

1.43 WHEN `IngestionStateStore.recover_stale_documents` resets documents whose leases have expired THEN it does not check whether the Chroma collection contains partial vectors for those documents, so a recovered document may be re-ingested from scratch while its orphaned partial vectors remain queryable in the index.

1.44 WHEN the backup/restore workflow is invoked (Stage 87) and the SQLite databases (`ingestion.sqlite3`, `semantic_cache.sqlite3`, etc.) are copied while ingestion is in progress THEN the WAL (write-ahead log) may not be checkpointed, producing a backup that is internally inconsistent and cannot be opened without SQLite's WAL recovery.

1.45 WHEN end-to-end pipeline tests (Stage 79) construct the RAG system directly via `MedEvidenceProductionRAGSystem` without going through `rag_project.composition.prepare_runtime` THEN the architecture gate invariant ("every production entry point must pass through the composition/runtime boundary") is violated, meaning integration tests validate a configuration that does not match production behavior.

---

### Expected Behavior (Correct)

**Stages 1–17: PDF Ingestion**

2.1 WHEN the `FileMonitor` scans a directory and detects two files with the same SHA-256 digest THEN the system SHALL flag both entries with `"duplicate": true` AND the ingestion pipeline SHALL process only the first alphabetically-sorted duplicate and skip the second, recording a `"skipped_duplicate"` event in the state store for the skipped file.

2.2 WHEN `robust_ingest_file` determines that an identical document version is already indexed AND the source file resolves to the same path as the processed target THEN the system SHALL NOT attempt to move or replace the file; it SHALL return `{"status": "skipped", ...}` without any file operation.

2.3 WHEN `DocumentClassifier.classify` returns its result THEN the `"sample_pages"` key SHALL contain only the count of sampled pages (an integer), not the full list of page indices, to prevent unbounded serialization into event records.

2.4 WHEN `DocumentClassifier.classify` evaluates document type THEN the threshold for `"text_based"` SHALL be `>= 0.60` (not `> 0.75`) so that a document with 60% or more native-text pages is classified as text-based, reducing unnecessary OCR invocation.

2.5 WHEN `PDFExtractor.extract_iter` skips a page because OCR was required but failed THEN the system SHALL write a page record to the state store with `extraction_status = "FAILED"` and `ocr_status = "failed"` so the page appears in audit queries and downstream citation code can detect the gap.

2.6 WHEN `PDFExtractor._extract_page_text` processes a page whose block layout suggests two or more columns (determined by a heuristic: multiple blocks at significantly different x-coordinates) THEN the system SHALL sort blocks by column first, then by vertical position within each column, so column A text precedes column B text in the output.

2.7 WHEN `OCRService._render_pixmap` must reduce scale to stay within `max_render_pixels` THEN the minimum permissible scale SHALL be `0.5` (not `0.25`), and if the image still exceeds the limit at `0.5` the system SHALL raise a `RuntimeError` rather than silently rendering a degraded image, so the page is handled as an OCR failure rather than producing unreliable dosage text.

2.8 WHEN OCR succeeds but the merged text equals the original native text THEN `extraction_method` SHALL remain `"native_text"` and `ocr_status` SHALL be set to `"not_required"` to prevent false OCR quality penalties.

2.9 WHEN `PDFExtractor._extract_figure_captions` detects a caption line longer than 500 characters THEN it SHALL truncate to the nearest sentence boundary (`.`) within the first 500 characters rather than truncating at exactly 1000, and SHALL log a warning that a long caption was encountered.

2.10 WHEN `PDFExtractor._extract_tables` encounters an exception from `page.find_tables()` THEN it SHALL distinguish `AttributeError` (PyMuPDF version does not support `find_tables`) from other exceptions: `AttributeError` is silently ignored, all other exceptions SHALL be logged at WARNING level with the page number.

2.11 WHEN `extract_page_number` searches for printed page labels THEN it SHALL search the full page text (not limited to the first 500 characters), including a scan of the last 200 characters where footers commonly appear.

**Stages 18–31: Text Processing**

2.12 WHEN `clean_text` normalizes Unicode THEN it SHALL preserve superscript and subscript characters (`\u2070–\u209f`, `\u00b2`, `\u00b3`) by converting them to their semantic equivalents with explicit notation (e.g., `²` → `^2`) rather than stripping them, so chemical formulas remain unambiguous.

2.13 WHEN `clean_text` processes text THEN it SHALL also strip soft-hyphen characters (`\u00ad`) that are invisible but disrupt tokenization, while leaving en-dash (`–`) and em-dash (`—`) intact.

2.14 WHEN `split_paragraphs` detects a line ending with a hyphen followed immediately by a newline and a continuation word THEN the system SHALL join the two parts into a single unhyphenated token (e.g., `"anti-\nhypertensive"` → `"antihypertensive"`), then proceed with paragraph splitting.

2.15 WHEN `detect_language` processes text that contains both Arabic and Latin characters in roughly equal proportions THEN the system SHALL return `"mixed"` rather than defaulting to either script's language, and downstream routing SHALL apply bilingual tokenization.

2.16 WHEN OCR output is available THEN the text-processing pipeline SHALL apply a domain-specific post-processing pass that corrects common OCR confusables in medical context: `"0"` (zero) and `"O"` (letter O) near numeric dosage patterns, `"l"` (lowercase L) and `"1"` (one) in numeric contexts, and `"rn"` misread as `"m"`.

2.17 WHEN `score_page_quality` evaluates OCR-produced text THEN it SHALL apply an OCR quality penalty of 0.15 to pages where `ocr_status == "completed"` and the raw OCR confidence is below 0.75, so low-confidence OCR pages receive lower quality scores and are retrieved with reduced priority.

**Stages 32–43: Chunking**

2.18 WHEN `SemanticChunker.chunk_pages` processes a page with no detectable headings THEN for each child chunk it SHALL still emit a minimal RAG-STRUCTURE marker that includes the page number, OCR status, and quality score, ensuring all chunks carry searchable provenance metadata.

2.19 WHEN `SemanticChunker.chunk_page_batches` processes pages in batches THEN the `document_chapter` carry-forward variable SHALL be threaded as an explicit parameter through `chunk_pages` or stored in a shared mutable container, so chapter context is preserved across batch boundaries.

2.20 WHEN `_stable_id` generates an identifier for a section hierarchy key THEN the key SHALL include the `document_id` prefix before hashing (e.g., `f"{document_id}|{chapter}|{section}"`) so that identical chapter/section titles in different documents produce different stable IDs.

2.21 WHEN `robust_ingest_file` deletes a partial index before re-ingestion using `delete_version` THEN it SHALL also call `VectorStore` to delete orphaned lexical index rows for the same `(document_id, version_id)` pair, ensuring vector and lexical parity before the new ingestion begins.

2.22 WHEN `SemanticChunker._compact_children` processes child text fragments THEN it SHALL discard any fragment that contains fewer than 20 characters OR consists entirely of heading-like text (matching `_is_heading_line`) that duplicates the section heading already encoded in the chunk metadata, to avoid junk single-phrase vectors.

**Stages 44–60: Embedding and Vector Storage**

2.23 WHEN `EmbeddingService` is in `test_mode` THEN `_test_embedding` SHALL generate vectors whose dimension matches `self.dimension` (the configured embedding dimension), so test vectors are dimensionally compatible with the production collection and do not corrupt the Chroma schema.

2.24 WHEN `EmbeddingService._transformers_embed_batch` calls `encode` THEN it SHALL set `normalize_embeddings=True` so all stored vectors are unit-length and cosine similarity computations in Chroma are correct.

2.25 WHEN `sanitize_model_text` truncates a text to the 12000-character limit THEN it SHALL truncate at the last sentence boundary (`.` or `\n`) before the limit rather than at a character boundary, preserving semantic completeness of the embedded text.

2.26 WHEN `VectorStore._resolve_dimension` creates a new Chroma collection THEN it SHALL store the resolved dimension in collection metadata and validate all subsequent inserts against it, refusing inserts with mismatched dimensions rather than silently accepting them.

2.27 WHEN `VectorStore._valid_vector` is called with `expected_dimension=0` THEN it SHALL treat `0` as "unresolved" and raise a `ValueError` rather than returning `True`, forcing callers to explicitly resolve dimension before validation.

2.28 WHEN `VectorStore._coerce_metadata` serializes `page_numbers` THEN it SHALL store it as a comma-separated string in a dedicated key `"page_numbers_csv"` (e.g., `"1,2"`) AND also retain the serialized JSON in `"page_numbers"`, so retrieval code that expects a list can parse `"page_numbers_csv"` without a JSON parse failure.

2.29 WHEN `VectorStore.reconcile_index` checks index parity THEN it SHALL reconcile counts per-document (not globally), and SHALL report per-document discrepancies rather than a single global sum, so individual document corruption is detectable.

2.30 WHEN `validate_document_index` counts vectors for a `(document_id, version_id)` pair THEN it SHALL count only vectors whose `index_state == "READY"` and whose `version_id` matches, excluding stale or partial vectors from the count, so the validation accurately reflects only the current committed ingestion.

**Stages 61–78: Retrieval and Answer Generation**

2.31 WHEN `HybridRetriever.retrieve` passes a `where` filter to the vector store THEN it SHALL first parse the filter to detect metadata fields that were serialized as strings during ingestion, and SHALL convert filter values to their serialized form (e.g., integer `1` → string comparison against `"page_numbers_csv"` containing `"1"`) to ensure the filter matches correctly.

2.32 WHEN `HybridRetriever.retrieve` encounters a lexical search error THEN it SHALL log the full exception at ERROR level including the query and filter parameters, so the failure is diagnosable without a debugger.

2.33 WHEN `HybridRetriever._intent_bonus` calculates a per-hit bonus THEN the cumulative bonus per document SHALL be capped at `0.3` to prevent any single document from monopolizing the top-k results.

2.34 WHEN `ContextBuilder.build` estimates token count THEN it SHALL use `max(len(text.split()), len(text) // 4)` where the character-based estimate (`len(text) // 4`) acts as a conservative floor for token-dense medical abbreviation text, preventing context-window overflow.

2.35 WHEN `ContextBuilder.build` evaluates whether to allow a high-quality hit past the budget THEN after deciding to allow it the code SHALL use `continue` (not `pass`) so the selection loop proceeds to evaluate remaining candidates.

2.36 WHEN `DeterministicAnswerGenerator.generate_answer` reads `page_numbers` from hit metadata THEN it SHALL parse it robustly: if the value is a string it SHALL attempt `json.loads(value)` first, then split on commas, before raising an error, ensuring citation generation is never blocked by serialization format differences.

2.37 WHEN `OllamaLLMClient.generate` receives an empty response from the model THEN it SHALL raise a `RuntimeError("LLM returned empty response")` rather than returning an empty string, so the answer service can fall back to the deterministic generator instead of silently emitting a blank answer.

2.38 WHEN retrieval assembles context THEN the retrieval filter SHALL include `{"version_id": {"$eq": current_version_id}}` so that only chunks from the currently active document version are returned, preventing evidence mixing from superseded and active versions.

**Stages 79–88: Testing, Monitoring, and Operations**

2.39 WHEN `IngestionStateStore.recover_stale_documents` resets stale leases THEN it SHALL also call `VectorStore.delete_version` for each stale document's partial vectors, so the re-ingestion starts from a clean index state.

2.40 WHEN any SQLite backup is performed THEN the system SHALL execute `PRAGMA wal_checkpoint(FULL)` on each database connection before copying the files, ensuring WAL pages are merged into the main database file and the backup is consistent.

2.41 WHEN end-to-end integration tests construct the RAG system THEN they SHALL call `rag_project.composition.prepare_runtime()` first, exactly as production does, so test configuration matches the production startup contract.

---

### Unchanged Behavior (Regression Prevention)

**Stages 1–17: PDF Ingestion**

3.1 WHEN a new PDF file not previously seen is submitted to the ingestion pipeline THEN the system SHALL CONTINUE TO run the full ingestion pipeline: classification, extraction, OCR (where required), chunking, embedding, and index publication.

3.2 WHEN a previously ingested PDF with the same content hash and matching `version_id` is re-submitted THEN the system SHALL CONTINUE TO skip re-ingestion and archive the duplicate, returning `{"status": "skipped", ...}`.

3.3 WHEN `PDFExtractor.extract_iter` processes a well-formed single-column text PDF THEN the system SHALL CONTINUE TO produce page extractions with correct text, table metadata, figure captions, and quality scores without invoking OCR.

3.4 WHEN OCR is required for a scanned page and RapidOCR succeeds with confidence ≥ 0.55 THEN the system SHALL CONTINUE TO merge OCR text with any native text, persist the result via `state_store.record_page`, and yield a `PageExtraction` with `ocr_status = "completed"`.

3.5 WHEN `validate_pdf_page_count` is called THEN the system SHALL CONTINUE TO reject PDFs exceeding the configured page limit and raise a security error before any page processing begins.

**Stages 18–31: Text Processing**

3.6 WHEN `clean_text` processes ordinary ASCII prose THEN the system SHALL CONTINUE TO normalize unicode, collapse multiple spaces to one, collapse multiple newlines to at most two, and strip leading/trailing whitespace.

3.7 WHEN `detect_language` processes purely Arabic text THEN the system SHALL CONTINUE TO return `"ar"` and apply Arabic diacritic removal and normalization.

3.8 WHEN `tokenize` processes a term like `"120/80"` THEN the system SHALL CONTINUE TO keep it as a single token `"120/80"` for correct blood-pressure range matching.

3.9 WHEN `keyword_overlap_score` evaluates query–document overlap THEN it SHALL CONTINUE TO use the medical alias lookup table so that `"diabetic"` and `"diabète"` are treated as equivalent tokens.

**Stages 32–43: Chunking**

3.10 WHEN `SemanticChunker.chunk_pages` processes a page with recognized chapter and section headings THEN it SHALL CONTINUE TO emit structured RAG-STRUCTURE markers that include `chapter_id`, `section_id`, `parent_id`, and `hierarchy_path`.

3.11 WHEN `SemanticChunker.chunk_pages` processes table blocks THEN it SHALL CONTINUE TO emit separate `"table"` representation chunks with `[TABLE]` prefixes and `table_id` metadata alongside prose chunks.

3.12 WHEN `SemanticChunker.chunk_pages` processes pages with figure captions THEN it SHALL CONTINUE TO emit separate `"figure_caption"` representation chunks with `[FIGURE CAPTION]` prefixes and `figure_id` metadata.

3.13 WHEN `chunk_page_batches` processes a 500-page document THEN it SHALL CONTINUE TO process pages in bounded batches without loading all pages or all chunks into memory simultaneously.

**Stages 44–60: Embedding and Vector Storage**

3.14 WHEN `EmbeddingService.embed_texts` is called with a batch of texts THEN it SHALL CONTINUE TO return vectors in the same order as the input texts.

3.15 WHEN `EmbeddingService.embed_texts` is called for texts already in the LRU cache THEN it SHALL CONTINUE TO return cached vectors without invoking the model backend.

3.16 WHEN `VectorStore.add_documents` is called THEN it SHALL CONTINUE TO atomically insert both the Chroma vector records and the lexical SQLite rows in the same logical transaction, rolling back both on failure.

3.17 WHEN `VectorStore.search` is called with a query embedding THEN it SHALL CONTINUE TO return results ordered by cosine distance, with metadata and document text intact.

3.18 WHEN `VectorStore.delete_version` removes vectors for a given version THEN it SHALL CONTINUE TO remove corresponding lexical rows from the SQLite index as well, maintaining parity.

**Stages 61–78: Retrieval and Answer Generation**

3.19 WHEN `HybridRetriever.retrieve` is called in `"hybrid"` mode THEN it SHALL CONTINUE TO combine vector and lexical results using the Reciprocal Rank Fusion algorithm with the configured `vector_weight`.

3.20 WHEN `HybridRetriever.retrieve` is called in `"vector"` mode and vector search fails THEN it SHALL CONTINUE TO fall back to lexical search as a last resort.

3.21 WHEN `ContextBuilder.build` selects evidence THEN it SHALL CONTINUE TO enforce the `max_per_document` limit so that no single document contributes more than `max_per_document` chunks to the assembled context.

3.22 WHEN `DeterministicAnswerGenerator.generate_answer` is called THEN it SHALL CONTINUE TO produce a `SynthesizedAnswer` object with confidence score, reasoning steps, and citations even when the LLM backend is unavailable, using the deterministic extraction path as fallback.

3.23 WHEN `OllamaLLMClient.generate` is called and the circuit breaker is open THEN it SHALL CONTINUE TO raise an appropriate error immediately rather than waiting for the full timeout.

**Stages 79–88: Testing, Monitoring, and Operations**

3.24 WHEN `IngestionStateStore.recover_stale_documents` is called THEN it SHALL CONTINUE TO reset only documents whose lease expiry timestamp is in the past.

3.25 WHEN the architecture gate (`scripts/architecture_gate.py`) is run THEN it SHALL CONTINUE TO enforce all existing boundary checks: thin `app.py` ceiling, composition/UI dependency direction, canonical application symbols, wildcard-import hygiene, and Python syntax across all project modules.

3.26 WHEN the ingestion pipeline encounters a `RuntimeError` during embedding THEN it SHALL CONTINUE TO move the failed PDF to the `failed_dir`, persist a `FAILED` status in the state store, and return `{"status": "failed", ...}` to the caller without re-raising.

3.27 WHEN `VectorStore.index_health_check` is called THEN it SHALL CONTINUE TO report counts for both the Chroma vector index and the SQLite lexical index, and flag any parity discrepancy.

---

## Bug Condition Summary

The primary bug conditions that drive this fix are:

```pascal
FUNCTION isBugCondition_FileHandling(X)
  INPUT: X of type IngestionRequest
  OUTPUT: boolean
  RETURN (X.source_path.resolve() == X.target_path.resolve() AND X.is_duplicate)
         OR (X.page has OCR failed AND no page record written)
         OR (X.document has two_column_layout AND sort_mode = "default")
END FUNCTION

FUNCTION isBugCondition_TextProcessing(X)
  INPUT: X of type PageText
  OUTPUT: boolean
  RETURN (X.contains_chemical_notation AND normalize_mode = "NFKC")
         OR (X.contains_hyphenated_line_break)
         OR (X.ocr_status = "completed" AND X.confidence < 0.75 AND quality_penalty = 0)
END FUNCTION

FUNCTION isBugCondition_Chunking(X)
  INPUT: X of type ChunkBatch
  OUTPUT: boolean
  RETURN (X.batch_index > 0 AND document_chapter_state NOT carried forward)
         OR (X.document_id_not_in_section_id_hash)
END FUNCTION

FUNCTION isBugCondition_Embedding(X)
  INPUT: X of type EmbeddingRequest
  OUTPUT: boolean
  RETURN (X.test_mode = True AND X.vector_dim != X.configured_dim)
         OR (X.normalize_embeddings = False)
         OR (X.page_numbers stored as JSON string AND filter expects list)
END FUNCTION

FUNCTION isBugCondition_Retrieval(X)
  INPUT: X of type RetrievalQuery
  OUTPUT: boolean
  RETURN (X.where_filter references string-serialized field)
         OR (X.token_budget_estimate uses word_count only)
         OR (X.loop uses pass instead of continue after budget exception)
END FUNCTION
```

**Fix Checking — FOR ALL X WHERE isBugCondition(X):**
```pascal
FOR ALL X WHERE isBugCondition_FileHandling(X) DO
  result ← ingest'(X)
  ASSERT result.file_not_deleted AND result.page_record_written_on_failure
         AND result.column_order_correct
END FOR

FOR ALL X WHERE isBugCondition_TextProcessing(X) DO
  result ← process_text'(X)
  ASSERT result.chemical_notation_preserved
         AND result.hyphen_joined
         AND result.ocr_quality_penalty_applied
END FOR

FOR ALL X WHERE isBugCondition_Chunking(X) DO
  result ← chunk'(X)
  ASSERT result.chapter_context_preserved_across_batches
         AND result.section_id_unique_per_document
END FOR

FOR ALL X WHERE isBugCondition_Embedding(X) DO
  result ← embed_and_store'(X)
  ASSERT result.test_vector_dim == result.production_vector_dim
         AND result.vectors_normalized
         AND result.page_numbers_parseable_from_metadata
END FOR

FOR ALL X WHERE isBugCondition_Retrieval(X) DO
  result ← retrieve'(X)
  ASSERT result.filter_matches_serialized_fields
         AND result.token_budget_not_exceeded
         AND result.all_candidates_evaluated
END FOR
```

**Preservation Checking — FOR ALL X WHERE NOT isBugCondition(X):**
```pascal
FOR ALL X WHERE NOT isBugCondition(X) DO
  ASSERT ingest(X) = ingest'(X)         // normal ingestion unchanged
         AND chunk(X) = chunk'(X)       // normal chunking unchanged
         AND embed(X) = embed'(X)       // normal embedding unchanged
         AND retrieve(X) = retrieve'(X) // normal retrieval unchanged
END FOR
```
