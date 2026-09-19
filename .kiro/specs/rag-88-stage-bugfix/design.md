# RAG 88-Stage Bugfix Design

## Overview

This design document formalises the bug-condition methodology for the systematic remediation of 45 defects identified across all 88 pipeline stages of the `mohaned_rag_med-master` RAG medical document ingestion and retrieval system.

The bugs are grouped into five pipeline zones:

| Zone | Stages | Bugs | Files Affected |
|------|--------|------|----------------|
| PDF Ingestion | 1–17 | 1.1–1.12 | `ingestion/file_monitor.py`, `ingestion/robust_ingestor.py`, `ingestion/state_store.py`, `parsing/pdf_extractor.py`, `ocr/ocr_service.py` |
| Text Processing | 18–31 | 1.13–1.19 | `parsing/pdf_extractor.py`, `utils/text_utils.py`, `intelligence/pdf_intelligence.py` |
| Chunking | 32–43 | 1.20–1.25 | `chunking/semantic_chunker.py` |
| Embedding & Vector Storage | 44–60 | 1.26–1.33 | `embeddings/embedding_service.py`, `knowledge/vector_store.py`, `ingestion/versioned_ingestor.py` |
| Retrieval & Answer Generation | 61–78 | 1.34–1.42 | `retrieval/hybrid_retriever.py`, `retrieval/context_builder.py`, `generation/deterministic_answer_generator.py`, `generation/ollama_llm_client.py` |
| Testing, Monitoring & Operations | 79–88 | 1.43–1.45 | `ingestion/state_store.py`, operational backup workflow, `testing/` |

The fix strategy is strictly **targeted and minimal**: each change patches exactly the identified defect, with no refactoring of surrounding correct code. Preservation of unchanged behaviors across all non-buggy inputs is a first-class constraint enforced by property-based tests.

---

## Glossary

- **Bug_Condition (C)**: A predicate `C(X) → bool` that returns `true` for inputs that trigger a specific defect and `false` for all other inputs.
- **Property (P)**: The desired correct output behavior for any input where `C(X) = true` after the fix is applied.
- **Preservation**: The guarantee that `F(X) = F′(X)` for all inputs where `C(X) = false`; i.e., the fix does not change correct behavior.
- **F**: The original unfixed function under investigation.
- **F′**: The fixed version of that function.
- **Counterexample**: A concrete input where `C(X) = true` that demonstrates the defect on unfixed code and the correct behavior on fixed code.
- **FileMonitor**: The `ingestion/file_monitor.py` component that watches directories for new PDF files.
- **robust_ingest_file**: The entry-point function in `ingestion/robust_ingestor.py` that drives the full ingestion pipeline for one file.
- **PDFExtractor**: `parsing/pdf_extractor.py` — extracts text, tables, captions, and page metadata from PDFs.
- **OCRService**: `ocr/ocr_service.py` — renders PDF pages as pixmaps and runs RapidOCR.
- **SemanticChunker**: `chunking/semantic_chunker.py` — splits page text into hierarchical RAG chunks.
- **EmbeddingService**: `embeddings/embedding_service.py` — encodes text to dense vectors.
- **VectorStore**: `knowledge/vector_store.py` — manages Chroma vector collection and SQLite lexical index.
- **HybridRetriever**: `retrieval/hybrid_retriever.py` — fuses vector and lexical results via RRF.
- **ContextBuilder**: `retrieval/context_builder.py` — assembles evidence within a token budget.
- **stable_id**: A deterministic chunk identifier derived from document hierarchy keys.
- **version_id**: A unique identifier for each ingestion run of a document; used to scope vector and lexical records.
- **WAL**: Write-Ahead Log — SQLite's journaling mode; must be checkpointed before a backup is consistent.

---

## Bug Details

### Zone A — PDF Ingestion (Bugs 1.1–1.12)

#### Bug Condition A1: Duplicate File Detection (Bug 1.1)

The `FileMonitor` does not compute content hashes. Two files with identical bytes but different filenames are treated as distinct new documents and both enter the pipeline.

**Formal Specification:**
```
FUNCTION isBugCondition_A1(scan_result)
  INPUT: scan_result — a pair of discovered file paths
  OUTPUT: boolean

  RETURN sha256(scan_result.file_a) == sha256(scan_result.file_b)
         AND scan_result.file_a.name != scan_result.file_b.name
         AND both_scheduled_for_ingestion(scan_result)
END FUNCTION
```

**Counterexamples:**
- `report_v1.pdf` and `report_copy.pdf` (identical bytes) both routed to ingestion → duplicate DB records.
- Same PDF uploaded twice under different names → two separate vector namespaces created.

#### Bug Condition A2: Self-Replace File Operation (Bug 1.2)

`robust_ingest_file` calls `file_path.replace(archived_path)` without checking whether `file_path` and `archived_path` resolve to the same path, destroying the only copy.

```
FUNCTION isBugCondition_A2(ingest_request)
  INPUT: ingest_request — {file_path, processed_dir, is_already_indexed}
  OUTPUT: boolean

  target = processed_dir / ingest_request.file_path.name
  RETURN ingest_request.is_already_indexed
         AND ingest_request.file_path.resolve() == target.resolve()
END FUNCTION
```

**Counterexample:** A PDF in `processed_dir/` is re-submitted; `replace()` clobbers the file with itself and may leave an empty or corrupt entry.

#### Bug Condition A3: OCR-Failed Pages Not Recorded (Bug 1.5)

When OCR is required but fails, `extract_iter` logs a warning and continues without writing a page record, leaving the page invisible to audit and citation code.

```
FUNCTION isBugCondition_A3(page_extraction_attempt)
  INPUT: page_extraction_attempt — {page_no, ocr_required, ocr_succeeded}
  OUTPUT: boolean

  RETURN page_extraction_attempt.ocr_required
         AND NOT page_extraction_attempt.ocr_succeeded
         AND NOT page_record_written(page_extraction_attempt.page_no)
END FUNCTION
```

**Counterexample:** Page 7 of a scanned PDF fails OCR; the page gap is not detectable in the state store; citation code produces wrong page references.

#### Bug Condition A4: Two-Column Layout Interleaving (Bug 1.6)

`_extract_page_text` uses `page.get_text("text", sort=True)` which merges columns horizontally, producing text like: `"Dose: 500mg   Clinical outcome: improvement"` where the two columns' content is interleaved line-by-line.

```
FUNCTION isBugCondition_A4(page)
  INPUT: page — PyMuPDF page object
  OUTPUT: boolean

  blocks = page.get_text("blocks")
  x_coords = [b[0] for b in blocks if b[6] == 0]  // text blocks only
  RETURN max(x_coords) - min(x_coords) > page.rect.width * 0.4
         AND extraction_mode == "default_sort"
END FUNCTION
```

**Counterexample:** A drug monograph with two-column dosage tables produces interleaved dosage instructions and pharmacokinetics text.

#### Bug Condition A5: OCR Scale Floor Too Low (Bug 1.8)

`_render_pixmap` reduces scale below `0.25` when the pixel count is too large, producing pixmaps so small that RapidOCR misreads single-digit dosage numerals.

```
FUNCTION isBugCondition_A5(render_request)
  INPUT: render_request — {page, configured_scale, max_render_pixels}
  OUTPUT: boolean

  required_reduction = configured_scale / 0.25
  RETURN (page.width * page.height * configured_scale^2) > max_render_pixels
         AND required_reduction > 1.0   // would need to go below 0.25
END FUNCTION
```

**Counterexample:** A large-format A3 PDF scan at default scale exceeds pixel budget; scale is reduced to `0.18`; RapidOCR reads `"5rng"` instead of `"5mg"`.

#### Bug Condition A6: Table Exception Swallowing (Bug 1.11)

`_extract_tables` catches all exceptions including `AttributeError` (older PyMuPDF without `find_tables`) with a bare `except Exception: return ""`, making genuine errors invisible.

```
FUNCTION isBugCondition_A6(table_extraction_attempt)
  INPUT: table_extraction_attempt — {page, exception_raised}
  OUTPUT: boolean

  RETURN table_extraction_attempt.exception_raised != None
         AND NOT isinstance(table_extraction_attempt.exception_raised, AttributeError)
         AND exception_silently_swallowed(table_extraction_attempt)
END FUNCTION
```

**Counterexample:** A corrupt PDF page raises `ValueError` inside `find_tables()`; the error is swallowed; the table data is silently missing from the chunk.

#### Bug Condition A7: Page Number Search Limited to 500 Chars (Bug 1.12)

`extract_page_number` only scans the first 500 characters, missing footers where the printed page number appears after a large title block.

```
FUNCTION isBugCondition_A7(page_text)
  INPUT: page_text — full extracted text of one page
  OUTPUT: boolean

  RETURN page_number_pattern_found(page_text[500:])
         AND NOT page_number_pattern_found(page_text[:500])
END FUNCTION
```

**Counterexample:** A textbook page begins with a full-width image caption taking 600 characters; page number `"— 47 —"` at position 612 is never found; `printed_page_number` is `null` for 30% of pages.

---

### Zone B — Text Processing (Bugs 1.13–1.19)

#### Bug Condition B1: NFKC Destroys Chemical Notation (Bug 1.13)

`clean_text` applies `unicodedata.normalize("NFKC", ...)` which converts `H₂O` → `H2O` and `²` → `2`, destroying formula structure.

```
FUNCTION isBugCondition_B1(text)
  INPUT: text — raw page text string
  OUTPUT: boolean

  RETURN any(char in text for char in SUPERSCRIPT_CHARS + SUBSCRIPT_CHARS)
         AND normalize_mode == "NFKC_unrestricted"
END FUNCTION

SUPERSCRIPT_CHARS = ['\u2070'..'\u2079', '\u00b2', '\u00b3', '\u00b9']
SUBSCRIPT_CHARS   = ['\u2080'..'\u2089']
```

**Counterexample:** `"CO₂ levels: 38 mmHg"` → after NFKC → `"CO2 levels: 38 mmHg"` — indistinguishable from plain integer `2`, breaking formula-aware search.

#### Bug Condition B2: Hyphenated Line-Break Not Joined (Bug 1.15)

`split_paragraphs` splits on `\n{2,}` but does not pre-join hyphen–newline sequences, so `"anti-\nhypertensive"` becomes two tokens in different paragraphs.

```
FUNCTION isBugCondition_B2(text)
  INPUT: text — page text after initial cleaning
  OUTPUT: boolean

  RETURN re.search(r'\w+-\n\w+', text) != None
         AND hyphen_newline_joining == False
END FUNCTION
```

**Counterexample:** Query `"antihypertensive"` returns no results because the index contains `"anti-"` and `"hypertensive"` as separate tokens in adjacent paragraphs.

#### Bug Condition B3: OCR Post-Correction Missing (Bug 1.18)

No domain-specific OCR correction pass exists; RapidOCR confusables (`0`/`O`, `l`/`1`, `rn`/`m`) in drug names propagate silently into the index.

```
FUNCTION isBugCondition_B3(ocr_text)
  INPUT: ocr_text — raw text from RapidOCR
  OUTPUT: boolean

  RETURN contains_numeric_drug_pattern(ocr_text)
         AND ocr_correction_pass_applied == False
END FUNCTION
```

**Counterexample:** `"Metf0rmin 500mg"` (zero instead of O) enters the index; queries for `"Metformin"` fail to find the chunk.

---

### Zone C — Chunking (Bugs 1.20–1.25)

#### Bug Condition C1: Chapter State Not Carried Across Batch Boundary (Bug 1.22)

`chunk_page_batches` calls `chunk_pages` independently per batch; the `document_chapter` carry-forward variable is reset to `None` at the start of each batch.

```
FUNCTION isBugCondition_C1(batch_call)
  INPUT: batch_call — {batch_index, preceding_chapter_state}
  OUTPUT: boolean

  RETURN batch_call.batch_index > 0
         AND batch_call.preceding_chapter_state != None
         AND chunk_pages_receives_chapter_state == False
END FUNCTION
```

**Counterexample:** Chapter "Pharmacology" spans pages 45–55 (batch boundary at page 50); pages 51–55 show `chapter = null` in chunk metadata, breaking section-level boosting.

#### Bug Condition C2: Cross-Document Section ID Collision (Bug 1.23)

`_stable_id` hashes `"{chapter}|{section}"` without including `document_id`; two documents with identical heading structures produce identical `section_id` and `parent_id` values.

```
FUNCTION isBugCondition_C2(chunk_pair)
  INPUT: chunk_pair — two chunks from different documents
  OUTPUT: boolean

  RETURN chunk_pair.doc_a.chapter == chunk_pair.doc_b.chapter
         AND chunk_pair.doc_a.section == chunk_pair.doc_b.section
         AND document_id_not_in_hash_key(chunk_pair)
END FUNCTION
```

**Counterexample:** Two drug monographs both have `"Chapter: Dosage | Section: Adults"` → same `section_id` → Chroma stores one over the other; hierarchy linking is broken.

#### Bug Condition C3: Lexical Rows Not Purged Before Re-Ingestion (Bug 1.24)

`delete_version` removes Chroma vectors but does not delete corresponding lexical SQLite rows for interrupted ingestions, leaving orphaned rows that create parity gaps.

```
FUNCTION isBugCondition_C3(re_ingest_request)
  INPUT: re_ingest_request — {document_id, version_id, prior_ingestion_interrupted}
  OUTPUT: boolean

  RETURN re_ingest_request.prior_ingestion_interrupted
         AND lexical_rows_for_version_exist(document_id, version_id)
         AND delete_version_does_not_clear_lexical == True
END FUNCTION
```

**Counterexample:** Ingestion of `cardiology.pdf` interrupted at chunk 150 of 300; re-ingestion clears Chroma but not SQLite; 150 stale lexical rows persist; reconcile_index reports false parity.

---

### Zone D — Embedding & Vector Storage (Bugs 1.26–1.33)

#### Bug Condition D1: Test-Mode Dimension Mismatch (Bug 1.26)

`_test_embedding` always generates 16-dimensional vectors regardless of the configured model dimension (e.g., 768), corrupting the Chroma collection schema.

```
FUNCTION isBugCondition_D1(embed_request)
  INPUT: embed_request — {test_mode, configured_dim}
  OUTPUT: boolean

  RETURN embed_request.test_mode == True
         AND test_embedding_dim != embed_request.configured_dim
END FUNCTION
```

**Counterexample:** Production model is `all-MiniLM-L6-v2` (384-dim); test mode inserts 16-dim vectors; first production query raises `DimensionMismatch` exception from Chroma.

#### Bug Condition D2: Embeddings Not Normalized (Bug 1.27)

`_transformers_embed_batch` calls `encode(normalize_embeddings=False)`; raw vectors are stored; cosine similarity computations degrade.

```
FUNCTION isBugCondition_D2(embed_call)
  INPUT: embed_call — {encode_kwargs}
  OUTPUT: boolean

  RETURN embed_call.encode_kwargs.get("normalize_embeddings") != True
END FUNCTION
```

**Counterexample:** Two semantically identical paragraphs with different lengths produce cosine scores of `0.6` instead of `~1.0` because their vector magnitudes differ.

#### Bug Condition D3: page_numbers Stored as JSON String (Bug 1.31)

`_coerce_metadata` serializes `page_numbers` list to a JSON string; retrieval code and citation builder receive `"[1, 2]"` instead of `[1, 2]`; Chroma `$where` filters fail silently; `TypeError` in citation builder.

```
FUNCTION isBugCondition_D3(metadata)
  INPUT: metadata — chunk metadata dict before Chroma insertion
  OUTPUT: boolean

  RETURN isinstance(metadata.get("page_numbers"), list)
         AND storage_form == "json_string_only"
         AND retrieval_filter_expects_scalar_match == True
END FUNCTION
```

**Counterexample:** Filter `{"page_numbers": {"$eq": 1}}` fails to match any chunk because stored value is `"[1]"` not `1`; citation builder raises `TypeError: 'str' object is not subscriptable`.

---

### Zone E — Retrieval & Answer Generation (Bugs 1.34–1.42)

#### Bug Condition E1: Chroma Filter on JSON-String Field (Bug 1.34)

`HybridRetriever` passes integer filter values against string-serialized metadata fields; Chroma silently returns empty results.

```
FUNCTION isBugCondition_E1(retrieval_query)
  INPUT: retrieval_query — {where_filter}
  OUTPUT: boolean

  RETURN any(field in STRING_SERIALIZED_FIELDS
             for field in retrieval_query.where_filter.keys())
         AND filter_not_adapted_to_serialization_form(retrieval_query)
END FUNCTION

STRING_SERIALIZED_FIELDS = ["page_numbers", "chapter_id", "section_id"]
```

#### Bug Condition E2: Token Budget Undercounting (Bug 1.37)

`ContextBuilder.build` estimates tokens as `len(text.split()) * 4 // 3`; dense medical abbreviations consume 2–3× more tokens per word than this formula assumes.

```
FUNCTION isBugCondition_E2(text)
  INPUT: text — candidate evidence text
  OUTPUT: boolean

  actual_tokens   = count_bpe_tokens(text)
  estimated_tokens = len(text.split()) * 4 // 3
  RETURN (actual_tokens - estimated_tokens) / actual_tokens > 0.3
         AND text_contains_abbreviation_density(text)
END FUNCTION
```

**Counterexample:** `"HR 72 bpm, BP 120/80 mmHg, SpO2 98%"` → `word_count=9`, `estimate=12`, `actual_bpe=22`; budget overrun by 83%.

#### Bug Condition E3: pass Instead of continue in Budget Loop (Bug 1.38)

In `ContextBuilder.build`, after allowing a high-quality hit past the budget via an exception block, `pass` is used instead of `continue`; the loop is never advanced and subsequent candidates are never evaluated.

```
FUNCTION isBugCondition_E3(context_build_loop)
  INPUT: context_build_loop — code path at budget exception branch
  OUTPUT: boolean

  RETURN context_build_loop.exception_branch_uses_pass == True
         AND candidates_after_exception_evaluated == False
END FUNCTION
```

**Counterexample:** After allowing hit #3 through the budget gate, loop stalls; hits #4–#10 are never considered; context is assembled from only 3 chunks even though 6 would fit.

#### Bug Condition E4: Empty LLM Response No Fallback (Bug 1.41)

`OllamaLLMClient.generate` returns an empty string when the model produces no output; the answer service emits a blank answer to the user.

```
FUNCTION isBugCondition_E4(llm_response)
  INPUT: llm_response — raw API response object
  OUTPUT: boolean

  RETURN extract_content(llm_response) == ""
         AND no_error_raised == True
END FUNCTION
```

---

### Zone F — Testing, Monitoring & Operations (Bugs 1.43–1.45)

#### Bug Condition F1: Stale-Lease Recovery Missing Vector Cleanup (Bug 1.43)

`recover_stale_documents` resets lease state without calling `delete_version`, leaving partial Chroma vectors queryable during re-ingestion.

```
FUNCTION isBugCondition_F1(recovery_call)
  INPUT: recovery_call — {stale_document_id, partial_vectors_in_chroma}
  OUTPUT: boolean

  RETURN recovery_call.partial_vectors_in_chroma == True
         AND delete_version_called_during_recovery == False
END FUNCTION
```

#### Bug Condition F2: SQLite Backup Without WAL Checkpoint (Bug 1.44)

Backup copies SQLite files while WAL is active, producing inconsistent backup files.

```
FUNCTION isBugCondition_F2(backup_operation)
  INPUT: backup_operation — {target_db, wal_active}
  OUTPUT: boolean

  RETURN backup_operation.wal_active == True
         AND wal_checkpoint_executed_before_copy == False
END FUNCTION
```

#### Bug Condition F3: Integration Tests Bypass prepare_runtime (Bug 1.45)

Integration tests instantiate `MedEvidenceProductionRAGSystem` directly, bypassing `rag_project.composition.prepare_runtime`, violating the production composition boundary.

```
FUNCTION isBugCondition_F3(test_setup)
  INPUT: test_setup — test class setUp / fixture
  OUTPUT: boolean

  RETURN test_setup.constructs_rag_system_directly == True
         AND test_setup.calls_prepare_runtime == False
END FUNCTION
```

---

## Expected Behavior

### Preservation Requirements

The following behaviors are established and correct. The fix MUST NOT change them.

**Unchanged Behaviors:**
- Normal single-copy PDF ingestion continues to run the full pipeline: classification → extraction → OCR → chunking → embedding → index publication (Req. 3.1).
- Already-indexed duplicate PDFs (matching hash and version_id, different path) continue to be archived and skipped with `{"status": "skipped"}` (Req. 3.2).
- Well-formed single-column text PDFs continue to extract text, tables, captions, and quality scores without invoking OCR (Req. 3.3).
- OCR pages with confidence ≥ 0.55 continue to persist via `state_store.record_page` and yield a `PageExtraction` with `ocr_status = "completed"` (Req. 3.4).
- `clean_text` continues to normalize ASCII prose, collapse whitespace, and strip leading/trailing whitespace (Req. 3.6).
- `detect_language` continues to return `"ar"` for purely Arabic text (Req. 3.7).
- The `"120/80"` token continues to be kept as a single unit by `tokenize` (Req. 3.8).
- `SemanticChunker` continues to emit structured RAG-STRUCTURE markers with `chapter_id`, `section_id`, `parent_id` for pages with headings (Req. 3.10).
- `EmbeddingService.embed_texts` continues to return vectors in input order and serves cached vectors from LRU cache (Reqs. 3.14–3.15).
- `VectorStore.add_documents` continues to atomically insert Chroma and lexical rows with rollback on failure (Req. 3.16).
- `delete_version` continues to remove both Chroma vectors and lexical rows for the targeted version (Req. 3.18).
- `HybridRetriever` continues to run RRF fusion in hybrid mode and fall back to lexical when vector fails (Reqs. 3.19–3.20).
- `ContextBuilder` continues to enforce `max_per_document` limit (Req. 3.21).
- `DeterministicAnswerGenerator` continues to produce `SynthesizedAnswer` with citations even when the LLM is unavailable (Req. 3.22).
- `recover_stale_documents` continues to reset only documents whose lease expiry is in the past (Req. 3.24).

**Scope:**
All inputs that do NOT satisfy any of the formal bug condition predicates above shall produce identical results before and after applying these fixes.

---

## Hypothesized Root Cause

Root causes are grouped by zone.

### Zone A — PDF Ingestion

1. **FileMonitor Has No Hash Index**: `file_monitor.py` tracks files by path alone; no SHA-256 registry is maintained. Fix: add a `dict[str, str]` mapping `sha256 → first_seen_path` alongside the existing path registry.

2. **Self-Replace Guard Missing**: `robust_ingestor.py` calls `file_path.replace(archived_path)` unconditionally; one-line path equality check before the `replace` call is all that is needed.

3. **OCR Failure Path Has No `record_page` Call**: `pdf_extractor.py` OCR exception handler calls `logger.warning` then `continue`; it needs a `state_store.record_page(page_no, status="FAILED")` call inserted before `continue`.

4. **Block Sorting Uses Default PyMuPDF Order**: `_extract_page_text` does not inspect block x-coordinates; a two-column detection heuristic (x-coordinate spread > 40% of page width) and sort-by-column-then-y logic must be added.

5. **Scale Floor Hardcoded at 0.25**: `ocr_service.py` has `MIN_RENDER_SCALE = 0.25`; change to `0.5` and add an explicit `RuntimeError` if the scaled pixel count still exceeds the limit.

6. **Bare `except Exception` in Table Extraction**: `_extract_tables` catches all exceptions; split into `except AttributeError: pass` (version compatibility) and `except Exception as e: logger.warning(...)` for all other errors.

7. **Page Number Search Capped at `[:500]`**: `extract_page_number` uses `text[:500]`; extend to scan full text with an additional regex pass over `text[-200:]` for footer-style page numbers.

### Zone B — Text Processing

8. **NFKC Applied Without Pre-Filtering**: `clean_text` applies NFKC unconditionally; a pre-pass that converts superscripts/subscripts to `^N`/`_N` notation before normalization is needed.

9. **No Hyphen–Newline Pre-Joining Step**: `split_paragraphs` splits on blank lines without first joining hyphenated line-breaks; add `re.sub(r'(\w)-\n(\w)', r'\1\2', text)` before the paragraph split.

10. **No OCR Correction Dictionary**: No post-processing dictionary exists in the OCR text path; a new `ocr_corrector.py` module with medical confusables must be added and called from `pdf_intelligence.py`.

### Zone C — Chunking

11. **Chapter State is a Local Variable**: `chunk_pages` initializes `document_chapter = None` at the top of each call; `chunk_page_batches` does not pass the previous batch's final chapter state. Fix: add `initial_chapter=None` parameter to `chunk_pages` and thread it through `chunk_page_batches`.

12. **Hash Key Excludes document_id**: `_stable_id` hashes `f"{chapter}|{section}"`; add `document_id` as the first component: `f"{document_id}|{chapter}|{section}"`.

13. **`delete_version` Only Targets Chroma**: `versioned_ingestor.py` (and `robust_ingestor.py`) calls `vector_store.delete_version()` which clears Chroma; a matching call to delete SQLite lexical rows for `(document_id, version_id)` must be added before re-ingestion starts.

### Zone D — Embedding & Vector Storage

14. **`_test_embedding` Has Hardcoded Dimension 16**: Change `_test_embedding` to generate vectors of length `self.dimension` using deterministic hashing extended to the full dimension.

15. **`encode` Called Without `normalize_embeddings=True`**: A single keyword argument addition to the `encode()` call.

16. **`_coerce_metadata` Serializes page_numbers to JSON Only**: Add a parallel `"page_numbers_csv"` key with comma-separated form; keep the JSON form for backward compatibility.

### Zone E — Retrieval & Answer Generation

17. **No Filter Adaptation Layer**: `HybridRetriever.retrieve` passes raw filter dicts to Chroma; a `_adapt_filter_for_serialized_fields` helper must translate integer/list filter values to their string-serialized equivalents.

18. **Word-Count Token Estimate Has No Floor**: `ContextBuilder.build` uses only word-count estimate; replace with `max(word_count_estimate, len(text) // 4)`.

19. **`pass` Instead of `continue`**: A one-character fix in `context_builder.py`.

20. **`OllamaLLMClient.generate` Returns Empty String Silently**: Add `if not _content: raise RuntimeError("LLM returned empty response")` after extracting the content.

### Zone F — Testing, Monitoring & Operations

21. **`recover_stale_documents` Does Not Call `delete_version`**: Add `vector_store.delete_version(doc_id, version_id)` for each stale document before resetting its lease state.

22. **Backup Workflow Has No WAL Checkpoint**: Add `conn.execute("PRAGMA wal_checkpoint(FULL)")` before each `shutil.copy` in the backup routine.

23. **Tests Bypass `prepare_runtime`**: Integration tests must be updated to call `prepare_runtime()` first and receive the system instance from it rather than constructing it directly.

---

## Correctness Properties

Property 1: Bug Condition — File Integrity During Ingestion

_For any_ `IngestionRequest` where `isBugCondition_A1` OR `isBugCondition_A2` holds, the fixed `FileMonitor` / `robust_ingest_file` SHALL: skip ingestion of the second content-duplicate (recording a `"skipped_duplicate"` event), and SHALL NOT move or overwrite a file when source and destination paths resolve identically.

**Validates: Requirements 2.1, 2.2**

---

Property 2: Bug Condition — OCR Failure Page Visibility

_For any_ page extraction attempt where `isBugCondition_A3` holds (OCR required, OCR failed), the fixed `PDFExtractor.extract_iter` SHALL write a page record to the state store with `extraction_status = "FAILED"` before moving to the next page, so the page appears in audit queries.

**Validates: Requirements 2.5**

---

Property 3: Bug Condition — Two-Column Text Order

_For any_ PDF page where `isBugCondition_A4` holds (multi-column layout detected), the fixed `_extract_page_text` SHALL produce text in reading order: all of column A (top-to-bottom) followed by all of column B (top-to-bottom), rather than interleaving blocks from both columns.

**Validates: Requirements 2.6**

---

Property 4: Bug Condition — OCR Scale Safety Floor

_For any_ render request where `isBugCondition_A5` holds (configured scale must be reduced below `0.5`), the fixed `OCRService._render_pixmap` SHALL raise `RuntimeError` rather than rendering at sub-`0.5` scale, so the page is recorded as an OCR failure rather than producing unreliable dosage text.

**Validates: Requirements 2.7**

---

Property 5: Bug Condition — Table Exception Transparency

_For any_ table extraction attempt where `isBugCondition_A6` holds (non-AttributeError exception raised), the fixed `_extract_tables` SHALL log the exception at WARNING level with the page number rather than swallowing it silently.

**Validates: Requirements 2.10**

---

Property 6: Bug Condition — Footer Page Number Detection

_For any_ page text where `isBugCondition_A7` holds (page number only in `text[500:]`), the fixed `extract_page_number` SHALL find and return the printed page number from the full text including the last 200 characters.

**Validates: Requirements 2.11**

---

Property 7: Bug Condition — Chemical Notation Preservation

_For any_ text where `isBugCondition_B1` holds (contains superscripts/subscripts), the fixed `clean_text` SHALL convert those characters to `^N`/`_N` notation rather than stripping them, preserving formula semantics.

**Validates: Requirements 2.12**

---

Property 8: Bug Condition — Hyphenated Line-Break Joining

_For any_ text where `isBugCondition_B2` holds (contains `\w+-\n\w+`), the fixed `split_paragraphs` SHALL join hyphenated line-breaks before splitting, producing a single unhyphenated token searchable as a whole term.

**Validates: Requirements 2.14**

---

Property 9: Bug Condition — OCR Confusable Correction

_For any_ OCR output where `isBugCondition_B3` holds (contains numeric drug dosage patterns with OCR confusables), the fixed pipeline SHALL apply the domain-specific correction pass before the text enters the chunk index.

**Validates: Requirements 2.16**

---

Property 10: Bug Condition — Chapter State Across Batch Boundaries

_For any_ batch call where `isBugCondition_C1` holds (batch_index > 0 with a non-null preceding chapter state), the fixed `chunk_page_batches` SHALL carry the `document_chapter` variable into the new batch call, so all chunks from that batch inherit the correct chapter context.

**Validates: Requirements 2.19**

---

Property 11: Bug Condition — Cross-Document Section ID Uniqueness

_For any_ chunk pair where `isBugCondition_C2` holds (same chapter/section titles, different documents), the fixed `_stable_id` SHALL produce distinct `section_id` values because `document_id` is included in the hash key.

**Validates: Requirements 2.20**

---

Property 12: Bug Condition — Lexical/Vector Parity Before Re-Ingestion

_For any_ re-ingestion request where `isBugCondition_C3` holds (interrupted prior ingestion with stale lexical rows), the fixed `robust_ingest_file` / `delete_version` SHALL delete those lexical rows before the new ingestion inserts fresh ones, ensuring parity.

**Validates: Requirements 2.21**

---

Property 13: Bug Condition — Test-Mode Embedding Dimension Compatibility

_For any_ test-mode embedding request where `isBugCondition_D1` holds (configured dim ≠ 16), the fixed `_test_embedding` SHALL return a vector of length `self.dimension`, preventing Chroma schema corruption.

**Validates: Requirements 2.23**

---

Property 14: Bug Condition — Unit-Length Embeddings

_For any_ embedding call where `isBugCondition_D2` holds (`normalize_embeddings` not set), the fixed `_transformers_embed_batch` SHALL call `encode(normalize_embeddings=True)`, producing unit-length vectors for correct cosine similarity.

**Validates: Requirements 2.24**

---

Property 15: Bug Condition — page_numbers Parseable from Metadata

_For any_ metadata dict where `isBugCondition_D3` holds (`page_numbers` is a list being coerced to string only), the fixed `_coerce_metadata` SHALL also store `"page_numbers_csv"` as a comma-separated string, enabling retrieval code to parse page numbers without a JSON parse error.

**Validates: Requirements 2.28**

---

Property 16: Bug Condition — Serialized-Field Filter Adaptation

_For any_ retrieval query where `isBugCondition_E1` holds (filter references a string-serialized field with a non-string value), the fixed `HybridRetriever` SHALL adapt the filter value to match the storage format before passing to Chroma, returning correct (non-empty) results.

**Validates: Requirements 2.31**

---

Property 17: Bug Condition — Token Budget Conservative Floor

_For any_ candidate text where `isBugCondition_E2` holds (medical abbreviation density causes word-count undercount > 30%), the fixed `ContextBuilder.build` SHALL use `max(word_count_estimate, len(text) // 4)` so the budget is never exceeded.

**Validates: Requirements 2.34**

---

Property 18: Bug Condition — Budget Exception Loop Continuity

_For any_ context build loop where `isBugCondition_E3` holds (high-quality hit allowed past budget via exception branch), the fixed code SHALL use `continue` so that all subsequent candidates are evaluated.

**Validates: Requirements 2.35**

---

Property 19: Bug Condition — Empty LLM Response Error

_For any_ LLM call where `isBugCondition_E4` holds (empty response returned), the fixed `OllamaLLMClient.generate` SHALL raise `RuntimeError("LLM returned empty response")`, triggering the deterministic fallback path.

**Validates: Requirements 2.37**

---

Property 20: Bug Condition — Stale-Lease Cleanup Includes Vectors

_For any_ stale-lease recovery where `isBugCondition_F1` holds (partial vectors exist for the stale document), the fixed `recover_stale_documents` SHALL call `delete_version` before resetting the lease, ensuring a clean index state for re-ingestion.

**Validates: Requirements 2.39**

---

Property 21: Bug Condition — WAL Checkpoint Before Backup

_For any_ SQLite backup operation where `isBugCondition_F2` holds (WAL is active), the fixed backup routine SHALL execute `PRAGMA wal_checkpoint(FULL)` on each database connection before copying the file, producing a consistent backup.

**Validates: Requirements 2.40**

---

Property 22: Bug Condition — Integration Tests Use Composition Boundary

_For any_ integration test setup where `isBugCondition_F3` holds (direct RAG system construction), the fixed test MUST call `prepare_runtime()` first, ensuring the test validates production-equivalent behavior.

**Validates: Requirements 2.41**

---

Property 23: Preservation — All Non-Buggy Inputs Unchanged

_For any_ input X where none of the 22 bug conditions above holds (`isBugCondition_Ak(X) = false` for all k), every fixed function F′ SHALL produce the same result as its original F, preserving all correct pipeline behaviors documented in Requirements 3.1–3.27.

**Validates: Requirements 3.1–3.27**

---

## Fix Implementation

### Zone A — PDF Ingestion

**File:** `rag_project/ingestion/file_monitor.py`

**Fix A1 — Duplicate File Detection:**
1. Add `self._content_hash_registry: dict[str, Path] = {}` to `__init__`.
2. After computing `sha256(file_path)`, check `if digest in self._content_hash_registry`: emit a `"skipped_duplicate"` state store event and skip scheduling.
3. Otherwise register `self._content_hash_registry[digest] = file_path` before scheduling.

---

**File:** `rag_project/ingestion/robust_ingestor.py`

**Fix A2 — Self-Replace Guard:**
1. Before `file_path.replace(archived_path)`, add:
   ```python
   if file_path.resolve() == archived_path.resolve():
       return {"status": "skipped", "reason": "source_is_destination"}
   ```

---

**File:** `rag_project/parsing/pdf_extractor.py`

**Fix A3 — OCR Failure Page Record:**
1. In the OCR exception handler inside `extract_iter`, before `continue`, add:
   ```python
   state_store.record_page(
       document_id, page_no,
       extraction_status="FAILED",
       ocr_status="failed"
   )
   ```

**Fix A4 — Two-Column Detection and Sort:**
1. Add `_is_two_column(page) -> bool` helper: compute x-coordinates of all text blocks; return `True` if `max_x - min_x > page.rect.width * 0.4` and there are ≥ 4 text blocks.
2. In `_extract_page_text`, when `_is_two_column(page)` is `True`, sort blocks by `(round(block.x0 / (page.rect.width * 0.5)), block.y0)` so left-column blocks come before right-column blocks within each row-band.

**Fix A6 — Table Exception Transparency:**
1. Replace `except Exception: return ""` with:
   ```python
   except AttributeError:
       return ""   # PyMuPDF version does not support find_tables
   except Exception as e:
       logger.warning("Table extraction failed on page %d: %s", page_no, e)
       return ""
   ```

**Fix A7 — Full-Text Page Number Search:**
1. Replace `text[:500]` with `text` in the regex search.
2. Add a second scan: `re.search(PAGE_NUMBER_PATTERN, text[-200:])` as a fallback when the primary scan fails.

---

**File:** `rag_project/ocr/ocr_service.py`

**Fix A5 — OCR Scale Floor:**
1. Change `MIN_RENDER_SCALE = 0.25` to `MIN_RENDER_SCALE = 0.5`.
2. After scale reduction, add:
   ```python
   if scale < MIN_RENDER_SCALE:
       raise RuntimeError(
           f"Page {page_no}: image too large even at scale {MIN_RENDER_SCALE}; "
           "treating as OCR failure"
       )
   ```

---

### Zone B — Text Processing

**File:** `rag_project/utils/text_utils.py`

**Fix B1 — Preserve Chemical Notation:**
1. Before `unicodedata.normalize("NFKC", text)`, apply:
   ```python
   text = _protect_superscripts(text)   # ² → ^2, ³ → ^3, ⁴ → ^4 …
   text = _protect_subscripts(text)     # ₂ → _2, ₃ → _3 …
   ```
2. Implement `_protect_superscripts` and `_protect_subscripts` using `str.translate` tables over `\u2070–\u2079`, `\u00b2`, `\u00b3`, `\u00b9` and `\u2080–\u2089`.

**Fix B2 — Hyphenated Line-Break Joining:**
1. At the top of `split_paragraphs`, add:
   ```python
   text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
   ```

---

**File:** `rag_project/ocr/ocr_corrector.py` _(new file)_

**Fix B3 — OCR Correction Dictionary:**
1. Create `OCR_CORRECTIONS` dict with medical confusables:
   - `r'(?<=\d)O(?=\d)'` → `'0'` (letter O between digits = zero)
   - `r'(?<=\d)l(?=\d)'` → `'1'` (lowercase L between digits = one)
   - `r'\brn\b'` → `'m'` for known medical abbreviation false-splits
   - Drug-name specific replacements (Metf0rmin, Am0xicillin patterns).
2. Expose `apply_ocr_corrections(text: str) -> str`.

**File:** `rag_project/intelligence/pdf_intelligence.py`

3. In `enrich_text`, after OCR text is available, call `apply_ocr_corrections(text)`.

---

### Zone C — Chunking

**File:** `rag_project/chunking/semantic_chunker.py`

**Fix C1 — Chapter State Carry-Forward:**
1. Add `initial_chapter: Optional[str] = None` parameter to `chunk_pages`.
2. Initialize `document_chapter = initial_chapter` instead of `None`.
3. In `chunk_page_batches`, thread the running chapter state:
   ```python
   current_chapter = None
   for batch in batches:
       chunks, current_chapter = chunk_pages(batch, initial_chapter=current_chapter)
   ```
4. `chunk_pages` must return `(chunks, final_chapter)` tuple.

**Fix C2 — Document-Scoped Stable ID:**
1. In `_stable_id`, change hash key from `f"{chapter}|{section}"` to `f"{document_id}|{chapter}|{section}"`.
2. Apply the same change to the `parent_id` key.

---

**File:** `rag_project/ingestion/robust_ingestor.py`
**File:** `rag_project/knowledge/vector_store.py`

**Fix C3 — Lexical Row Purge Before Re-Ingestion:**
1. In `robust_ingestor.py`, after `vector_store.delete_version(document_id, version_id)`, add:
   ```python
   vector_store.delete_lexical_version(document_id, version_id)
   ```
2. In `vector_store.py`, implement `delete_lexical_version(document_id, version_id)` that executes:
   ```sql
   DELETE FROM lexical_index
    WHERE document_id = ? AND version_id = ?
   ```

---

### Zone D — Embedding & Vector Storage

**File:** `rag_project/embeddings/embedding_service.py`

**Fix D1 — Test-Mode Dimension Match:**
1. In `_test_embedding`, replace the hardcoded 16-element list with:
   ```python
   seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
   rng  = random.Random(seed)
   vec  = [rng.uniform(-1, 1) for _ in range(self.dimension)]
   magnitude = sum(v**2 for v in vec) ** 0.5
   return [v / magnitude for v in vec]
   ```

**Fix D2 — L2 Normalization:**
1. In `_transformers_embed_batch`, change `encode(...)` to `encode(..., normalize_embeddings=True)`.

---

**File:** `rag_project/knowledge/vector_store.py`

**Fix D3 — page_numbers CSV Storage:**
1. In `_coerce_metadata`, for `page_numbers` lists:
   ```python
   metadata["page_numbers_csv"] = ",".join(str(p) for p in value)
   metadata["page_numbers"] = json.dumps(value)   # keep JSON form
   ```
2. In retrieval code that reads `page_numbers`, use `_parse_page_numbers(meta)`:
   ```python
   def _parse_page_numbers(meta):
       csv_val = meta.get("page_numbers_csv", "")
       if csv_val:
           return [int(x) for x in csv_val.split(",") if x]
       json_val = meta.get("page_numbers", "[]")
       if isinstance(json_val, list):
           return json_val
       return json.loads(json_val)
   ```

---

### Zone E — Retrieval & Answer Generation

**File:** `rag_project/retrieval/hybrid_retriever.py`

**Fix E1 — Filter Adaptation:**
1. Add `_adapt_filter(where_filter: dict) -> dict` helper:
   - For fields in `STRING_SERIALIZED_FIELDS`, convert integer `$eq` values to check against `page_numbers_csv` using a `$contains` or CSV-membership pattern.
2. Call `_adapt_filter` on every `where_filter` before passing to vector and lexical stores.

**Fix E2 already addressed in ContextBuilder — see below.**

---

**File:** `rag_project/retrieval/context_builder.py`

**Fix E2 — Conservative Token Floor:**
1. Replace:
   ```python
   estimated = len(text.split()) * 4 // 3
   ```
   with:
   ```python
   estimated = max(len(text.split()) * 4 // 3, len(text) // 4)
   ```

**Fix E3 — `pass` → `continue`:**
1. In the budget exception branch, change `pass` to `continue`.

---

**File:** `rag_project/generation/ollama_llm_client.py`

**Fix E4 — Empty LLM Response:**
1. After `_content = extract_content(response)`, add:
   ```python
   if not _content:
       raise RuntimeError("LLM returned empty response")
   ```

---

### Zone F — Testing, Monitoring & Operations

**File:** `rag_project/ingestion/state_store.py`

**Fix F1 — Stale-Lease Recovery with Vector Cleanup:**
1. In `recover_stale_documents`, before resetting each stale document's lease:
   ```python
   self._vector_store.delete_version(doc_id, version_id)
   self._vector_store.delete_lexical_version(doc_id, version_id)
   ```

---

**Backup Workflow** (operational scripts or `rag_project/scripts/`)

**Fix F2 — WAL Checkpoint Before Backup:**
1. Before each `shutil.copy(db_path, backup_path)`:
   ```python
   with sqlite3.connect(db_path) as conn:
       conn.execute("PRAGMA wal_checkpoint(FULL)")
   ```

---

**File:** All affected integration test files under `rag_project/testing/`

**Fix F3 — Enforce Composition Boundary:**
1. Replace direct `MedEvidenceProductionRAGSystem(...)` construction with:
   ```python
   from rag_project.composition import prepare_runtime
   system = prepare_runtime(config=test_config)
   ```

---

## Testing Strategy

### Validation Approach

Testing follows a **two-phase strategy**:

**Phase 1 — Exploratory (run against UNFIXED code):** Write tests that express each bug condition and assert the correct post-fix behavior. Run these on the unfixed codebase to confirm they fail and to observe the failure mode, validating the root cause analysis.

**Phase 2 — Fix Checking + Preservation (run against FIXED code):** The same tests must pass after applying the fix. Property-based tests additionally verify that all non-buggy inputs continue to produce identical results.

---

### Exploratory Bug Condition Checking

**Goal:** Surface counterexamples that demonstrate each bug on unfixed code, confirming or refuting the root cause analysis. If tests pass on unfixed code, the root cause analysis must be revised.

**Test Cases (expected to FAIL on unfixed code):**

1. **Duplicate File Skipping** (`test_filemonitor_skips_content_duplicate`): Create two files with identical bytes; assert only one ingestion is scheduled. *Expected failure: both are scheduled.*

2. **Self-Replace Guard** (`test_robust_ingest_no_self_replace`): Submit a file already in `processed_dir`; assert `file_path` still exists after the call. *Expected failure: file is deleted/corrupted.*

3. **OCR Failure Page Record** (`test_ocr_failure_writes_page_record`): Mock OCR to raise an exception; assert `state_store.record_page` was called with `extraction_status="FAILED"`. *Expected failure: no record written.*

4. **Two-Column Order** (`test_two_column_text_column_a_before_column_b`): Feed a synthetic two-column page; assert all column-A text precedes first column-B token. *Expected failure: columns interleaved.*

5. **OCR Scale Floor** (`test_ocr_render_raises_at_sub_0_5_scale`): Provide an oversized page; assert `RuntimeError` is raised rather than a sub-`0.5` scale render. *Expected failure: render silently at `0.18`.*

6. **Non-AttributeError Table Exception Logged** (`test_table_extract_non_attr_error_logged`): Mock `find_tables` to raise `ValueError`; assert `logger.warning` was called. *Expected failure: warning not called.*

7. **Footer Page Number Detection** (`test_extract_page_number_footer_position`): Provide text with page number at position 600; assert function returns the correct number. *Expected failure: returns `None`.*

8. **Chemical Notation Preserved** (`test_clean_text_preserves_superscripts`): Input `"CO₂ 38mmHg"`, assert output contains `"CO_2"` or `"CO^2"`, not `"CO2"` with bare digit. *Expected failure: outputs `"CO2"`.*

9. **Hyphen Line-Break Joined** (`test_split_paragraphs_joins_hyphenated_linebreak`): Input `"anti-\nhypertensive therapy"`, assert output contains `"antihypertensive"`. *Expected failure: split into two tokens.*

10. **OCR Correction Applied** (`test_ocr_correction_fixes_metformin`): Input `"Metf0rmin 500mg"`, assert output is `"Metformin 500mg"`. *Expected failure: confusable uncorrected.*

11. **Chapter Carry-Forward** (`test_chunk_page_batches_carries_chapter_across_boundary`): Create a 10-page document with a chapter heading on page 5; chunk in batches of 6; assert pages 7–10 in batch 2 show the chapter heading in their metadata. *Expected failure: `chapter=null` in batch 2.*

12. **Section ID Uniqueness** (`test_stable_id_unique_across_documents`): Create chunks from two documents with identical chapter/section titles; assert `section_id` differs. *Expected failure: IDs collide.*

13. **Lexical Purge Before Re-Ingest** (`test_delete_version_clears_lexical_rows`): Insert 10 lexical rows, call `delete_lexical_version`; assert 0 rows remain. *Expected failure: rows persist.*

14. **Test-Mode Dimension Match** (`test_test_embedding_matches_configured_dimension`): Instantiate `EmbeddingService(test_mode=True, dimension=384)`; assert `len(_test_embedding("x")) == 384`. *Expected failure: length is 16.*

15. **Normalized Embeddings** (`test_embeddings_are_unit_length`): Embed a batch; assert each vector's L2 norm ≈ 1.0. *Expected failure: norms vary.*

16. **page_numbers CSV Parseable** (`test_coerce_metadata_adds_csv_key`): Coerce `{"page_numbers": [1, 2, 3]}`; assert `"page_numbers_csv" == "1,2,3"`. *Expected failure: key absent.*

17. **Filter Adaptation Returns Results** (`test_hybrid_retriever_filter_on_page_numbers`): Index a chunk with `page_numbers=[5]`; retrieve with `where={"page_numbers": 5}`; assert non-empty results. *Expected failure: empty results.*

18. **Token Budget Not Exceeded** (`test_context_builder_abbreviation_dense_text`): Feed abbreviation-dense text; assert assembled context token count ≤ budget × 1.05. *Expected failure: over budget.*

19. **All Candidates Evaluated** (`test_context_builder_pass_vs_continue`): Provide 10 candidates, budget tight; assert all 10 were evaluated (loop ran 10 iterations). *Expected failure: loop stalls after exception branch.*

20. **Empty LLM Response Raises** (`test_ollama_empty_response_raises`): Mock API to return `""`; assert `RuntimeError` is raised. *Expected failure: empty string returned silently.*

21. **Recovery Clears Vectors** (`test_recover_stale_calls_delete_version`): Insert partial vectors, mark document stale, call `recover_stale_documents`; assert `delete_version` was called and Chroma has 0 vectors for that document. *Expected failure: vectors remain.*

22. **WAL Checkpoint Before Backup** (`test_backup_executes_wal_checkpoint`): Mock `sqlite3.connect`; assert `PRAGMA wal_checkpoint(FULL)` was executed before the file copy. *Expected failure: PRAGMA not called.*

23. **Integration Test Uses prepare_runtime** (`test_integration_uses_compose_boundary`): Assert that `prepare_runtime()` is called during test setup by inspecting the call stack or using a mock sentinel. *Expected failure: assertion fails because `prepare_runtime` is bypassed.*

---

### Fix Checking

**Goal:** Verify that for all inputs where any bug condition holds, the fixed function produces the expected behavior (Properties 1–22).

**Pseudocode:**
```
FOR ALL X WHERE isBugCondition_Ak(X) DO
  result := F'_k(X)
  ASSERT Property_k(result)
END FOR
```

The 23 unit tests listed above serve as the primary fix-checking suite. After applying the fix, all 23 must pass.

---

### Preservation Checking

**Goal:** Verify that for all inputs where NO bug condition holds, every fixed function produces the same result as the original (Property 23).

**Pseudocode:**
```
FOR ALL X WHERE NOT any(isBugCondition_Ak(X)) DO
  ASSERT F_k(X) = F'_k(X)
END FOR
```

**Testing Approach:** Property-based testing (PBT) is used for preservation because it generates thousands of random inputs across the valid input domain, catching edge cases that hand-crafted tests miss. The PBT library generates random inputs, applies both the original and fixed function, and asserts their outputs are equal.

**Property-Based Test Cases:**

1. **File Handling Preservation** (`prop_normal_ingestion_unchanged`): Generate random single-copy PDFs (no content duplicates, source ≠ destination); assert `ingest'(X) = ingest(X)`.

2. **Text Normalization Preservation** (`prop_clean_text_ascii_unchanged`): Generate random ASCII prose strings (no superscripts/subscripts, no hyphen–newline); assert `clean_text'(X) = clean_text(X)`.

3. **Paragraph Splitting Preservation** (`prop_split_paragraphs_no_hyphen_unchanged`): Generate paragraph text without hyphen–newline sequences; assert `split_paragraphs'(X) = split_paragraphs(X)`.

4. **Chunking Preservation** (`prop_single_batch_chunking_unchanged`): Generate single-batch page sets (no cross-batch boundary); assert chunking output is identical.

5. **Embedding Preservation** (`prop_production_embeddings_normalized`): Generate random text; assert `embed'(X)` produces unit-length vectors and order matches input order.

6. **Retrieval Preservation** (`prop_retrieval_no_serialized_field_filter`): Generate queries with no filters on string-serialized fields; assert `retrieve'(X) = retrieve(X)` (same result set, same order).

7. **Context Budget Preservation** (`prop_context_builder_non_abbreviation_text`): Generate non-abbreviation-dense text; assert token estimate is unchanged.

8. **OCR Correction Idempotency** (`prop_ocr_correction_clean_text_unchanged`): Generate text with no OCR confusable patterns; assert `apply_ocr_corrections'(X) = X`.

---

### Unit Tests

- Test each `isBugCondition_*` predicate independently with boundary inputs.
- Test `_is_two_column` helper with single-column, two-column, and edge-case (3 blocks) pages.
- Test `_protect_superscripts` / `_protect_subscripts` with all Unicode ranges.
- Test `_stable_id` with same chapter/section across different `document_id` values.
- Test `delete_lexical_version` SQL directly against an in-memory SQLite database.
- Test `_test_embedding` for all configured dimensions (128, 256, 384, 768).
- Test `_adapt_filter` for each STRING_SERIALIZED_FIELDS field.
- Test `_parse_page_numbers` with JSON string, CSV string, and list inputs.
- Test `OllamaLLMClient.generate` with empty, whitespace-only, and valid response bodies.
- Test `recover_stale_documents` with and without partial Chroma vectors.

### Property-Based Tests

- Generate random PDF page objects (mocked) and verify two-column detection heuristic is consistent: if `_is_two_column(page) = True` then column A text always precedes column B text in the output.
- Generate random embedding batches and verify L2 norms are all within `[0.999, 1.001]` after normalization.
- Generate random metadata dicts and verify `_parse_page_numbers(_coerce_metadata(meta))` round-trips correctly for any integer list.
- Generate random token-budget scenarios and verify `context_builder` never exceeds budget + 5% tolerance after the fix.
- Generate random chapter-sequence page batches and verify chapter carry-forward produces consistent chapter metadata across batch boundaries.

### Integration Tests

- Full ingestion of `DC_endocrino_version_2024__portrait.pdf` via `prepare_runtime()` → verify all 88 stages complete, all chunks have non-null chapter metadata, all `page_numbers_csv` fields are present.
- Re-ingestion of the same PDF after a simulated interruption → verify lexical/vector parity before and after.
- End-to-end query for a term split across a hyphen–newline boundary → verify retrieval returns the expected chunk.
- End-to-end query with a page-number filter → verify results are non-empty and citations are correctly formatted.
- Backup operation during active ingestion → verify backup SQLite files open without WAL recovery errors.
