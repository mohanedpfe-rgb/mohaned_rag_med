"""Production-path completion probes for the 17-phase diagnostic system.

Each probe records evidence that a real production component executed.  The
module deliberately avoids label-only certification and keeps CI deterministic.
"""
from __future__ import annotations

import json
import os
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any

import fitz

from rag_project.testing.deep_diagnostics import PhaseResult
from rag_project.testing.advanced_phases import _cleanup_store, _embedding, _result

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"
CORPUS = TESTS / "support" / "gold_sets" / "diagnostic_independent_corpus.jsonl"
GOLD = TESTS / "support" / "gold_sets" / "diagnostic_independent_gold.jsonl"


def _finish(result: PhaseResult) -> PhaseResult:
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


def _png_bytes(width: int = 500, height: int = 700) -> bytes:
    """Build a tiny deterministic RGB PNG without adding a new dependency."""
    rows: list[bytes] = []
    for y in range(height):
        row = bytearray()
        row.append(0)
        for x in range(width):
            if 30 <= x < width - 30 and 30 <= y < height - 50:
                value = 215
            else:
                value = 255
            row.extend((value, value, value))
        rows.append(bytes(row))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)

    signature = b"\x89PNG\r\n\x1a\n"
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return signature + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"".join(rows), level=6)) + chunk(b"IEND", b"")


class _DeterministicOCR:
    """Minimal production OCRService-compatible adapter used to exercise the OCR branch."""

    def __init__(self) -> None:
        self.calls = 0

    def ocr_page_object(self, page: fitz.Page, page_index: int, *, force: bool = False) -> tuple[str, float]:
        self.calls += 1
        return "Scanned medical page OCR text: diabetes mellitus and HbA1c.", 0.99


def _make_pdf(path: Path, kind: str) -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    if kind == "text":
        page.insert_text((55, 70), "Chapter 1 — Diabetes mellitus")
        page.insert_text((55, 105), "Diabetes mellitus is a chronic metabolic disease. Diagnosis uses plasma glucose and HbA1c.")
    elif kind == "table":
        page.insert_text((55, 70), "Table 1: Diagnostic thresholds")
        rows = ["Measure | Threshold", "HbA1c | 6.5 %", "Fasting glucose | 126 mg/dL", "Random glucose | 200 mg/dL"]
        y = 110
        for row in rows:
            page.insert_text((55, y), row)
            y += 24
        page.insert_text((55, y + 12), "Figure 1: glucose regulation pathway")
    elif kind == "image_heavy":
        page.insert_image(fitz.Rect(35, 45, 560, 780), stream=_png_bytes())
    elif kind == "unicode_layout":
        page.insert_text((55, 70), "Chapitre 2 — Diabète / طب السكري")
        page.insert_text((55, 105), "α-glucose • HbA1c ≥ 6,5 % — signes cliniques et suivi.")
    doc.save(path)
    doc.close()


def phase7_real_pdf_lab(phase: Any) -> PhaseResult:
    result = _result(phase)
    try:
        from rag_project.parsing.pdf_extractor import PDFExtractor

        cases = ["text", "table", "image_heavy", "unicode_layout"]
        outcomes = []
        with tempfile.TemporaryDirectory(prefix="rag_phase7_pdf_") as td:
            root = Path(td)
            extractor = PDFExtractor(ocr_enabled=False)
            for kind in cases:
                pdf = root / f"{kind}.pdf"
                _make_pdf(pdf, kind)
                started = time.perf_counter()
                pages = extractor.extract(pdf, document_id=f"phase7-{kind}")
                elapsed_ms = (time.perf_counter() - started) * 1000
                outcomes.append({
                    "case": kind,
                    "pages": len(pages),
                    "text_chars": sum(len(p.text or "") for p in pages),
                    "table_count": sum(int(p.table_count or 0) for p in pages),
                    "figure_count": sum(int(p.image_count or 0) for p in pages),
                    "ocr_required": any(bool(p.ocr_required) for p in pages),
                    "ocr_status": [p.ocr_status for p in pages],
                    "page_types": [p.page_type for p in pages],
                    "elapsed_ms": round(elapsed_ms, 3),
                })

            # Exercise the actual OCR decision/merge branch with a deterministic adapter;
            # no external OCR model or network access is required in CI.
            ocr_probe = _DeterministicOCR()
            ocr_extractor = PDFExtractor(ocr_enabled=True, ocr_service=ocr_probe)
            ocr_pages = ocr_extractor.extract(root / "image_heavy.pdf", document_id="phase7-ocr")
            ocr_outcome = {
                "case": "image_heavy_ocr_enabled",
                "ocr_calls": ocr_probe.calls,
                "ocr_status": [p.ocr_status for p in ocr_pages],
                "extraction_methods": [p.extraction_method for p in ocr_pages],
                "text_chars": sum(len(p.text or "") for p in ocr_pages),
            }

            invalid = root / "malformed.pdf"
            invalid.write_bytes(b"not a pdf")
            malformed_failed = False
            try:
                extractor.extract(invalid, document_id="phase7-malformed")
            except Exception:
                malformed_failed = True

        checks = {
            "text_pdf_extracted": outcomes[0]["pages"] == 1 and outcomes[0]["text_chars"] > 40,
            "table_content_preserved": outcomes[1]["text_chars"] > 60,
            "image_heavy_classified": outcomes[2]["ocr_required"] is True or "image_heavy" in outcomes[2]["page_types"],
            "unicode_layout_extracted": outcomes[3]["text_chars"] > 20,
            "ocr_branch_executed": ocr_outcome["ocr_calls"] >= 1 and "success" in ocr_outcome["ocr_status"],
            "ocr_text_merged": ocr_outcome["text_chars"] > 20 and "ocr" in ocr_outcome["extraction_methods"],
            "malformed_pdf_rejected": malformed_failed,
        }
        result.details = {
            "evidence_level": "real_pdf_extractor",
            "cases": outcomes,
            "ocr_probe": ocr_outcome,
            "checks": checks,
            "malformed_pdf_rejected": malformed_failed,
            "production_component": "rag_project.parsing.pdf_extractor.PDFExtractor",
            "variant_results": outcomes + [ocr_outcome],
            "real_pdf_objects": True,
        }
        result.score = sum(checks.values()) / len(checks)
        result.status = "PASS" if result.score == 1.0 else "FAIL"
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "phase 7 real PDF laboratory", "exception": type(exc).__name__, "message": str(exc)})
    return _finish(result)


def phase14_real_benchmark(phase: Any) -> PhaseResult:
    result = _result(phase)
    try:
        from rag_project.parsing.pdf_extractor import PDFExtractor
        from rag_project.chunking.semantic_chunker import SemanticChunker
        from rag_project.storage.vector_store import VectorStore

        samples: dict[str, list[float]] = {"pdf_extract": [], "chunking": [], "index_build": [], "lexical_retrieval": [], "semantic_retrieval": []}
        with tempfile.TemporaryDirectory(prefix="rag_phase14_perf_") as td:
            root = Path(td)
            pdf = root / "benchmark.pdf"
            _make_pdf(pdf, "table")
            extractor = PDFExtractor(ocr_enabled=False)
            for repetition in range(7):
                t = time.perf_counter(); pages = extractor.extract(pdf, document_id=f"phase14-perf-{repetition}"); samples["pdf_extract"].append((time.perf_counter()-t)*1000)
                t = time.perf_counter(); chunks = SemanticChunker(chunk_size=220, chunk_overlap=30).chunk_pages(pages); samples["chunking"].append((time.perf_counter()-t)*1000)
                store = VectorStore(root / f"idx-{repetition}", collection_name="perf")
                docs = [c.text for c in chunks]
                metas = [{**dict(c.metadata or {}), "document_id": c.doc_id, "chunk_id": f"{c.doc_id}:{c.chunk_index}"} for c in chunks]
                embs = [_embedding(c.text) for c in chunks]
                ids = [f"{c.doc_id}:{c.chunk_index}" for c in chunks]
                t = time.perf_counter(); store.add_documents(docs, metas, embs, ids); samples["index_build"].append((time.perf_counter()-t)*1000)
                t = time.perf_counter(); store.search_lexical("HbA1c threshold", n_results=3); samples["lexical_retrieval"].append((time.perf_counter()-t)*1000)
                t = time.perf_counter(); store.search(_embedding("HbA1c threshold"), n_results=3); samples["semantic_retrieval"].append((time.perf_counter()-t)*1000)
                del store
        metrics = {
            stage: {
                "samples": len(vals),
                "p50_ms": round(statistics.median(vals), 3),
                "p95_ms": round(sorted(vals)[max(0, int(len(vals)*0.95)-1)], 3),
                "max_ms": round(max(vals), 3),
            }
            for stage, vals in samples.items()
        }
        result.details = {
            "evidence_level": "real_pdf_to_retrieval_benchmark",
            "stage_metrics": metrics,
            "stages": list(samples),
            "minimum_samples_per_stage": min(len(v) for v in samples.values()),
            "repetitions": 7,
            "pipeline_exercised": ["PDFExtractor", "SemanticChunker", "VectorStore.add_documents", "VectorStore.search_lexical", "VectorStore.search"],
            "production_components": ["PDFExtractor", "SemanticChunker", "VectorStore"],
        }
        result.score = 1.0 if all(metrics[s]["samples"] >= 7 for s in metrics) else 0.0
        result.status = "PASS" if result.score == 1.0 else "FAIL"
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "phase 14 real benchmark", "exception": type(exc).__name__, "message": str(exc)})
    return _finish(result)


def phase15_resource_stability(phase: Any) -> PhaseResult:
    result = _result(phase)
    try:
        child = ROOT / "scripts" / "diagnostic_resource_workload.py"
        if not child.exists():
            raise FileNotFoundError(child)
        duration = float(os.getenv("DIAGNOSTIC_RESOURCE_SECONDS", "20"))
        requested_mode = os.getenv("DIAGNOSTIC_RESOURCE_MODE", "bounded").strip().lower()
        if requested_mode == "24h":
            duration = max(duration, 86400.0)
        cmd = [sys.executable, str(child), "--duration", str(max(5.0, duration))]
        started = time.perf_counter()
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=max(30, int(duration)+20))
        elapsed = time.perf_counter()-started
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        result.details = {
            "evidence_level": "real_subprocess_resource_observation",
            "requested_seconds": duration,
            "requested_mode": requested_mode,
            "observed_seconds": payload.get("observed_seconds"),
            "sample_count": payload.get("sample_count"),
            "iterations": payload.get("iterations"),
            "repetitions": payload.get("iterations", 0),
            "rss_first_bytes": payload.get("rss_first_bytes"),
            "rss_last_bytes": payload.get("rss_last_bytes"),
            "rss_peak_bytes": payload.get("rss_peak_bytes"),
            "fd_first": payload.get("fd_first"),
            "fd_last": payload.get("fd_last"),
            "fd_delta": payload.get("fd_delta"),
            "exit_code": proc.returncode,
            "workload": payload.get("workload"),
            "pipeline_exercised": ["fixture chunk creation", "VectorStore add/search", "index cleanup", "RSS sampling", "FD sampling"],
            "long_running_24h_mode_supported": True,
            "certification_mode": "bounded_smoke" if requested_mode != "24h" else "24h_observation",
            "elapsed_probe_seconds": elapsed,
        }
        rss_delta = payload.get("rss_delta_bytes")
        fd_delta = payload.get("fd_delta")
        result.score = 1.0 if proc.returncode == 0 and payload.get("sample_count",0) >= 3 and payload.get("iterations",0) >= 3 and (rss_delta is None or rss_delta < 64*1024*1024) and (fd_delta is None or abs(fd_delta) <= 2) else 0.0
        result.status = "PASS" if result.score == 1.0 else "FAIL"
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "phase 15 resource stability", "exception": type(exc).__name__, "message": str(exc)})
    return _finish(result)


def phase10_real_capability(phase: Any) -> PhaseResult:
    from rag_project.testing.strict_phases import phase10_production_generation
    return phase10_production_generation(phase)


def phase16_real_pipeline(phase: Any) -> PhaseResult:
    result = _result(phase)
    tmp = None
    try:
        from rag_project.parsing.pdf_extractor import PDFExtractor
        from rag_project.chunking.semantic_chunker import SemanticChunker
        from rag_project.storage.vector_store import VectorStore
        if not CORPUS.exists() or not GOLD.exists():
            raise FileNotFoundError("separate corpus and gold files are required")
        corpus = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
        gold = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
        tmp = Path(tempfile.mkdtemp(prefix="rag_phase16_pdf_pipeline_"))
        pdf_dir = tmp / "pdf"; pdf_dir.mkdir()
        extractor = PDFExtractor(ocr_enabled=False)
        store = VectorStore(tmp / "index", collection_name="phase16")
        all_docs=[]; all_metas=[]; all_embs=[]; all_ids=[]
        extraction_records=[]
        for row in corpus:
            pdf = pdf_dir / f"{row['doc_id']}.pdf"
            doc = fitz.open(); page=doc.new_page(); page.insert_text((50,60), row['text']); doc.save(pdf); doc.close()
            pages = extractor.extract(pdf, document_id=row['doc_id'])
            chunks = SemanticChunker(chunk_size=220, chunk_overlap=30).chunk_pages(pages)
            extraction_records.append({"doc_id":row["doc_id"], "pages":len(pages), "chunks":len(chunks), "text_chars":sum(len(p.text or "") for p in pages)})
            for chunk in chunks:
                cid=f"{chunk.doc_id}:{chunk.chunk_index}"
                all_docs.append(chunk.text)
                all_metas.append({**dict(chunk.metadata or {}), "document_id":chunk.doc_id,"chunk_id":cid})
                all_embs.append(_embedding(chunk.text)); all_ids.append(cid)
        store.add_documents(all_docs, all_metas, all_embs, all_ids)
        rows=[]
        for case in gold:
            expected=set(case['expected_doc_ids'])
            lex=store.search_lexical(case['question'], n_results=min(8,len(all_docs)))
            sem=store.search(_embedding(case['question']), n_results=min(8,len(all_docs)))
            lex_ids=[str(x) for x in (lex.get('ids') or [[]])[0]]
            sem_ids=[str(x) for x in (sem.get('ids') or [[]])[0]]
            ids=lex_ids+sem_ids
            docs={v.split(':',1)[0] for v in ids}
            hit=bool(expected & docs)
            rows.append({'id':case['id'],'hit':hit,'expected':sorted(expected),'lexical_top3':lex_ids[:3],'semantic_top3':sem_ids[:3]})
        recall=sum(int(r['hit']) for r in rows)/max(1,len(rows))
        result.details={
            'evidence_level':'real_pdf_extraction_to_storage_retrieval',
            'corpus_documents':len(corpus),
            'indexed_chunks':len(all_docs),
            'gold_cases':len(gold),
            'retrieval_recall':recall,
            'gold_labels_independent_of_corpus_text':True,
            'extraction_records':extraction_records,
            'pipeline_exercised':['real PDF creation','PDFExtractor.extract','SemanticChunker.chunk_pages','VectorStore.add_documents','VectorStore.search_lexical','VectorStore.search'],
            'production_components':['PDFExtractor','SemanticChunker','VectorStore'],
            'results':rows,
            'clinical_correctness_claimed':False,
        }
        result.score=recall
        result.status='PASS' if recall>=0.8 else 'FAIL'
    except Exception as exc:
        result.status='FAIL'
        result.failures.append({'location':'phase 16 real PDF pipeline','exception':type(exc).__name__,'message':str(exc)})
    finally:
        if tmp is not None: _cleanup_store(tmp)
    return _finish(result)
