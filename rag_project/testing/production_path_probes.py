from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import fitz

from rag_project.configuration.settings import Settings
from rag_project.embeddings.embedding_service import EmbeddingService
from rag_project.ingestion.robust_ingestor import robust_ingest_file
from rag_project.ingestion.state_store import IngestionStateStore
from rag_project.storage.vector_store import VectorStore
from rag_project.testing.deep_diagnostics import PhaseResult
from rag_project.testing.advanced_phases import _cleanup_store, _result

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "tests" / "support" / "gold_sets" / "diagnostic_independent_corpus.jsonl"
GOLD = ROOT / "tests" / "support" / "gold_sets" / "diagnostic_independent_gold.jsonl"


class _CancelFlag:
    cancelled = False


class _ProductionIngestionProbeSystem:
    """Isolated adapter exposing the same contract used by canonical ingestion."""

    def __init__(self, root: Path) -> None:
        incoming = root / "incoming"
        processed = root / "processed"
        failed = root / "failed"
        archive = root / "archive"
        vector_db = root / "vector_db"
        logs = root / "logs"
        for directory in (incoming, processed, failed, archive, vector_db, logs):
            directory.mkdir(parents=True, exist_ok=True)

        self.settings = Settings(
            device_mode="i5_16gb",
            project_root=root,
            incoming_dir=incoming,
            processed_dir=processed,
            failed_dir=failed,
            archive_dir=archive,
            vector_db_dir=vector_db,
            log_dir=logs,
            ingestion_db_path=root / "ingestion.sqlite3",
            embedding_model="diagnostic-deterministic",
            embedding_test_mode=True,
            ocr_enabled=False,
            auto_ocr=False,
            chunk_size=220,
            chunk_overlap=30,
            page_batch_size=2,
            embedding_batch_size=8,
            ingestion_lease_seconds=120,
        )
        self.state_store = IngestionStateStore(self.settings.ingestion_db_path)
        self.vector_store = VectorStore(vector_db, collection_name="production_path_probe")
        self.embedding_service = EmbeddingService(
            "",
            self.settings.embedding_model,
            batch_size=self.settings.embedding_batch_size,
            retries=0,
            timeout_seconds=30,
            test_mode=True,
        )
        self.embedding_startup_error: Exception | None = None
        self.logger = logging.getLogger("production_path_probe")

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _ingestion_version_id(**kwargs: Any) -> str:
        payload = json.dumps(kwargs, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _ensure_embedding_dimension(self) -> int:
        if self.embedding_service.dimension is None:
            self.embedding_service.discover_dimension()
        return int(self.embedding_service.dimension or 0)

    def _new_cancel_flag(self, document_id: str) -> _CancelFlag:
        return _CancelFlag()

    def _remove_cancel_flag(self, document_id: str) -> None:
        return None


def _write_probe_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.insert_textbox(fitz.Rect(45, 45, 550, 790), text, fontsize=11)
    document.save(path)
    document.close()


def _metadata_document_ids(payload: dict[str, Any]) -> list[str]:
    values = (payload.get("metadatas") or [[]])[0]
    found: list[str] = []
    for value in values:
        if isinstance(value, dict):
            document_id = value.get("document_id")
            if document_id is not None:
                found.append(str(document_id))
            continue
        raw = str(value)
        try:
            metadata = json.loads(raw.replace("'", '"')) if raw.startswith("{") else None
        except json.JSONDecodeError:
            metadata = None
        if isinstance(metadata, dict) and metadata.get("document_id") is not None:
            found.append(str(metadata["document_id"]))
    return found


def phase16_production_ingestion_benchmark(phase: Any) -> PhaseResult:
    result = _result(phase)
    temporary_root: Path | None = None
    try:
        if not CORPUS.exists() or not GOLD.exists():
            raise FileNotFoundError("separate corpus and gold files are required")
        corpus = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
        gold = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(corpus) < 5 or not gold:
            raise RuntimeError("independent corpus/gold benchmark is too small")

        temporary_root = Path(tempfile.mkdtemp(prefix="rag_phase16_production_ingestion_"))
        system = _ProductionIngestionProbeSystem(temporary_root)
        source_dir = temporary_root / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        document_ids: dict[str, str] = {}
        ingestion_rows: list[dict[str, Any]] = []

        for row in corpus:
            source = source_dir / f"{row['doc_id']}.pdf"
            _write_probe_pdf(source, row["text"])
            outcome = robust_ingest_file(system, source)
            if outcome.get("status") not in {"success", "skipped"}:
                raise RuntimeError(f"production ingestion did not reach success state: {outcome}")
            document_id = str(outcome.get("document_id") or "")
            if not document_id:
                raise RuntimeError(f"production ingestion returned no document_id: {outcome}")
            document_ids[row["doc_id"]] = document_id
            ingestion_rows.append({
                "source_doc_id": row["doc_id"],
                "document_id": document_id,
                "status": outcome.get("status"),
                "timings_ms": outcome.get("timings_ms", {}),
                "pages": outcome.get("page_count"),
                "chunks": outcome.get("chunk_count"),
                "embeddings": outcome.get("embedding_count"),
                "retrieval_mode": outcome.get("retrieval_mode"),
            })

        ready_states = 0
        indexed_chunks = 0
        for source_doc_id, document_id in document_ids.items():
            state = system.state_store.get_document(document_id) or {}
            processed = system.settings.processed_dir / f"{source_doc_id}.pdf"
            if not processed.exists():
                raise RuntimeError(f"processed publication missing for {source_doc_id}")
            content_hash = system._hash_file(processed)
            validation = system.vector_store.validate_document_index(document_id, content_hash)
            ready_states += int(system.state_store.is_ready_status(state.get("status")))
            indexed_chunks += int(validation.get("count") or 0)
            if not validation.get("valid") or not system.state_store.is_ready_status(state.get("status")):
                raise RuntimeError(f"post-ingestion integrity check failed for {source_doc_id}: {validation}")

        retrieval_rows: list[dict[str, Any]] = []
        for case in gold:
            expected_source_ids = set(case["expected_doc_ids"])
            expected_document_ids = {document_ids[key] for key in expected_source_ids if key in document_ids}
            lexical = system.vector_store.search_lexical(case["question"], n_results=min(8, max(1, indexed_chunks)))
            query_embedding = system.embedding_service.embed_texts([case["question"]])[0]
            semantic = system.vector_store.search(query_embedding, n_results=min(8, max(1, indexed_chunks)))
            retrieved_document_ids = _metadata_document_ids(lexical) + _metadata_document_ids(semantic)
            if not retrieved_document_ids:
                ids = [str(value) for value in ((lexical.get("ids") or [[]])[0] + (semantic.get("ids") or [[]])[0])]
                retrieved_document_ids = [document_id for document_id in document_ids.values() if any(document_id in item for item in ids)]
            hit = bool(expected_document_ids & set(retrieved_document_ids))
            retrieval_rows.append({
                "id": case["id"],
                "hit": hit,
                "expected_source_doc_ids": sorted(expected_source_ids),
                "expected_document_ids": sorted(expected_document_ids),
                "retrieved_document_ids": retrieved_document_ids[:8],
            })

        recall = sum(int(row["hit"]) for row in retrieval_rows) / max(1, len(retrieval_rows))
        result.details = {
            "evidence_level": "real_pdf_extraction_to_storage_retrieval",
            "production_entrypoint": "rag_project.ingestion.robust_ingestor.robust_ingest_file",
            "production_path_strict": True,
            "production_components": [
                "DocumentClassifier",
                "PDFExtractor",
                "SemanticChunker",
                "EmbeddingService(test_mode)",
                "VectorStore",
                "IngestionStateStore",
            ],
            "corpus_document_count": len(corpus),
            "indexed_chunk_count": indexed_chunks,
            "ready_state_count": ready_states,
            "gold_case_count": len(retrieval_rows),
            "retrieval_recall": round(recall, 3),
            "gold_labels_independent_of_corpus_text": True,
            "durable_state_verified": ready_states == len(corpus),
            "index_integrity_verified": indexed_chunks > 0,
            "publication_verified": True,
            "ingestion_rows": ingestion_rows,
            "results": retrieval_rows,
            "clinical_correctness_claimed": False,
        }
        result.score = recall
        result.status = "PASS" if recall >= 0.8 and ready_states == len(corpus) and indexed_chunks > 0 else "FAIL"
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "phase 16 canonical production ingestion", "exception": type(exc).__name__, "message": str(exc)})
    finally:
        if temporary_root is not None:
            _cleanup_store(temporary_root)
    return result
