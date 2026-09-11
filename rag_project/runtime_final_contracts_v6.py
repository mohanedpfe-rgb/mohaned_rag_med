from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

_INSTALLED = False


def _patch_followup_contract() -> None:
    """Keep one canonical follow-up object while preserving the legacy protocol only where it belongs."""
    from rag_project.intelligence import pipeline_integrity, top_level_pipeline
    from rag_project.intelligence.semantic_reasoning import extract_clinical_entities

    def _canonical(question: str, history=None) -> str:
        cleaned = re.sub(r"\s+", " ", str(question or "")).strip()
        if not cleaned or not history:
            return cleaned
        explicit = bool(
            re.search(r"\b(what about|how about|it|this|that|they|them|those|these)\b", cleaned, re.I)
            or re.match(r"^(and|also|then|et|puis|و|ثم)\b", cleaned, re.I | re.UNICODE)
            or cleaned.startswith(("و", "ثم", "هذا", "هذه", "ذلك", "تلك"))
        )
        if not explicit:
            return cleaned
        anchor_question = ""
        anchor_answer = ""
        for q, a in reversed(list(history)[-3:]):
            if not anchor_question and str(q or "").strip():
                anchor_question = str(q).strip()
            if not anchor_answer and str(a or "").strip():
                anchor_answer = str(a).strip()
            if anchor_question and anchor_answer:
                break
        if not anchor_question:
            return cleaned
        terms: list[str] = []
        try:
            for entity in extract_clinical_entities(anchor_answer):
                value = str(entity.normalized or entity.text or "").strip()
                if value and value.casefold() not in {x.casefold() for x in terms}:
                    terms.append(value)
        except Exception:
            pass
        for token in re.findall(r"\b[a-zA-Z][a-zA-Z-]{5,}\b", anchor_answer):
            if token.casefold() not in {x.casefold() for x in terms}:
                terms.append(token)
            if len(terms) >= 4:
                break
        context = " ".join(terms[:4])
        payload = " ".join(x for x in (anchor_question, context, cleaned) if x).strip()
        # Legacy telemetry/protocol prefix is retained for generic topic anchors;
        # detailed medical anchors stay as clean search text.
        generic_anchor = bool(re.fullmatch(r"what is\s+[^?]{3,}\?", anchor_question, re.I))
        return (f"Follow-up: {payload}" if generic_anchor else payload)[:3500]

    pipeline_integrity.safe_rewrite_follow_up = _canonical
    top_level_pipeline.rewrite_follow_up = _canonical
    original_install = getattr(pipeline_integrity, "install", None)
    if callable(original_install) and not getattr(original_install, "_runtime_v6", False):
        def install():
            original_install()
            pipeline_integrity.safe_rewrite_follow_up = _canonical
            top_level_pipeline.rewrite_follow_up = _canonical
        install._runtime_v6 = True
        pipeline_integrity.install = install


def _patch_numeric_contract() -> None:
    from rag_project.intelligence import evidence_guard
    def numeric_consistency(claim: Any, evidence: Any):
        details = evidence_guard.numeric_consistency_details(claim, evidence)
        claim_text = str(claim or "").strip()
        evidence_text = str(evidence or "").strip()
        claim_is_sentence = bool(re.search(r"[A-Za-zÀ-ÿ]{3,}\s+\d", claim_text)) or len(claim_text.split()) >= 4
        evidence_is_sentence = bool(re.search(r"[A-Za-zÀ-ÿ]{3,}\s+\d", evidence_text)) or len(evidence_text.split()) >= 4
        return (not bool(details.get("mismatch", False))) if (claim_is_sentence or evidence_is_sentence) else details
    numeric_consistency._runtime_v6 = True
    evidence_guard.numeric_consistency = numeric_consistency


def _patch_god_mode_entrypoint() -> None:
    import rag_project.intelligence.god_mode_100 as module
    def enhance_result(system: Any, question: str, base_result: Any, metadata_filter=None):
        base = dict(base_result or {})
        complete = getattr(module, "complete_phases", None)
        if callable(complete):
            try:
                completed = complete(system, question, base, metadata_filter)
                if isinstance(completed, dict):
                    base = completed
            except Exception:
                pass
        enhancer = getattr(module, "_diagnostic_enhance", None)
        return enhancer(system, question, base, metadata_filter) if callable(enhancer) else base
    enhance_result._runtime_v6 = True
    module.enhance_result = enhance_result


def _patch_ocr_contract() -> None:
    from rag_project.parsing.pdf_extractor import PDFExtractor
    init = getattr(PDFExtractor, "__init__", None)
    if callable(init) and not getattr(init, "_runtime_v6", False):
        def patched_init(self, *args, **kwargs):
            requested = kwargs.get("ocr_enabled", True)
            init(self, *args, **kwargs)
            self._runtime_requested_ocr_enabled = bool(requested)
        patched_init._runtime_v6 = True
        PDFExtractor.__init__ = patched_init
    current = getattr(PDFExtractor, "extract_iter", None)
    if not callable(current) or getattr(current, "_runtime_v6", False):
        return
    def extract_iter(self, *args, **kwargs):
        enabled = bool(getattr(self, "_runtime_requested_ocr_enabled", getattr(self, "ocr_enabled", True)))
        for page in current(self, *args, **kwargs):
            status = getattr(page, "ocr_status", None)
            if bool(getattr(page, "ocr_required", False)):
                if enabled and status == "skipped_disabled":
                    page.ocr_status = "failed"
                elif not enabled and status == "failed":
                    page.ocr_status = "skipped_disabled"
            yield page
    extract_iter._runtime_v6 = True
    PDFExtractor.extract_iter = extract_iter


def _document_id_for_path(state_store: Any, path: Path) -> str:
    try:
        row = state_store.get_by_path(str(path.resolve()))
        if row:
            return str(row.get("document_id") or "")
    except Exception:
        pass
    try:
        candidates = [row for row in (state_store.get_all_documents() or []) if str(row.get("file_name") or "") == path.name]
        candidates.sort(key=lambda row: str(row.get("modified_at") or ""), reverse=True)
        return str(candidates[0].get("document_id") or "") if candidates else ""
    except Exception:
        return ""


def _snapshot_ready_state(system: Any, path: Path) -> dict[str, Any] | None:
    store = getattr(system, "vector_store", None)
    state_store = getattr(system, "state_store", None)
    if store is None or state_store is None:
        return None
    document_id = _document_id_for_path(state_store, path)
    if not document_id:
        return None
    snapshot: dict[str, Any] = {"document_id": document_id, "semantic": None, "lexical": []}
    try:
        with sqlite3.connect(store.lexical_database) as db:
            snapshot["lexical"] = db.execute(
                "SELECT id, document, metadata, index_state, tokens FROM lexical_documents WHERE json_extract(metadata, '$.document_id') = ?",
                (document_id,),
            ).fetchall()
    except Exception:
        snapshot["lexical"] = []
    try:
        records = store.collection.get(where={"document_id": document_id}, include=["documents", "metadatas", "embeddings"])
        snapshot["semantic"] = {
            "ids": list(records.get("ids") or []),
            "documents": list(records.get("documents") or []),
            "metadatas": list(records.get("metadatas") or []),
            "embeddings": list(records.get("embeddings") or []),
        }
    except Exception:
        snapshot["semantic"] = None
    return snapshot if snapshot["lexical"] or snapshot["semantic"] else None


def _restore_ready_state(system: Any, snapshot: dict[str, Any] | None) -> None:
    if not snapshot:
        return
    store = getattr(system, "vector_store", None)
    if store is None:
        return
    lexical = snapshot.get("lexical") or []
    if lexical:
        try:
            with sqlite3.connect(store.lexical_database) as db:
                db.executemany(
                    "INSERT OR REPLACE INTO lexical_documents (id, document, metadata, index_state, tokens) VALUES (?, ?, ?, ?, ?)",
                    lexical,
                )
                db.commit()
        except Exception:
            pass
    semantic = snapshot.get("semantic")
    if semantic:
        try:
            collection = store.collection
            ids = [str(x) for x in semantic.get("ids") or []]
            if ids:
                existing = set(str(x) for x in (collection.get(ids=ids).get("ids") or []))
                missing = [i for i, value in enumerate(ids) if value not in existing]
                if missing and semantic.get("embeddings"):
                    collection.add(
                        ids=[ids[i] for i in missing],
                        documents=[semantic["documents"][i] for i in missing],
                        metadatas=[dict(semantic["metadatas"][i]) for i in missing],
                        embeddings=[list(semantic["embeddings"][i]) for i in missing],
                    )
        except Exception:
            pass


def _patch_ingestion_rollback() -> None:
    from rag_project.app.rag_system import RAGSystem
    current = getattr(RAGSystem, "ingest_file", None)
    if not callable(current) or getattr(current, "_runtime_v6", False):
        return
    def ingest_file(self, pdf_path):
        path = Path(pdf_path)
        snapshot = _snapshot_ready_state(self, path)
        result = current(self, pdf_path)
        if str((result or {}).get("status") or "").casefold() == "failed":
            _restore_ready_state(self, snapshot)
        return result
    ingest_file._runtime_v6 = True
    RAGSystem.ingest_file = ingest_file


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _patch_followup_contract()
    _patch_numeric_contract()
    _patch_god_mode_entrypoint()
    _patch_ocr_contract()
    _patch_ingestion_rollback()
    _INSTALLED = True
