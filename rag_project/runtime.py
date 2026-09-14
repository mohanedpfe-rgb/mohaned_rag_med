from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from typing import Any


_INSTALL_LOCK = threading.RLock()
_INSTALL_APPLICATION_CONTRACT_LOCK = threading.RLock()
_INSTALLED = False
_INSTALL_PROVENANCE: list[dict[str, Any]] = []


def _install_ingestion_compatibility() -> None:
    """Deprecated compatibility hook; ingestion helpers now belong to state_store."""
    return None


def _load_installers() -> tuple[Callable[[], None], ...]:
    """Return infrastructure installers in one deterministic order.

    This remains a compatibility composition root for low-level infrastructure.
    It does not install answer contracts or wrap application/legacy methods.
    """
    from rag_project.runtime_hardening import install as hardening
    from rag_project.runtime_hardening_extra import install as hardening_extra
    from rag_project.runtime_recovery import install as recovery
    from rag_project.runtime_quality_gate import install as quality_gate
    from rag_project.runtime_final_gate import install as final_gate
    from rag_project.runtime_stability import install as stability
    from rag_project.runtime_stability_v2 import install as stability_v2
    from rag_project.runtime_stability_v3 import install as stability_v3
    from rag_project.runtime_stability_v4 import install as stability_v4
    from rag_project.runtime_stability_v5 import install as stability_v5
    from rag_project.runtime_stability_v6 import install as stability_v6
    from rag_project.runtime_stability_v7 import install as stability_v7
    from rag_project.runtime_stability_v8 import install as stability_v8
    from rag_project.storage.vector_store_runtime import install as vector_store
    from rag_project.runtime_chroma_distance_fix import install as chroma_distance_fix
    from rag_project.intelligence.deep_pdf_contract import install as deep_pdf_contract
    from rag_project.intelligence.structure_cleanup_contract import install as structure_cleanup
    from rag_project.intelligence.deep_pdf_finalizer import install as deep_pdf_finalizer
    from rag_project.intelligence.structure_anchor_runtime import install as structure_anchor_runtime
    from rag_project.intelligence.deep_pdf_finalizer_v2 import install as deep_pdf_finalizer_v2
    from rag_project.intelligence.deep_pdf_finalizer_v3 import install as deep_pdf_finalizer_v3
    from rag_project.intelligence.deep_pdf_finalizer_v4 import install as deep_pdf_finalizer_v4
    from rag_project.runtime_contract_compat import install as runtime_contract_compat
    from rag_project.runtime_final_contracts import install as runtime_final_contracts
    from rag_project.runtime_final_contracts_v2 import install as runtime_final_contracts_v2
    from rag_project.runtime_final_contracts_v3 import install as runtime_final_contracts_v3
    from rag_project.runtime_final_contracts_v4 import install as runtime_final_contracts_v4
    from rag_project.runtime_final_contracts_v5 import install as runtime_final_contracts_v5
    from rag_project.runtime_final_contracts_v7 import install as runtime_final_contracts_v7
    from rag_project.runtime_final_contracts_v8 import install as runtime_final_contracts_v8
    from rag_project.runtime_chroma_metadata_fix import install as chroma_metadata_fix
    from rag_project.runtime_hierarchy_path_fix import install as hierarchy_path_fix
    from rag_project.runtime_version_rollback_fix import install as version_rollback_fix
    from rag_project.runtime_answer_recovery_contract import install as answer_recovery_contract
    from rag_project.runtime_deep_contract_fix import install as deep_contract_fix
    from rag_project.runtime_post_contract_fix import install as post_contract_fix
    from rag_project.runtime_functionality_deep_fix import install as functionality_deep_fix
    from rag_project.runtime_functionality_state_fix import install as functionality_state_fix
    from rag_project.runtime_functionality_safety_fix import install as functionality_safety_fix
    from rag_project.runtime_functionality_routing_fix import install as functionality_routing_fix
    from rag_project.runtime_functionality_scope_fix import install as functionality_scope_fix
    from rag_project.runtime_functionality_comparison_template_fix import install as functionality_comparison_template_fix
    from rag_project.runtime_functionality_numeric_range_fix import install as functionality_numeric_range_fix
    from rag_project.runtime_functionality_final_audit_fix import install as functionality_final_audit_fix
    from rag_project.runtime_production_contract_fix import install as production_contract_fix
    from rag_project.runtime_ready_publication_fix import install as ready_publication_fix
    from rag_project.runtime_transactional_rollback_fix import install as transactional_rollback_fix
    from rag_project.runtime_chroma_lifecycle_fix import install as chroma_lifecycle_fix
    from rag_project.runtime_page_identity_fix import install as page_identity_fix
    from rag_project.runtime_phase16_contract_fix import install as phase16_contract_fix
    from rag_project.runtime_ocr_state_fix import install as ocr_state_fix
    from rag_project.runtime_phase16_evidence_fix import install as phase16_evidence_fix
    from rag_project.runtime_version_transaction_fix import install as version_transaction_fix
    from rag_project.runtime_transaction_numpy_fix import install as transaction_numpy_fix
    from rag_project.runtime_failed_publication_cleanup import install as failed_publication_cleanup
    from rag_project.runtime_post_index_publication_contract import install as post_index_publication_contract
    from rag_project.runtime_post_index_publication_contract_v2 import install as post_index_publication_contract_v2
    from rag_project.runtime_terminal_state_guard import install as terminal_state_guard
    from rag_project.runtime_cancel_flag_fix import install as cancel_flag_fix

    return (
        vector_store, chroma_distance_fix, hardening, hardening_extra, recovery, quality_gate, final_gate,
        stability, stability_v2, stability_v3, stability_v4, stability_v5, stability_v6,
        stability_v7, stability_v8, deep_pdf_contract, structure_cleanup,
        deep_pdf_finalizer, structure_anchor_runtime, deep_pdf_finalizer_v2,
        deep_pdf_finalizer_v3, deep_pdf_finalizer_v4, runtime_contract_compat,
        runtime_final_contracts, runtime_final_contracts_v2, runtime_final_contracts_v3,
        runtime_final_contracts_v4, runtime_final_contracts_v5, runtime_final_contracts_v7,
        runtime_final_contracts_v8, chroma_metadata_fix, hierarchy_path_fix,
        version_rollback_fix, answer_recovery_contract, deep_contract_fix,
        post_contract_fix, functionality_deep_fix, functionality_state_fix,
        functionality_safety_fix, functionality_routing_fix, functionality_scope_fix,
        functionality_comparison_template_fix, functionality_numeric_range_fix,
        functionality_final_audit_fix, production_contract_fix, ready_publication_fix,
        transactional_rollback_fix, chroma_lifecycle_fix, page_identity_fix,
        phase16_contract_fix, ocr_state_fix, phase16_evidence_fix,
        version_transaction_fix, transaction_numpy_fix, failed_publication_cleanup,
        post_index_publication_contract, post_index_publication_contract_v2,
        terminal_state_guard, cancel_flag_fix,
    )


def install() -> None:
    """Install low-level infrastructure policy once and record provenance."""
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        _INSTALL_PROVENANCE.clear()
        _install_ingestion_compatibility()
        for installer in _load_installers():
            name = getattr(installer, "__module__", "unknown") + "." + getattr(installer, "__name__", "install")
            started = time.perf_counter()
            entry: dict[str, Any] = {"installer": name, "status": "RUNNING"}
            try:
                installer()
            except Exception as exc:
                entry.update({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)})
                entry["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
                _INSTALL_PROVENANCE.append(entry)
                raise
            entry["status"] = "OK"
            entry["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
            _INSTALL_PROVENANCE.append(entry)
            if os.getenv("RAG_RUNTIME_TRACE", "").strip().lower() in {"1", "true", "yes"}:
                print(f"[runtime] {name}: OK ({entry['elapsed_ms']} ms)")
        _INSTALLED = True


def installation_report() -> dict[str, Any]:
    return {
        "installed": bool(_INSTALLED),
        "installer_count": len(_INSTALL_PROVENANCE),
        "failed": [dict(item) for item in _INSTALL_PROVENANCE if item.get("status") == "FAILED"],
        "installers": [dict(item) for item in _INSTALL_PROVENANCE],
    }


def _install_ready_only_lexical_boundary() -> None:
    """Deprecated compatibility hook. Ready-only semantics live in VectorStore."""
    return None


def install_application_contracts() -> dict[str, object]:
    """Install only non-behavioral application contract registration.

    The production contract is applied at the canonical answer boundary in
    application_answer_service.answer(). This function must never monkey-patch
    top_level_pipeline, god_mode_100, or the legacy ProductionRAGSystem class.
    """
    from rag_project.intelligence.pipeline_integrity import install as pipeline_integrity
    from rag_project.ingestion.ingestion_contract import install as ingestion_contract
    from rag_project.canonical_runtime import install as canonical_runtime
    from rag_project.intelligence.production_contract_v2 import CONTRACT_VERSION

    with _INSTALL_APPLICATION_CONTRACT_LOCK:
        pipeline_integrity()
        ingestion_contract()
        canonical = canonical_runtime()
        return {
            "pipeline_integrity": True,
            "production_contract": {"version": CONTRACT_VERSION, "behavioral_patch": False},
            "ingestion_contract": True,
            "canonical_runtime": canonical,
        }
