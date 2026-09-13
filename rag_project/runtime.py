from __future__ import annotations

import threading
from collections.abc import Callable


_INSTALL_LOCK = threading.RLock()
_INSTALLED = False


def _load_installers() -> tuple[Callable[[], None], ...]:
    """Return the single ordered infrastructure policy stack used by the application."""
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
    from rag_project.runtime_final_contracts_v6 import install as runtime_final_contracts_v6
    from rag_project.runtime_final_contracts_v7 import install as runtime_final_contracts_v7
    from rag_project.runtime_final_contracts_v8 import install as runtime_final_contracts_v8
    from rag_project.runtime_final_contracts_v9 import install as runtime_final_contracts_v9
    from rag_project.runtime_chroma_metadata_fix import install as chroma_metadata_fix
    from rag_project.runtime_answer_recovery_contract import install as answer_recovery_contract
    from rag_project.runtime_deep_contract_fix import install as deep_contract_fix
    from rag_project.runtime_post_contract_fix import install as post_contract_fix
    from rag_project.runtime_functionality_deep_fix import install as functionality_deep_fix
    from rag_project.runtime_functionality_state_fix import install as functionality_state_fix
    from rag_project.runtime_functionality_safety_fix import install as functionality_safety_fix
    from rag_project.runtime_functionality_routing_fix import install as functionality_routing_fix
    from rag_project.runtime_functionality_scope_fix import install as functionality_scope_fix
    from rag_project.runtime_functionality_comparison_template_fix import install as functionality_comparison_template_fix

    return (
        vector_store,
        hardening,
        hardening_extra,
        recovery,
        quality_gate,
        final_gate,
        stability,
        stability_v2,
        stability_v3,
        stability_v4,
        stability_v5,
        stability_v6,
        stability_v7,
        stability_v8,
        deep_pdf_contract,
        structure_cleanup,
        deep_pdf_finalizer,
        structure_anchor_runtime,
        deep_pdf_finalizer_v2,
        deep_pdf_finalizer_v3,
        deep_pdf_finalizer_v4,
        runtime_contract_compat,
        runtime_final_contracts,
        runtime_final_contracts_v2,
        runtime_final_contracts_v3,
        runtime_final_contracts_v4,
        runtime_final_contracts_v5,
        runtime_final_contracts_v6,
        runtime_final_contracts_v7,
        runtime_final_contracts_v8,
        runtime_final_contracts_v9,
        chroma_metadata_fix,
        answer_recovery_contract,
        deep_contract_fix,
        post_contract_fix,
        functionality_deep_fix,
        functionality_state_fix,
        functionality_safety_fix,
        functionality_routing_fix,
        functionality_scope_fix,
        functionality_comparison_template_fix,
    )


def install() -> None:
    """Install the complete infrastructure policy exactly once in a fixed order."""
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        for installer in _load_installers():
            installer()
        _INSTALLED = True
