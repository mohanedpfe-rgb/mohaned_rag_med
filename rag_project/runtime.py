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
