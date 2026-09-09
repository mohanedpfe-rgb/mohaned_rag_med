from __future__ import annotations

import threading


_INSTALL_LOCK = threading.RLock()
_INSTALLED = False


def install() -> None:
    """Install cross-cutting infrastructure policies exactly once.

    Answer behavior is owned by ``ProductionRAGSystem`` and is never installed by
    global monkey-patching.  The runtime layer is limited to ingestion/index
    safety policies that existing low-level components depend on.
    """
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        from rag_project.runtime_hardening import install as install_hardening
        from rag_project.runtime_hardening_extra import install as install_extra
        from rag_project.runtime_recovery import install as install_recovery
        from rag_project.runtime_quality_gate import install as install_quality
        from rag_project.runtime_final_gate import install as install_final

        install_hardening()
        install_extra()
        install_recovery()
        install_quality()
        install_final()
        _INSTALLED = True
