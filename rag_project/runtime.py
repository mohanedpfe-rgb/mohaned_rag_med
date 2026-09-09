from __future__ import annotations

import threading


_INSTALL_LOCK = threading.RLock()
_INSTALLED = False


def install() -> None:
    """Install infrastructure policies exactly once.

    Answer behavior is composed explicitly by ProductionRAGSystem; this function
    no longer monkey-patches RAGSystem.answer at import/startup time.
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
        from rag_project.intelligence.god_mode import install as install_god_mode

        install_hardening()
        install_extra()
        install_recovery()
        install_quality()
        install_final()
        install_god_mode()
        _INSTALLED = True
