from __future__ import annotations

import threading


_INSTALL_LOCK = threading.RLock()
_INSTALLED = False


def install() -> None:
    """Install the complete production runtime hardening and intelligence stack exactly once."""
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
        # Must be last: the intelligence layer consumes the already-hardened
        # RAGSystem, VectorStore, parser and model clients.
        install_god_mode()
        _INSTALLED = True
