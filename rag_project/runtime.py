from __future__ import annotations

import threading


_INSTALL_LOCK = threading.RLock()
_INSTALLED = False


def install() -> None:
    """Install the production runtime policy exactly once, in deterministic order."""
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return

        from rag_project.runtime_hardening import install as install_hardening
        from rag_project.runtime_hardening_extra import install as install_extra
        from rag_project.runtime_recovery import install as install_recovery
        from rag_project.runtime_quality_gate import install as install_quality
        from rag_project.runtime_final_gate import install as install_final
        from rag_project.runtime_stability import install as install_stability
        from rag_project.runtime_stability_v2 import install as install_stability_v2
        from rag_project.runtime_stability_v3 import install as install_stability_v3
        from rag_project.runtime_stability_v4 import install as install_stability_v4
        from rag_project.runtime_stability_v5 import install as install_stability_v5
        from rag_project.runtime_stability_v6 import install as install_stability_v6
        from rag_project.runtime_stability_v7 import install as install_stability_v7
        from rag_project.runtime_stability_v8 import install as install_stability_v8

        install_hardening()
        install_extra()
        install_recovery()
        install_quality()
        install_final()
        install_stability()
        install_stability_v2()
        install_stability_v3()
        install_stability_v4()
        install_stability_v5()
        install_stability_v6()
        install_stability_v7()
        # v8 closes UI-level array truth, Ollama health, job retention, and upload-arrival races.
        install_stability_v8()
        _INSTALLED = True
