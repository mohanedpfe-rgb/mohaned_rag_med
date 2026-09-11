from __future__ import annotations

_INSTALLED = False


def install() -> None:
    """Apply the final canonical contract boundary after legacy adapters."""
    global _INSTALLED
    if _INSTALLED:
        return
    from rag_project.runtime_canonical_contract_guard import install as canonical_guard

    canonical_guard()
    _INSTALLED = True
