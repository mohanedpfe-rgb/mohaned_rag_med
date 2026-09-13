"""Compatibility boundary for the retired runtime-v7 contract patch layer.

Runtime-v7 used to overwrite authoritative follow-up and numeric APIs with
legacy protocol shapes. Those public contracts are now owned by their normal
modules and must never be monkey-patched from this compatibility layer.
"""
from __future__ import annotations

_INSTALLED = False


def install() -> None:
    """Keep the legacy compatibility layer inert.

    The module remains importable because the runtime installer stack includes
    it, but it deliberately performs no mutation. Canonical contracts are
    installed by their owning modules after infrastructure startup.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True


__all__ = ["install"]
