"""Deprecated compatibility module.

Production contracts now live in their owning ingestion, evidence, and storage
modules. This module intentionally performs no monkey-patching.
"""
from __future__ import annotations


def install() -> None:
    return None


__all__ = ["install"]
