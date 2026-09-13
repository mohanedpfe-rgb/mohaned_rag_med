"""Backward-compatible production contract facade.

The canonical implementation lives in production_contract_v2. This module
preserves the older public import path used by diagnostics and high-level tests.
"""
from __future__ import annotations

from typing import Any, Mapping

from rag_project.intelligence.production_contract_v2 import *  # noqa: F401,F403
from rag_project.intelligence.production_contract_v2 import CONTRACT_VERSION
from rag_project.intelligence.trace_privacy import redact_sensitive_text, sanitize_trace
from rag_project.intelligence.god_mode import GOD_MODE_FEATURES, report as god_mode_report


def validate_feature_contract() -> dict[str, Any]:
    features = tuple(GOD_MODE_FEATURES)
    unique = len(features) == len(set(features))
    return {
        "feature_count": len(features),
        "unique_names": unique,
        "all_resolved": unique and len(features) == 44,
        "expected": 44,
        "errors": [] if unique and len(features) == 44 else ["feature_registry_invalid"],
    }


def production_readiness(*, feature_contract: bool = True, tests_green: bool = False, index_ready: bool = False, privacy_controls: bool = True, medical_safety: bool = True) -> dict[str, Any]:
    status = bool(feature_contract and tests_green and index_ready and privacy_controls and medical_safety)
    return {
        "ready": status,
        "feature_contract": bool(feature_contract),
        "tests_green": bool(tests_green),
        "index_ready": bool(index_ready),
        "privacy_controls": bool(privacy_controls),
        "medical_safety": bool(medical_safety),
        "contract_version": CONTRACT_VERSION,
        "god_mode": god_mode_report(),
    }


def install() -> None:
    from rag_project.intelligence.production_contract_v2 import install as install_v2
    install_v2()


__all__ = [
    "CONTRACT_VERSION", "redact_sensitive_text", "sanitize_trace",
    "validate_feature_contract", "production_readiness", "install",
]
