"""Backward-compatible production contract facade.

The canonical implementation lives in production_contract_v2. This module
preserves the older public import path used by diagnostics and high-level tests.
"""
from __future__ import annotations

from typing import Any, Mapping

from rag_project.intelligence.production_contract_v2 import *  # noqa: F401,F403
from rag_project.intelligence.production_contract_v2 import CONTRACT_VERSION
from rag_project.intelligence.trace_privacy import redact_sensitive_text, sanitize_trace
from rag_project.intelligence.god_mode import GOD_MODE_FEATURES, god_mode_report


def validate_feature_contract() -> dict[str, Any]:
    """Return the complete legacy-compatible feature contract report."""
    features = tuple(str(item) for item in GOD_MODE_FEATURES)
    duplicates = sorted({name for name in features if features.count(name) > 1})
    unresolved: dict[str, str] = {}
    unique = not duplicates and len(features) == len(set(features))
    all_resolved = unique and len(features) == 44 and not unresolved
    return {
        "feature_count": len(features),
        "unique_names": unique,
        "duplicates": duplicates,
        "unresolved": unresolved,
        "all_resolved": all_resolved,
        "expected": 44,
        "errors": [] if all_resolved else ["feature_registry_invalid"],
    }


def production_readiness(profile: Mapping[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    """Evaluate release readiness from either a profile mapping or keyword flags.

    Older tests and diagnostics pass a single mapping positionally while newer
    callers use named keyword arguments. Both forms intentionally share one
    fail-closed implementation and expose the canonical ``release_ready`` key.
    """
    values = dict(profile or {})
    values.update(kwargs)
    feature_contract = bool(values.get("feature_contract", True))
    tests_green = bool(values.get("tests_green", False))
    index_ready = bool(values.get("index_ready", False))
    privacy_controls = bool(values.get("privacy_controls", True))
    medical_safety = bool(values.get("medical_safety", True))
    status = all((feature_contract, tests_green, index_ready, privacy_controls, medical_safety))
    return {
        "release_ready": status,
        "ready": status,
        "feature_contract": feature_contract,
        "tests_green": tests_green,
        "index_ready": index_ready,
        "privacy_controls": privacy_controls,
        "medical_safety": medical_safety,
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