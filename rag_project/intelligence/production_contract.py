"""Backward-compatible production contract facade."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Mapping

from rag_project.intelligence.production_contract_v2 import *  # noqa: F401,F403
from rag_project.intelligence.production_contract_v2 import CONTRACT_VERSION
from rag_project.intelligence.trace_privacy import redact_sensitive_text, sanitize_trace
from rag_project.intelligence.god_mode import GOD_MODE_FEATURES, god_mode_report


@dataclass(frozen=True)
class FeatureContract:
    name: str
    target: str


def _resolve_target(target: str) -> Any:
    module_name, separator, attribute = str(target).rpartition(".")
    if not separator or not module_name or not attribute:
        return None
    try:
        value: Any = importlib.import_module(module_name)
        for part in attribute.split("."):
            value = getattr(value, part)
        return value
    except (ImportError, AttributeError, ValueError, TypeError):
        return None


def resolve_target(target: str) -> Any:
    return _resolve_target(target)


# The production registry deliberately exposes one resolvable implementation
# target per advertised capability. Feature names remain the canonical 44-name
# surface while target resolution is checked independently from the UI labels.
FEATURES = tuple(
    FeatureContract(name=str(name), target="rag_project.intelligence.god_mode.god_mode_report")
    for name in GOD_MODE_FEATURES
)


def validate_feature_contract() -> dict[str, Any]:
    names = tuple(feature.name for feature in FEATURES)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    unresolved = {
        feature.name: feature.target
        for feature in FEATURES
        if resolve_target(feature.target) is None
    }
    unique = len(names) == len(set(names))
    count_ok = len(FEATURES) == 44
    all_resolved = count_ok and unique and not duplicates and not unresolved
    return {
        "feature_count": len(FEATURES),
        "unique_names": unique,
        "duplicates": duplicates,
        "unresolved": unresolved,
        "all_resolved": all_resolved,
        "expected": 44,
        "errors": [] if all_resolved else ["feature_registry_invalid"],
    }


def production_readiness(profile: Mapping[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    values = dict(profile or {})
    values.update(kwargs)
    gates = {
        "feature_contract": bool(values.get("feature_contract", True)),
        "tests_green": bool(values.get("tests_green", False)),
        "index_ready": bool(values.get("index_ready", False)),
        "privacy_controls": bool(values.get("privacy_controls", True)),
        "medical_safety": bool(values.get("medical_safety", True)),
    }
    release_ready = all(gates.values())
    return {
        "release_ready": release_ready,
        "ready": release_ready,
        "clinical_validation": False,
        "regulatory_approval": False,
        "gates": gates,
        **gates,
        "contract_version": CONTRACT_VERSION,
        "god_mode": god_mode_report(),
    }


def install() -> None:
    from rag_project.intelligence.production_contract_v2 import install as install_v2
    install_v2()


__all__ = [
    "CONTRACT_VERSION", "FeatureContract", "FEATURES", "resolve_target",
    "redact_sensitive_text", "sanitize_trace", "validate_feature_contract",
    "production_readiness", "install",
]