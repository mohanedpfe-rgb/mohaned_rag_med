from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence


@dataclass(frozen=True)
class CalibratedConfidence:
    raw: float
    calibrated: float
    level: str
    factors: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def calibrate_confidence(*, retrieval: float, rerank: float, entailment: float, entity_coverage: float, source_agreement: float, contradiction: float, safety_conflict: float, ocr_penalty: float = 0.0) -> CalibratedConfidence:
    factors = {
        "retrieval": max(0.0, min(1.0, retrieval)),
        "rerank": max(0.0, min(1.0, rerank)),
        "entailment": max(0.0, min(1.0, entailment)),
        "entity_coverage": max(0.0, min(1.0, entity_coverage)),
        "source_agreement": max(0.0, min(1.0, source_agreement)),
        "contradiction": max(0.0, min(1.0, contradiction)),
        "safety_conflict": max(0.0, min(1.0, safety_conflict)),
        "ocr_penalty": max(0.0, min(1.0, ocr_penalty)),
    }
    raw = (
        0.18 * factors["retrieval"] +
        0.18 * factors["rerank"] +
        0.24 * factors["entailment"] +
        0.16 * factors["entity_coverage"] +
        0.10 * factors["source_agreement"] -
        0.08 * factors["contradiction"] -
        0.04 * factors["safety_conflict"] -
        0.02 * factors["ocr_penalty"]
    )
    # Conservative calibration: uncertainty near the decision boundary is compressed.
    calibrated = max(0.0, min(1.0, 0.5 + (raw - 0.5) * 0.90))
    level = "high" if calibrated >= 0.78 else "medium" if calibrated >= 0.52 else "low"
    return CalibratedConfidence(round(raw, 4), round(calibrated, 4), level, {k: round(v, 4) for k, v in factors.items()})


def confidence_gate(confidence: CalibratedConfidence, *, required: float = 0.50) -> bool:
    return confidence.calibrated >= required
