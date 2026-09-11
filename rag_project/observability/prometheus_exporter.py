"""Dependency-free Prometheus text exposition for MedEvidence Pro."""
from __future__ import annotations

from typing import Any


def _metric_name(value: str) -> str:
    return "medevidence_" + "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value)


def export_metrics(snapshot: dict[str, Any], alerts: list[str] | None = None) -> str:
    """Render the stable metrics subset using Prometheus exposition format."""
    lines: list[str] = []
    latency = snapshot.get("latency_percentiles") or {}
    for percentile in ("p50", "p95", "p99"):
        value = float(latency.get(percentile, 0.0) or 0.0)
        name = _metric_name(f"latency_{percentile}_ms")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {value}")

    usage = snapshot.get("path_usage") or {}
    for path, count in usage.items():
        safe = str(path).replace('"', '_')
        lines.append(f"# TYPE medevidence_path_usage counter")
        lines.append(f"medevidence_path_usage{{path=\"{safe}\"}} {int(count)}")

    feedback = snapshot.get("feedback") or {}
    for kind, count in feedback.items():
        safe = str(kind).replace('"', '_')
        lines.append("# TYPE medevidence_feedback_total counter")
        lines.append(f"medevidence_feedback_total{{type=\"{safe}\"}} {int(count)}")

    error_rate = float((snapshot.get("error_rates") or {}).get("negative_feedback_rate", 0.0) or 0.0)
    lines.append("# TYPE medevidence_negative_feedback_rate gauge")
    lines.append(f"medevidence_negative_feedback_rate {error_rate}")

    alert_count = len(alerts or [])
    lines.append("# TYPE medevidence_alerts gauge")
    lines.append(f"medevidence_alerts {alert_count}")
    return "\n".join(lines) + "\n"


__all__ = ["export_metrics"]
