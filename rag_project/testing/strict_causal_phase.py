"""Additional causal-graph controls for authoritative Phase 13."""
from __future__ import annotations

from typing import Any

from rag_project.testing.deep_diagnostics import PhaseResult


def strict_causal_phase(phase: Any, results: dict[int, PhaseResult]) -> PhaseResult:
    from rag_project.testing import production_diagnostic_probes as probes
    from rag_project.testing import runner

    base = probes.phase13_known_causal_graph(phase, results)
    try:
        later_pair = None
        for left in runner.PHASES:
            for right in runner.PHASES:
                if right.number <= left.number:
                    continue
                if left.number not in set(right.dependencies):
                    later_pair = (left, right)
                    break
            if later_pair:
                break
        if not later_pair:
            raise RuntimeError("unable to construct an independent non-dependency phase pair")
        left, right = later_pair
        negative = {
            left.number: PhaseResult(left.number, left.key, left.name, status="FAIL", failures=[{
                "location": "rag_project/module_alpha.py:10",
                "exception": "SharedFailure",
                "message": "same controlled failure",
            }]),
            right.number: PhaseResult(right.number, right.key, right.name, status="FAIL", failures=[{
                "location": "rag_project/module_beta.py:20",
                "exception": "SharedFailure",
                "message": "same controlled failure",
            }]),
        }
        graph = runner._hardened_causal_graph(phase, negative)
        negative_edges = {(edge["from"], edge["to"]) for edge in graph.details.get("edges", [])}
        forbidden = (f"p{left.number}f0", f"p{right.number}f0")
        unrelated_independence_verified = forbidden not in negative_edges
        base.details.update({
            "negative_control_phase_pair": [left.number, right.number],
            "negative_control_declared_dependency": False,
            "negative_control_shared_exception": True,
            "negative_control_shared_module": False,
            "negative_control_edge_absent": unrelated_independence_verified,
            "causal_false_positive_control_verified": unrelated_independence_verified,
            "authoritative_implementation": "strict_causal_phase",
        })
        base.status = "PASS" if base.status == "PASS" and unrelated_independence_verified else "FAIL"
        base.score = 1.0 if base.status == "PASS" else 0.0
        if base.status == "FAIL":
            base.failures.append({"location": "phase 13 causal false-positive control", "exception": "CausalFalsePositiveControlFailure", "message": str(base.details)})
    except Exception as exc:
        base.status = "FAIL"
        base.score = 0.0
        base.failures.append({"location": "phase 13 causal false-positive control", "exception": type(exc).__name__, "message": str(exc)})
    return base


__all__ = ["strict_causal_phase"]
