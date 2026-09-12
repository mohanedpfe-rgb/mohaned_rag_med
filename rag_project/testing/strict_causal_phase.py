"""Additional causal-graph controls for authoritative Phase 13."""
from __future__ import annotations

import re
from typing import Any

from rag_project.testing.deep_diagnostics import PhaseResult


def _phase_is_ancestor(phases: tuple[Any, ...], ancestor: int, descendant: int) -> bool:
    graph = {item.number: set(item.dependencies) for item in phases}
    pending = list(graph.get(descendant, set()))
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        if current == ancestor:
            return True
        pending.extend(graph.get(current, set()))
    return False


def _strict_dependency_graph(phase: Any, results: dict[int, PhaseResult]) -> tuple[list[dict[str, Any]], list[str]]:
    from rag_project.testing import runner

    phases = tuple(runner.PHASES)
    nodes: list[dict[str, Any]] = []
    for number, phase_result in sorted(results.items()):
        for index, failure in enumerate(phase_result.failures):
            nodes.append({
                "id": f"p{number}f{index}",
                "phase": number,
                "location": failure.get("location"),
                "exception": failure.get("exception"),
                "message": failure.get("message"),
            })

    edges: list[dict[str, Any]] = []
    for left in nodes:
        for right in nodes:
            if left["id"] == right["id"] or left["phase"] >= right["phase"]:
                continue
            dependency_path = _phase_is_ancestor(phases, left["phase"], right["phase"])
            if not dependency_path:
                continue
            same_exception = bool(left.get("exception")) and left["exception"] == right.get("exception")
            left_terms = set(re.findall(r"[a-z_]{5,}", str(left.get("message") or "").casefold()))
            right_terms = set(re.findall(r"[a-z_]{5,}", str(right.get("message") or "").casefold()))
            shared_terms = bool(left_terms & right_terms)
            if not (same_exception or shared_terms):
                continue
            reasons = ["dependency_reachability"]
            if same_exception:
                reasons.append("same_exception")
            if shared_terms:
                reasons.append("shared_failure_terms")
            confidence = 0.97 if same_exception and dependency_path else 0.90
            edges.append({"from": left["id"], "to": right["id"], "reason": reasons, "confidence": confidence})

    roots = [node["id"] for node in nodes if not any(edge["to"] == node["id"] for edge in edges)]
    return edges, roots


def strict_causal_phase(phase: Any, results: dict[int, PhaseResult]) -> PhaseResult:
    from rag_project.testing import production_diagnostic_probes as probes
    from rag_project.testing import runner

    base = probes.phase13_known_causal_graph(phase, results)
    try:
        strict_edges, strict_roots = _strict_dependency_graph(phase, {
            number: PhaseResult(
                item.number,
                item.key,
                item.name,
                status=item.status,
                failures=list(item.failures),
            )
            for number, item in results.items()
        })

        observed_edges = {(edge["from"], edge["to"]) for edge in strict_edges}
        known_required = {("p5f0", "p9f0"), ("p9f0", "p10f0"), ("p5f0", "p10f0")}
        known_runtime_chain_verified = bool(base.details.get("known_failure_injection_verified")) and known_required.issubset(observed_edges)

        phase_pair = None
        for left in runner.PHASES:
            for right in runner.PHASES:
                if right.number <= left.number:
                    continue
                if left.number not in set(right.dependencies):
                    phase_pair = (left, right)
                    break
            if phase_pair:
                break
        if phase_pair is None:
            raise RuntimeError("unable to construct an independent non-dependency phase pair")
        left, right = phase_pair

        negative_unrelated = {
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
        unrelated_edges, _ = _strict_dependency_graph(phase, negative_unrelated)
        unrelated_forbidden = (f"p{left.number}f0", f"p{right.number}f0")
        unrelated_independence_verified = unrelated_forbidden not in {(e["from"], e["to"]) for e in unrelated_edges}

        shared_module_left = next((item for item in runner.PHASES if item.number not in {left.number, right.number}), None)
        shared_module_pair_verified = False
        if shared_module_left is not None:
            negative_shared_module = {
                shared_module_left.number: PhaseResult(shared_module_left.number, shared_module_left.key, shared_module_left.name, status="FAIL", failures=[{
                    "location": "rag_project/shared_module.py:10",
                    "exception": "SharedFailure",
                    "message": "same controlled failure",
                }]),
                right.number: PhaseResult(right.number, right.key, right.name, status="FAIL", failures=[{
                    "location": "rag_project/shared_module.py:20",
                    "exception": "SharedFailure",
                    "message": "same controlled failure",
                }]),
            }
            if not _phase_is_ancestor(tuple(runner.PHASES), shared_module_left.number, right.number):
                shared_module_edges, _ = _strict_dependency_graph(phase, negative_shared_module)
                shared_module_forbidden = (f"p{shared_module_left.number}f0", f"p{right.number}f0")
                shared_module_pair_verified = shared_module_forbidden not in {(e["from"], e["to"]) for e in shared_module_edges}

        base.details.update({
            "evidence_level": "dependency_supported_causal_graph",
            "algorithm": "dependency-reachability plus shared exception/terms; non-dependency similarity cannot create an edge",
            "edges": strict_edges,
            "candidate_roots": strict_roots,
            "strict_causal_edges": len(strict_edges),
            "known_runtime_chain_verified": known_runtime_chain_verified,
            "negative_control_phase_pair": [left.number, right.number],
            "negative_control_declared_dependency": False,
            "negative_control_edge_absent": unrelated_independence_verified,
            "negative_control_shared_module_edge_absent": shared_module_pair_verified,
            "causal_false_positive_control_verified": unrelated_independence_verified and shared_module_pair_verified,
            "authoritative_implementation": "strict_causal_phase",
        })
        base.status = "PASS" if base.status == "PASS" and known_runtime_chain_verified and unrelated_independence_verified and shared_module_pair_verified else "FAIL"
        base.score = 1.0 if base.status == "PASS" else 0.0
        if base.status == "FAIL":
            base.failures.append({"location": "phase 13 dependency-supported causal contract", "exception": "CausalGraphContractFailure", "message": str(base.details)})
    except Exception as exc:
        base.status = "FAIL"
        base.score = 0.0
        base.failures.append({"location": "phase 13 dependency-supported causal contract", "exception": type(exc).__name__, "message": str(exc)})
    return base


__all__ = ["strict_causal_phase"]
