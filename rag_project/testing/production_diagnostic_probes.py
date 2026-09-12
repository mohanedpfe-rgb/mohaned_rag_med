from __future__ import annotations

import hashlib
import re
from typing import Any

from rag_project.testing import runner as base_runner
from rag_project.testing.deep_diagnostics import PhaseResult


def phase12_stable_fingerprinting(phase: Any, results: dict[int, PhaseResult]) -> PhaseResult:
    """Validate fingerprint determinism, equivalence normalization, and discrimination."""
    base = base_runner._hardened_fingerprinting(phase, results)
    try:
        a = {1: PhaseResult(1, "a", "a", status="FAIL", failures=[{"location": "rag_project/x.py:10", "exception": "ValueError", "message": "timeout at 123ms object=0xabc"}])}
        b = {1: PhaseResult(1, "a", "a", status="FAIL", failures=[{"location": "rag_project/x.py:99", "exception": "ValueError", "message": "timeout at 456ms object=0xdef"}])}
        c = {1: PhaseResult(1, "a", "a", status="FAIL", failures=[{"location": "rag_project/y.py:10", "exception": "ValueError", "message": "timeout at 123ms object=0xabc"}])}
        fa = base_runner._hardened_fingerprinting(phase, a).details["fingerprints"][0]["fingerprint"]
        fb = base_runner._hardened_fingerprinting(phase, b).details["fingerprints"][0]["fingerprint"]
        fc = base_runner._hardened_fingerprinting(phase, c).details["fingerprints"][0]["fingerprint"]
        stable = fa == fb
        discriminates = fa != fc
        base.details.update({"self_test_equivalent_inputs_same": stable, "self_test_different_module_different": discriminates, "self_test_algorithm_digest": hashlib.sha256(str(base.details["algorithm"]).encode()).hexdigest()[:16], "normalization_contract": "line numbers, numeric timing values, and pointer addresses do not change equivalent failure identity"})
        base.status = "PASS" if base.status == "PASS" and stable and discriminates else "FAIL"
        base.score = 1.0 if base.status == "PASS" else 0.0
        if base.status == "FAIL":
            base.failures.append({"location": "phase 12 fingerprint self-test", "exception": "FingerprintContractFailure", "message": str(base.details)})
    except Exception as exc:
        base.status = "FAIL"
        base.score = 0.0
        base.failures.append({"location": "phase 12 fingerprint self-test", "exception": type(exc).__name__, "message": str(exc)})
    return base


def phase13_known_causal_graph(phase: Any, results: dict[int, PhaseResult]) -> PhaseResult:
    """Validate that declared dependencies produce the expected causal direction."""
    base = base_runner._hardened_causal_graph(phase, results)
    synthetic = {
        5: PhaseResult(5, "cross_layer_invariants", "Cross-layer", status="FAIL", failures=[{"location": "rag_project/storage/vector_store.py", "exception": "IdentityConservationFailure", "message": "document identity dropped before retrieval"}]),
        9: PhaseResult(9, "retrieval_microscope", "Retrieval", status="FAIL", failures=[{"location": "rag_project/storage/vector_store.py", "exception": "IdentityConservationFailure", "message": "document identity dropped before ranking"}]),
        10: PhaseResult(10, "rag_causality", "Answer", status="FAIL", failures=[{"location": "rag_project/intelligence/med_evidence_pro.py", "exception": "IdentityConservationFailure", "message": "document identity unavailable for answer evidence"}]),
    }
    try:
        known = base_runner._hardened_causal_graph(phase, synthetic)
        edges = {(edge["from"], edge["to"]) for edge in known.details["edges"]}
        expected_chain = {("p5f0", "p9f0"), ("p9f0", "p10f0")}
        chain_present = expected_chain.issubset(edges)
        root_present = "p5f0" in set(known.details["candidate_roots"])
        base.details.update({"known_causal_fixture_verified": chain_present and root_present, "known_fixture_expected_edges": sorted(expected_chain), "known_fixture_observed_edges": sorted(edges)})
        base.status = "PASS" if base.status == "PASS" and chain_present and root_present else "FAIL"
        base.score = 1.0 if base.status == "PASS" else 0.0
        if base.status == "FAIL":
            base.failures.append({"location": "phase 13 causal self-test", "exception": "CausalGraphContractFailure", "message": str(base.details)})
    except Exception as exc:
        base.status = "FAIL"
        base.score = 0.0
        base.failures.append({"location": "phase 13 causal self-test", "exception": type(exc).__name__, "message": str(exc)})
    return base


def phase17_strict_completion(phase: Any, results: dict[int, PhaseResult]) -> PhaseResult:
    """Add a phase-by-phase evidence contract on top of the existing strict gate."""
    base = base_runner._phase17_strict(phase, results)
    required: dict[int, tuple[str, ...]] = {
        1: ("domains",),
        2: ("collected_tests",),
        3: ("chain",),
        4: ("input_contract", "transformation_contract", "output_contract"),
        5: ("violations",),
        6: ("field_survival", "token_survival"),
        7: ("variant_results",),
        8: ("checks",),
        9: ("lexical_recall_at_3", "semantic_recall_at_3"),
        10: ("canonical_engine_executed", "answer_generated", "verification_allow"),
        11: ("mutants_applicable", "kill_score"),
        12: ("unique_fingerprints", "self_test_equivalent_inputs_same", "self_test_different_module_different"),
        13: ("nodes", "edges", "known_causal_fixture_verified"),
        14: ("stage_metrics", "repetitions"),
        15: ("sample_count", "repetitions", "pipeline_exercised"),
        16: ("durable_state_verified", "index_integrity_verified", "retrieval_recall"),
    }
    evidence_failures: list[dict[str, Any]] = []
    for number, keys in required.items():
        item = results.get(number)
        if item is None:
            evidence_failures.append({"phase": number, "required": "phase result"})
            continue
        if item.status != "PASS":
            evidence_failures.append({"phase": number, "required": "PASS status", "actual": item.status})
        if item.failures:
            evidence_failures.append({"phase": number, "required": "zero recorded failures", "actual_failures": len(item.failures)})
        details = item.details or {}
        for key in keys:
            if key not in details or details[key] in (None, "", [], {}):
                evidence_failures.append({"phase": number, "required": key})
    unique = sorted({entry["phase"] for entry in evidence_failures})
    base.details.setdefault("phase_by_phase_evidence_contract", {})
    base.details["phase_by_phase_evidence_contract"] = {"required_phases": list(required), "failures": evidence_failures}
    base.details["implementation_coverage"] = "17/17" if not evidence_failures and base.status == "PASS" else f"{17 - len(unique)}/17"
    base.details["fully_implemented_phase_numbers"] = [] if evidence_failures or base.status != "PASS" else list(range(1, 18))
    if evidence_failures:
        base.details.setdefault("evidence_failures", []).extend(evidence_failures)
        base.details["runtime_non_pass_phases"] = sorted(set(base.details.get("runtime_non_pass_phases", [])) | set(unique))
        base.status = "FAIL"
        base.score = 0.0
        base.failures.append({"location": "phase 17 phase-by-phase evidence contract", "exception": "Incomplete17PhaseImplementation", "message": str(evidence_failures)})
    return base
