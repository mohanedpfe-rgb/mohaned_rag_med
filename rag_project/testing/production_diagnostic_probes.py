from __future__ import annotations

import hashlib
import os
from typing import Any

from rag_project.testing.deep_diagnostics import PhaseResult


def _runner():
    from rag_project.testing import runner
    return runner


def phase12_stable_fingerprinting(phase: Any, results: dict[int, PhaseResult]) -> PhaseResult:
    runner = _runner(); base = runner._hardened_fingerprinting(phase, results)
    try:
        a = {1: PhaseResult(1, "a", "a", status="FAIL", failures=[{"location": "rag_project/x.py:10", "exception": "ValueError", "message": "timeout at 123ms object=0xabc"}])}
        b = {1: PhaseResult(1, "a", "a", status="FAIL", failures=[{"location": "rag_project/x.py:99", "exception": "ValueError", "message": "timeout at 456ms object=0xdef"}])}
        c = {1: PhaseResult(1, "a", "a", status="FAIL", failures=[{"location": "rag_project/y.py:10", "exception": "ValueError", "message": "timeout at 123ms object=0xabc"}])}
        fa = runner._hardened_fingerprinting(phase, a).details["fingerprints"][0]["fingerprint"]; fb = runner._hardened_fingerprinting(phase, b).details["fingerprints"][0]["fingerprint"]; fc = runner._hardened_fingerprinting(phase, c).details["fingerprints"][0]["fingerprint"]
        stable = fa == fb; discriminates = fa != fc
        base.details.update({"self_test_equivalent_inputs_same": stable, "self_test_different_module_different": discriminates, "self_test_algorithm_digest": hashlib.sha256(str(base.details["algorithm"]).encode()).hexdigest()[:16], "normalization_contract": "line numbers, numeric timing values, and pointer addresses do not change equivalent failure identity"})
        base.status = "PASS" if base.status == "PASS" and stable and discriminates else "FAIL"; base.score = 1.0 if base.status == "PASS" else 0.0
        if base.status == "FAIL": base.failures.append({"location": "phase 12 fingerprint self-test", "exception": "FingerprintContractFailure", "message": str(base.details)})
    except Exception as exc:
        base.status = "FAIL"; base.score = 0.0; base.failures.append({"location": "phase 12 fingerprint self-test", "exception": type(exc).__name__, "message": str(exc)})
    return base


def phase13_known_causal_graph(phase: Any, results: dict[int, PhaseResult]) -> PhaseResult:
    runner = _runner(); base = runner._hardened_causal_graph(phase, results)
    synthetic = {
        5: PhaseResult(5, "cross_layer_invariants", "Cross-layer", status="FAIL", failures=[{"location": "rag_project/storage/vector_store.py", "exception": "IdentityConservationFailure", "message": "document identity dropped before retrieval"}]),
        9: PhaseResult(9, "retrieval_microscope", "Retrieval", status="FAIL", failures=[{"location": "rag_project/storage/vector_store.py", "exception": "IdentityConservationFailure", "message": "document identity dropped before ranking"}]),
        10: PhaseResult(10, "rag_causality", "Answer", status="FAIL", failures=[{"location": "rag_project/intelligence/med_evidence_pro.py", "exception": "IdentityConservationFailure", "message": "document identity unavailable for answer evidence"}]),
    }
    try:
        known = runner._hardened_causal_graph(phase, synthetic); edges = {(edge["from"], edge["to"]) for edge in known.details["edges"]}; expected_chain = {("p5f0", "p9f0"), ("p9f0", "p10f0")}; chain_present = expected_chain.issubset(edges); root_present = "p5f0" in set(known.details["candidate_roots"])
        base.details.update({"known_causal_fixture_verified": chain_present and root_present, "known_fixture_expected_edges": sorted(expected_chain), "known_fixture_observed_edges": sorted(edges)})
        base.status = "PASS" if base.status == "PASS" and chain_present and root_present else "FAIL"; base.score = 1.0 if base.status == "PASS" else 0.0
        if base.status == "FAIL": base.failures.append({"location": "phase 13 causal self-test", "exception": "CausalGraphContractFailure", "message": str(base.details)})
    except Exception as exc:
        base.status = "FAIL"; base.score = 0.0; base.failures.append({"location": "phase 13 causal self-test", "exception": type(exc).__name__, "message": str(exc)})
    return base


def phase17_strict_completion(phase: Any, results: dict[int, PhaseResult]) -> PhaseResult:
    runner = _runner(); base = runner._base_phase17_strict(phase, results)
    required: dict[int, tuple[str, ...]] = {
        1: ("domains",), 2: ("collected_tests",), 3: ("chain",), 4: ("input_contract", "transformation_contract", "output_contract"),
        5: ("violations",), 6: ("field_survival", "token_survival"), 7: ("variant_results",), 8: ("checks",),
        9: ("lexical_recall_at_3", "semantic_recall_at_3"), 10: ("canonical_engine_executed", "answer_generated", "verification_allow"),
        11: ("mutants_applicable", "kill_score"), 12: ("unique_fingerprints", "self_test_equivalent_inputs_same", "self_test_different_module_different"),
        13: ("nodes", "edges", "known_causal_fixture_verified"), 14: ("stage_metrics", "repetitions"),
        15: ("sample_count", "repetitions", "pipeline_exercised"), 16: ("durable_state_verified", "index_integrity_verified", "retrieval_recall"),
    }
    evidence_failures: list[dict[str, Any]] = []
    for number, keys in required.items():
        item = results.get(number)
        if item is None:
            evidence_failures.append({"phase": number, "required": "phase result"}); continue
        if item.status != "PASS": evidence_failures.append({"phase": number, "required": "PASS status", "actual": item.status})
        if item.failures: evidence_failures.append({"phase": number, "required": "zero recorded failures", "actual_failures": len(item.failures)})
        details = item.details or {}
        for key in keys:
            if key not in details or details[key] in (None, "", [], {}): evidence_failures.append({"phase": number, "required": key})
    require_live_ocr = os.getenv("REQUIRE_REAL_OCR", "0").strip().lower() in {"1", "true", "yes", "on"}
    if require_live_ocr and not (results.get(7) and results[7].details.get("real_ocr_verified") is True): evidence_failures.append({"phase": 7, "required": "real_ocr_verified=true"})
    require_live_llm = os.getenv("REQUIRE_LIVE_OLLAMA", "0").strip().lower() in {"1", "true", "yes", "on"}
    if require_live_llm and not (results.get(10) and results[10].details.get("live_ollama_verified") is True): evidence_failures.append({"phase": 10, "required": "live_ollama_verified=true"})
    require_24h = os.getenv("REQUIRE_24H_CERTIFICATION", "0").strip().lower() in {"1", "true", "yes", "on"}
    if require_24h:
        p15 = results.get(15).details if results.get(15) else {}
        if p15.get("certification_mode") != "24h_observation" or float(p15.get("observed_seconds") or 0) < 86400: evidence_failures.append({"phase": 15, "required": "24h_observation with >=86400 observed_seconds"})
    unique = sorted({entry["phase"] for entry in evidence_failures})
    base.details["phase_by_phase_evidence_contract"] = {"required_phases": list(required), "failures": evidence_failures}
    base.details["implementation_coverage"] = "17/17" if not evidence_failures and base.status == "PASS" else f"{17 - len(unique)}/17"
    base.details["fully_implemented_phase_numbers"] = [] if evidence_failures or base.status != "PASS" else list(range(1, 18))
    base.details["live_certification_requirements"] = {"real_ocr": require_live_ocr, "live_ollama": require_live_llm, "24h_resources": require_24h}
    if evidence_failures:
        base.details.setdefault("evidence_failures", []).extend(evidence_failures); base.details["runtime_non_pass_phases"] = sorted(set(base.details.get("runtime_non_pass_phases", [])) | set(unique)); base.status = "FAIL"; base.score = 0.0
        base.failures.append({"location": "phase 17 phase-by-phase evidence contract", "exception": "Incomplete17PhaseImplementation", "message": str(evidence_failures)})
    return base
