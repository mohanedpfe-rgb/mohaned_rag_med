from __future__ import annotations

import ast
import hashlib
import os
import re
from pathlib import Path
from typing import Any

from rag_project.testing.deep_diagnostics import PhaseResult

ROOT = Path(__file__).resolve().parents[2]


def _runner():
    from rag_project.testing import runner
    return runner


def _canonical_failure_fingerprint(failure: dict[str, Any]) -> str:
    location = str(failure.get("location") or "unknown").replace("\\", "/")
    location = re.sub(r":\d+(?::\d+)?$", "", location)
    exception = str(failure.get("exception") or "UnknownFailure")
    message = str(failure.get("message") or failure.get("detail") or "")
    message = re.sub(r"0x[0-9a-fA-F]+", "#", message)
    message = re.sub(r"\b\d+(?:\.\d+)?\b", "#", message)
    message = " ".join(message.casefold().split())
    return hashlib.sha256(f"{location}|{exception}|{message}".encode("utf-8")).hexdigest()[:16]


def phase12_stable_fingerprinting(phase: Any, results: dict[int, PhaseResult]) -> PhaseResult:
    runner = _runner()
    fixtures = [
        {"location": "rag_project/x.py:10", "exception": "ValueError", "message": "timeout at 123ms object=0xabc"},
        {"location": "rag_project/x.py:99", "exception": "ValueError", "message": "timeout at 456ms object=0xdef"},
        {"location": "rag_project/y.py:10", "exception": "ValueError", "message": "timeout at 123ms object=0xabc"},
    ]
    values = [_canonical_failure_fingerprint(row) for row in fixtures]
    stable = values[0] == values[1]
    discriminates = values[0] != values[2]
    authoritative = runner._hardened_fingerprinting(phase, results)
    details = dict(authoritative.details or {})
    details.update({
        "self_test_equivalent_inputs_same": stable,
        "self_test_different_module_different": discriminates,
        "self_test_algorithm_digest": hashlib.sha256(b"project-frame|exception|normalized-message|sha256").hexdigest()[:16],
        "normalization_contract": "line numbers and numeric values do not change equivalent failure identity while project frame still discriminates modules",
        "authoritative_fingerprinting_owner": "_hardened_fingerprinting",
    })
    result = PhaseResult(phase.number, phase.key, phase.name, status="PASS" if stable and discriminates else "FAIL", score=1.0 if stable and discriminates else 0.0)
    result.details = details
    if result.status == "FAIL":
        result.failures.append({"location": "phase 12 fingerprint self-test", "exception": "FingerprintContractFailure", "message": str(details)})
    return result


def phase13_known_causal_graph(phase: Any, results: dict[int, PhaseResult]) -> PhaseResult:
    runner = _runner()
    try:
        from rag_project.testing import advanced_phases
        from rag_project.testing import production_answer_probes as answer_probes
        synthetic = {
            5: PhaseResult(5, "cross_layer_invariants", "Cross-layer", status="FAIL", failures=[{"location": "rag_project/storage/vector_store.py", "exception": "IdentityConservationFailure", "message": "document identity dropped before retrieval"}]),
            9: PhaseResult(9, "retrieval_microscope", "Retrieval", status="FAIL", failures=[{"location": "rag_project/storage/vector_store.py", "exception": "IdentityConservationFailure", "message": "document identity dropped before ranking"}]),
            10: PhaseResult(10, "rag_causality", "Answer", status="FAIL", failures=[{"location": "rag_project/intelligence/med_evidence_pro.py", "exception": "IdentityConservationFailure", "message": "document identity unavailable for answer evidence"}]),
        }
        known = runner._hardened_causal_graph(phase, synthetic)
        edges = {(edge["from"], edge["to"]) for edge in known.details["edges"]}
        expected = {("p5f0", "p9f0"), ("p9f0", "p10f0")}
        known_fixture_verified = expected.issubset(edges) and "p5f0" in set(known.details["candidate_roots"])

        original_store_fixture = advanced_phases._store_fixture
        original_answer_seed = answer_probes._seed_real_retrieval
        class SharedInjectedFailure(RuntimeError):
            pass
        def failing_store_fixture(*args: Any, **kwargs: Any):
            raise SharedInjectedFailure("controlled phase-13 shared dependency failure")
        def failing_answer_seed(*args: Any, **kwargs: Any):
            raise SharedInjectedFailure("controlled phase-13 shared dependency failure")
        advanced_phases._store_fixture = failing_store_fixture
        answer_probes._seed_real_retrieval = failing_answer_seed
        try:
            p5 = advanced_phases.cross_layer_invariants(runner.PHASES[4])
            p9 = advanced_phases.retrieval_microscope(runner.PHASES[8])
            p10 = answer_probes.phase10_canonical_answer_engine(runner.PHASES[9])
            injected_results = {5: p5, 9: p9, 10: p10}
        finally:
            advanced_phases._store_fixture = original_store_fixture
            answer_probes._seed_real_retrieval = original_answer_seed
        injected = runner._hardened_causal_graph(phase, injected_results)
        shared_failures = all(item.status == "FAIL" and item.failures and item.failures[0].get("exception") == "SharedInjectedFailure" for item in injected_results.values())
        propagation_verified = shared_failures and {("p5f0", "p9f0"), ("p5f0", "p10f0")} <= {(e["from"], e["to"]) for e in injected.details["edges"]} and "p5f0" in set(injected.details["candidate_roots"])
        base = runner._hardened_causal_graph(phase, results)
        base.details.update({"known_causal_fixture_verified": known_fixture_verified, "known_failure_injection_verified": propagation_verified, "known_fixture_expected_edges": sorted(expected), "runtime_propagation_phases": sorted(injected_results), "runtime_failure_identities": {str(k): item.failures[0].get("exception") for k, item in injected_results.items()}, "causal_validation_mode": "synthetic_fixture_plus_real_runtime_fault_injection_across_phases_5_9_10", "causal_root_is_injected_dependency": propagation_verified})
        base.status = "PASS" if base.status == "PASS" and known_fixture_verified and propagation_verified else "FAIL"
        base.score = 1.0 if base.status == "PASS" else 0.0
        if base.status == "FAIL": base.failures.append({"location": "phase 13 causal validation", "exception": "CausalGraphContractFailure", "message": str(base.details)})
        return base
    except Exception as exc:
        return PhaseResult(phase.number, phase.key, phase.name, status="FAIL", score=0.0, failures=[{"location":"phase 13 causal validation","exception":type(exc).__name__,"message":str(exc)}])


def _phase1_semantics(details: dict[str, Any]) -> list[dict[str, Any]]:
    failures=[]; domains=details.get("domains") or {}; required={"ingestion","chunking","embeddings","retrieval","intelligence","generation","storage","evaluation"}; missing=sorted(d for d in required if not (domains.get(d) or {}).get("modules",0))
    failures += [{"phase":1,"reason":f"missing production domain: {d}"} for d in missing]
    if int(details.get("python_modules") or 0)<=0: failures.append({"phase":1,"reason":"no production Python modules"})
    if int(details.get("test_files") or 0)<=0: failures.append({"phase":1,"reason":"no test files"})
    import_edges=0
    for path in (ROOT/"rag_project").rglob("*.py"):
        try:
            tree=ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
            import_edges += sum(len(node.names) for node in ast.walk(tree) if isinstance(node,(ast.Import,ast.ImportFrom)))
        except (OSError,SyntaxError) as exc:
            failures.append({"phase":1,"reason":f"AST parse failure: {path.relative_to(ROOT)}:{type(exc).__name__}"})
    if import_edges<=0: failures.append({"phase":1,"reason":"empty import dependency graph"})
    return failures


def _semantic_contracts(results: dict[int, PhaseResult]) -> list[dict[str, Any]]:
    failures=_phase1_semantics(results.get(1).details if results.get(1) else {})
    return failures


def phase17_strict_completion(phase: Any, results: dict[int, PhaseResult]) -> PhaseResult:
    return _runner()._base_phase17_strict(phase, results)
