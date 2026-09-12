from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
from typing import Any

from rag_project.testing.deep_diagnostics import PhaseResult

ROOT = Path(__file__).resolve().parents[2]


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
        known_failure_injection_verified = False
        injected_edges: set[tuple[str, str]] = set()
        try:
            from rag_project.testing import advanced_phases
            original_store_fixture = advanced_phases._store_fixture
            class SharedInjectedFailure(RuntimeError):
                pass
            def failing_store_fixture(*args: Any, **kwargs: Any):
                raise SharedInjectedFailure("controlled phase-13 shared dependency failure")
            advanced_phases._store_fixture = failing_store_fixture
            try:
                injected_results = {
                    5: advanced_phases.cross_layer_invariants(runner.PHASES[4]),
                    9: advanced_phases.retrieval_microscope(runner.PHASES[8]),
                    10: advanced_phases.rag_causality(runner.PHASES[9]),
                }
            finally:
                advanced_phases._store_fixture = original_store_fixture
            injected = runner._hardened_causal_graph(phase, injected_results)
            injected_edges = {(edge["from"], edge["to"]) for edge in injected.details["edges"]}
            injected_roots = set(injected.details["candidate_roots"])
            failures_are_shared = all(
                item.status == "FAIL"
                and (item.failures and item.failures[0].get("exception") == "SharedInjectedFailure")
                for item in injected_results.values()
            )
            known_failure_injection_verified = failures_are_shared and {("p5f0", "p9f0"), ("p5f0", "p10f0")} <= injected_edges and "p5f0" in injected_roots
        except Exception:
            known_failure_injection_verified = False
        base.details.update({
            "known_causal_fixture_verified": chain_present and root_present,
            "known_fixture_expected_edges": sorted(expected_chain),
            "known_fixture_observed_edges": sorted(edges),
            "known_failure_injection_verified": known_failure_injection_verified,
            "known_failure_injection_observed_edges": sorted(injected_edges),
            "causal_validation_mode": "synthetic_fixture_plus_real_production_phase_fault_injection",
        })
        base.status = "PASS" if base.status == "PASS" and chain_present and root_present and known_failure_injection_verified else "FAIL"; base.score = 1.0 if base.status == "PASS" else 0.0
        if base.status == "FAIL": base.failures.append({"location": "phase 13 causal self-test", "exception": "CausalGraphContractFailure", "message": str(base.details)})
    except Exception as exc:
        base.status = "FAIL"; base.score = 0.0; base.failures.append({"location": "phase 13 causal self-test", "exception": type(exc).__name__, "message": str(exc)})
    return base


def _architecture_semantic_contract(details: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    required_domains = {"ingestion", "chunking", "embeddings", "retrieval", "intelligence", "generation", "storage", "evaluation"}
    domains = details.get("domains") or {}
    missing_domains = sorted(domain for domain in required_domains if not (domains.get(domain) or {}).get("modules", 0))
    if missing_domains:
        failures.append(f"missing production domains: {missing_domains}")
    if int(details.get("python_modules") or 0) <= 0:
        failures.append("architecture reports no production Python modules")
    if int(details.get("test_files") or 0) <= 0:
        failures.append("architecture reports no test files")
    parse_errors = []
    import_edges = 0
    for path in (ROOT / "rag_project").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_edges += len(node.names)
        except (OSError, SyntaxError) as exc:
            parse_errors.append(f"{path.relative_to(ROOT)}:{type(exc).__name__}")
    if parse_errors:
        failures.append(f"production AST parse failures: {parse_errors[:5]}")
    if import_edges <= 0:
        failures.append("architecture dependency graph is empty")
    return failures


def _semantic_phase_contracts(results: dict[int, PhaseResult]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    p1 = results.get(1)
    if p1:
        for message in _architecture_semantic_contract(p1.details or {}):
            failures.append({"phase": 1, "reason": message})

    p7 = results.get(7)
    if p7:
        d = p7.details or {}
        if not d.get("real_pdf_objects"):
            failures.append({"phase": 7, "reason": "real PDF objects were not exercised"})
        if not d.get("malformed_pdf_rejected"):
            failures.append({"phase": 7, "reason": "malformed PDF rejection was not demonstrated"})
        if len(d.get("variant_results") or []) < 5:
            failures.append({"phase": 7, "reason": "adversarial document matrix is too small"})

    p8 = results.get(8)
    if p8:
        checks = p8.details.get("checks") or {}
        if len(checks) < 4 or not all(bool(value) for value in checks.values()):
            failures.append({"phase": 8, "reason": "metamorphic invariants are incomplete or failing"})
        if not p8.details.get("production_functions"):
            failures.append({"phase": 8, "reason": "metamorphic phase did not report executed production functions"})

    p9 = results.get(9)
    if p9 and not p9.details.get("gold_labels_independent_of_corpus_text", False):
        failures.append({"phase": 9, "reason": "retrieval labels are not independently sourced"})

    p10 = results.get(10)
    if p10:
        d = p10.details or {}
        if d.get("retrieval_stub_used") is not False:
            failures.append({"phase": 10, "reason": "answer phase did not prove non-stub retrieval"})
        orchestration = set(d.get("production_orchestration") or [])
        required_orchestration = {"MultiTierRetriever", "HybridRetriever", "VectorStore", "EvidenceCompiler", "AnswerCascade", "ActiveVerifier"}
        if not required_orchestration.issubset(orchestration):
            failures.append({"phase": 10, "reason": f"production answer orchestration incomplete: {sorted(required_orchestration - orchestration)}"})

    p11 = results.get(11)
    if p11:
        d = p11.details or {}
        if int(d.get("mutants_applicable") or 0) < 8 or float(d.get("kill_score") or 0) < 1.0:
            failures.append({"phase": 11, "reason": "mutation suite does not have at least eight applicable fully killed mutants"})
        targets = {str(row.get("target")) for row in d.get("mutation_results") or [] if row.get("target")}
        if len(targets) < 2:
            failures.append({"phase": 11, "reason": "mutation suite does not span multiple production modules"})

    p13 = results.get(13)
    if p13 and not p13.details.get("known_causal_fixture_verified"):
        failures.append({"phase": 13, "reason": "causal graph has no verified known-causal fixture"})
    if p13 and not p13.details.get("known_failure_injection_verified", False):
        failures.append({"phase": 13, "reason": "causal graph has no real failure-injection propagation experiment"})

    p14 = results.get(14)
    if p14:
        d = p14.details or {}
        metrics = d.get("stage_metrics") or {}
        if not metrics or any(int(row.get("samples") or 0) < 5 for row in metrics.values()):
            failures.append({"phase": 14, "reason": "performance intelligence lacks >=5 samples for every measured stage"})
        baseline_path = ROOT / "tests" / "support" / "performance_baseline.json"
        if not baseline_path.exists():
            failures.append({"phase": 14, "reason": "performance regression baseline file is missing"})

    p15 = results.get(15)
    if p15:
        d = p15.details or {}
        if int(d.get("sample_count") or 0) < 3 or int(d.get("repetitions") or 0) < 3:
            failures.append({"phase": 15, "reason": "resource observation has insufficient independent samples"})
        if not d.get("pipeline_exercised"):
            failures.append({"phase": 15, "reason": "resource phase does not identify exercised production pipeline"})

    p16 = results.get(16)
    if p16:
        d = p16.details or {}
        for key in ("durable_state_verified", "index_integrity_verified", "gold_labels_independent_of_corpus_text"):
            if d.get(key) is not True:
                failures.append({"phase": 16, "reason": f"missing production ingestion evidence: {key}"})
        if float(d.get("retrieval_recall") or 0) < 0.80:
            failures.append({"phase": 16, "reason": "production ingestion retrieval recall is below 0.80"})
    return failures


def phase17_strict_completion(phase: Any, results: dict[int, PhaseResult]) -> PhaseResult:
    runner = _runner(); base = runner._base_phase17_strict(phase, results)
    required: dict[int, tuple[str, ...]] = {
        1: ("domains",), 2: ("collected_tests",), 3: ("chain",), 4: ("input_contract", "transformation_contract", "output_contract"),
        5: ("violations",), 6: ("field_surfaces", "measurements"), 7: ("variant_results",), 8: ("checks",),
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

    evidence_failures.extend(_semantic_phase_contracts(results))

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
    base.details["semantic_evidence_failures"] = evidence_failures
    base.details["implementation_coverage"] = "17/17" if not evidence_failures and base.status == "PASS" else f"{17 - len(unique)}/17"
    base.details["fully_implemented_phase_numbers"] = [] if evidence_failures or base.status != "PASS" else list(range(1, 18))
    base.details["live_certification_requirements"] = {"real_ocr": require_live_ocr, "live_ollama": require_live_llm, "24h_resources": require_24h}
    if evidence_failures:
        base.details.setdefault("evidence_failures", []).extend(evidence_failures); base.details["runtime_non_pass_phases"] = sorted(set(base.details.get("runtime_non_pass_phases", [])) | set(unique)); base.status = "FAIL"; base.score = 0.0
        base.failures.append({"location": "phase 17 semantic evidence contract", "exception": "Incomplete17PhaseImplementation", "message": str(evidence_failures)})
    return base
