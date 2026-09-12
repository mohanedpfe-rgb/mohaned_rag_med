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
        fixtures = [
            {"location": "rag_project/x.py:10", "exception": "ValueError", "message": "timeout at 123ms object=0xabc"},
            {"location": "rag_project/x.py:99", "exception": "ValueError", "message": "timeout at 456ms object=0xdef"},
            {"location": "rag_project/y.py:10", "exception": "ValueError", "message": "timeout at 123ms object=0xabc"},
        ]
        fp = []
        for failure in fixtures:
            sample = {1: PhaseResult(1, "a", "a", status="FAIL", failures=[failure])}
            fp.append(runner._hardened_fingerprinting(phase, sample).details["fingerprints"][0]["fingerprint"])
        stable = fp[0] == fp[1]; discriminates = fp[0] != fp[2]
        base.details.update({
            "self_test_equivalent_inputs_same": stable,
            "self_test_different_module_different": discriminates,
            "self_test_algorithm_digest": hashlib.sha256(str(base.details["algorithm"]).encode()).hexdigest()[:16],
            "normalization_contract": "line numbers and numeric values do not change equivalent failure identity while project frame still discriminates modules",
        })
        base.status = "PASS" if base.status == "PASS" and stable and discriminates else "FAIL"; base.score = 1.0 if base.status == "PASS" else 0.0
        if base.status == "FAIL": base.failures.append({"location": "phase 12 fingerprint self-test", "exception": "FingerprintContractFailure", "message": str(base.details)})
    except Exception as exc:
        base.status = "FAIL"; base.score = 0.0; base.failures.append({"location": "phase 12 fingerprint self-test", "exception": type(exc).__name__, "message": str(exc)})
    return base


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
        injected_edges = {(edge["from"], edge["to"]) for edge in injected.details["edges"]}
        shared_failures = all(
            item.status == "FAIL" and item.failures and item.failures[0].get("exception") == "SharedInjectedFailure"
            for item in injected_results.values()
        )
        runtime_locations = {
            5: any("cross_layer_invariants" in str(f.get("location")) for f in p5.failures),
            9: any("production representation chain" in str(f.get("location")) or "production" in str(f.get("location")) for f in p9.failures),
            10: any("phase 10 canonical answer engine" in str(f.get("location")) for f in p10.failures),
        }
        propagation_verified = (
            shared_failures
            and all(runtime_locations.values())
            and {("p5f0", "p9f0"), ("p5f0", "p10f0")} <= injected_edges
            and "p5f0" in set(injected.details["candidate_roots"])
        )

        base = runner._hardened_causal_graph(phase, results)
        base.details.update({
            "known_causal_fixture_verified": known_fixture_verified,
            "known_failure_injection_verified": propagation_verified,
            "known_fixture_expected_edges": sorted(expected),
            "known_fixture_observed_edges": sorted(edges),
            "known_failure_injection_observed_edges": sorted(injected_edges),
            "runtime_propagation_phases": sorted(injected_results),
            "runtime_failure_identities": {str(k): item.failures[0].get("exception") for k, item in injected_results.items()},
            "runtime_failure_locations_verified": runtime_locations,
            "causal_validation_mode": "synthetic_fixture_plus_real_runtime_fault_injection_across_phases_5_9_10",
            "causal_root_is_injected_dependency": propagation_verified,
        })
        base.status = "PASS" if base.status == "PASS" and known_fixture_verified and propagation_verified else "FAIL"; base.score = 1.0 if base.status == "PASS" else 0.0
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
    p7=results.get(7); d=p7.details if p7 else {}
    if p7 and (not d.get("real_pdf_objects") or not d.get("malformed_pdf_rejected") or len(d.get("variant_results") or [])<5): failures.append({"phase":7,"reason":"PDF adversarial evidence incomplete"})
    p8=results.get(8); d=p8.details if p8 else {}; checks=d.get("checks") or {}
    if p8 and (len(checks)<6 or not all(bool(v) for v in checks.values()) or not d.get("production_functions") or not d.get("end_to_end_answer_path_executed")): failures.append({"phase":8,"reason":"end-to-end metamorphic evidence incomplete"})
    p9=results.get(9)
    if p9 and (not p9.details.get("gold_labels_independent_of_corpus_text", p9.details.get("gold_independent_of_corpus_text", False)) or p9.details.get("lexical_recall_at_3",0)<.8 or p9.details.get("semantic_recall_at_3",0)<.8): failures.append({"phase":9,"reason":"independent retrieval evidence below threshold"})
    p10=results.get(10); d=p10.details if p10 else {}; required_orchestration={"MultiTierRetriever","HybridRetriever","VectorStore","EvidenceCompiler","AnswerCascade","ActiveVerifier"}
    if p10 and (d.get("retrieval_stub_used") is not False or not required_orchestration.issubset(set(d.get("production_orchestration") or [])) or not d.get("ollama_protocol_roundtrip_verified")): failures.append({"phase":10,"reason":"canonical answer + Ollama client protocol evidence incomplete"})
    p11=results.get(11); d=p11.details if p11 else {}; targets=set(d.get("mutation_targets") or [])
    if p11 and (int(d.get("mutants_applicable") or 0)<12 or float(d.get("kill_score") or 0)<1.0 or len(targets)<3): failures.append({"phase":11,"reason":"multi-module mutation coverage incomplete"})
    p13=results.get(13); d=p13.details if p13 else {}
    if p13 and (not d.get("known_causal_fixture_verified") or not d.get("known_failure_injection_verified") or not d.get("causal_root_is_injected_dependency")): failures.append({"phase":13,"reason":"causal runtime propagation validation incomplete"})
    p14=results.get(14); d=p14.details if p14 else {}; metrics=d.get("stage_metrics") or {}; comparisons=d.get("regression_comparisons") or {}
    if p14 and (not metrics or any(int(v.get("samples") or 0)<5 for v in metrics.values()) or not comparisons or not d.get("regression_pass") or not (ROOT/"tests/support/performance_baseline.json").exists()): failures.append({"phase":14,"reason":"performance baseline/regression evidence incomplete"})
    p15=results.get(15); d=p15.details if p15 else {}
    if p15 and (int(d.get("sample_count") or 0)<3 or int(d.get("repetitions") or 0)<3 or not d.get("pipeline_exercised") or not d.get("certification_mode")): failures.append({"phase":15,"reason":"resource evidence incomplete"})
    p16=results.get(16); d=p16.details if p16 else {}
    if p16 and (not d.get("durable_state_verified") or not d.get("index_integrity_verified") or not d.get("gold_labels_independent_of_corpus_text") or float(d.get("retrieval_recall") or 0)<.8): failures.append({"phase":16,"reason":"production ingestion evidence incomplete"})
    return failures


def phase17_strict_completion(phase: Any, results: dict[int, PhaseResult]) -> PhaseResult:
    try:
        from rag_project.testing.full_metamorphic_probes import run_full_metamorphic_suite
        results[8] = run_full_metamorphic_suite(phase)
    except Exception as exc:
        results[8] = PhaseResult(8,"metamorphic","Metamorphic stability",status="FAIL",failures=[{"location":"phase 8 authoritative recheck","exception":type(exc).__name__,"message":str(exc)}])
    try:
        from rag_project.testing.full_mutation_probes import run_full_mutation_suite
        results[11] = run_full_mutation_suite(phase)
    except Exception as exc:
        results[11] = PhaseResult(11,"mutation","Mutation detection",status="FAIL",failures=[{"location":"phase 11 authoritative recheck","exception":type(exc).__name__,"message":str(exc)}])

    runner=_runner(); base=runner._base_phase17_strict(phase,results)
    required_keys={1:("domains",),2:("collected_tests",),3:("chain",),4:("input_contract","transformation_contract","output_contract"),5:("violations",),6:("field_surfaces","measurements"),7:("variant_results",),8:("checks","end_to_end_answer_path_executed"),9:("lexical_recall_at_3","semantic_recall_at_3"),10:("canonical_engine_executed","answer_generated","verification_allow","ollama_protocol_roundtrip_verified"),11:("mutants_applicable","kill_score","mutation_targets"),12:("unique_fingerprints","self_test_equivalent_inputs_same","self_test_different_module_different"),13:("nodes","edges","known_causal_fixture_verified","causal_root_is_injected_dependency"),14:("stage_metrics","repetitions","regression_comparisons","regression_pass"),15:("sample_count","repetitions","pipeline_exercised","certification_mode"),16:("durable_state_verified","index_integrity_verified","retrieval_recall")}
    evidence_failures=[]
    for number,keys in required_keys.items():
        item=results.get(number)
        if item is None: evidence_failures.append({"phase":number,"required":"phase result"}); continue
        if item.status!="PASS": evidence_failures.append({"phase":number,"required":"PASS status","actual":item.status})
        if item.failures: evidence_failures.append({"phase":number,"required":"zero recorded failures","actual_failures":len(item.failures)})
        for key in keys:
            if key not in (item.details or {}) or item.details[key] in (None,"",[],{}): evidence_failures.append({"phase":number,"required":key})
    evidence_failures.extend(_semantic_contracts(results))
    if os.getenv("REQUIRE_REAL_OCR","0").lower() in {"1","true","yes","on"} and not (results.get(7) and results[7].details.get("real_ocr_verified")): evidence_failures.append({"phase":7,"required":"real_ocr_verified=true"})
    if os.getenv("REQUIRE_LIVE_OLLAMA","0").lower() in {"1","true","yes","on"} and not (results.get(10) and results[10].details.get("live_ollama_verified")): evidence_failures.append({"phase":10,"required":"live_ollama_verified=true"})
    if os.getenv("REQUIRE_24H_CERTIFICATION","0").lower() in {"1","true","yes","on"}:
        p15=results.get(15).details if results.get(15) else {}
        if p15.get("certification_mode")!="24h_observation" or float(p15.get("observed_seconds") or 0)<86400: evidence_failures.append({"phase":15,"required":"24h_observation >=86400s"})
    unique=sorted({int(x["phase"]) for x in evidence_failures})
    base.details["phase_by_phase_evidence_contract"]={"required_phases":list(required_keys),"failures":evidence_failures}
    base.details["semantic_evidence_failures"]=evidence_failures
    base.details["implementation_coverage"]="17/17" if not evidence_failures and base.status=="PASS" else f"{17-len(unique)}/17"
    base.details["fully_implemented_phase_numbers"]=[] if evidence_failures or base.status!="PASS" else list(range(1,18))
    if evidence_failures:
        base.status="FAIL"; base.score=0.0; base.details["runtime_non_pass_phases"]=sorted(set(base.details.get("runtime_non_pass_phases",[]))|set(unique)); base.failures.append({"location":"phase 17 semantic evidence contract","exception":"Incomplete17PhaseImplementation","message":str(evidence_failures)})
    return base
