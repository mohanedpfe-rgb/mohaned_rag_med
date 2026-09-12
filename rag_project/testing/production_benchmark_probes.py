from __future__ import annotations

import hashlib
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import fitz

from rag_project.ingestion.robust_ingestor import robust_ingest_file
from rag_project.testing.advanced_phases import _result
from rag_project.testing.production_path_probes import _ProductionIngestionProbeSystem
from rag_project.testing.deep_diagnostics import PhaseResult

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "tests" / "support" / "performance_baseline.json"


def _write_pdf(path: Path, repetition: int) -> None:
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.insert_textbox(
        fitz.Rect(45, 45, 550, 790),
        f"Performance benchmark {repetition}\nDiabetes mellitus is a chronic metabolic disease. HbA1c is used for diagnosis and monitoring. Fasting glucose is also clinically relevant.",
        fontsize=11,
    )
    document.save(path)
    document.close()


def _stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "samples": len(values),
        "p50_ms": round(statistics.median(values), 3),
        "p95_ms": round(ordered[max(0, int(len(values) * 0.95) - 1)], 3),
        "max_ms": round(max(values), 3),
    }


def _load_baseline() -> dict[str, Any]:
    if not BASELINE.exists():
        raise FileNotFoundError(BASELINE)
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError("performance baseline is empty")
    return payload


def _environment_provenance() -> dict[str, Any]:
    lockfile = ROOT / "requirements.lock"
    try:
        lock_digest = hashlib.sha256(lockfile.read_bytes()).hexdigest() if lockfile.exists() else None
    except OSError:
        lock_digest = None
    try:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, timeout=10, check=True).stdout.strip()
    except Exception:
        git_head = None
    return {
        "git_head_sha": git_head,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device_mode": __import__("os").getenv("DEVICE_MODE"),
        "embedding_mode": __import__("os").getenv("EMBEDDING_TEST_MODE"),
        "requirements_lock_sha256": lock_digest,
    }


def _regression_metrics(metrics: dict[str, dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    failures: list[str] = []
    for stage, current in metrics.items():
        ref = baseline.get(stage)
        if not isinstance(ref, dict) or ref.get("p95_ms") in (None, ""):
            failures.append(f"missing baseline p95_ms for {stage}")
            continue
        try:
            ceiling = float(ref["p95_ms"])
        except (TypeError, ValueError):
            failures.append(f"invalid baseline p95_ms for {stage}")
            continue
        observed = float(current["p95_ms"])
        ratio = observed / max(ceiling, 1e-9)
        comparison[stage] = {
            "observed_p95_ms": observed,
            "baseline_p95_ms": ceiling,
            "ratio": round(ratio, 3),
            "budget_multiplier": 1.25,
            "within_budget": observed <= ceiling * 1.25,
            "baseline_basis": ref.get("basis", "unspecified"),
        }
        if observed > ceiling * 1.25:
            failures.append(f"{stage}: observed p95 {observed:.3f}ms exceeds baseline budget {ceiling * 1.25:.3f}ms")
    return {"comparisons": comparison, "failures": failures, "pass": not failures}


def phase14_production_benchmark(phase: Any) -> PhaseResult:
    result = _result(phase)
    root: Path | None = None
    try:
        stage_samples = {
            "canonical_ingestion": [],
            "lexical_retrieval": [],
            "semantic_retrieval": [],
            "post_ingestion_validation": [],
        }
        repetitions = 7
        root = Path(tempfile.mkdtemp(prefix="rag_phase14_production_benchmark_"))
        for repetition in range(repetitions):
            system_root = root / f"rep-{repetition}"
            system = _ProductionIngestionProbeSystem(system_root)
            source_dir = system_root / "source"
            source_dir.mkdir(parents=True, exist_ok=True)
            source = source_dir / f"benchmark_{repetition}.pdf"
            _write_pdf(source, repetition)

            started = time.perf_counter()
            outcome = robust_ingest_file(system, source)
            stage_samples["canonical_ingestion"].append((time.perf_counter() - started) * 1000)
            if outcome.get("status") != "success":
                raise RuntimeError(f"benchmark ingestion failed: {outcome}")
            document_id = str(outcome["document_id"])
            processed = system.settings.processed_dir / source.name
            content_hash = system._hash_file(processed)
            started = time.perf_counter()
            validation = system.vector_store.validate_document_index(document_id, content_hash)
            stage_samples["post_ingestion_validation"].append((time.perf_counter() - started) * 1000)
            if not validation.get("valid"):
                raise RuntimeError(f"benchmark validation failed: {validation}")

            started = time.perf_counter()
            system.vector_store.search_lexical("HbA1c diagnosis", n_results=3)
            stage_samples["lexical_retrieval"].append((time.perf_counter() - started) * 1000)

            query_vector = system.embedding_service.embed_texts(["HbA1c diagnosis"])[0]
            started = time.perf_counter()
            system.vector_store.search(query_vector, n_results=3)
            stage_samples["semantic_retrieval"].append((time.perf_counter() - started) * 1000)

        metrics = {stage: _stats(values) for stage, values in stage_samples.items()}
        baseline = _load_baseline()
        regression = _regression_metrics(metrics, baseline)
        baseline_type = baseline.get("baseline_type")
        historical = bool(baseline.get("historical_measurement"))
        result.details = {
            "evidence_level": "real_pdf_to_retrieval_benchmark",
            "production_path_strict": True,
            "production_entrypoint": "rag_project.ingestion.robust_ingestor.robust_ingest_file",
            "stage_metrics": metrics,
            "repetitions": repetitions,
            "minimum_samples_per_stage": min(row["samples"] for row in metrics.values()),
            "pipeline_exercised": [
                "DocumentClassifier",
                "PDFExtractor",
                "SemanticChunker",
                "EmbeddingService(test_mode)",
                "VectorStore.add_documents",
                "VectorStore.validate_document_index",
                "VectorStore.search_lexical",
                "VectorStore.search",
            ],
            "baseline_path": str(BASELINE.relative_to(ROOT)),
            "baseline_loaded": True,
            "baseline_type": baseline_type,
            "historical_baseline_available": historical,
            "baseline_schema_version": baseline.get("schema_version"),
            "baseline_environment": baseline.get("environment"),
            "benchmark_environment": _environment_provenance(),
            "regression_budget_multiplier": 1.25,
            "regression_comparisons": regression["comparisons"],
            "regression_failures": regression["failures"],
            "regression_pass": regression["pass"],
        }
        result.score = 1.0 if result.details["minimum_samples_per_stage"] >= repetitions and regression["pass"] else 0.0
        result.status = "PASS" if result.score == 1.0 else "FAIL"
        if result.status == "FAIL":
            result.failures.append({
                "location": "phase 14 production benchmark baseline comparison",
                "exception": "PerformanceRegression",
                "message": str(regression["failures"]),
            })
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "phase 14 canonical production benchmark", "exception": type(exc).__name__, "message": str(exc)})
    finally:
        if root is not None:
            import shutil
            shutil.rmtree(root, ignore_errors=True)
    return result
