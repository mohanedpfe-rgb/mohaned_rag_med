from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from rag_project.evaluation.dataset import EVAL_VERSION, default_eval_root
from rag_project.evaluation.types import EvalReport


RETRIEVAL_DISPLAY_KEYS = [
    ("recall_at_3", "Recall@3"),
    ("recall_at_6", "Recall@6"),
    ("recall_at_10", "Recall@10"),
    ("mrr", "MRR"),
    ("ndcg_at_6", "nDCG@6"),
]

GENERATION_DISPLAY_KEYS = [
    ("faithfulness", "Faithfulness"),
    ("answer_acceptability", "Answer Acceptability"),
    ("rouge_l_f1_best", "ROUGE-L F1"),
    ("citation_validity", "Citation Validity"),
    ("negative_refusal_rate", "Neg Refusal Rate"),
]

LATENCY_DISPLAY_KEYS = [
    ("retrieval_ms", "Retrieval (ms)"),
    ("rerank_ms", "Rerank (ms)"),
    ("generation_ms", "Generation (ms)"),
    ("total_ms", "Total (ms)"),
]


def _fmt(value: float, decimals: int = 2) -> str:
    if value is None:
        return "-"
    if value >= 100:
        return f"{value:.0f}"
    if decimals == 0:
        return f"{value:.0f}"
    return f"{value:.{decimals}f}"


def _delta(before: Optional[float], after: float, decimals: int = 2) -> str:
    if before is None:
        return f"{_fmt(after, decimals)}"
    diff = after - before
    sign = "+" if diff > 0 else ""
    return f"{_fmt(after, decimals)} ({sign}{_fmt(diff, decimals)})"


def format_report_summary(report: EvalReport,
                          baseline: Optional[EvalReport] = None) -> str:
    lines: list[str] = []
    title_tag = ""
    if baseline is not None:
        title_tag = f" vs {baseline.label}"
    lines.append(f"=== {report.label.upper()}{title_tag} ===")
    lines.append(
        f"N={report.dataset_size} questions, dataset_version={report.dataset_version}, "
        f"model={report.configuration.get('generation_model', '?')}, "
        f"chunk={report.configuration.get('chunk_size', '?')}"
    )
    lines.append("")
    lines.append("RETRIEVAL (mean)")
    lines.append(f"  {'Metric':<14}  {'Value':>8}")
    base_ret = baseline.retrieval.mean if baseline else {}
    for key, label in RETRIEVAL_DISPLAY_KEYS:
        before = base_ret.get(key)
        after = report.retrieval.mean.get(key, 0.0)
        lines.append(f"  {label:<14}  {_delta(before, after, 3):>14}")
    lines.append("")
    lines.append("GENERATION (mean)")
    base_gen = baseline.generation.mean if baseline else {}
    lines.append(f"  {'Metric':<20}  {'Value':>14}")
    for key, label in GENERATION_DISPLAY_KEYS:
        before = base_gen.get(key)
        after = report.generation.mean.get(key, 0.0)
        if key == "negative_refusal_rate" and not report.generation.mean.get(key) and before is None:
            continue
        lines.append(f"  {label:<20}  {_delta(before, after, 3):>14}")
    if report.generation.mean.get("llm_judge_score") is not None:
        lines.append(
            f"  {'LLM Judge Score':<20}  "
            f"{_fmt(report.generation.mean['llm_judge_score'], 3):>14}"
        )
    lines.append("")
    lines.append("LATENCY (p50 / p95)")
    base_lat_p50 = baseline.latency_ms.p50 if baseline else {}
    base_lat_p95 = baseline.latency_ms.p95 if baseline else {}
    lines.append(f"  {'Stage':<14}  {'p50':>12} / {'p95':>12}")
    for key, label in LATENCY_DISPLAY_KEYS:
        p50 = report.latency_ms.p50.get(key, 0.0)
        p95 = report.latency_ms.p95.get(key, 0.0)
        bp50 = base_lat_p50.get(key)
        bp95 = base_lat_p95.get(key)
        lines.append(
            f"  {label:<14}  "
            f"{_delta(bp50, p50, 0):>12} / {_delta(bp95, p95, 0):>12}"
        )
    total_list = [r.timing.total_ms for r in report.individual_results if r.timing.total_ms > 0]
    if total_list:
        qpm = 60000.0 / (sum(total_list) / max(len(total_list), 1))
        lines.append(f"  Throughput:    {qpm:.1f} questions/min")
    lines.append("")
    lines.append("CATEGORY BREAKDOWN (Recall@6 / Faithfulness)")
    cat_combined: Dict[str, tuple[float, float]] = {}
    for cat, metrics in report.retrieval.by_category.items():
        recall = metrics.get("recall_at_6", 0.0)
        faith = report.generation.by_category.get(cat, {}).get("faithfulness", 0.0)
        cat_combined[cat] = (recall, faith)
    for cat in sorted(cat_combined):
        recall, faith = cat_combined[cat]
        lines.append(f"  {cat:<28} R@6={_fmt(recall,3)}  Faith={_fmt(faith,3)}")
    if baseline is not None:
        regressions = detect_regressions(baseline, report)
        if regressions:
            lines.append("")
            lines.append("⚠️  REGRESSIONS DETECTED:")
            for metric, (before, after, threshold) in regressions:
                lines.append(
                    f"  - {metric}: {_fmt(before,3)} → {_fmt(after,3)} "
                    f"(delta={_fmt(after-before,3)}, threshold={_fmt(threshold,3)})"
                )
        else:
            lines.append("")
            lines.append("✅ No regressions beyond thresholds.")
    return "\n".join(lines)


DEFAULT_REGRESSION_THRESHOLDS: Dict[str, float] = {
    "retrieval.recall_at_6": -0.03,
    "retrieval.mrr": -0.03,
    "retrieval.ndcg_at_6": -0.03,
    "generation.faithfulness": -0.05,
    "generation.answer_acceptability": -0.05,
    "latency_ms.total_ms.p95": 500.0,
}


def detect_regressions(
    baseline: EvalReport,
    current: EvalReport,
    thresholds: Optional[Dict[str, float]] = None,
) -> list[tuple[str, float, float, float]]:
    thresholds = dict(DEFAULT_REGRESSION_THRESHOLDS, **(thresholds or {}))
    regressions: list[tuple[str, float, float, float]] = []
    prefixes: Dict[str, tuple[Dict[str, float], Dict[str, float]]] = {
        "retrieval": (baseline.retrieval.mean, current.retrieval.mean),
        "generation": (baseline.generation.mean, current.generation.mean),
    }
    for prefix, (base, curr) in prefixes.items():
        for key, threshold in thresholds.items():
            if not key.startswith(prefix + "."):
                continue
            metric_name = key[len(prefix) + 1:]
            before = base.get(metric_name)
            after = curr.get(metric_name)
            if before is None or after is None:
                continue
            if threshold < 0:
                if (after - before) < threshold:
                    regressions.append((key, before, after, threshold))
            else:
                if (after - before) > threshold:
                    regressions.append((key, before, after, threshold))
    latency_keys = [k for k in thresholds if k.startswith("latency_ms.")]
    for key in latency_keys:
        parts = key.split(".")
        if len(parts) != 3:
            continue
        _, metric, stat = parts
        base_map = (
            baseline.latency_ms.p50 if stat == "p50"
            else baseline.latency_ms.p95 if stat == "p95"
            else baseline.latency_ms.mean
        )
        curr_map = (
            current.latency_ms.p50 if stat == "p50"
            else current.latency_ms.p95 if stat == "p95"
            else current.latency_ms.mean
        )
        before = base_map.get(metric)
        after = curr_map.get(metric)
        if before is None or after is None:
            continue
        threshold = thresholds[key]
        if (after - before) > threshold:
            regressions.append((key, before, after, threshold))
    return regressions


def compare_reports(
    label: str,
    *reports: EvalReport,
    thresholds: Optional[Dict[str, float]] = None,
) -> str:
    if not reports:
        return "(no reports to compare)"
    baseline = reports[0]
    others = reports[1:]
    out_parts = [format_report_summary(baseline)]
    for report in others:
        out_parts.append("")
        out_parts.append(format_report_summary(report, baseline=baseline))
    return "\n".join(out_parts)


def load_report_file(label: str, dataset_version: str = EVAL_VERSION) -> Optional[dict]:
    path = default_eval_root() / dataset_version / f"report_{label.lower()}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
