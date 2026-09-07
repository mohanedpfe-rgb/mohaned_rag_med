from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from rag_project.configuration.settings import Settings
from rag_project.evaluation.dataset import EVAL_VERSION, dataset_summary, load_dataset
from rag_project.evaluation.report import (
    format_report_summary,
    load_report_file,
)
from rag_project.evaluation.runner import run_eval
from rag_project.evaluation.types import EvalReport


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the RAG eval harness on a dataset version and print a report."
    )
    parser.add_argument("--label", type=str, default="baseline",
                        help="Report label (e.g. baseline, step_b_after, step_c_after)")
    parser.add_argument("--dataset-version", type=str, default=EVAL_VERSION)
    parser.add_argument("--dataset-filename", type=str, default="dataset.jsonl")
    parser.add_argument("--llm-judge", action="store_true",
                        help="Run LLM-as-judge calibration once on this run")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Override model for LLM judge (defaults to generation_model)")
    parser.add_argument("--dataset-summary", action="store_true",
                        help="Print dataset summary and exit")
    parser.add_argument("--baseline-label", type=str, default=None,
                        help="Compare against a previous report label")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save JSON report to disk")
    parser.add_argument("--only-reviewed", action="store_true",
                        help="Evaluate only human-curated or reviewed questions")
    args = parser.parse_args(argv)

    if args.dataset_summary:
        questions = load_dataset(
            version=args.dataset_version,
            filename=args.dataset_filename,
            only_reviewed=args.only_reviewed,
        )
        print(json.dumps(dataset_summary(questions), indent=2, ensure_ascii=False))
        return 0

    settings = Settings.from_env()
    questions = load_dataset(
        version=args.dataset_version,
        filename=args.dataset_filename,
        only_reviewed=args.only_reviewed,
    )
    if not questions:
        print(
            f"ERROR: No questions in eval dataset v{args.dataset_version}/"
            f"{args.dataset_filename}.\n"
            f"       Run `python -m rag_project.evaluation.generate_silver` first, "
            f"then review and fill manual slots."
        , file=sys.stderr)
        return 2

    report = run_eval(
        rag_system=None,
        questions=questions,
        label=args.label,
        dataset_version=args.dataset_version,
        run_llm_judge=args.llm_judge,
        judge_model=args.judge_model,
        save_report=not args.no_save,
        settings=settings,
    )
    baseline = None
    if args.baseline_label:
        loaded = load_report_file(args.baseline_label, dataset_version=args.dataset_version)
        if loaded is None:
            print(f"WARNING: baseline report '{args.baseline_label}' not found, skipping comparison.",
                  file=sys.stderr)
        else:
            baseline = EvalReport.from_dict(loaded) if loaded is not None else None
    print()
    print(format_report_summary(report, baseline=baseline))
    return 0


if __name__ == "__main__":
    sys.exit(main())
