from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_random_exponential,
)

from rag_project.app.rag_system import RAGSystem, _generate_with_citations
from rag_project.configuration.settings import Settings
from rag_project.evaluation.dataset import EVAL_VERSION, default_eval_root
from rag_project.evaluation.metrics import (
    _mean,
    _percentile,
    build_llm_judge_prompt,
    compute_generation_metrics,
    compute_retrieval_metrics,
    parse_llm_judge_output,
)
from rag_project.evaluation.types import (
    AggregatedMetrics,
    EvalQuestion,
    EvalReport,
    EvalResult,
    GenerationMetrics,
    RetrievalMetrics,
    StageTiming,
)
from rag_project.generation.llm_client import OllamaLLMClient
from rag_project.retrieval.hybrid_retriever import RetrievalHit
from rag_project.retrieval.metadata_filter import MetadataFilter
from rag_project.retrieval.query_rewriter import QueryRewriter
from rag_project.utils.logger import build_logger


_TimerCallback = Callable[[str, float], None]


class _TimingContext:
    def __init__(self, cb: _TimerCallback, label: str):
        self._cb = cb
        self._label = label
        self._start: float | None = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._start is not None:
            elapsed_ms = (time.perf_counter() - self._start) * 1000.0
            self._cb(self._label, elapsed_ms)
        return False


class _InstrumentedRAG:
    def __init__(self, system: RAGSystem, stage_times: Dict[str, float]):
        self.system = system
        self._stage_times = stage_times

    def _record(self, label: str, ms: float) -> None:
        self._stage_times[label] = self._stage_times.get(label, 0.0) + ms

    def timer(self, label: str) -> _TimingContext:
        return _TimingContext(self._record, label)

    def retrieve_with_timing(
        self, query: str, top_k: int = 10
    ) -> List[RetrievalHit]:
        rewritten = QueryRewriter.rewrite(query, [], llm=self.system.llm)
        filter_query = MetadataFilter.build(None)
        with self.timer("retrieval"):
            raw_top_k = max(self.system.settings.top_k, 6)
            candidate_count = max(raw_top_k * 5, 20)
            retrieval_top_k = max(top_k, raw_top_k)
            hits = self.system.retriever.retrieve(
                rewritten, top_k=retrieval_top_k, where=filter_query
            )
        with self.timer("rerank"):
            reranked = self.system.reranker.rerank(rewritten, hits)
        return reranked[:top_k]

    def answer_with_timing(
        self, question: str
    ) -> Tuple[str, List[RetrievalHit], set[int]]:
        with self.timer("total"):
            rewritten = QueryRewriter.rewrite(question, self.system.conversation_memory.history, llm=self.system.llm)
            filter_query = MetadataFilter.build(None)
            with self.timer("retrieval"):
                hits = self.system.retriever.retrieve(
                    rewritten,
                    top_k=max(self.system.settings.top_k, 6),
                    where=filter_query,
                )
            with self.timer("rerank"):
                reranked = self.system.reranker.rerank(rewritten, hits)
            context, selected_hits = self.system.context_builder.build(reranked)
            try:
                with self.timer("generation"):
                    answer, cited_markers = self._generate(question, context, selected_hits)
                    answer, _, _, _ = self.system.apply_grounding_guard(
                        answer, rewritten, selected_hits
                    )
            except RuntimeError:
                answer = self._fallback_answer(selected_hits)
                cited_markers = set(range(1, len(selected_hits) + 1))
        return answer, selected_hits, cited_markers

    def _generate(
        self, question: str, context: str, selected_hits: Sequence[RetrievalHit]
    ) -> Tuple[str, set[int]]:
        conversation_context = self.system.conversation_memory.prompt_context()
        return _generate_with_citations(
            self.system.llm,
            question=question,
            context=context,
            selected_hits=selected_hits,
            conversation_context=conversation_context,
            temperature=self.system.settings.temperature,
        )

    @staticmethod
    def _fallback_answer(selected_hits: Sequence[RetrievalHit]) -> str:
        if not selected_hits:
            return "I could not find sufficient evidence in the indexed documents."
        return (
            "The language model is currently unavailable. The most relevant indexed "
            "evidence is provided below; verify it against the cited pages.\n\n"
            + "\n\n".join(
                f"[S{i + 1}] {hit.text}" for i, hit in enumerate(selected_hits)
            )
        )


def _aggregate_retrieval(
    results: Sequence[EvalResult],
) -> AggregatedMetrics:
    names = [
        "recall_at_3",
        "recall_at_6",
        "recall_at_10",
        "mrr",
        "ndcg_at_6",
        "chunk_id_exact_match",
    ]
    values: Dict[str, List[float]] = {n: [] for n in names}
    for r in results:
        values["recall_at_3"].append(r.retrieval.recall_at_3)
        values["recall_at_6"].append(r.retrieval.recall_at_6)
        values["recall_at_10"].append(r.retrieval.recall_at_10)
        values["mrr"].append(r.retrieval.mrr)
        values["ndcg_at_6"].append(r.retrieval.ndcg_at_6)
        values["chunk_id_exact_match"].append(1.0 if r.retrieval.chunk_id_exact_match else 0.0)
    return _build_aggregated(values, results, lambda r: r.category.value,
                            lambda r: r.difficulty.value)


def _aggregate_generation(
    results: Sequence[EvalResult],
) -> AggregatedMetrics:
    names = [
        "faithfulness",
        "keyphrase_precision",
        "forbidden_penalty",
        "answer_acceptability",
        "rouge_l_f1_best",
        "citation_validity",
        "negative_refusal_rate",
    ]
    values: Dict[str, List[float]] = {n: [] for n in names}
    llm_judge_values: List[float] = []
    for r in results:
        values["faithfulness"].append(r.generation.faithfulness)
        values["keyphrase_precision"].append(r.generation.keyphrase_precision)
        values["forbidden_penalty"].append(r.generation.forbidden_penalty)
        values["answer_acceptability"].append(r.generation.answer_acceptability)
        values["rouge_l_f1_best"].append(r.generation.rouge_l_f1_best)
        values["citation_validity"].append(r.generation.citation_validity)
        if r.generation.negative_refusal is not None:
            values["negative_refusal_rate"].append(
                1.0 if r.generation.negative_refusal else 0.0
            )
        if r.generation.llm_judge_score is not None:
            llm_judge_values.append(r.generation.llm_judge_score)
    agg = _build_aggregated(values, results, lambda r: r.category.value,
                            lambda r: r.difficulty.value)
    if llm_judge_values:
        agg.mean["llm_judge_score"] = _mean(llm_judge_values)
        agg.p50["llm_judge_score"] = _percentile(llm_judge_values, 50)
        agg.p95["llm_judge_score"] = _percentile(llm_judge_values, 95)
    return agg


def _aggregate_latency(
    results: Sequence[EvalResult],
) -> AggregatedMetrics:
    names = ["retrieval_ms", "rerank_ms", "generation_ms", "total_ms"]
    values: Dict[str, List[float]] = {n: [] for n in names}
    for r in results:
        values["retrieval_ms"].append(r.timing.retrieval_ms)
        values["rerank_ms"].append(r.timing.rerank_ms)
        values["generation_ms"].append(r.timing.generation_ms)
        values["total_ms"].append(r.timing.total_ms)
    return _build_aggregated(values, results, lambda r: r.category.value,
                            lambda r: r.difficulty.value)


def _build_aggregated(
    values: Dict[str, List[float]],
    results: Sequence[EvalResult],
    category_key,
    difficulty_key,
) -> AggregatedMetrics:
    mean_dict: Dict[str, float] = {}
    p50_dict: Dict[str, float] = {}
    p95_dict: Dict[str, float] = {}
    for name, lst in values.items():
        if not lst:
            continue
        mean_dict[name] = _mean(lst)
        p50_dict[name] = _percentile(lst, 50)
        p95_dict[name] = _percentile(lst, 95)

    by_category: Dict[str, Dict[str, float]] = {}
    by_difficulty: Dict[str, Dict[str, float]] = {}
    for r in results:
        cat = category_key(r)
        diff = difficulty_key(r)
        by_category.setdefault(cat, {})
        by_difficulty.setdefault(diff, {})

    for metric_name, all_values in values.items():
        if not all_values:
            continue
        cat_lists: Dict[str, List[float]] = {}
        diff_lists: Dict[str, List[float]] = {}
        for r in results:
            cat = category_key(r)
            diff = difficulty_key(r)
            value_map = {
                "recall_at_3": r.retrieval.recall_at_3,
                "recall_at_6": r.retrieval.recall_at_6,
                "recall_at_10": r.retrieval.recall_at_10,
                "mrr": r.retrieval.mrr,
                "ndcg_at_6": r.retrieval.ndcg_at_6,
                "chunk_id_exact_match": 1.0 if r.retrieval.chunk_id_exact_match else 0.0,
                "faithfulness": r.generation.faithfulness,
                "keyphrase_precision": r.generation.keyphrase_precision,
                "forbidden_penalty": r.generation.forbidden_penalty,
                "answer_acceptability": r.generation.answer_acceptability,
                "rouge_l_f1_best": r.generation.rouge_l_f1_best,
                "citation_validity": r.generation.citation_validity,
                "negative_refusal_rate": (
                    1.0 if r.generation.negative_refusal else 0.0
                    if r.generation.negative_refusal is not None else None
                ),
                "retrieval_ms": r.timing.retrieval_ms,
                "rerank_ms": r.timing.rerank_ms,
                "generation_ms": r.timing.generation_ms,
                "total_ms": r.timing.total_ms,
            }
            val = value_map.get(metric_name)
            if val is None:
                continue
            cat_lists.setdefault(cat, []).append(val)
            diff_lists.setdefault(diff, []).append(val)
        for cat, lst in cat_lists.items():
            by_category[cat][metric_name] = _mean(lst)
        for diff, lst in diff_lists.items():
            by_difficulty[diff][metric_name] = _mean(lst)

    return AggregatedMetrics(
        mean=mean_dict,
        p50=p50_dict,
        p95=p95_dict,
        by_category=by_category,
        by_difficulty=by_difficulty,
    )


def _run_llm_judge_calibration(
    results: List[EvalResult],
    questions_by_id: Dict[str, EvalQuestion],
    hits_by_question_id: Dict[str, List[RetrievalHit]],
    answers_by_question_id: Dict[str, str],
    judge_llm: OllamaLLMClient,
    logger,
) -> int:
    logger.info("Running LLM-as-judge calibration (one-shot, %d questions)...", len(results))

    judge_retries: Dict[str, int] = {"total": 0}

    def _judge_before_sleep(retry_state):
        judge_retries["total"] += 1
        attempt = retry_state.attempt_number
        exc = retry_state.outcome.exception()
        logger.warning("Judge retry attempt %d failed with: %s", attempt, exc)

    @retry(
        stop=stop_after_attempt(5) | stop_after_delay(120),
        wait=wait_random_exponential(min=1, max=30),
        retry=retry_if_exception_type((requests.RequestException, RuntimeError)),
        before_sleep=_judge_before_sleep,
        reraise=True,
    )
    def _judge_one(user_prompt: str, system_prompt: str) -> str:
        return judge_llm.generate(
            user_prompt, system_prompt=system_prompt, temperature=0.0
        )

    for i, result in enumerate(results):
        q = questions_by_id.get(result.question_id)
        if q is None:
            continue
        hits = hits_by_question_id.get(result.question_id, [])
        answer = answers_by_question_id.get(result.question_id, result.raw_answer)
        system_prompt, user_prompt = build_llm_judge_prompt(q, answer, hits)
        try:
            raw = _judge_one(user_prompt, system_prompt)
            score, explanation = parse_llm_judge_output(raw)
            result.generation.llm_judge_score = score
            result.generation.llm_judge_explanation = explanation
        except RuntimeError as exc:
            logger.warning("LLM-judge failed for %s: %s", result.question_id, exc)
            result.generation.llm_judge_score = None
            result.generation.llm_judge_explanation = f"judge_error: {exc}"
            result.error = f"judge_exhausted_retries: {exc}"

    return judge_retries["total"]


def run_eval(
    rag_system: RAGSystem | None = None,
    questions: Sequence[EvalQuestion] | None = None,
    label: str = "baseline",
    dataset_version: str = EVAL_VERSION,
    run_llm_judge: bool = False,
    judge_model: str | None = None,
    save_report: bool = True,
    settings: Settings | None = None,
) -> EvalReport:
    if settings is None:
        settings = Settings.from_env()
    if rag_system is None:
        rag_system = RAGSystem(settings)
    log_dir = settings.log_dir
    logger = build_logger(f"eval_{label.lower()}", log_dir)
    from rag_project.evaluation.dataset import load_dataset
    dataset_in = list(questions) if questions is not None else load_dataset(dataset_version)
    dataset = [
        q for q in dataset_in
        if not (q.metadata and q.metadata.get("manual_slot"))
    ]
    skipped_manual = len(dataset_in) - len(dataset)
    if not dataset:
        raise RuntimeError(
            f"No eval questions (after skipping {skipped_manual} manual slots) "
            f"found for version {dataset_version}. "
            f"Run generate_silver.py first or populate data/eval/{dataset_version}/dataset.jsonl."
        )
    logger.info(
        "Starting eval run label=%s n=%d (skipped %d manual adversarial slots)",
        label, len(dataset), skipped_manual,
    )

    results: List[EvalResult] = []
    questions_by_id: Dict[str, EvalQuestion] = {q.id: q for q in dataset}
    hits_by_question_id: Dict[str, List[RetrievalHit]] = {}
    answers_by_question_id: Dict[str, str] = {}

    for i, q in enumerate(dataset):
        stage_times: Dict[str, float] = {}
        inst = _InstrumentedRAG(rag_system, stage_times)
        retrieval_metrics: RetrievalMetrics | None = None
        generation_metrics: GenerationMetrics | None = None
        selected_hits: List[RetrievalHit] = []
        raw_answer = ""
        error: str | None = None
        retrieved_chunk_ids: List[str] = []
        retrieved_scores: List[float] = []
        try:
            all_hits = inst.retrieve_with_timing(q.question, top_k=10)
            retrieval_metrics = compute_retrieval_metrics(q, all_hits)
            hits_by_question_id[q.id] = list(all_hits)
            retrieved_chunk_ids = [
                str(hit.metadata.get("chunk_id", hit.doc_id)) for hit in all_hits
            ]
            retrieved_scores = [float(hit.score) for hit in all_hits]
            raw_answer, selected_hits, cited_markers = inst.answer_with_timing(q.question)
            answers_by_question_id[q.id] = raw_answer
            generation_metrics = compute_generation_metrics(q, raw_answer, selected_hits)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.error("Eval question %s failed: %s", q.id, error)
            if retrieval_metrics is None:
                retrieval_metrics = RetrievalMetrics()
            if generation_metrics is None:
                generation_metrics = GenerationMetrics()

        timing = StageTiming(
            retrieval_ms=stage_times.get("retrieval", 0.0),
            rerank_ms=stage_times.get("rerank", 0.0),
            generation_ms=stage_times.get("generation", 0.0),
            total_ms=stage_times.get("total", stage_times.get("retrieval", 0.0)
                                     + stage_times.get("rerank", 0.0)
                                     + stage_times.get("generation", 0.0)),
        )
        if timing.total_ms < 0.01:
            timing.total_ms = timing.retrieval_ms + timing.rerank_ms + timing.generation_ms
        results.append(EvalResult(
            question_id=q.id,
            category=q.category,
            difficulty=q.difficulty,
            retrieval=retrieval_metrics,
            generation=generation_metrics,
            timing=timing,
            raw_answer=raw_answer,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_scores=retrieved_scores,
            error=error,
        ))
        if (i + 1) % 10 == 0 or i == len(dataset) - 1:
            logger.info("Progress: %d/%d questions evaluated", i + 1, len(dataset))

    total_retries_encountered: int = 0
    if run_llm_judge:
        judge_client = OllamaLLMClient(
            base_url=settings.ollama_base_url,
            model=judge_model or settings.generation_model,
            timeout_seconds=max(settings.generation_timeout_seconds, 120.0),
        )
        total_retries_encountered = _run_llm_judge_calibration(
            results, questions_by_id, hits_by_question_id,
            answers_by_question_id, judge_client, logger,
        )

    report = EvalReport(
        label=label,
        dataset_version=dataset_version,
        dataset_size=len(dataset),
        retrieval=_aggregate_retrieval(results),
        generation=_aggregate_generation(results),
        latency_ms=_aggregate_latency(results),
        individual_results=results,
        created_at=datetime.now(timezone.utc).isoformat(),
        configuration={
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k,
            "embedding_model": settings.embedding_model,
            "generation_model": settings.generation_model,
            "temperature": settings.temperature,
            "lexical_mode": settings.lexical_mode,
            "llm_judge_ran": run_llm_judge,
            "judge_model": judge_model or settings.generation_model if run_llm_judge else None,
            "retries_encountered": total_retries_encountered,
        },
    )
    if save_report:
        report_dir = default_eval_root() / dataset_version
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / f"report_{label.lower()}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2, ensure_ascii=False)
        logger.info("Eval report saved to %s", path)
    return report
