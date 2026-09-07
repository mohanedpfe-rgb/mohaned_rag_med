from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from rag_project.evaluation.types import (
    EvalCategory,
    EvalQuestion,
    GenerationMetrics,
    RetrievalMetrics,
)
from rag_project.retrieval.hybrid_retriever import RetrievalHit
from rag_project.utils.text_utils import meaningful_tokens, normalize_arabic, tokenize


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def _pages_from_hits(hits: Sequence[RetrievalHit]) -> List[List[int]]:
    pages_per_hit: List[List[int]] = []
    for hit in hits:
        metadata = hit.metadata or {}
        pages = metadata.get("page_numbers") or metadata.get("source_pages") or []
        if isinstance(pages, int):
            pages_per_hit.append([pages])
        else:
            pages_per_hit.append([int(p) for p in pages])
    return pages_per_hit


def compute_recall_at_k(
    relevant_pages: Iterable[int],
    hits: Sequence[RetrievalHit],
    k: int,
) -> float:
    relevant = set(int(p) for p in relevant_pages)
    if not relevant:
        return 0.0
    pages_per_hit = _pages_from_hits(hits[: max(0, k)])
    retrieved_pages: set[int] = set()
    for pages in pages_per_hit:
        retrieved_pages.update(pages)
    return len(retrieved_pages & relevant) / len(relevant)


def compute_mrr(
    relevant_pages: Iterable[int],
    hits: Sequence[RetrievalHit],
) -> tuple[float, Optional[int]]:
    relevant = set(int(p) for p in relevant_pages)
    if not relevant:
        return 0.0, None
    pages_per_hit = _pages_from_hits(hits)
    for rank, pages in enumerate(pages_per_hit, start=1):
        if set(pages) & relevant:
            return 1.0 / rank, rank
    return 0.0, None


def _graded_relevance(pages: List[int], relevant: set[int]) -> int:
    overlap = set(pages) & relevant
    if not overlap:
        return 0
    if overlap == set(pages):
        return 3
    return 1


def compute_ndcg_at_k(
    relevant_pages: Iterable[int],
    hits: Sequence[RetrievalHit],
    k: int,
) -> float:
    relevant = set(int(p) for p in relevant_pages)
    if not relevant:
        return 0.0
    pages_per_hit = _pages_from_hits(hits[: max(0, k)])
    dcg = 0.0
    for i, pages in enumerate(pages_per_hit):
        rel = _graded_relevance(pages, relevant)
        dcg += rel / math.log2(i + 2)
    ideal = sorted(
        [_graded_relevance(pages, relevant) for pages in pages_per_hit],
        reverse=True,
    )
    idcg = 0.0
    for i, rel in enumerate(ideal):
        idcg += rel / math.log2(i + 2)
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def chunk_id_exact_match(
    relevant_chunk_ids: Iterable[str],
    hits: Sequence[RetrievalHit],
    k: int = 6,
) -> bool:
    expected = {str(i) for i in relevant_chunk_ids}
    if not expected:
        return False
    for hit in hits[:k]:
        metadata = hit.metadata or {}
        chunk_id = str(metadata.get("chunk_id", ""))
        if chunk_id in expected:
            return True
    return False


def compute_retrieval_metrics(
    question: EvalQuestion,
    hits: Sequence[RetrievalHit],
) -> RetrievalMetrics:
    pages_per_hit = _pages_from_hits(hits)
    retrieved_page_union: List[int] = sorted(
        {page for pages in pages_per_hit for page in pages}
    )
    mrr, first_rank = compute_mrr(question.relevant_page_numbers, hits)
    return RetrievalMetrics(
        recall_at_3=compute_recall_at_k(question.relevant_page_numbers, hits, 3),
        recall_at_6=compute_recall_at_k(question.relevant_page_numbers, hits, 6),
        recall_at_10=compute_recall_at_k(question.relevant_page_numbers, hits, 10),
        mrr=mrr,
        ndcg_at_6=compute_ndcg_at_k(question.relevant_page_numbers, hits, 6),
        chunk_id_exact_match=chunk_id_exact_match(
            question.relevant_chunk_ids, hits, k=6
        ),
        retrieved_page_union=retrieved_page_union,
        rank_of_first_relevant=first_rank,
    )


# ---------------------------------------------------------------------------
# Generation metrics (rule-based)
# ---------------------------------------------------------------------------


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0] * (len(b) + 1)
        for j, other in enumerate(b):
            if token == other:
                current[j + 1] = previous[j] + 1
            else:
                current[j + 1] = max(current[j], previous[j + 1])
        previous = current
    return previous[-1]


def rouge_l_f1(candidate: str, reference: str) -> float:
    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(cand_tokens, ref_tokens)
    precision = lcs / max(len(cand_tokens), 1)
    recall = lcs / max(len(ref_tokens), 1)
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def best_rouge_l(candidate: str, references: Sequence[str]) -> float:
    if not references:
        return 0.0
    return max(rouge_l_f1(candidate, ref) for ref in references)


def answer_acceptability(candidate: str, references: Sequence[str],
                        rouge_threshold: float = 0.5) -> float:
    if not references:
        return 0.0
    candidate_lower = candidate.casefold()
    for ref in references:
        if not ref:
            continue
        if ref.casefold() in candidate_lower:
            return 1.0
    if best_rouge_l(candidate, references) >= rouge_threshold:
        return 1.0
    return 0.0


def _casefold_find(haystack: str, needles: Iterable[str]) -> List[str]:
    hay = normalize_arabic(haystack).casefold()
    return [
        n for n in needles
        if n and normalize_arabic(n).casefold() in hay
    ]


def compute_keyphrase_precision(
    answer: str,
    evidence_text: str,
    required_keyphrases: Sequence[str],
) -> float:
    if not required_keyphrases:
        return 1.0
    in_answer = set(_casefold_find(answer, required_keyphrases))
    in_evidence = set(_casefold_find(evidence_text, required_keyphrases))
    numerator = 0
    for kp in required_keyphrases:
        if kp in in_answer and kp in in_evidence:
            numerator += 1
    denominator = len(required_keyphrases)
    return numerator / denominator


def compute_forbidden_penalty(answer: str, forbidden: Sequence[str]) -> float:
    matches = _casefold_find(answer, forbidden)
    return 1.0 if matches else 0.0


NEGATIVE_REFUSAL_PATTERN = re.compile(
    r"\b(not\s+(found|contain|available|in\s+the\s+document|in\s+corpus|in\s+evidence)"
    r"|cannot?\s+(answer|find|determine|provide)"
    r"|no\s+(information|evidence|relevant\s+document|data|mention)"
    r"|insufficient\s+(information|evidence|data)"
    r"|(don't|don’t|do\s+not)\s+know"
    r"|(je\s+ne\s+sais\s+pas|pas\s+d'information|information\s+non\s+disponible)"
    r"|(لا\s+توجد?\s+(?:معلومات|أدلة)|لا\s+أعرف|المعلومات\s+غير\s+متوفرة))\b",
    re.IGNORECASE | re.UNICODE,
)


def is_negative_refusal(answer: str) -> bool:
    return bool(NEGATIVE_REFUSAL_PATTERN.search(answer or ""))


CITATION_MARKER_RE = re.compile(r"\[S(\d+)\]")


def compute_citation_validity(
    answer: str,
    evidence_hits: Sequence[Any],
    required_keyphrases: Sequence[str],
) -> float:
    markers = CITATION_MARKER_RE.findall(answer or "")
    if not markers:
        return 0.0
    valid = 0
    for raw in markers:
        try:
            idx = int(raw) - 1
        except ValueError:
            continue
        if idx < 0 or idx >= len(evidence_hits):
            continue
        hit_text = getattr(evidence_hits[idx], "text", "")
        if required_keyphrases:
            if _casefold_find(hit_text, required_keyphrases):
                valid += 1
        else:
            valid += 1
    return valid / len(markers)


def compute_generation_metrics(
    question: EvalQuestion,
    answer: str,
    evidence_hits: Sequence[Any],
) -> GenerationMetrics:
    evidence_text = "\n\n".join(
        getattr(h, "text", "") or (h.metadata or {}).get("text", "")
        if hasattr(h, "metadata")
        else str(h)
        for h in evidence_hits
    )
    keyphrase_precision = compute_keyphrase_precision(
        answer, evidence_text, question.required_keyphrases
    )
    forbidden_penalty = compute_forbidden_penalty(
        answer, question.forbidden_keyphrases
    )
    faithfulness = max(0.0, keyphrase_precision - forbidden_penalty)
    clean_answer = re.sub(r"\s*Sources?:.*", "", answer or "", flags=re.IGNORECASE | re.DOTALL).strip()
    acceptability = answer_acceptability(clean_answer, question.acceptable_answers)
    rouge = best_rouge_l(clean_answer, question.acceptable_answers)
    citation_validity = compute_citation_validity(
        answer, evidence_hits, question.required_keyphrases
    )
    is_negative = question.category == EvalCategory.NEGATIVE_CONTROL
    refusal = is_negative_refusal(answer) if is_negative else None
    return GenerationMetrics(
        faithfulness=faithfulness,
        keyphrase_precision=keyphrase_precision,
        forbidden_penalty=forbidden_penalty,
        answer_acceptability=acceptability,
        rouge_l_f1_best=rouge,
        citation_validity=citation_validity,
        negative_refusal=refusal,
        llm_judge_score=None,
        llm_judge_explanation=None,
    )


# ---------------------------------------------------------------------------
# LLM-as-judge calibration (run once per dataset version, not per step)
# ---------------------------------------------------------------------------


LLM_JUDGE_SYSTEM_PROMPT = (
    "You are a strict, reference-only grading judge for a RAG system. "
    "Given a question, the RAG's generated answer, the ground-truth "
    "acceptable answers, and the raw evidence text the RAG was allowed "
    "to use, output ONLY a JSON object with two fields:\n"
    "  score: number between 0.0 and 1.0\n"
    "  explanation: short string explaining the score\n"
    "Scoring rubric:\n"
    "  1.0 = answer is fully factually supported by the evidence, matches "
    "or is more specific than any acceptable reference, cites correctly, "
    "invents nothing, has no contradictions.\n"
    "  0.7 = mostly correct, minor paraphrasing drift but no substantive errors.\n"
    "  0.4 = partially correct, omits key details or adds weak unsupported claims.\n"
    "  0.1 = mostly wrong, significant inventions or contradictions with evidence.\n"
    "  0.0 = entirely fabricated, wrong, or refuses when evidence is sufficient.\n"
    "For NEGATIVE_CONTROL questions (no answer in corpus), assign 1.0 only if "
    "the answer clearly says the information is not present; 0.0 if it hallucinates."
)


LLM_JUDGE_USER_TEMPLATE = """
CATEGORY: {category}
QUESTION: {question}
GROUND-TRUTH ACCEPTABLE ANSWERS (use as semantic reference only):
{references}
RAW EVIDENCE THE RAG USED:
{evidence}
RAG'S GENERATED ANSWER TO GRADE:
{answer}
"""


def build_llm_judge_prompt(
    question: EvalQuestion,
    answer: str,
    evidence_hits: Sequence[Any],
) -> tuple[str, str]:
    references = "\n".join(
        f"  [{i + 1}] {ref}" for i, ref in enumerate(question.acceptable_answers)
    ) or "  (none provided)"
    evidence = "\n\n".join(
        f"--- Evidence chunk {i + 1} ---\n{getattr(h, 'text', '')}"
        for i, h in enumerate(evidence_hits)
    )
    user_prompt = LLM_JUDGE_USER_TEMPLATE.format(
        category=question.category.value,
        question=question.question,
        references=references,
        evidence=evidence or "(empty)",
        answer=answer or "(empty)",
    )
    return LLM_JUDGE_SYSTEM_PROMPT, user_prompt


def parse_llm_judge_output(raw: str) -> tuple[Optional[float], Optional[str]]:
    if not raw:
        return None, None
    first_json = re.search(r"\{[\s\S]*?\}", raw)
    if not first_json:
        return None, raw.strip()[:200]
    try:
        import json
        obj = json.loads(first_json.group(0))
        score = obj.get("score")
        if isinstance(score, (int, float)):
            score = float(max(0.0, min(1.0, score)))
        else:
            score = None
        explanation = str(obj.get("explanation", "")) or None
        return score, explanation
    except (ValueError, TypeError, KeyError):
        return None, raw.strip()[:200]


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ordered[int(k)]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
