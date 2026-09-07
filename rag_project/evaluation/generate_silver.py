from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rag_project.configuration.settings import Settings
from rag_project.evaluation.dataset import (
    EVAL_VERSION,
    append_questions,
    dataset_summary,
    load_dataset,
    save_dataset,
)
from rag_project.evaluation.types import (
    EvalCategory,
    EvalDifficulty,
    EvalQuestion,
    EvalSource,
)
from rag_project.generation.llm_client import OllamaLLMClient
from rag_project.storage.vector_store import VectorStore
from rag_project.utils.logger import build_logger
from rag_project.utils.text_utils import meaningful_tokens

_EVAL_GENERIC_KEYPHRASES = {
    "texte", "textes", "fois", "selon", "section", "chapitre",
    "partie", "question", "réponse", "reference", "référence",
}


SILVER_SYSTEM_PROMPT = (
    "You are a textbook question writer. Your output must be a JSON array. "
    "Do not write prose outside the JSON array. The array must contain "
    "exactly 2 objects, each with keys: question, answer, keyphrases, "
    "difficulty, category.\n"
    "Rules:\n"
    "1. The answer MUST appear verbatim as a contiguous substring of the "
    "source excerpt. Do not paraphrase, shorten, or rephrase the answer "
    "beyond what the excerpt already says.\n"
    "2. The question must be answerable using ONLY the excerpt.\n"
    "3. The answer should be a complete, self-contained sentence or two "
    "from the excerpt, not just a number or single word.\n"
    "4. keyphrases is an array of 3-6 French/English strings that MUST "
    "appear in both the excerpt and the answer, e.g. ['metformine', "
    "'première ligne', 'diabète de type 2'].\n"
    "5. difficulty is one of: easy, medium.\n"
    "6. category is one of: factual_recall, dosage_calculation, "
    "contraindication.\n"
    "7. Do not reference page numbers, chapters, or tables in the question "
    "or answer."
)


SILVER_USER_TEMPLATE = """
SOURCE EXCERPT (page {page} of {filename}):
-----
{chunk_text}
-----
"""


TABLE_SYSTEM_PROMPT = (
    "You are a table-data question writer. Output a JSON array of exactly 1 "
    "object with keys: question, answer, keyphrases, difficulty, category. "
    "Rules:\n"
    "1. The answer must come verbatim from the table below.\n"
    "2. The question must ask about a specific cell value, threshold, or "
    "row-to-row comparison.\n"
    "3. category must be table_lookup.\n"
    "4. difficulty must be hard."
)


@dataclass
class SampledChunk:
    chunk_id: str
    document_id: str
    file_name: str
    page_numbers: List[int]
    text: str


def _all_ready_chunks_from_store(vector_db_dir: Path) -> List[SampledChunk]:
    import sqlite3
    store = VectorStore(vector_db_dir)
    chroma_count = store.collection.count()
    all_chunks: List[SampledChunk] = []
    if chroma_count > 0:
        peek = chroma_count
        sample = store.collection.peek(peek)
        ids = sample.get("ids") or []
        docs = sample.get("documents") or []
        metas = sample.get("metadatas") or []
        for chunk_id, doc, meta in zip(ids, docs, metas):
            if not isinstance(meta, dict):
                continue
            if meta.get("index_state") != "READY":
                continue
            pages = meta.get("page_numbers") or meta.get("source_pages") or []
            if isinstance(pages, int):
                pages_list = [pages]
            else:
                pages_list = [int(p) for p in pages]
            all_chunks.append(SampledChunk(
                chunk_id=str(chunk_id),
                document_id=str(meta.get("document_id", "")),
                file_name=str(meta.get("file_name", "unknown.pdf")),
                page_numbers=pages_list,
                text=str(doc or ""),
            ))
    if all_chunks:
        return all_chunks
    lexical_path = Path(vector_db_dir) / "lexical.sqlite3"
    if not lexical_path.exists():
        return []
    conn = sqlite3.connect(str(lexical_path))
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT chunk_id, document_id, document, metadata FROM lexical_chunks "
            "WHERE json_extract(metadata, '$.index_state') = 'READY'"
        )
        for chunk_id, document_id, document, metadata_json in cur.fetchall():
            try:
                meta = json.loads(metadata_json) if metadata_json else {}
            except (ValueError, TypeError):
                meta = {}
            pages = meta.get("page_numbers") or meta.get("source_pages") or []
            if isinstance(pages, int):
                pages_list = [pages]
            else:
                pages_list = [int(p) for p in pages]
            all_chunks.append(SampledChunk(
                chunk_id=str(chunk_id),
                document_id=str(meta.get("document_id", document_id)),
                file_name=str(meta.get("file_name", "unknown.pdf")),
                page_numbers=pages_list,
                text=str(document or ""),
            ))
    finally:
        conn.close()
    return all_chunks


def _sample_chunks_from_store(
    vector_db_dir: Path,
    count: int,
    rng: random.Random,
) -> List[SampledChunk]:
    all_chunks = _all_ready_chunks_from_store(vector_db_dir)
    by_page: Dict[int, List[SampledChunk]] = {}
    for sc in all_chunks:
        if not (50 < len(sc.text.strip()) < 5000):
            continue
        anchor_page = sc.page_numbers[0] if sc.page_numbers else 0
        by_page.setdefault(anchor_page, []).append(sc)
    strata = sorted(by_page.keys())
    if not strata:
        return []
    selected: List[SampledChunk] = []
    step = max(1, len(strata) // max(count, 1))
    for i in range(0, len(strata), step):
        page = strata[i]
        choices = by_page[page]
        selected.append(rng.choice(choices))
        if len(selected) >= count:
            break
    if len(selected) < count:
        flat = [c for lst in by_page.values() for c in lst if c not in selected]
        rng.shuffle(flat)
        selected.extend(flat[: count - len(selected)])
    return selected


def _sample_table_chunks(
    vector_db_dir: Path,
    count: int,
    rng: random.Random,
) -> List[SampledChunk]:
    all_chunks = _all_ready_chunks_from_store(vector_db_dir)
    candidates: List[SampledChunk] = []
    for sc in all_chunks:
        if sc.text.count(" | ") < 3:
            continue
        candidates.append(sc)
    rng.shuffle(candidates)
    return candidates[:count]


_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json_objects(raw: str, expect_array: bool = True) -> List[Dict[str, Any]]:
    if not raw:
        return []
    if expect_array:
        match = _JSON_ARRAY_RE.search(raw)
    else:
        match = _JSON_OBJECT_RE.search(raw)
    if not match:
        return []
    try:
        obj = json.loads(match.group(0))
    except (ValueError, TypeError):
        return []
    if expect_array and isinstance(obj, list):
        return [o for o in obj if isinstance(o, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []


def _answer_is_verbatim(answer: str, chunk_text: str) -> bool:
    if not answer or not chunk_text:
        return False
    stripped = re.sub(r"\s+", " ", answer).strip().casefold()
    haystack = re.sub(r"\s+", " ", chunk_text).strip().casefold()
    if len(stripped) < 20:
        return False
    return stripped in haystack


def _keyphrases_in_text(keyphrases: Sequence[str], text: str) -> bool:
    if not keyphrases:
        return False
    hay = text.casefold()
    return all(k.casefold() in hay for k in keyphrases if k)


def _validated_pairs(
    llm_output: str,
    chunk: SampledChunk,
    idx: int,
    start_qid: int,
    max_per_chunk: int = 2,
) -> List[EvalQuestion]:
    objects = _extract_json_objects(llm_output, expect_array=True)
    if not objects:
        return []
    out: List[EvalQuestion] = []
    difficulty_map = {"easy": EvalDifficulty.EASY, "medium": EvalDifficulty.MEDIUM,
                      "hard": EvalDifficulty.HARD}
    category_map = {
        "factual_recall": EvalCategory.FACTUAL_RECALL,
        "table_lookup": EvalCategory.TABLE_LOOKUP,
        "dosage_calculation": EvalCategory.DOSAGE_CALCULATION,
        "contraindication": EvalCategory.CONTRAINDICATION,
        "cross_section_comparison": EvalCategory.CROSS_SECTION_COMPARISON,
        "multi_hop_reasoning": EvalCategory.MULTI_HOP_REASONING,
    }
    for j, obj in enumerate(objects):
        if len(out) >= max_per_chunk:
            break
        q = str(obj.get("question", "")).strip()
        a = str(obj.get("answer", "")).strip()
        kps = obj.get("keyphrases") or []
        if not isinstance(kps, list):
            continue
        kp_strs = [str(k).strip() for k in kps if str(k).strip()]
        diff = difficulty_map.get(str(obj.get("difficulty", "")).lower(), EvalDifficulty.MEDIUM)
        cat = category_map.get(str(obj.get("category", "")).lower(), EvalCategory.FACTUAL_RECALL)
        if len(q) < 8 or len(a) < 20:
            continue
        if not _answer_is_verbatim(a, chunk.text):
            continue
        if not _keyphrases_in_text(kp_strs, chunk.text) or not _keyphrases_in_text(kp_strs, a):
            continue
        qid = f"q_{start_qid + idx * max_per_chunk + j:04d}"
        out.append(EvalQuestion(
            id=qid,
            category=cat,
            difficulty=diff,
            question=q,
            relevant_chunk_ids=[chunk.chunk_id],
            relevant_page_numbers=list(chunk.page_numbers),
            acceptable_answers=[a],
            required_keyphrases=kp_strs,
            forbidden_keyphrases=[],
            source_type=EvalSource.LLM_SILVER_DRAFT,
            notes=(
                f"LLM-generated from chunk {chunk.chunk_id} "
                f"({chunk.file_name}, pages {chunk.page_numbers}); "
                f"REQUIRES HUMAN REVIEW before treating as gold."
            ),
            metadata={
                "source_chunk_id": chunk.chunk_id,
                "source_document_id": chunk.document_id,
                "source_file_name": chunk.file_name,
                "generated_by": "llm_pass2",
                "source_text_excerpt": chunk.text[:1200],
            },
        ))
    return out


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text or "") if s.strip()]


def _extract_top_keyphrases(text: str, max_kp: int = 6) -> List[str]:
    tokens = meaningful_tokens(text)
    counts: Dict[str, int] = {}
    for t in tokens:
        if len(t) < 3 or t in _EVAL_GENERIC_KEYPHRASES:
            continue
        counts[t] = counts.get(t, 0) + 1
    sorted_tokens = sorted(counts.items(), key=lambda kv: (-kv[1], -len(kv[0])))
    return [t for t, _ in sorted_tokens[:max_kp]]


def _template_pairs_from_chunk(
    chunk: SampledChunk, idx: int, start_qid: int,
) -> List[EvalQuestion]:
    sentences = _split_sentences(chunk.text)
    if len(sentences) < 2:
        return []
    out: List[EvalQuestion] = []
    templates = [
        (
            "What does the text explain about {kp0}?",
            EvalDifficulty.EASY,
            EvalCategory.FACTUAL_RECALL,
            slice(0, min(3, len(sentences))),
        ),
        (
            "How are {kp1} and {kp2} related in the text?",
            EvalDifficulty.MEDIUM,
            EvalCategory.FACTUAL_RECALL,
            slice(max(0, len(sentences) // 3),
                  min(len(sentences), len(sentences) // 3 + 4)),
        ),
    ]
    kps = _extract_top_keyphrases(chunk.text, max_kp=5)
    if len(kps) < 2:
        kps = (kps + ["content", "information", "recommendation"])[:5]
    for pidx, (template, diff, cat, sent_slice) in enumerate(templates):
        answer_sentences = sentences[sent_slice]
        if not answer_sentences:
            continue
        answer = answer_sentences[0].strip()
        if len(answer.split()) < 40 and len(answer_sentences) > 1:
            answer = " ".join(answer_sentences[:2]).strip()
        if len(answer.split()) < 40 or len(answer.split()) > 80:
            continue
        try:
            question = template.format(
                kp0=kps[0], kp1=kps[1], kp2=kps[2] if len(kps) > 2 else kps[0],
            )
        except (IndexError, KeyError):
            continue
        qid = f"q_{start_qid + idx * 2 + pidx:04d}"
        required = _extract_top_keyphrases(answer, max_kp=4)
        norm_answer = re.sub(r"\s+", " ", answer).strip().casefold()
        norm_chunk = re.sub(r"\s+", " ", chunk.text).strip().casefold()
        if norm_answer not in norm_chunk:
            continue
        out.append(EvalQuestion(
            id=qid,
            category=cat,
            difficulty=diff,
            question=question,
            relevant_chunk_ids=[chunk.chunk_id],
            relevant_page_numbers=list(chunk.page_numbers),
            acceptable_answers=[answer],
            required_keyphrases=required,
            forbidden_keyphrases=[],
            source_type=EvalSource.LLM_SILVER_DRAFT,
            notes=(
                f"Template-generated from chunk {chunk.chunk_id} "
                f"({chunk.file_name}, pages {chunk.page_numbers}); "
                f"ground-truth answer is verbatim substring of source chunk. "
                f"REQUIRES HUMAN REVIEW: rewrite the question to be natural, "
                f"verify page numbers, then set source_type=llm_silver_reviewed."
            ),
            metadata={
                "source_chunk_id": chunk.chunk_id,
                "source_document_id": chunk.document_id,
                "source_file_name": chunk.file_name,
                "generated_by": "template_pass1",
                "source_text_excerpt": chunk.text[:1200],
            },
        ))
    return out


def _template_table_pairs(
    chunk: SampledChunk, idx: int, start_qid: int,
) -> List[EvalQuestion]:
    lines = [ln.strip() for ln in chunk.text.splitlines() if " | " in ln.strip()]
    if len(lines) < 2:
        return []
    kps = _extract_top_keyphrases(chunk.text, max_kp=5)
    qid = f"q_{start_qid + idx:04d}"
    answer = "\n".join(lines[: min(len(lines), 6)])
    if len(answer) < 30:
        return []
    first_line_cells = [c.strip() for c in lines[0].split("|") if c.strip()]
    topic = " vs ".join(first_line_cells[:2]) if len(first_line_cells) >= 2 else (
        kps[0] if kps else "parameters"
    )
    question = (
        f"From the table comparing {topic} (pages {chunk.page_numbers}), "
        f"what are the tabulated values for {kps[0] if kps else 'each entry'}?"
    )
    required = kps[:3]
    return [EvalQuestion(
        id=qid,
        category=EvalCategory.TABLE_LOOKUP,
        difficulty=EvalDifficulty.HARD,
        question=question,
        relevant_chunk_ids=[chunk.chunk_id],
        relevant_page_numbers=list(chunk.page_numbers),
        acceptable_answers=[answer],
        required_keyphrases=required,
        forbidden_keyphrases=[],
        source_type=EvalSource.LLM_SILVER_DRAFT,
        notes=(
            f"Template-generated TABLE_LOOKUP from chunk {chunk.chunk_id} "
            f"({chunk.file_name}, pages {chunk.page_numbers}). "
            f"REQUIRES HUMAN REVIEW: rewrite to ask a specific cell/row question, "
            f"then set source_type=llm_silver_reviewed."
        ),
        metadata={
            "source_chunk_id": chunk.chunk_id,
            "source_document_id": chunk.document_id,
            "source_file_name": chunk.file_name,
            "generated_by": "template_table_pass1",
            "source_text_excerpt": chunk.text[:1200],
        },
    )]


MANUAL_TEMPLATE_NOTES = (
    "Manual adversarial slot: fill question, acceptable_answers, "
    "required_keyphrases, relevant_page_numbers. Then change source_type "
    "from adversarial_manual to human_curated."
)


def _build_manual_placeholders(start_id: int, count: int = 30) -> List[EvalQuestion]:
    questions: List[EvalQuestion] = []
    for i in range(count):
        qid = f"q_manual_{start_id + i:04d}"
        if i < 10:
            category = EvalCategory.TABLE_LOOKUP
            difficulty = EvalDifficulty.HARD
            notes = (
                f"{MANUAL_TEMPLATE_NOTES} Type: numeric/table lookup. "
                f"Target: specific HbA1c, BP, BMI thresholds, dosages, "
                f"or lab value rows from tables."
            )
        elif i < 20:
            category = EvalCategory.MULTI_HOP_REASONING
            difficulty = EvalDifficulty.HARD
            notes = (
                f"{MANUAL_TEMPLATE_NOTES} Type: cross-section synthesis. "
                f"Target: link concepts from 2+ distinct chapters/sections "
                f"(e.g. compare 2 drug classes on contraindications)."
            )
        else:
            category = EvalCategory.NEGATIVE_CONTROL
            difficulty = EvalDifficulty.EASY
            notes = (
                f"{MANUAL_TEMPLATE_NOTES} Type: no-answer-in-corpus negative. "
                f"Target: post-2024 guidelines/therapies not in this 9th edition."
            )
        placeholder = {
            EvalCategory.TABLE_LOOKUP: "numeric table question",
            EvalCategory.MULTI_HOP_REASONING: "cross-section synthesis question",
            EvalCategory.NEGATIVE_CONTROL: "impossible/out-of-corpus question",
        }[category]
        questions.append(EvalQuestion(
            id=qid,
            category=category,
            difficulty=difficulty,
            question=f"(FILL IN MANUALLY: {placeholder})",
            relevant_chunk_ids=[],
            relevant_page_numbers=[],
            acceptable_answers=["(FILL IN MANUALLY: reference answer(s))"],
            required_keyphrases=[],
            forbidden_keyphrases=[],
            source_type=EvalSource.ADVERSARIAL_MANUAL,
            notes=notes,
            metadata={"manual_slot": True, "slot_index": i},
        ))
    return questions


def _dedupe_similar(questions: List[EvalQuestion],
                   jaccard_threshold: float = 0.7) -> List[EvalQuestion]:
    kept: List[EvalQuestion] = []
    for q in questions:
        q_tokens = set(meaningful_tokens(q.question))
        duplicate = False
        for kept_q in kept:
            k_tokens = set(meaningful_tokens(kept_q.question))
            if not q_tokens or not k_tokens:
                continue
            inter = len(q_tokens & k_tokens)
            union = len(q_tokens | k_tokens)
            if union > 0 and inter / union > jaccard_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(q)
    return kept


def generate_silver_dataset(
    settings: Settings | None = None,
    target_silver: int = 70,
    target_table: int = 10,
    manual_slots: int = 30,
    seed: int = 42,
    version: str = EVAL_VERSION,
    overwrite: bool = False,
    use_llm: bool = False,
) -> Tuple[List[EvalQuestion], Dict[str, Any]]:
    settings = settings or Settings.from_env()
    log_dir = settings.log_dir
    logger = build_logger("eval_silver_gen", log_dir)
    rng = random.Random(seed)
    vector_db_dir = settings.vector_db_dir

    logger.info("Sampling chunks from vector store at %s", vector_db_dir)
    chunks = _sample_chunks_from_store(
        vector_db_dir, count=max(target_silver // 2, 40), rng=rng
    )
    table_chunks = _sample_table_chunks(
        vector_db_dir, count=max(target_table * 2, 15), rng=rng
    )
    logger.info("Sampled %d general chunks, %d table-heavy chunks",
                len(chunks), len(table_chunks))
    if not chunks and not table_chunks:
        raise RuntimeError(
            "No READY chunks found in the vector store. Ingest at least one "
            "PDF before generating the eval set."
        )

    silver: List[EvalQuestion] = []
    start_qid = 1
    llm_errors = 0
    validation_rejections = 0
    llm = None

    if use_llm:
        llm = OllamaLLMClient(
            base_url=settings.ollama_base_url,
            model=settings.generation_model,
            timeout_seconds=max(settings.generation_timeout_seconds, 120.0),
        )
        for idx, chunk in enumerate(chunks):
            if len(silver) >= target_silver:
                break
            prompt = SILVER_USER_TEMPLATE.format(
                page=chunk.page_numbers[0] if chunk.page_numbers else "?",
                filename=chunk.file_name,
                chunk_text=chunk.text[:3500],
            )
            try:
                raw = llm.generate(prompt, system_prompt=SILVER_SYSTEM_PROMPT, temperature=0.3)
            except RuntimeError as exc:
                llm_errors += 1
                logger.warning("LLM failed for chunk %s (%s), falling back to template",
                               chunk.chunk_id, exc)
                silver.extend(_template_pairs_from_chunk(chunk, idx, start_qid))
                continue
            pairs = _validated_pairs(raw, chunk, idx, start_qid)
            if not pairs:
                validation_rejections += 1
                silver.extend(_template_pairs_from_chunk(chunk, idx, start_qid))
                continue
            silver.extend(pairs)
        logger.info("LLM pass yielded %d valid silver pairs (LLM errs=%d, rejects=%d)",
                    len(silver), llm_errors, validation_rejections)
    else:
        for idx, chunk in enumerate(chunks):
            if len(silver) >= target_silver:
                break
            silver.extend(_template_pairs_from_chunk(chunk, idx, start_qid))
        logger.info("Template pass produced %d factual_recall silver drafts",
                    len(silver))

    table_pairs: List[EvalQuestion] = []
    table_start = start_qid + 1000
    for idx, chunk in enumerate(table_chunks):
        if len(table_pairs) >= target_table:
            break
        used_llm = False
        if use_llm and llm is not None:
            prompt = SILVER_USER_TEMPLATE.format(
                page=chunk.page_numbers[0] if chunk.page_numbers else "?",
                filename=chunk.file_name,
                chunk_text=chunk.text[:3500],
            )
            try:
                raw = llm.generate(prompt, system_prompt=TABLE_SYSTEM_PROMPT, temperature=0.2)
                pairs = _validated_pairs(raw, chunk, idx, table_start, max_per_chunk=1)
                for p in pairs:
                    p.category = EvalCategory.TABLE_LOOKUP
                    p.difficulty = EvalDifficulty.HARD
                if pairs:
                    table_pairs.extend(pairs)
                    used_llm = True
            except RuntimeError:
                pass
        if not used_llm:
            table_pairs.extend(_template_table_pairs(chunk, idx, table_start))
    logger.info("Table-heavy chunks produced %d table_lookup silver drafts",
                len(table_pairs))

    while len(silver) < target_silver:
        extra_needed = max((target_silver - len(silver) + 1) // 2, 5)
        more = _sample_chunks_from_store(
            vector_db_dir, count=extra_needed, rng=rng
        )
        if not more:
            break
        seen_chunk_ids = {
            q.metadata.get("source_chunk_id") for q in silver + table_pairs
        }
        new_added = 0
        for j, chunk in enumerate(more):
            if chunk.chunk_id in seen_chunk_ids:
                continue
            draft_idx = 10000 + len(silver) + j
            for p in _template_pairs_from_chunk(chunk, draft_idx, start_qid + 5000):
                silver.append(p)
                new_added += 1
                if len(silver) >= target_silver:
                    break
            if len(silver) >= target_silver:
                break
        if new_added == 0:
            break

    silver_deduped = _dedupe_similar(silver + table_pairs)
    if len(silver_deduped) > target_silver + target_table:
        silver_deduped = silver_deduped[: target_silver + target_table]
    manual = _build_manual_placeholders(start_id=2000, count=manual_slots)
    all_questions: List[EvalQuestion] = []
    seen_question_ids: set[str] = set()
    for question in silver_deduped + manual:
        if question.id in seen_question_ids:
            continue
        seen_question_ids.add(question.id)
        all_questions.append(question)
    if overwrite:
        path = save_dataset(all_questions, version=version)
    else:
        path = append_questions(all_questions, version=version)
    summary = dataset_summary(load_dataset(version=version))
    logger.info("Dataset written to %s", path)
    logger.info("Summary: %s", json.dumps(summary, indent=2, ensure_ascii=False))
    stats = {
        "silver_generated": len(silver_deduped),
        "silver_factual": sum(
            1 for q in silver_deduped if q.category == EvalCategory.FACTUAL_RECALL
        ),
        "silver_table": sum(
            1 for q in silver_deduped if q.category == EvalCategory.TABLE_LOOKUP
        ),
        "manual_slots": len(manual),
        "manual_table_slots": sum(
            1 for q in manual if q.category == EvalCategory.TABLE_LOOKUP
        ),
        "manual_multihop_slots": sum(
            1 for q in manual if q.category == EvalCategory.MULTI_HOP_REASONING
        ),
        "manual_negative_slots": sum(
            1 for q in manual if q.category == EvalCategory.NEGATIVE_CONTROL
        ),
        "llm_used": use_llm,
        "llm_errors": llm_errors,
        "validation_rejections": validation_rejections,
        "dataset_path": str(path),
        "next_steps": (
            "1) Review llm_silver_draft rows; rewrite questions to be natural, "
            "verify page numbers; set source_type to llm_silver_reviewed. "
            "2) Fill in 30 manual adversarial slots (q_manual_*). "
            "3) Run `python -m rag_project.evaluation --label baseline` "
            "with --llm-judge for baseline calibration."
        ),
    }
    return all_questions, stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate silver Q&A drafts + manual adversarial slots for eval set v1."
    )
    parser.add_argument("--silver", type=int, default=70,
                        help="Target silver pairs (before dedupe)")
    parser.add_argument("--table", type=int, default=10,
                        help="Target table-lookup silver pairs")
    parser.add_argument("--manual", type=int, default=30,
                        help="Number of manual adversarial slots to insert")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", type=str, default=EVAL_VERSION)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite dataset instead of appending")
    parser.add_argument("--use-llm", action="store_true",
                        help="Slow: call Ollama per chunk (else fast template-based pass)")
    parser.add_argument("--summary", action="store_true",
                        help="Print current dataset summary and exit")
    args = parser.parse_args(argv)

    if args.summary:
        questions = load_dataset(version=args.version)
        print(json.dumps(dataset_summary(questions), indent=2, ensure_ascii=False))
        return 0

    _, stats = generate_silver_dataset(
        target_silver=args.silver,
        target_table=args.table,
        manual_slots=args.manual,
        seed=args.seed,
        version=args.version,
        overwrite=args.overwrite,
        use_llm=args.use_llm,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
