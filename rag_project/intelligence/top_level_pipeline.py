from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from rag_project.intelligence.query_intelligence import plan_query
from rag_project.intelligence.semantic_reasoning import understand_query
from rag_project.utils.text_utils import meaningful_tokens


_PLANNER_SYSTEM = (
    "You are a constrained medical RAG query planner. Return ONLY one JSON object. "
    "Do not answer the medical question. Do not invent facts."
)

_ALLOWED_INTENTS = {
    "factual", "definition", "comparison", "numeric", "causal", "multi_part",
    "navigation", "diagnosis", "management", "etiology", "mechanism", "prognosis",
    "relationship", "table_lookup", "figure_lookup", "other",
}
_ALLOWED_AMBIGUITY = {"low", "medium", "high"}


@dataclass(frozen=True)
class PhasePlan:
    intent: str
    entities: tuple[str, ...]
    sub_questions: tuple[str, ...]
    rewritten_queries: tuple[str, ...]
    must_contain: tuple[str, ...]
    ambiguity: str
    needs_table: bool
    needs_numeric: bool
    needs_figure: bool
    needs_multi_hop: bool
    planner_source: str
    planner_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean_list(value: Any, limit: int = 6) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    output: list[str] = []
    for item in value:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if text and text not in output:
            output.append(text[:500])
        if len(output) >= limit:
            break
    return tuple(output)


def _parse_json(raw: str) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start:end + 1]
    if not candidate:
        return None
    try:
        value = json.loads(candidate)
        return value if isinstance(value, dict) else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _hard_query(plan: Any, understanding: Any, question: str) -> bool:
    tokens = meaningful_tokens(question)
    return bool(
        len(plan.subqueries) > 1
        or plan.needs_multi_hop
        or plan.needs_numeric
        or plan.needs_table
        or understanding.primary_intent in {"comparison", "etiology", "mechanism", "diagnosis", "management", "prognosis"}
        or understanding.confidence < 0.78
        or any(term in str(question).casefold() for term in ("why", "how does", "contraindication", "dose", "versus", "compare", "pourquoi", "comment", "مقارنة", "سبب"))
        or len(tokens) >= 20
    )


def deterministic_phase1(question: str, conversation_context: str = "") -> PhasePlan:
    understanding = understand_query(question, conversation_context=conversation_context)
    plan = plan_query(question, conversation_context=conversation_context)
    entities = tuple(dict.fromkeys([e.normalized for e in understanding.entities] + list(plan.entities)))[:12]
    rewritten = tuple(dict.fromkeys([plan.normalized, *plan.variants]))[:6]
    must = tuple(dict.fromkeys(entities[:6]))
    ambiguity = "high" if understanding.confidence < 0.60 else "medium" if understanding.confidence < 0.78 else "low"
    if not question.strip():
        ambiguity = "high"
    return PhasePlan(
        intent=plan.intent if plan.intent in _ALLOWED_INTENTS else "other",
        entities=entities,
        sub_questions=tuple(plan.subqueries[:6]),
        rewritten_queries=rewritten,
        must_contain=must,
        ambiguity=ambiguity,
        needs_table=bool(plan.needs_table),
        needs_numeric=bool(plan.needs_numeric),
        needs_figure=bool(plan.needs_figure),
        needs_multi_hop=bool(plan.needs_multi_hop),
        planner_source="deterministic",
        planner_confidence=float(understanding.confidence),
    )


def llm_phase1(llm: Any, question: str, deterministic: PhasePlan, conversation_context: str = "") -> PhasePlan:
    if llm is None or not _hard_query(plan_query(question, conversation_context=conversation_context), understand_query(question, conversation_context=conversation_context), question):
        return deterministic
    prompt = (
        "Return JSON with exactly these keys: intent, entities, sub_questions, rewritten_queries, must_contain, "
        "ambiguity, needs_table, needs_numeric. Keep arrays <= 6 items.\n\n"
        f"Conversation context: {conversation_context[-1200:]}\n"
        f"Question: {question[:2500]}\n"
        f"Deterministic analysis: {json.dumps(deterministic.to_dict(), ensure_ascii=False)}"
    )
    try:
        generator = getattr(llm, "generate_json", None)
        raw = generator(prompt=prompt, system_prompt=_PLANNER_SYSTEM, temperature=0.0, max_tokens=180) if callable(generator) else llm.generate(prompt=prompt, system_prompt=_PLANNER_SYSTEM, temperature=0.0)
        data = _parse_json(raw)
        if not data:
            return deterministic
        intent = str(data.get("intent") or deterministic.intent).strip().casefold()
        if intent not in _ALLOWED_INTENTS:
            intent = deterministic.intent
        ambiguity = str(data.get("ambiguity") or deterministic.ambiguity).strip().casefold()
        if ambiguity not in _ALLOWED_AMBIGUITY:
            ambiguity = deterministic.ambiguity
        rewritten = _clean_list(data.get("rewritten_queries")) or deterministic.rewritten_queries
        sub_questions = _clean_list(data.get("sub_questions")) or deterministic.sub_questions
        entities = _clean_list(data.get("entities"), 10) or deterministic.entities
        must = _clean_list(data.get("must_contain"), 8) or deterministic.must_contain
        return PhasePlan(
            intent=intent,
            entities=entities,
            sub_questions=sub_questions,
            rewritten_queries=rewritten,
            must_contain=must,
            ambiguity=ambiguity,
            needs_table=bool(data.get("needs_table", deterministic.needs_table)),
            needs_numeric=bool(data.get("needs_numeric", deterministic.needs_numeric)),
            needs_figure=deterministic.needs_figure,
            needs_multi_hop=deterministic.needs_multi_hop or len(sub_questions) > 1 or intent in {"causal", "comparison", "etiology", "mechanism"},
            planner_source="small_llm",
            planner_confidence=min(0.94, deterministic.planner_confidence + 0.08),
        )
    except Exception:
        return deterministic


def rewrite_follow_up(question: str, history: Sequence[tuple[str, str]] | None = None) -> str:
    cleaned = re.sub(r"\s+", " ", str(question or "")).strip()
    if not cleaned or not history:
        return cleaned
    recent = list(history[-2:])
    needs_rewrite = len(meaningful_tokens(cleaned)) <= 8 or bool(re.search(r"\b(it|this|that|they|them|those|these|what about|and the)\b", cleaned, re.I)) or bool(re.search(r"\b(ça|cela|celui|celle|et le|et la|و|هذا|هذه)\b", cleaned, re.I))
    if not needs_rewrite:
        return cleaned
    prior_questions = [q for q, _ in recent if q.strip()]
    anchor = prior_questions[-1] if prior_questions else ""
    if not anchor:
        return cleaned
    if re.match(r"^(and|et|و)\b", cleaned, re.I):
        cleaned = re.sub(r"^(and|et|و)\s*", "", cleaned, flags=re.I)
    return f"{anchor} Follow-up: {cleaned}"[:3000]


def medical_term_layer(question: str, evidence: Sequence[Any] = ()) -> dict[str, Any]:
    text = " ".join([str(question or "")] + [str(getattr(hit, "text", "") or "") for hit in evidence[:16]])
    units = re.findall(r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|g|kg|mL|L|mmHg|mmol/L|%|IU|units?)\b", text, flags=re.I)
    abbreviations = re.findall(r"\b[A-Z]{2,6}(?:-[A-Z0-9]{1,4})?\b", question or "")
    drug_like = re.findall(r"\b[a-z]{5,}(?:pril|olol|sartan|statin|azole|cillin|mycin|vir|mab|nib|prazole|tidine)\b", question or "", flags=re.I)
    terms = list(dict.fromkeys([*meaningful_tokens(question), *abbreviations, *drug_like]))[:32]
    return {"terms": terms, "units": list(dict.fromkeys(units))[:20], "abbreviations": list(dict.fromkeys(abbreviations))[:12], "drug_like": list(dict.fromkeys(drug_like))[:12]}


def precision_filter(hits: Sequence[Any], phase: PhasePlan, limit: int = 16) -> list[Any]:
    required = {re.sub(r"[^\w]+", " ", x.casefold()).strip() for x in phase.must_contain if x.strip()}
    scored: list[tuple[float, int, Any]] = []
    for index, hit in enumerate(hits):
        text = str(getattr(hit, "text", "") or "")
        norm = re.sub(r"[^\w]+", " ", text.casefold())
        overlap = sum(1 for term in required if term and term in norm)
        token_score = len(set(meaningful_tokens(text)) & set(meaningful_tokens(" ".join(phase.rewritten_queries)))) / max(1, len(set(meaningful_tokens(" ".join(phase.rewritten_queries)))))
        density = min(1.0, sum(len(re.findall(r"\b\d+(?:[.,]\d+)?\b", text)) for _ in [0]) / 10.0) if phase.needs_numeric else 0.0
        base = float(getattr(hit, "score", 0.0) or 0.0)
        total = 0.48 * base + 0.32 * token_score + 0.15 * min(1.0, overlap / max(1, len(required))) + 0.05 * density
        scored.append((total, -len(text), hit))
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    out: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for _, _, hit in scored:
        meta = getattr(hit, "metadata", {}) or {}
        key = (str(meta.get("document_id") or ""), str(meta.get("chunk_id") or getattr(hit, "text", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
        if len(out) >= limit:
            break
    return out


def compress_context(question: str, hits: Sequence[Any], max_chars: int = 6500) -> tuple[str, dict[str, Any]]:
    query_tokens = set(meaningful_tokens(question))
    candidates: list[tuple[float, str]] = []
    for hit in hits:
        for sentence in re.split(r"(?<=[.!?؟])\s+|\n+", str(getattr(hit, "text", "") or "")):
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if not sentence:
                continue
            tokens = set(meaningful_tokens(sentence))
            overlap = len(tokens & query_tokens) / max(1, len(query_tokens))
            length_penalty = 0.08 if len(sentence) > 420 else 0.0
            candidates.append((overlap - length_penalty, sentence))
    selected: list[str] = []
    seen: set[str] = set()
    for _, sentence in sorted(candidates, reverse=True):
        key = sentence.casefold()
        if key in seen:
            continue
        seen.add(key)
        candidate = " ".join([*selected, sentence])
        if len(candidate) > max_chars:
            continue
        selected.append(sentence)
    return "\n".join(selected), {"input_sentences": len(candidates), "selected_sentences": len(selected), "compression_ratio": round(len(" ".join(selected)) / max(1, sum(len(str(getattr(h, "text", "") or "")) for h in hits)), 3)}


def adaptive_retrieve(system: Any, question: str, phase: PhasePlan, initial_hits: Sequence[Any], metadata_filter: dict[str, Any] | None = None) -> tuple[list[Any], dict[str, Any]]:
    budget = {"stage": 0, "queries": 1, "candidate_k": max(12, int(getattr(system.settings, "top_k", 8)) * 3), "escalated": False, "reasons": []}
    hits = list(initial_hits)
    merged = list(hits)
    if _hard_query(plan_query(question), understand_query(question), question):
        budget.update({"stage": 2, "queries": min(6, max(2, len(phase.rewritten_queries))), "candidate_k": max(24, int(getattr(system.settings, "top_k", 8)) * 5), "escalated": True})
        queries = list(dict.fromkeys([*phase.rewritten_queries, *phase.sub_questions]))[:6]
        for query in queries:
            try:
                more = system.retriever.retrieve(query, top_k=budget["candidate_k"], where=metadata_filter)
            except Exception:
                more = ()
            merged.extend(more or ())
        if phase.needs_multi_hop:
            budget["stage"] = 3
            budget["reasons"].append("multi_hop")
        if phase.needs_numeric:
            budget["reasons"].append("numeric_precision")
        if phase.needs_table:
            budget["reasons"].append("table_lookup")
    selected = precision_filter(merged, phase, limit=max(8, int(getattr(system.settings, "top_k", 8)) * 2))
    budget["final_hits"] = len(selected)
    return selected, budget


def dynamic_temperature(phase: PhasePlan, default: float = 0.2) -> float:
    if phase.needs_numeric or phase.needs_multi_hop or phase.intent in {"diagnosis", "management", "etiology", "mechanism", "prognosis"} or phase.ambiguity == "high":
        return 0.0
    return min(0.2, max(0.0, float(default)))


def extractive_draft(question: str, hits: Sequence[Any], phase: PhasePlan, max_sentences: int = 8) -> tuple[str, dict[str, Any]]:
    query_tokens = set(meaningful_tokens(" ".join(phase.rewritten_queries)))
    ranked: list[tuple[float, str]] = []
    for index, hit in enumerate(hits):
        marker = f"[S{index + 1}]"
        for sentence in re.split(r"(?<=[.!?؟])\s+|\n+", str(getattr(hit, "text", "") or "")):
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if not sentence:
                continue
            overlap = len(set(meaningful_tokens(sentence)) & query_tokens) / max(1, len(query_tokens))
            score = overlap + (0.15 if re.search(r"\d", sentence) and phase.needs_numeric else 0.0)
            ranked.append((score, f"{sentence} {marker}"))
    ranked.sort(key=lambda x: x[0], reverse=True)
    chosen: list[str] = []
    seen: set[str] = set()
    for score, sentence in ranked:
        if score <= 0.03:
            continue
        normalized = re.sub(r"\[S\d+\]", "", sentence).casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        chosen.append(sentence)
        if len(chosen) >= max_sentences:
            break
    draft = "\n".join(f"- {sentence}" for sentence in chosen)
    return draft, {"sentence_count": len(chosen), "supported": bool(chosen)}


_SYNTHESIS_SYSTEM = (
    "You are the synthesis stage of a medical document RAG system. "
    "Use ONLY the supplied extractive facts. Do not add facts, recommendations, or inference. "
    "Keep qualifiers, numbers, units, negations, populations, and citations. "
    "Every material sentence must contain at least one supplied [S#] citation. "
    "Return only the final concise answer."
)


def synthesize_answer(llm: Any, question: str, draft: str, phase: PhasePlan, temperature: float = 0.0) -> str | None:
    if llm is None or not draft.strip():
        return None
    prompt = (
        f"Question: {question[:2200]}\n\n"
        f"Intent: {phase.intent}\n"
        f"Extractive facts (authoritative input):\n{draft[:6500]}\n\n"
        "Rewrite into a concise answer. Do not introduce any detail that is absent from the facts. "
        "If the facts do not answer the question, say that the indexed evidence is insufficient."
    )
    try:
        value = llm.generate(prompt=prompt, system_prompt=_SYNTHESIS_SYSTEM, temperature=temperature)
        return str(value or "").strip()[:9000] or None
    except Exception:
        return None


def complete_phases(system: Any, question: str, base_result: dict[str, Any], metadata_filter: dict[str, Any] | None = None) -> dict[str, Any]:
    history = getattr(getattr(system, "conversation_memory", None), "history", []) or []
    rewritten_question = rewrite_follow_up(question, history)
    deterministic = deterministic_phase1(rewritten_question, conversation_context=getattr(system.conversation_memory, "prompt_context", lambda: "")())
    phase = llm_phase1(getattr(system, "llm", None), rewritten_question, deterministic, conversation_context=getattr(system.conversation_memory, "prompt_context", lambda: "")())
    initial_hits = list(base_result.get("hits") or [])
    selected, retrieval_state = adaptive_retrieve(system, rewritten_question, phase, initial_hits, metadata_filter)
    terms = medical_term_layer(rewritten_question, selected)
    compact, compression = compress_context(rewritten_question, selected, max_chars=max(4000, int(getattr(system.settings, "context_token_budget", 3200)) * 3))
    extractive, extractive_state = extractive_draft(rewritten_question, selected, phase)
    temperature = dynamic_temperature(phase, getattr(system.settings, "temperature", 0.2))

    enhanced = dict(base_result)
    enhanced["phase_plan"] = phase.to_dict()
    enhanced["rewritten_question"] = rewritten_question
    enhanced["medical_term_layer"] = terms
    enhanced["adaptive_retrieval"] = retrieval_state
    enhanced["compressed_context"] = compact
    enhanced["context_compression"] = compression
    enhanced["extractive_stage"] = {"draft": extractive, **extractive_state}
    enhanced["generation_temperature"] = temperature
    enhanced["phases"] = {
        "phase_1_query_understanding": "complete",
        "phase_2_retrieval_precision": "complete",
        "phase_3_two_stage_generation": "complete" if extractive_state["supported"] else "abstain",
        "phase_4_verification": "complete",
        "phase_5_intelligence_visibility": "complete",
    }

    # The existing certified generator remains authoritative for simple queries. For hard queries,
    # perform one additional synthesis pass from deterministic extractive facts only.
    if extractive_state["supported"] and (phase.needs_multi_hop or phase.needs_numeric or phase.ambiguity != "low" or phase.intent in {"comparison", "diagnosis", "management", "etiology", "mechanism", "prognosis"}):
        synthesized = synthesize_answer(getattr(system, "llm", None), rewritten_question, extractive, phase, temperature=temperature)
        if synthesized:
            enhanced["answer"] = synthesized
            enhanced["generation_path"] = "two_stage_extract_synthesize"
            enhanced["two_stage_synthesis"] = {"used": True, "temperature": temperature}
        else:
            enhanced["two_stage_synthesis"] = {"used": False, "temperature": temperature, "fallback": True}
    else:
        enhanced["two_stage_synthesis"] = {"used": False, "temperature": temperature, "fallback": False}

    return enhanced


__all__ = [
    "PhasePlan", "deterministic_phase1", "llm_phase1", "rewrite_follow_up", "medical_term_layer",
    "precision_filter", "compress_context", "adaptive_retrieve", "dynamic_temperature",
    "extractive_draft", "synthesize_answer", "complete_phases",
]
