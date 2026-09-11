"""MedEvidence Pro: evidence-first cascade runtime.

This module is the single production answer engine.  It deliberately keeps
routing, retrieval policy, evidence compression, answer generation, verification,
formatting, and feedback in one deterministic orchestration layer so the small
local LLM is used only when the evidence requires synthesis.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from rag_project.intelligence.evidence_guard import grounding_decision, verify_claims
from rag_project.intelligence.final_answer_contract import verify_final_answer
from rag_project.intelligence.query_intelligence import plan_query
from rag_project.intelligence.semantic_reasoning import extract_clinical_entities, understand_query
from rag_project.retrieval.hybrid_retriever import RetrievalHit
from rag_project.utils.text_utils import meaningful_tokens


MEDICAL_TERMS = {
    "drug", "medicine", "medication", "dose", "dosage", "treatment", "therapy", "symptom",
    "disease", "condition", "diagnosis", "patient", "clinical", "medical", "health", "syndrome",
    "hypertension", "diabetes", "infection", "cancer", "anemia", "pain", "fever", "heart", "kidney",
    "liver", "lung", "blood", "pressure", "pregnancy", "child", "pediatric", "adult", "contraindication",
    "interaction", "side effect", "adverse", "prognosis", "mechanism", "pathophysiology", "anatomy",
    "physiology", "laboratory", "lab", "mg", "mcg", "ml", "mmhg", "bpm", "ecg", "ct", "mri", "xray",
    "hb", "hba1c", "egfr", "gfr", "nsaid", "antibiotic", "vaccine", "imaging", "screening", "surgery",
}

SYNONYMS = {
    "htn": ("hypertension", "high blood pressure"),
    "mi": ("myocardial infarction", "heart attack", "acute myocardial infarction"),
    "dm2": ("type 2 diabetes", "type 2 diabetes mellitus", "t2dm"),
    "hba1c": ("hemoglobin a1c", "glycated hemoglobin", "a1c"),
    "ckd": ("chronic kidney disease", "renal impairment", "kidney disease"),
    "gfr": ("glomerular filtration rate", "egfr", "estimated glomerular filtration rate"),
    "chf": ("heart failure", "congestive heart failure"),
    "nsaid": ("nonsteroidal anti-inflammatory drug", "nonsteroidal anti-inflammatory drugs"),
    "copd": ("chronic obstructive pulmonary disease" ,),
    "pe": ("pulmonary embolism",),
    "dvt": ("deep vein thrombosis",),
    "aki": ("acute kidney injury",),
    "tb": ("tuberculosis",),
    "uti": ("urinary tract infection",),
    "bid": ("twice daily", "two times a day"),
    "tid": ("three times daily", "three times a day"),
    "qid": ("four times daily", "four times a day"),
    "iv": ("intravenous",),
    "im": ("intramuscular",),
    "po": ("oral", "by mouth"),
}

BLOCK_PATTERNS = (
    r"\bhow to (?:make|synthesize|manufacture|cook)\b.*\b(?:drug|meth|heroin|cocaine|fentanyl)\b",
    r"\b(?:synthesize|manufacture|produce|cook)\b.*\b(?:opioid|amphetamine|methamphetamine|heroin|cocaine)\b",
    r"\b(?:help|instructions?|steps?)\b.*\b(?:kill myself|suicide|self[- ]harm)\b",
    r"\b(?:how|ways?)\b.*\b(?:hurt|poison|kill)\b.*\b(?:someone|person)\b",
)
EMERGENCY_TERMS = (
    "can't breathe", "cannot breathe", "severe chest pain", "chest pain", "anaphylaxis",
    "overdose", "poisoning", "poisoned", "unconscious", "seizure", "stroke symptoms",
    "major bleeding", "heavy bleeding", "suicidal", "suicide attempt", "difficulty breathing",
    "ne peut pas respirer", "douleur thoracique sévère", "surdosage", "anaphylaxie",
    "لا أستطيع التنفس", "ألم صدر شديد", "جرعة زائدة", "تسمم", "نزيف شديد",
)
REAL_PATIENT_HINTS = ("i have", "my symptoms", "my patient", "patient has", "j'ai", "mon patient", "لدي", "عندي")
HIGH_RIGOR_HINTS = ("pregnant", "pregnancy", "pediatric", "child", "infant", "newborn", "renal failure", "kidney failure", "pregnancy", "حامل", "طفل", "رضيع")


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _hash_query(text: str) -> str:
    return hashlib.sha256(_norm(text).encode("utf-8")).hexdigest()


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for part in re.split(r"(?<=[.!?؟])\s+|\n+", str(text or "")):
        part = re.sub(r"^[-*•\s]+", "", re.sub(r"\s+", " ", part).strip())
        if len(part) >= 18:
            out.append(part)
    return out


@dataclass(frozen=True)
class SafetyDecision:
    action: str
    reason: str = ""
    confidence_threshold: float = 0.75
    emergency: bool = False
    real_patient: bool = False
    high_rigor: bool = False
    scope_confidence: float = 1.0


class SafetyGate:
    def check(self, query: str, context: str = "") -> SafetyDecision:
        q = _norm(query)
        for pattern in BLOCK_PATTERNS:
            if re.search(pattern, q, flags=re.I | re.UNICODE):
                return SafetyDecision("BLOCK", "harmful_or_illicit_request", 0.95)
        has_medical = bool(set(meaningful_tokens(q)) & MEDICAL_TERMS) or bool(extract_clinical_entities(query))
        emergency = any(term in q for term in EMERGENCY_TERMS)
        real_patient = any(term in q for term in REAL_PATIENT_HINTS)
        high_rigor = any(term in q for term in HIGH_RIGOR_HINTS)
        if not has_medical:
            return SafetyDecision("ABSTAIN", "outside_medical_scope", 0.95, emergency, real_patient, high_rigor, 0.25)
        threshold = 0.90 if real_patient else 0.85 if high_rigor else 0.75
        if emergency:
            # Emergency language is never blocked by the local safety gate.  The answer
            # remains evidence-first but carries an escalation flag in the formatter.
            return SafetyDecision("PROCEED", "emergency_signal", threshold, True, real_patient, high_rigor)
        return SafetyDecision("PROCEED", "in_scope", threshold, False, real_patient, high_rigor)


@dataclass(frozen=True)
class RouteMetadata:
    intent: str
    complexity: float
    entities: tuple[str, ...]
    numeric_sensitivity: bool
    temporal_sensitivity: bool
    conditional_context: tuple[str, ...]
    is_follow_up: bool
    confidence_threshold: float
    template_type: str | None
    retrieval_timeout_ms: int
    needs_multi_hop: bool
    query_variants: tuple[str, ...] = ()


class QueryRouter:
    _NUMERIC = ("dose", "dosage", "mg", "mcg", "ml", "how much", "how many", "range", "normal value", "frequency", "جرعة", "ملغ", "قيمة")
    _COMPARISON = ("compare", "versus", " vs ", "difference", "differences", "مقارنة", "فرق")
    _MANAGEMENT = ("treat", "treatment", "therapy", "management", "contraindication", "traitement", "علاج", "موانع")
    _MECHANISM = ("mechanism", "pathway", "pathophysiology", "mécanisme", "آلية")
    _TABLE = ("table", "list", "side effects", "rows", "columns", "tableau", "جدول", "قائمة")

    def route(self, question: str, context: str, safety: SafetyDecision) -> RouteMetadata:
        q = _norm(question)
        plan = plan_query(question, conversation_context=context)
        semantic = understand_query(question, conversation_context=context)
        entities = tuple(dict.fromkeys([e.normalized for e in semantic.entities] + list(plan.entities)))[:16]
        tokens = meaningful_tokens(q)
        numeric = plan.needs_numeric or any(term in q for term in self._NUMERIC)
        comparison = plan.intent == "comparison" or any(term in q for term in self._COMPARISON)
        management = plan.intent == "management" or any(term in q for term in self._MANAGEMENT)
        mechanism = plan.intent in {"mechanism", "etiology", "causal"} or any(term in q for term in self._MECHANISM)
        table = plan.needs_table or any(term in q for term in self._TABLE)
        temporal = any(term in q for term in ("latest", "recent", "new", "current", "updated", "récent", "nouveau", "حديث", "جديد"))
        conditional = tuple(term for term in HIGH_RIGOR_HINTS if term in q)
        complexity = 0.05 + min(0.25, len(entities) * 0.06) + min(0.20, len(tokens) * 0.008)
        complexity += 0.16 if numeric else 0.0
        complexity += 0.15 if comparison else 0.0
        complexity += 0.20 if management or mechanism else 0.0
        complexity += 0.15 if len(plan.subqueries) > 1 else 0.0
        complexity += 0.10 if temporal else 0.0
        complexity = min(1.0, complexity)
        if numeric:
            intent = "numeric"
        elif comparison:
            intent = "comparison"
        elif management:
            intent = "management"
        elif mechanism:
            intent = "mechanism"
        elif table:
            intent = "table"
        else:
            intent = plan.intent or semantic.primary_intent or "factual"
        template = "dosage" if numeric else "comparison" if comparison else "table" if table else "mechanism" if mechanism else None
        follow_up = bool(re.search(r"\b(it|this|that|they|them|what about|how about|and|also)\b|^(و|ثم|et|puis)\b", q))
        variants = tuple(dict.fromkeys([plan.normalized, *plan.variants]))[:8]
        return RouteMetadata(
            intent=intent,
            complexity=complexity,
            entities=entities,
            numeric_sensitivity=numeric,
            temporal_sensitivity=temporal,
            conditional_context=conditional,
            is_follow_up=follow_up,
            confidence_threshold=safety.confidence_threshold,
            template_type=template,
            retrieval_timeout_ms=1200 if complexity < 0.65 else 2000,
            needs_multi_hop=plan.needs_multi_hop or complexity >= 0.65,
            query_variants=variants,
        )


class SemanticCache:
    def __init__(self, db_path: Path, ttl_seconds: float = 604800.0):
        self.db_path = Path(db_path)
        self.ttl = float(ttl_seconds)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS retrieval_cache (key TEXT PRIMARY KEY, query TEXT NOT NULL, payload TEXT NOT NULL, created REAL NOT NULL)")
            db.commit()

    def get(self, query: str) -> list[dict[str, Any]] | None:
        key = _hash_query(query)
        try:
            with sqlite3.connect(self.db_path) as db:
                row = db.execute("SELECT payload, created FROM retrieval_cache WHERE key=?", (key,)).fetchone()
            if not row:
                return None
            if self.ttl > 0 and time.time() - float(row[1]) > self.ttl:
                self.delete(query)
                return None
            value = json.loads(row[0])
            return value if isinstance(value, list) else None
        except (OSError, sqlite3.Error, ValueError, TypeError, json.JSONDecodeError):
            return None

    def put(self, query: str, hits: Sequence[RetrievalHit]) -> None:
        payload = []
        for hit in hits[:24]:
            payload.append({"doc_id": hit.doc_id, "text": hit.text, "metadata": hit.metadata, "score": float(hit.score), "vector_score": float(hit.vector_score), "lexical_score": float(hit.lexical_score)})
        try:
            with sqlite3.connect(self.db_path) as db:
                db.execute("INSERT OR REPLACE INTO retrieval_cache(key,query,payload,created) VALUES(?,?,?,?)", (_hash_query(query), query[:3000], json.dumps(payload, ensure_ascii=False), time.time()))
                db.commit()
        except sqlite3.Error:
            pass

    def delete(self, query: str) -> None:
        try:
            with sqlite3.connect(self.db_path) as db:
                db.execute("DELETE FROM retrieval_cache WHERE key=?", (_hash_query(query),))
                db.commit()
        except sqlite3.Error:
            pass

    @staticmethod
    def restore(payload: Sequence[dict[str, Any]]) -> list[RetrievalHit]:
        out = []
        for item in payload or []:
            try:
                out.append(RetrievalHit(str(item.get("doc_id", "unknown")), str(item.get("text", "")), dict(item.get("metadata") or {}), float(item.get("score", 0.0)), float(item.get("vector_score", 0.0)), float(item.get("lexical_score", 0.0))))
            except (TypeError, ValueError):
                continue
        return out


class MedicalKnowledgeLayer:
    """Optional structured local KB.

    It is intentionally schema-tolerant: if users provide a SQLite DB with
    tables named drugs, interactions, contraindications, or guidelines, the
    runtime will query them. An absent DB is a normal empty state, never a failure.
    """
    def __init__(self, db_path: Path | None):
        self.db_path = Path(db_path) if db_path else None

    def lookup(self, entities: Sequence[str], question: str) -> dict[str, Any]:
        if not self.db_path or not self.db_path.exists() or not entities:
            return {"available": False, "records": [], "matched_entities": []}
        terms = tuple(dict.fromkeys(_norm(x) for x in entities if x))[:12]
        records: list[dict[str, Any]] = []
        matched: set[str] = set()
        try:
            with sqlite3.connect(self.db_path) as db:
                tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
                for table in ("drugs", "interactions", "contraindications", "guidelines"):
                    if table not in tables:
                        continue
                    cols = [row[1] for row in db.execute(f'PRAGMA table_info("{table}")').fetchall()]
                    if not cols:
                        continue
                    search_col = next((c for c in ("name", "drug", "entity", "condition", "title") if c in cols), cols[0])
                    value_col = next((c for c in ("description", "details", "text", "guidance", "value") if c in cols), cols[-1])
                    for term in terms:
                        try:
                            rows = db.execute(f'SELECT "{search_col}", "{value_col}" FROM "{table}" WHERE lower("{search_col}") LIKE ? LIMIT 8', (f"%{term}%",)).fetchall()
                        except sqlite3.Error:
                            rows = []
                        for key, value in rows:
                            matched.add(term)
                            records.append({"type": table, "entity": str(key), "text": str(value)})
        except sqlite3.Error:
            return {"available": False, "records": [], "matched_entities": []}
        return {"available": bool(records), "records": records[:40], "matched_entities": sorted(matched)}


class MultiTierRetriever:
    def __init__(self, system: Any):
        self.system = system
        settings = getattr(system, "settings", None)
        project_root = getattr(settings, "project_root", Path.cwd())
        self.cache = SemanticCache(Path(project_root) / "data" / "med_evidence_cache.sqlite3", ttl_seconds=604800.0)

    @staticmethod
    def _lexical_hits(retriever: Any, query: str, top_k: int, where: dict[str, Any] | None) -> list[RetrievalHit]:
        try:
            lexical_query = " ".join(meaningful_tokens(query)).strip()
            raw = retriever._lexical(lexical_query, max(20, top_k * 5), where)
            ids, documents, metadatas, distances = retriever._unpack_results(raw)
            hits: list[RetrievalHit] = []
            for idx, item_id in enumerate(ids[:top_k]):
                meta = metadatas[idx] if idx < len(metadatas) else {}
                dist = distances[idx] if idx < len(distances) else 1e9
                score = math.exp(-max(0.0, dist))
                hits.append(RetrievalHit(str(meta.get("document_id", item_id)), documents[idx] if idx < len(documents) else "", meta, score, 0.0, score))
            return hits
        except Exception:
            return []

    @staticmethod
    def _confidence(hits: Sequence[RetrievalHit], entities: Sequence[str]) -> float:
        if not hits:
            return 0.0
        top = max(0.0, min(1.0, float(hits[0].score)))
        source_diversity = len({str((h.metadata or {}).get("document_id") or h.doc_id) for h in hits[:5]}) / max(1, min(5, len(hits)))
        entity_text = " ".join(h.text.casefold() for h in hits[:5])
        coverage = sum(1 for e in entities if _norm(e) in entity_text) / max(1, len(entities)) if entities else 1.0
        high = sum(1 for h in hits[:5] if float(h.score) >= 0.35) / max(1, min(5, len(hits)))
        return min(1.0, 0.35 * top + 0.2 * source_diversity + 0.25 * coverage + 0.2 * high)

    @staticmethod
    def _merge(hits: Iterable[RetrievalHit], limit: int = 16) -> list[RetrievalHit]:
        by_key: dict[tuple[str, str], RetrievalHit] = {}
        for hit in hits:
            key = (str(hit.doc_id), hashlib.sha1(hit.text.encode("utf-8", "ignore")).hexdigest()[:12])
            old = by_key.get(key)
            if old is None or hit.score > old.score:
                by_key[key] = hit
        return sorted(by_key.values(), key=lambda x: x.score, reverse=True)[:limit]

    def retrieve(self, question: str, route: RouteMetadata, where: dict[str, Any] | None = None) -> tuple[list[RetrievalHit], dict[str, Any]]:
        cached = self.cache.get(question)
        if cached:
            restored = self.cache.restore(cached)
            return restored, {"tier": "CACHE", "cache_hit": True, "early_exit": True, "tier0_confidence": self._confidence(restored, route.entities), "retrieval_latency_ms": 0.2, "candidate_count": len(restored)}
        started = time.perf_counter()
        retriever = getattr(self.system, "retriever", None)
        if retriever is None:
            return [], {"tier": "NONE", "cache_hit": False, "early_exit": False, "candidate_count": 0, "retrieval_latency_ms": 0.0}
        tier0_queries = []
        q_norm = _norm(question)
        tier0_queries.append(question)
        for token, variants in SYNONYMS.items():
            if re.search(rf"\b{re.escape(token)}\b", q_norm):
                tier0_queries.extend(variants)
        tier0_queries = list(dict.fromkeys(tier0_queries))[:4]
        tier0: list[RetrievalHit] = []
        with ThreadPoolExecutor(max_workers=min(4, len(tier0_queries))) as pool:
            futures = [pool.submit(self._lexical_hits, retriever, q, max(12, getattr(getattr(self.system, "settings", None), "top_k", 8) * 3), where) for q in tier0_queries]
            for future in as_completed(futures):
                try:
                    tier0.extend(future.result())
                except Exception:
                    pass
        tier0 = self._merge(tier0, 16)
        c0 = self._confidence(tier0, route.entities)
        if c0 >= 0.75 and (not route.entities or c0 >= 0.75):
            self.cache.put(question, tier0)
            return tier0, {"tier": "TIER0_EXIT", "cache_hit": False, "early_exit": True, "tier0_confidence": round(c0, 4), "tier1_confidence": None, "retrieval_latency_ms": round((time.perf_counter() - started) * 1000, 2), "candidate_count": len(tier0), "queries": len(tier0_queries)}
        tier1_queries = list(dict.fromkeys([question, *route.query_variants[:4]]))[:5]
        tier1: list[RetrievalHit] = list(tier0)
        with ThreadPoolExecutor(max_workers=min(4, len(tier1_queries))) as pool:
            futures = [pool.submit(retriever.retrieve, q, max(10, getattr(getattr(self.system, "settings", None), "top_k", 8) * 3), where) for q in tier1_queries]
            for future in as_completed(futures):
                try:
                    tier1.extend(future.result() or [])
                except Exception:
                    pass
        tier1 = self._merge(tier1, 20)
        c1 = self._confidence(tier1, route.entities)
        if c1 >= 0.80 or (not route.needs_multi_hop and len(tier1) >= 3):
            self.cache.put(question, tier1)
            return tier1, {"tier": "TIER1", "cache_hit": False, "early_exit": c1 >= 0.80, "tier0_confidence": round(c0, 4), "tier1_confidence": round(c1, 4), "retrieval_latency_ms": round((time.perf_counter() - started) * 1000, 2), "candidate_count": len(tier1), "queries": len(tier1_queries)}
        if route.needs_multi_hop and route.complexity > 0.60:
            subqueries = list(dict.fromkeys([*route.query_variants, f"{question} mechanism", f"{question} causes", f"{question} treatment"]))[:8]
            tier2 = list(tier1)
            with ThreadPoolExecutor(max_workers=min(4, len(subqueries))) as pool:
                futures = [pool.submit(retriever.retrieve, q, max(10, getattr(getattr(self.system, "settings", None), "top_k", 8) * 3), where) for q in subqueries]
                for future in as_completed(futures):
                    try:
                        tier2.extend(future.result() or [])
                    except Exception:
                        pass
            tier2 = self._merge(tier2, 24)
            self.cache.put(question, tier2)
            return tier2, {"tier": "TIER2", "cache_hit": False, "early_exit": False, "tier0_confidence": round(c0, 4), "tier1_confidence": round(c1, 4), "retrieval_latency_ms": round((time.perf_counter() - started) * 1000, 2), "candidate_count": len(tier2), "queries": len(subqueries)}
        self.cache.put(question, tier1)
        return tier1, {"tier": "TIER1", "cache_hit": False, "early_exit": False, "tier0_confidence": round(c0, 4), "tier1_confidence": round(c1, 4), "retrieval_latency_ms": round((time.perf_counter() - started) * 1000, 2), "candidate_count": len(tier1), "queries": len(tier1_queries)}


@dataclass(frozen=True)
class EvidenceClaim:
    text: str
    source_numbers: tuple[int, ...]
    numeric: bool = False
    contradiction: bool = False
    confidence: float = 0.0


class EvidenceCompiler:
    def compile(self, question: str, hits: Sequence[RetrievalHit], route: RouteMetadata, structured: dict[str, Any]) -> dict[str, Any]:
        query_tokens = set(meaningful_tokens(question))
        ranked: list[tuple[float, str, int]] = []
        for index, hit in enumerate(hits[:24], 1):
            for sentence in _sentences(hit.text):
                overlap = len(set(meaningful_tokens(sentence)) & query_tokens) / max(1, len(query_tokens))
                numeric_bonus = 0.12 if route.numeric_sensitivity and re.search(r"\d", sentence) else 0.0
                score = 0.55 * max(0.0, min(1.0, hit.score)) + 0.35 * overlap + numeric_bonus
                if score >= 0.06:
                    ranked.append((score, sentence, index))
        ranked.sort(key=lambda x: x[0], reverse=True)
        claims: list[EvidenceClaim] = []
        seen: set[str] = set()
        for score, sentence, idx in ranked:
            key = _norm(sentence)
            if key in seen:
                continue
            seen.add(key)
            claims.append(EvidenceClaim(sentence, (idx,), bool(re.search(r"\d", sentence)), False, min(1.0, score)))
            if len(claims) >= 8:
                break
        numeric_values = [m.group(0) for c in claims for m in re.finditer(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|kg|mL|ml|L|mmHg|mmol/L|%|IU|units?)\b", c.text, flags=re.I)]
        contradiction = self._detect_contradiction(claims)
        compressed = "\n".join(f"- {c.text} [S{c.source_numbers[0]}]" for c in claims)
        return {"claims": claims, "compressed": compressed[:6500], "numeric_values": numeric_values[:32], "contradiction": contradiction, "structured": structured, "claim_count": len(claims), "compression_chars": len(compressed[:6500])}

    @staticmethod
    def _detect_contradiction(claims: Sequence[EvidenceClaim]) -> dict[str, Any]:
        numeric_claims = []
        for claim in claims:
            nums = [m.group(0).casefold() for m in re.finditer(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|kg|mL|ml|L|mmHg|mmol/L|%|IU|units?)\b", claim.text, flags=re.I)]
            if nums:
                numeric_claims.append((nums, claim.text))
        conflicts = []
        for i, (left, _) in enumerate(numeric_claims):
            for right, _ in numeric_claims[i + 1 :]:
                if left and right and left != right:
                    # Only flag exact-looking dose conflicts; broad ranges are left for the verifier.
                    if any(a.replace(" ", "") != b.replace(" ", "") for a in left for b in right if re.search(r"(?:mg|mcg|µg|g|kg)", a)):
                        conflicts.append({"left": left, "right": right})
        return {"has_contradiction": bool(conflicts), "conflicts": conflicts[:8], "agreement": 0.65 if conflicts else 1.0}


class AnswerCascade:
    def __init__(self, system: Any):
        self.system = system

    @staticmethod
    def _extractive(compiled: dict[str, Any], max_sentences: int = 6) -> str:
        rows = compiled.get("claims") or []
        return "\n".join(f"- {claim.text} [S{claim.source_numbers[0]}]" for claim in rows[:max_sentences])

    @staticmethod
    def _template(compiled: dict[str, Any], route: RouteMetadata) -> str | None:
        claims = compiled.get("claims") or []
        if not claims:
            return None
        if route.template_type == "comparison":
            left = claims[0].text
            right = claims[1].text if len(claims) > 1 else "Evidence for the comparison was limited."
            return f"- Key evidence for the first item: {left} [S{claims[0].source_numbers[0]}]\n- Key evidence for the second/comparison item: {right} [S{claims[1].source_numbers[0] if len(claims) > 1 else claims[0].source_numbers[0]}]"
        if route.template_type == "dosage":
            rows = [c for c in claims if c.numeric]
            if not rows:
                return None
            return "\n".join(f"- {row.text} [S{row.source_numbers[0]}]" for row in rows[:6])
        if route.template_type == "table":
            return "\n".join(f"- {row.text} [S{row.source_numbers[0]}]" for row in claims[:8])
        if route.template_type == "mechanism":
            return "\n".join(f"- {row.text} [S{row.source_numbers[0]}]" for row in claims[:6])
        return None

    def _llm(self, question: str, evidence: str, route: RouteMetadata) -> str | None:
        llm = getattr(self.system, "llm", None)
        if llm is None or not evidence.strip():
            return None
        system_prompt = (
            "You are MedEvidence Pro's constrained synthesis stage. Use ONLY the supplied evidence. "
            "Every factual sentence must end in an existing [S#] citation. Do not introduce a new number, "
            "unit, diagnosis, recommendation, cause, population, severity, timing, or contraindication. "
            "Preserve negation exactly. If the evidence is insufficient, say so briefly. Return only the answer."
        )
        prompt = f"Question: {question[:2600]}\nIntent: {route.intent}\n\nEvidence:\n{evidence[:6500]}"
        try:
            value = llm.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.0)
            value = str(value or "").strip()
            return value[:9000] if value else None
        except Exception:
            return None

    @staticmethod
    def _citation_complete(answer: str, hit_count: int) -> bool:
        if not answer:
            return False
        sentences = _sentences(answer)
        if not sentences:
            return False
        markers = [int(x) for x in re.findall(r"\[S(\d+)\]", answer, flags=re.I)]
        return bool(markers) and all(1 <= n <= hit_count for n in markers) and all(re.search(r"\[S\d+\]", s, flags=re.I) for s in sentences)

    def generate(self, question: str, route: RouteMetadata, compiled: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        claims = compiled.get("claims") or []
        if not claims:
            return "", "ABSTAIN", {"attempted": False}
        evidence = str(compiled.get("compressed") or "")
        c = float(sum(x.confidence for x in claims[:3]) / max(1, min(3, len(claims))))
        if route.complexity < 0.35 and c >= 0.45:
            answer = self._extractive(compiled)
            return answer, "PATH_A_EXTRACTIVE", {"attempted": False, "confidence": c}
        templated = self._template(compiled, route)
        if templated and route.complexity < 0.65:
            return templated, "PATH_B_TEMPLATE", {"attempted": False, "confidence": c}
        synthesized = self._llm(question, evidence, route)
        if synthesized and self._citation_complete(synthesized, max(1, len(getattr(self.system, "_med_selected_hits", [])))):
            return synthesized, "PATH_C_CONSTRAINED_LLM", {"attempted": True, "confidence": c}
        if route.complexity < 0.8:
            fallback = templated or self._extractive(compiled)
            if fallback:
                return fallback, "PATH_HYBRID_FALLBACK", {"attempted": bool(synthesized), "confidence": c}
        return "", "PATH_D_ABSTAIN", {"attempted": bool(synthesized), "confidence": c}


class ActiveVerifier:
    def verify(self, answer: str, hits: Sequence[RetrievalHit], route: RouteMetadata, compiled: dict[str, Any]) -> dict[str, Any]:
        blocks = [str(h.text or "") for h in hits]
        marker_ids = [f"S{i+1}" for i in range(len(hits))]
        checks = list(verify_claims(answer, blocks, marker_ids)) if answer and blocks else []
        ground = grounding_decision(checks, min_supported_ratio=0.70) if checks else {"allow": False, "supported_ratio": 0.0}
        try:
            final = dict(verify_final_answer(answer, hits)) if answer else {"allow": False, "checked": True}
        except Exception:
            final = {"allow": bool(ground.get("allow")), "checked": True, "supported_ratio": float(ground.get("supported_ratio", 0.0))}
        numeric_values = compiled.get("numeric_values") or []
        numeric_mismatch = False
        if route.numeric_sensitivity and numeric_values:
            answer_nums = re.findall(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|kg|mL|ml|L|mmHg|mmol/L|%|IU|units?)\b", answer, flags=re.I)
            for value in answer_nums:
                normalized = re.sub(r"\s+", "", value).casefold()
                if normalized not in {re.sub(r"\s+", "", x).casefold() for x in numeric_values}:
                    numeric_mismatch = True
                    break
        contradiction = dict(compiled.get("contradiction") or {})
        allow = bool(ground.get("allow")) and bool(final.get("allow", True)) and not numeric_mismatch
        return {
            "allow": allow,
            "checked": True,
            "grounding": dict(ground),
            "final_answer": final,
            "numeric_mismatch": numeric_mismatch,
            "contradiction": contradiction,
            "supported_ratio": float(ground.get("supported_ratio", 0.0) or 0.0),
            "claim_count": len(checks),
            "blocked_claims": sum(1 for c in checks if getattr(c, "status", "") in {"UNSUPPORTED", "WEAK", "NUMERIC_MISMATCH", "CONTRADICTED"}),
        }


class ResponseFormatter:
    def format(self, answer: str, citations: Sequence[Any], safety: SafetyDecision, route: RouteMetadata, verification: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
        final_answer = str(answer or "").strip()
        disclaimer = "This is an evidence-based study aid, not a substitute for clinician assessment or emergency care."
        if safety.emergency:
            disclaimer = "Emergency symptoms may require urgent in-person assessment; this evidence-based answer does not replace emergency care."
        confidence = float(verification.get("supported_ratio", 0.0) or 0.0)
        level = "high" if confidence >= 0.85 else "medium" if confidence >= 0.70 else "low"
        return {
            "answer": final_answer,
            "citations": list(citations),
            "disclaimer": disclaimer,
            "emergency_flag": safety.emergency,
            "confidence": {"level": level, "evidence_confidence": confidence, "threshold": safety.confidence_threshold},
            "route": asdict(route),
            "verification": verification,
            "provenance": provenance,
        }


class FeedbackLogger:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS answer_feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL NOT NULL, query TEXT NOT NULL, status TEXT NOT NULL, path TEXT, confidence REAL, latency_ms REAL, needs_review INTEGER NOT NULL, payload TEXT NOT NULL)")
            db.commit()

    def log(self, query: str, result: dict[str, Any], latency_ms: float) -> None:
        status = str(result.get("status", "UNKNOWN"))
        confidence = float((result.get("confidence") or {}).get("evidence_confidence", 0.0) or 0.0)
        needs_review = bool(result.get("needs_review") or status in {"GENERATION_ABSTAIN", "ANSWER_UNAVAILABLE", "BLOCK", "ABSTAIN"} or confidence < 0.70 or (result.get("verification") or {}).get("contradiction", {}).get("has_contradiction"))
        payload = json.dumps({k: v for k, v in result.items() if k not in {"hits"}}, ensure_ascii=False, default=str)[:20000]
        try:
            with sqlite3.connect(self.db_path) as db:
                db.execute("INSERT INTO answer_feedback(created,query,status,path,confidence,latency_ms,needs_review,payload) VALUES(?,?,?,?,?,?,?,?)", (time.time(), query[:3000], status, str(result.get("generation_path", "")), confidence, float(latency_ms), int(needs_review), payload))
                db.commit()
        except sqlite3.Error:
            pass


class MedEvidenceProEngine:
    VERSION = "1.0.0"

    def __init__(self, system: Any):
        self.system = system
        settings = getattr(system, "settings", None)
        root = Path(getattr(settings, "project_root", Path.cwd()))
        self.safety = SafetyGate()
        self.router = QueryRouter()
        self.retrieval = MultiTierRetriever(system)
        self.knowledge = MedicalKnowledgeLayer(root / "data" / "medical_knowledge.sqlite3")
        self.compiler = EvidenceCompiler()
        self.cascade = AnswerCascade(system)
        self.verifier = ActiveVerifier()
        self.formatter = ResponseFormatter()
        self.feedback = FeedbackLogger(root / "data" / "med_evidence_feedback.sqlite3")
        self.selected_hits: list[RetrievalHit] = []

    def answer(self, question: str, metadata_filter: dict[str, Any] | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        clean = re.sub(r"\s+", " ", str(question or "")).strip()[:3500]
        if not clean:
            return {"status": "LOW_QUALITY_QUERY", "answer": "Please provide a precise medical question.", "citations": [], "hits": [], "confidence": {"level": "none", "evidence_confidence": 0.0}}
        memory = getattr(self.system, "conversation_memory", None)
        context = getattr(memory, "prompt_context", lambda: "")() if memory is not None else ""
        safety = self.safety.check(clean, context)
        if safety.action != "PROCEED":
            result = {"status": safety.action, "answer": "I cannot safely answer that request within the medical evidence scope.", "citations": [], "hits": [], "confidence": {"level": "none", "evidence_confidence": 0.0}, "safety": asdict(safety)}
            self.feedback.log(clean, result, (time.perf_counter() - started) * 1000)
            return result
        route = self.router.route(clean, context, safety)
        hits, retrieval_state = self.retrieval.retrieve(clean, route, metadata_filter)
        self.selected_hits = hits
        setattr(self.system, "_med_selected_hits", hits)
        knowledge = self.knowledge.lookup(route.entities, clean)
        compiled = self.compiler.compile(clean, hits, route, knowledge)
        if not hits or not compiled.get("claims"):
            result = {"status": "NOT_SUPPORTED", "answer": "I could not find sufficient indexed evidence to answer this question safely.", "citations": [], "hits": hits, "confidence": {"level": "none", "evidence_confidence": 0.0}, "safety": asdict(safety), "route": asdict(route), "retrieval": retrieval_state, "evidence": {"claim_count": 0}}
            self.feedback.log(clean, result, (time.perf_counter() - started) * 1000)
            return result
        answer, generation_path, generation_meta = self.cascade.generate(clean, route, compiled)
        verification = self.verifier.verify(answer, hits, route, compiled) if answer else {"allow": False, "checked": True, "supported_ratio": 0.0, "claim_count": 0, "blocked_claims": 0, "numeric_mismatch": False, "contradiction": compiled.get("contradiction", {})}
        if not verification.get("allow"):
            # For a low-complexity answer, the exact evidence sentence path is the safe fallback.
            fallback = self.cascade._extractive(compiled, max_sentences=6)
            fallback_verification = self.verifier.verify(fallback, hits, route, compiled) if fallback else verification
            if fallback and fallback_verification.get("allow") and (route.complexity < 0.80 or generation_path == "PATH_HYBRID_FALLBACK"):
                answer, generation_path, generation_meta = fallback, "PATH_A_VERIFIED_FALLBACK", {"attempted": True, "fallback": True}
                verification = fallback_verification
            else:
                result = {"status": "GENERATION_ABSTAIN", "answer": "The evidence was retrieved, but the requested synthesis could not be safely verified without adding unsupported medical content.", "citations": [], "hits": hits, "confidence": {"level": "low", "evidence_confidence": verification.get("supported_ratio", 0.0)}, "safety": asdict(safety), "route": asdict(route), "retrieval": retrieval_state, "evidence": {"claim_count": compiled.get("claim_count", 0), "compressed_context": compiled.get("compressed", "")}, "verification": verification, "generation_path": generation_path, "generation_meta": generation_meta, "needs_review": True}
                self.feedback.log(clean, result, (time.perf_counter() - started) * 1000)
                return result
        citations: list[Any] = []
        try:
            built = self.system.citation_manager.build(hits) or []
            citations = self.system.citation_manager.validate(built, hits) or []
        except Exception:
            citations = []
        formatted = self.formatter.format(answer, citations, safety, route, verification, {"claims": [asdict(c) for c in compiled.get("claims", [])]})
        status = "SUCCESS_WITH_WARNINGS" if safety.emergency or compiled.get("contradiction", {}).get("has_contradiction") or not citations else "SUCCESS"
        latency_ms = (time.perf_counter() - started) * 1000
        result = {
            "query_id": f"medevidence-{int(time.time() * 1000)}",
            "status": status,
            "answer": formatted["answer"],
            "citations": citations,
            "hits": hits,
            "confidence": formatted["confidence"],
            "safety": formatted["emergency_flag"] and {**asdict(safety), "action": "PROCEED"} or asdict(safety),
            "route": formatted["route"],
            "retrieval": retrieval_state,
            "structured_knowledge": knowledge,
            "evidence": {"claim_count": compiled.get("claim_count", 0), "compressed_context": compiled.get("compressed", ""), "numeric_values": compiled.get("numeric_values", [])},
            "verification": verification,
            "generation_path": generation_path,
            "generation_meta": generation_meta,
            "disclaimer": formatted["disclaimer"],
            "emergency_flag": formatted["emergency_flag"],
            "evidence_first": True,
            "document_aware": True,
            "med_evidence_pro_version": self.VERSION,
            "canonical_pipeline_executed": True,
            "pipeline_authority": "rag_project.intelligence.med_evidence_pro.MedEvidenceProEngine",
            "phases": {
                "phase_0_safety_gate": "complete",
                "phase_1_query_router": "complete",
                "phase_2a_multi_tier_retrieval": retrieval_state.get("tier", "complete"),
                "phase_2b_structured_knowledge": "complete",
                "phase_3_evidence_compiler": "complete",
                "phase_4_answer_cascade": generation_path,
                "phase_5_active_verification": "complete",
                "phase_6_response_formatter": "complete",
                "phase_7_logging_feedback": "complete",
            },
            "query_trace": {
                "mode": "med_evidence_pro",
                "routing": asdict(route),
                "retrieval": retrieval_state,
                "generation": {"path": generation_path, **generation_meta},
                "verification": {"allow": verification.get("allow"), "supported_ratio": verification.get("supported_ratio"), "numeric_mismatch": verification.get("numeric_mismatch"), "contradiction": verification.get("contradiction")},
                "timings_ms": {"total": round(latency_ms, 2)},
            },
            "needs_review": bool(status != "SUCCESS" or formatted["confidence"]["evidence_confidence"] < 0.70),
        }
        self.feedback.log(clean, result, latency_ms)
        return result


def enhanced_med_evidence_answer(system: Any, question: str, metadata_filter: dict[str, Any] | None = None) -> dict[str, Any]:
    engine = MedEvidenceProEngine(system)
    return engine.answer(question, metadata_filter)


__all__ = [
    "MedEvidenceProEngine", "enhanced_med_evidence_answer", "SafetyGate", "QueryRouter", "RouteMetadata",
    "SemanticCache", "MedicalKnowledgeLayer", "MultiTierRetriever", "EvidenceCompiler", "AnswerCascade",
    "ActiveVerifier", "ResponseFormatter", "FeedbackLogger",
]
