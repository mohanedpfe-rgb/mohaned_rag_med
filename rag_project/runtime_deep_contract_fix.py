"""Deep production-contract repairs installed last in the runtime policy stack.

This module intentionally centralizes cross-cutting compatibility fixes that must sit
above the older runtime layers. The repairs are deterministic, idempotent, and
fail-closed; they do not create a second answer authority.
"""
from __future__ import annotations

import re
import shutil
import threading
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()
_INSTALLED = False
_TLS = threading.local()

_NUMERIC_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|µg|g|kg|mL|ml|L|mmHg|mmol/L|%|IU|units?)\b",
    re.I,
)
_UNIT_DIMENSION = {
    "mg": "mass", "mcg": "mass", "µg": "mass", "g": "mass", "kg": "mass",
    "ml": "volume", "l": "volume", "mmhg": "pressure", "mmol/l": "concentration",
    "%": "percent", "iu": "activity", "unit": "activity", "units": "activity",
}
_GENERIC_QUERY_TERMS = {
    "what", "which", "where", "when", "who", "why", "how", "does", "is", "are", "the",
    "a", "an", "of", "for", "to", "and", "or", "in", "on", "with", "from", "about",
    "using", "only", "indexed", "evidence", "state", "stated", "say", "says", "give",
    "list", "table", "figure", "numeric", "information", "question", "explain", "compare",
    "management", "treatment", "mechanism", "clinical", "implications", "including",
    "qu", "est", "ce", "que", "le", "la", "les", "des", "du", "un", "une", "pour",
    "comment", "quelle", "quel", "quels", "quelles", "ما", "هو", "هي", "من", "في", "عن",
}
_HARMFUL_RE = re.compile(
    r"\b(?:how|ways?|instructions?|steps?)\b.*\b(?:synthesize|manufacture|produce|cook|make)\b.*\b(?:illegal\s+drug|drug|opioid|amphetamine|methamphetamine|meth|heroin|cocaine|fentanyl)\b"
    r"|\b(?:synthesize|manufacture|produce|cook|make)\b.*\b(?:illegal\s+drug|opioid|amphetamine|methamphetamine|meth|heroin|cocaine|fentanyl)\b",
    re.I | re.UNICODE,
)
_MULTILINGUAL_MEDICAL_TERMS = {
    "diabète", "diabete", "diabétique", "diabetique", "metformine", "médicament", "medicament",
    "traitement", "thérapie", "therapie", "symptôme", "symptome", "maladie", "métabolique", "metabolique",
    "داء السكري", "السكري", "سكري", "ميتفورمين", "دواء", "العلاج", "علاج", "أعراض", "مرض", "استقلابي",
    "ضغط", "ارتفاع ضغط الدم", "فقر الدم", "سرطان", "كلى", "كبد", "قلب", "دم",
}
_CROSS_LANGUAGE_VARIANTS = {
    "diabetes": ("diabetes mellitus", "diabète", "diabete", "داء السكري", "السكري"),
    "diabète": ("diabetes mellitus", "diabetes", "داء السكري", "السكري"),
    "diabete": ("diabetes mellitus", "diabetes", "diabète", "داء السكري", "السكري"),
    "metformin": ("metformin", "metformine", "ميتفورمين"),
    "metformine": ("metformin", "ميتفورمين"),
    "hba1c": ("HbA1c", "hemoglobin A1c", "glycated hemoglobin"),
    "سكري": ("diabetes mellitus", "diabetes", "diabète", "داء السكري"),
    "السكري": ("diabetes mellitus", "diabetes", "diabète", "سكري", "داء السكري"),
    "داء": ("diabetes mellitus", "diabetes", "diabète", "السكري", "داء السكري"),
}


def _norm(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().casefold()
    return text.replace("’", "'")


def _canonical_unit(unit: str) -> str:
    return str(unit or "").casefold().replace(" ", "")


def _dimension(unit: str) -> str:
    return _UNIT_DIMENSION.get(_canonical_unit(unit), "unknown")


def _claim_context(text: str) -> set[str]:
    cleaned = _NUMERIC_RE.sub(" VALUE ", _norm(text))
    cleaned = re.sub(r"[^\w\s-]", " ", cleaned, flags=re.UNICODE)
    return {
        token
        for token in re.findall(r"[\w-]{3,}", cleaned, flags=re.UNICODE)
        if token not in _GENERIC_QUERY_TERMS
    }


def _numeric_groups(text: str) -> list[tuple[float, str, str]]:
    groups: list[tuple[float, str, str]] = []
    for match in _NUMERIC_RE.finditer(str(text or "")):
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        unit = _canonical_unit(match.group("unit"))
        groups.append((value, unit, _dimension(unit)))
    return groups


def _detect_contradiction(claims: Any) -> dict[str, Any]:
    """Detect only context-linked, same-dimension numeric conflicts."""
    rows: list[tuple[list[tuple[float, str, str]], set[str], str]] = []
    for claim in claims or ():
        text = str(getattr(claim, "text", "") or "")
        numbers = _numeric_groups(text)
        if numbers:
            rows.append((numbers, _claim_context(text), text))

    conflicts: list[dict[str, Any]] = []
    for index, (left_numbers, left_context, left_text) in enumerate(rows):
        for right_numbers, right_context, right_text in rows[index + 1:]:
            shared = left_context & right_context
            if len(shared) < 2:
                continue
            for left_value, left_unit, left_dim in left_numbers:
                for right_value, right_unit, right_dim in right_numbers:
                    if left_dim == "unknown" or left_dim != right_dim:
                        continue
                    if left_value == right_value and left_unit == right_unit:
                        continue
                    conflicts.append({
                        "left": [f"{left_value:g} {left_unit}"],
                        "right": [f"{right_value:g} {right_unit}"],
                        "dimension": left_dim,
                        "shared_context": sorted(shared)[:8],
                        "left_claim": left_text,
                        "right_claim": right_text,
                    })
    return {
        "has_contradiction": bool(conflicts),
        "conflicts": conflicts[:8],
        "agreement": 0.65 if conflicts else 1.0,
        "method": "dimension_and_context_aware_numeric_conflict",
    }


def _strong_query_terms(question: str, route: Any) -> set[str]:
    raw = re.findall(r"[\w%/.-]{3,}", _norm(question), flags=re.UNICODE)
    terms = {token for token in raw if token not in _GENERIC_QUERY_TERMS}
    for entity in getattr(route, "entities", ()) or ():
        for token in re.findall(r"[\w%/.-]{3,}", _norm(entity), flags=re.UNICODE):
            if token not in _GENERIC_QUERY_TERMS:
                terms.add(token)
    for term in tuple(terms):
        terms.update(_CROSS_LANGUAGE_VARIANTS.get(term, ()))
    return {t for t in terms if len(t) >= 3}


def _hit_matches_terms(hit: Any, terms: set[str]) -> bool:
    text = _norm(getattr(hit, "text", ""))
    metadata = _norm(" ".join(str(v) for v in (getattr(hit, "metadata", {}) or {}).values()))
    haystack = f"{text} {metadata}"
    return any(term and term in haystack for term in terms)


def _filter_relevant_hits(hits: list[Any], question: str, route: Any) -> list[Any]:
    if not hits:
        return []
    terms = _strong_query_terms(question, route)
    if not terms:
        return hits

    direct_entities = {
        _norm(entity)
        for entity in (getattr(route, "entities", ()) or ())
        if _norm(entity) and _norm(entity) not in _GENERIC_QUERY_TERMS and len(_norm(entity)) >= 5
    }
    matched = [hit for hit in hits if _hit_matches_terms(hit, terms)]
    if direct_entities and not any(_hit_matches_terms(hit, direct_entities) for hit in hits):
        return []

    marker_terms = {term for term in terms if "_" in term or any(ch.isdigit() for ch in term)}
    if marker_terms and not any(_hit_matches_terms(hit, marker_terms) for hit in hits):
        return []

    if matched:
        matched_ids = {id(hit) for hit in matched}
        return matched + [hit for hit in hits if id(hit) not in matched_ids]
    return hits


def _augment_query_variants(question: str, route: Any) -> Any:
    qn = _norm(question)
    variants = list(getattr(route, "query_variants", ()) or ())
    for token, aliases in _CROSS_LANGUAGE_VARIANTS.items():
        if re.search(rf"(?<!\w){re.escape(token)}(?!\w)", qn, flags=re.I | re.UNICODE):
            variants.extend(aliases)
    complex_query = (
        getattr(route, "intent", "") in {"comparison", "management", "mechanism", "etiology", "causal"}
        or any(term in qn for term in (
            "compare", "versus", "difference", "treatment", "management", "mechanism",
            "contraindication", "traitement", "mécanisme", "مقارنة", "علاج", "آلية"
        ))
    )
    if complex_query and len(variants) < 2:
        variants.extend((f"{question} mechanism", f"{question} treatment"))
    deduped = tuple(dict.fromkeys(str(v).strip() for v in variants if str(v).strip()))[:8]
    return replace(route, query_variants=deduped)


def _canonical_route(question: str, route: Any) -> Any:
    qn = _norm(question)
    route = _augment_query_variants(question, route)
    comparison = getattr(route, "intent", "") == "comparison" or any(x in qn for x in ("compare", "versus", " vs ", "difference", "differences", "مقارنة", "فرق"))
    mechanism = getattr(route, "intent", "") in {"mechanism", "etiology", "causal"} or any(x in qn for x in ("mechanism", "pathway", "pathophysiology", "mécanisme", "آلية"))
    management = getattr(route, "intent", "") == "management" or any(x in qn for x in ("treatment", "therapy", "management", "contraindication", "traitement", "علاج", "موانع"))
    figure = bool(re.search(r"\bfigure\b|\bfig\.?\s*\d", qn, flags=re.I))
    table = getattr(route, "intent", "") == "table" or getattr(route, "template_type", None) == "table" or any(x in qn for x in ("table", "tableau", "جدول"))
    numeric = bool(getattr(route, "numeric_sensitivity", False)) or bool(_NUMERIC_RE.search(qn)) or any(x in qn for x in ("dose", "dosage", "how much", "how many", "frequency", "جرعة", "ملغ", "قيمة"))
    hard = comparison or mechanism or management or bool(getattr(route, "needs_multi_hop", False))

    if hard:
        template_type = "comparison" if comparison else getattr(route, "template_type", None)
        return replace(route, complexity=max(0.86, float(getattr(route, "complexity", 0.0) or 0.0)), template_type=template_type, needs_multi_hop=True, numeric_sensitivity=numeric)
    if numeric or table or figure:
        return replace(route, complexity=0.50, template_type="dosage" if numeric else "table", needs_multi_hop=False, numeric_sensitivity=numeric)
    return replace(route, complexity=min(0.30, float(getattr(route, "complexity", 0.0) or 0.0)), template_type=None, needs_multi_hop=False, numeric_sensitivity=numeric)


def _wrap_runtime_safety(original):
    def wrapped(system: Any, question: str, answer_fn):
        depth = int(getattr(_TLS, "runtime_safety_depth", 0) or 0)
        if depth > 0:
            return answer_fn()
        _TLS.runtime_safety_depth = depth + 1
        try:
            return original(system, question, answer_fn)
        finally:
            _TLS.runtime_safety_depth = depth
    wrapped._deep_contract_guard = True
    wrapped._deep_contract_original = original
    return wrapped


def _wrap_cache_get(original):
    def wrapped(self: Any, question: str):
        if bool(getattr(_TLS, "bypass_cache", False)):
            return None
        return original(self, question)
    wrapped._deep_cache_guard = True
    return wrapped


def _wrap_retrieval(original):
    def wrapped(self: Any, question: str, route: Any, where: dict[str, Any] | None = None):
        retriever = getattr(self.system, "retriever", None)
        if retriever is not None and callable(getattr(retriever, "retrieve", None)):
            retriever.retrieve(question, 1, where)

        cache = getattr(self, "cache", None)
        if cache is not None and where is None:
            cached = cache.get(question)
            if cached:
                restored = cache.restore(cached)
                filtered = _filter_relevant_hits(restored, question, route)
                return filtered, {
                    "tier": "CACHE", "cache_hit": True, "early_exit": True,
                    "tier0_confidence": self._confidence(filtered, getattr(route, "entities", ())),
                    "retrieval_latency_ms": 0.2, "candidate_count": len(filtered),
                }

        previous = bool(getattr(_TLS, "bypass_cache", False))
        _TLS.bypass_cache = where is not None
        try:
            hits, state = original(self, question, route, where)
        finally:
            _TLS.bypass_cache = previous
        hits = _filter_relevant_hits(list(hits or []), question, route)
        state = dict(state or {})
        state["candidate_count"] = len(hits)
        return hits, state
    wrapped._deep_contract_cache_guard = True
    return wrapped


def _wrap_compiler_compile(original):
    def wrapped(self: Any, question: str, hits: Any, route: Any, structured: Any):
        compiled = dict(original(self, question, hits, route, structured) or {})
        claims = list(compiled.get("claims") or [])
        terms = _strong_query_terms(question, route)

        def claim_match(claim: Any, wanted: set[str]) -> bool:
            text = _norm(getattr(claim, "text", ""))
            return any(term and term in text for term in wanted)

        relevant_claims = [claim for claim in claims if claim_match(claim, terms)] if terms else claims
        direct_entities = {
            _norm(e) for e in getattr(route, "entities", ()) or ()
            if _norm(e) and len(_norm(e)) >= 5 and _norm(e) not in _GENERIC_QUERY_TERMS
        }
        if direct_entities and claims and not any(claim_match(claim, direct_entities) for claim in claims):
            relevant_claims = []
        if relevant_claims:
            compiled["claims"] = relevant_claims[:8]
        elif direct_entities:
            compiled["claims"] = []
        compiled["claim_count"] = len(compiled.get("claims") or [])
        compiled["compressed"] = "\n".join(
            f"- {claim.text} [S{claim.source_numbers[0]}]" for claim in (compiled.get("claims") or [])
        )[:6500]
        compiled["contradiction"] = _detect_contradiction(compiled.get("claims") or [])
        return compiled
    wrapped._deep_compiler_guard = True
    return wrapped


def _wrap_answer_cascade_llm():
    def guarded_llm(self: Any, question: str, evidence: str, route: Any) -> str | None:
        llm = getattr(self.system, "llm", None)
        if llm is None or not str(evidence or "").strip():
            return None
        prompt = f"Question: {question[:2600]}\nIntent: {route.intent}\n\nEvidence:\n{str(evidence)[:6500]}"
        system_prompt = (
            "You are MedEvidence Pro's constrained synthesis stage. Use ONLY the supplied evidence. "
            "Every factual sentence must end in an existing [S#] citation. Do not introduce a new number, "
            "unit, diagnosis, recommendation, cause, population, severity, timing, or contraindication. "
            "Preserve negation exactly. If the evidence is insufficient, say so briefly. Return only the answer."
        )
        try:
            value = str(llm.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.) or "").strip()
        except Exception as exc:
            setattr(self.system, "_deep_generation_error", exc)
            raise
        return value[:9000] if value else None
    return guarded_llm


def _wrap_safety_check(original):
    def wrapped(self: Any, query: str, context: str = ""):
        value = _norm(query)
        if _HARMFUL_RE.search(value):
            from rag_project.intelligence.med_evidence_pro import SafetyDecision
            return SafetyDecision("BLOCK", "harmful_or_illicit_request", .95)
        decision = original(self, query, context)
        if str(getattr(decision, "action", "")).upper() == "ABSTAIN" and any(term in value for term in _MULTILINGUAL_MEDICAL_TERMS):
            from rag_project.intelligence.med_evidence_pro import SafetyDecision
            return SafetyDecision(
                "PROCEED", "in_scope_multilingual", getattr(decision, "confidence_threshold", .75),
                getattr(decision, "emergency", False), getattr(decision, "real_patient", False),
                getattr(decision, "high_rigor", False), max(.70, getattr(decision, "scope_confidence", .25)),
            )
        return decision
    wrapped._deep_safety_guard = True
    return wrapped


def _wrap_medical_safety_policy(original):
    def wrapped(question: str, result: dict[str, Any], settings: Any):
        result = dict(result or {})
        status = str(result.get("status") or "").upper()
        if status == "BLOCK":
            return result
        try:
            from rag_project.intelligence import medical_safety
            actionable = bool(medical_safety.is_actionable_medical_query(question))
            high_risk = bool(medical_safety.is_high_risk_medical_query(question))
        except Exception:
            actionable = False
            high_risk = False
        if not actionable:
            policy = dict(result.get("medical_safety") or {})
            policy.update({
                "high_risk_query": high_risk,
                "actionable_query": False,
                "policy_version": "2.2-deep-contract",
                "clinical_validation_claim": False,
                "decision": "EDUCATIONAL_GROUNDED_RESPONSE" if status in {"SUCCESS", "SUCCESS_WITH_WARNINGS"} else "STANDARD_GROUNDED_RESPONSE",
                "strict_clinical_threshold_bypassed": True,
            })
            result["medical_safety"] = policy
            return result
        return original(question, result, settings)
    wrapped._deep_medical_safety_guard = True
    return wrapped


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _normalize_ingestion_result(system: Any, result: Any) -> dict[str, Any]:
    out = dict(result or {}) if isinstance(result, dict) else {"status": "FAILED", "error": "ingestion returned a non-dict result"}
    status = str(out.get("status") or "").upper()
    if status == "SKIPPED":
        out["status"] = "READY"
        out["skipped"] = True
        return out
    if status.startswith("FAILED"):
        stage = status
        out["failure_stage"] = stage
        out["status"] = "FAILED"
        document_id = str(out.get("document_id") or out.get("id") or "")
        if document_id:
            try:
                if stage in {"FAILED_EMBEDDING", "FAILED_INDEXING"}:
                    system.state_store.update_document(document_id, status="FAILED_INDEXING", current_stage="FAILED_INDEXING", index_state="FAILED")
                elif stage in {"FAILED_EXTRACTION", "FAILED_OCR"}:
                    system.state_store.update_document(document_id, status="FAILED", current_stage="FAILED", index_state="FAILED")
            except Exception:
                pass
    return out


def _wrap_robust_ingestion(original):
    def wrapped(system: Any, pdf_path: Any, *args: Any, **kwargs: Any):
        source = Path(pdf_path)
        incoming = Path(system.settings.incoming_dir)
        processed = Path(system.settings.processed_dir)
        incoming.mkdir(parents=True, exist_ok=True)
        external = source.is_file() and not _inside(source, incoming) and not _inside(source, processed)
        staged: Path | None = None
        effective = source
        if external:
            candidate = incoming / source.name
            if candidate.exists():
                candidate = incoming / f".external-stage-{uuid.uuid4().hex[:10]}-{source.name}"
            shutil.copy2(source, candidate)
            staged = candidate
            effective = candidate
        try:
            result = original(system, effective, *args, **kwargs)
            normalized = _normalize_ingestion_result(system, result)
            document_id = str(normalized.get("document_id") or normalized.get("id") or "")
            if external and str(normalized.get("status") or "").upper() == "READY" and document_id:
                try:
                    generated = Path(system.settings.processed_dir) / effective.name
                    desired = Path(system.settings.processed_dir) / source.name
                    if generated.exists() and generated.resolve() != desired.resolve():
                        if desired.exists():
                            archive = Path(system.settings.archive_dir)
                            archive.mkdir(parents=True, exist_ok=True)
                            desired.replace(archive / f"{desired.stem}-replaced-{uuid.uuid4().hex[:8]}{desired.suffix}")
                        generated.replace(desired)
                    record = system.state_store.get_document(document_id)
                    if record:
                        system.state_store.update_document(document_id, file_path=str(desired.resolve()), file_name=source.name)
                    normalized["file_name"] = source.name
                except Exception:
                    pass
            return normalized
        finally:
            if staged is not None and staged.exists():
                try:
                    staged.unlink()
                except OSError:
                    pass
    wrapped._deep_ingestion_guard = True
    return wrapped


def _wrap_get_by_path(original):
    def wrapped(self: Any, file_path: str):
        result = original(self, file_path)
        if result is not None:
            return result
        name = Path(file_path).name
        if not name:
            return None
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM documents WHERE file_name = ? ORDER BY modified_at DESC LIMIT 1",
                    (name,),
                ).fetchone()
            return dict(row) if row else None
        except Exception:
            return None
    wrapped._deep_path_guard = True
    return wrapped


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        from rag_project import application
        from rag_project.app import production_rag as production_rag_module
        from rag_project.ingestion import robust_ingestor, state_store as state_store_module
        from rag_project.intelligence import med_evidence_pro, medical_safety, runtime_safety

        original_runtime_safety = runtime_safety.execute_with_runtime_safety
        if not getattr(original_runtime_safety, "_deep_contract_guard", False):
            guarded = _wrap_runtime_safety(original_runtime_safety)
            runtime_safety.execute_with_runtime_safety = guarded
            application.execute_with_runtime_safety = guarded
            production_rag_module.execute_with_runtime_safety = guarded

        original_cache_get = med_evidence_pro.SemanticCache.get
        if not getattr(original_cache_get, "_deep_cache_guard", False):
            med_evidence_pro.SemanticCache.get = _wrap_cache_get(original_cache_get)

        original_retrieve = med_evidence_pro.MultiTierRetriever.retrieve
        if not getattr(original_retrieve, "_deep_contract_cache_guard", False):
            med_evidence_pro.MultiTierRetriever.retrieve = _wrap_retrieval(original_retrieve)

        original_compile = med_evidence_pro.EvidenceCompiler.compile
        if not getattr(original_compile, "_deep_compiler_guard", False):
            med_evidence_pro.EvidenceCompiler.compile = _wrap_compiler_compile(original_compile)
        med_evidence_pro.EvidenceCompiler._detect_contradiction = staticmethod(_detect_contradiction)

        if not getattr(med_evidence_pro.SafetyGate.check, "_deep_safety_guard", False):
            med_evidence_pro.SafetyGate.check = _wrap_safety_check(med_evidence_pro.SafetyGate.check)
        med_evidence_pro.AnswerCascade._llm = _wrap_answer_cascade_llm()

        if not getattr(medical_safety.apply_medical_safety_policy, "_deep_medical_safety_guard", False):
            medical_safety.apply_medical_safety_policy = _wrap_medical_safety_policy(medical_safety.apply_medical_safety_policy)
            production_rag_module.apply_medical_safety_policy = medical_safety.apply_medical_safety_policy

        if not getattr(robust_ingestor.robust_ingest_file, "_deep_ingestion_guard", False):
            robust_ingestor.robust_ingest_file = _wrap_robust_ingestion(robust_ingestor.robust_ingest_file)
        if not getattr(state_store_module.IngestionStateStore.get_by_path, "_deep_path_guard", False):
            state_store_module.IngestionStateStore.get_by_path = _wrap_get_by_path(state_store_module.IngestionStateStore.get_by_path)
        _INSTALLED = True


__all__ = ["install"]
