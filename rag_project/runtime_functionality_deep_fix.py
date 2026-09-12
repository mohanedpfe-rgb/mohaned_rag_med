"""Final functionality-only repairs for deterministic answer behavior."""
from __future__ import annotations

import re
import threading
from dataclasses import replace
from typing import Any

_LOCK = threading.RLock()
_INSTALLED = False
_TLS = threading.local()


def _wrap_retrieval_cache_fallthrough(original):
    def wrapped(self: Any, question: str, route: Any, where: dict[str, Any] | None = None):
        hits, state = original(self, question, route, where)
        state = dict(state or {})
        if where is None and str(state.get("tier", "")).upper() == "CACHE" and not hits:
            cache = getattr(self, "cache", None)
            delete = getattr(cache, "delete", None)
            if callable(delete):
                delete(question)
            return original(self, question, route, where)
        return hits, state
    wrapped._functionality_cache_fallthrough = True
    return wrapped


def _wrap_retrieval_skip_health_probe(original):
    def wrapped(self: Any, question: str, route: Any, where: dict[str, Any] | None = None):
        retriever = getattr(getattr(self, "system", None), "retriever", None)
        method = getattr(retriever, "retrieve", None)
        if not callable(method):
            return original(self, question, route, where)
        called = {"probe": False}
        original_method = method
        def proxy(query: Any, top_k: Any = 8, filter_where: Any = None, *args: Any, **kwargs: Any):
            if not called["probe"] and query == question and top_k == 1 and filter_where == where and not args and not kwargs:
                called["probe"] = True
                return []
            return original_method(query, top_k, filter_where, *args, **kwargs)
        try:
            retriever.retrieve = proxy
            return original(self, question, route, where)
        finally:
            retriever.retrieve = original_method
    wrapped._functionality_probe_guard = True
    return wrapped


def _citation_complete_without_shared_state(original) -> Any:
    def wrapped(answer: str, hit_count: int) -> bool:
        expected_ids = getattr(_TLS, "citation_ids", None)
        markers = {int(x) for x in re.findall(r"\[S(\d+)\]", str(answer or ""), flags=re.I)}
        if expected_ids is not None:
            try:
                allowed_ids = {int(x) for x in expected_ids}
            except (TypeError, ValueError):
                allowed_ids = set()
            if not markers or not markers.issubset(allowed_ids):
                return False
        expected = getattr(_TLS, "citation_limit", None)
        return original(answer, int(expected if expected is not None else hit_count))
    wrapped._functionality_citation_guard = True
    return wrapped


def _wrap_generate(original):
    def wrapped(self: Any, question: str, route: Any, compiled: dict[str, Any]):
        claims = list(compiled.get("claims") or [])
        source_numbers = [int(number) for claim in claims for number in (getattr(claim, "source_numbers", ()) or ()) if isinstance(number, int) or str(number).isdigit()]
        previous_limit = getattr(_TLS, "citation_limit", None)
        previous_ids = getattr(_TLS, "citation_ids", None)
        _TLS.citation_limit = max(source_numbers, default=0)
        _TLS.citation_ids = set(source_numbers)
        try:
            return original(self, question, route, compiled)
        finally:
            if previous_limit is None:
                try: del _TLS.citation_limit
                except AttributeError: pass
            else: _TLS.citation_limit = previous_limit
            if previous_ids is None:
                try: del _TLS.citation_ids
                except AttributeError: pass
            else: _TLS.citation_ids = previous_ids
    wrapped._functionality_generate_guard = True
    return wrapped

_NUMERIC_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|µg|g|kg|mL|ml|L|mmHg|mmol/L|%|IU|units?)\b", re.I)
_UNIT_SCALE = {"kg":1_000_000.0,"g":1_000.0,"mg":1.0,"mcg":0.001,"µg":0.001,"l":1_000.0,"ml":1.0,"mmhg":1.0,"mmol/l":1.0,"%":1.0,"iu":1.0,"unit":1.0,"units":1.0}
_UNIT_DIMENSION = {"kg":"mass","g":"mass","mg":"mass","mcg":"mass","µg":"mass","l":"volume","ml":"volume","mmhg":"pressure","mmol/l":"concentration","%":"percent","iu":"activity","unit":"activity","units":"activity"}


def _numeric_unit_equivalent(left: str, right: str) -> bool:
    a = _NUMERIC_RE.fullmatch(str(left or "").strip()); b = _NUMERIC_RE.fullmatch(str(right or "").strip())
    if not a or not b: return False
    ua = a.group("unit").casefold().replace(" ", ""); ub = b.group("unit").casefold().replace(" ", "")
    if _UNIT_DIMENSION.get(ua) != _UNIT_DIMENSION.get(ub): return False
    sa = _UNIT_SCALE.get(ua); sb = _UNIT_SCALE.get(ub)
    if sa is None or sb is None: return False
    try: va = float(a.group("value"))*sa; vb = float(b.group("value"))*sb
    except (TypeError, ValueError): return False
    return abs(va-vb) <= 1e-9*max(1.0,abs(va),abs(vb))


def _wrap_numeric_verifier(original):
    def wrapped(self: Any, answer: str, hits: Any, route: Any, compiled: dict[str, Any]):
        result = dict(original(self, answer, hits, route, compiled) or {})
        if not result.get("numeric_mismatch") or not answer or not hits: return result
        answer_values = [m.group(0) for m in _NUMERIC_RE.finditer(answer)]
        evidence_values = [m.group(0) for hit in hits for m in _NUMERIC_RE.finditer(str(getattr(hit, "text", "") or ""))]
        if not answer_values or not evidence_values: return result
        if not all(any(_numeric_unit_equivalent(a, e) for e in evidence_values) for a in answer_values): return result
        grounding = result.get("grounding") if isinstance(result.get("grounding"), dict) else {}
        final = result.get("final_answer") if isinstance(result.get("final_answer"), dict) else {}
        result["numeric_mismatch"] = False
        result["allow"] = bool(grounding.get("allow")) and bool(final.get("allow", True))
        return result
    wrapped._functionality_numeric_guard = True
    return wrapped


def _filter_false_numeric_contradictions(original):
    def wrapped(claims: Any):
        result = dict(original(claims) or {}); conflicts = []
        for conflict in result.get("conflicts") or []:
            left=[str(value) for value in conflict.get("left") or ()]; right=[str(value) for value in conflict.get("right") or ()]
            comparable=[(a,b) for a in left for b in right if _NUMERIC_RE.fullmatch(a.strip()) and _NUMERIC_RE.fullmatch(b.strip())]
            if comparable and all(_numeric_unit_equivalent(a,b) for a,b in comparable): continue
            conflicts.append(conflict)
        result["conflicts"]=conflicts[:8]; result["has_contradiction"]=bool(conflicts); result["agreement"]=0.65 if conflicts else 1.0
        return result
    wrapped._functionality_unit_contradiction_guard = True
    return wrapped


def _wrap_contextual_numeric_contradictions(original):
    """Keep numeric conflicts only when the compared claims discuss the same fact."""
    def wrapped(claims: Any):
        claim_list = list(claims or ())
        base = dict(original(claim_list) or {})
        try:
            from rag_project.intelligence import evidence_guard
            extract = evidence_guard.extract_measurements
            compatible = evidence_guard._measurement_compatible
            token_fn = evidence_guard.meaningful_tokens
        except Exception:
            return base
        numeric_claims = []
        for claim in claim_list:
            text = str(getattr(claim, "text", "") or "")
            measurements = extract(text)
            if measurements:
                stripped = re.sub(r"\[S\d+\]", " ", text)
                stripped = evidence_guard.MEASURE.sub(" ", stripped)
                tokens = set(token_fn(stripped))
                numeric_claims.append((measurements, tokens, text))
        conflicts = []
        for i, (left_values, left_tokens, _) in enumerate(numeric_claims):
            for right_values, right_tokens, _ in numeric_claims[i + 1:]:
                shared = left_tokens & right_tokens
                if len(shared) < 2:
                    continue
                comparable = [(a, b) for a in left_values for b in right_values]
                mismatching = [(a, b) for a, b in comparable if not compatible(a, b)]
                if not mismatching:
                    continue
                conflicts.append({
                    "left": [f"{value} {unit}" for value, unit in left_values],
                    "right": [f"{value} {unit}" for value, unit in right_values],
                    "shared_terms": sorted(shared)[:8],
                })
                if len(conflicts) >= 8:
                    return {"has_contradiction": True, "conflicts": conflicts, "agreement": .65}
        return {"has_contradiction": bool(conflicts), "conflicts": conflicts, "agreement": .65 if conflicts else 1.0}
    wrapped._functionality_contextual_numeric_guard = True
    return wrapped


def _functionality_sentences(text: Any) -> list[str]:
    out=[]
    for part in re.split(r"(?<=[.!?؟])\s+|\n+", str(text or "")):
        part=re.sub(r"^[-*•\s]+","",re.sub(r"\s+"," ",part).strip())
        if len(part)>=2: out.append(part)
    return out


def _wrap_route(original):
    def wrapped(self: Any, question: str, context: str, safety: Any):
        route=original(self,question,context,safety); q=re.sub(r"\s+"," ",str(question or "")).strip().casefold()
        explicit=bool(re.search(r"\b(?:what about|how about|it|this|that|they|them|also)\b",q) or re.match(r"^(?:and|et|puis|و|ثم)\b",q,flags=re.I|re.UNICODE))
        return replace(route,is_follow_up=explicit) if bool(getattr(route,"is_follow_up",False))!=explicit else route
    wrapped._functionality_followup_guard=True
    return wrapped


def _wrap_contradiction_detection(original):
    """Avoid false contradictions caused by unrelated negation with weak context overlap."""
    def wrapped(claim: Any, evidence_blocks: Any):
        blocks = [evidence_blocks] if isinstance(evidence_blocks, str) else list(evidence_blocks or ())
        result = bool(original(claim, blocks))
        if not result:
            return False
        claim_text = str(claim or "").casefold()
        claim_tokens = set(re.findall(r"[\w-]{3,}", claim_text, flags=re.UNICODE))
        try:
            from rag_project.intelligence import evidence_guard
            semantic_support = evidence_guard.semantic_support
        except Exception:
            semantic_support = None
        for block in blocks:
            text = str(block or "").casefold()
            shared = claim_tokens & set(re.findall(r"[\w-]{3,}", text, flags=re.UNICODE))
            explicit = bool(
                (re.search(r"\b(?:contraindicated|avoid|should not|without|absent|absence|negative|no)\b", claim_text) and re.search(r"\b(?:indicated|recommended|should|with|present|detected|positive|has)\b", text))
                or (re.search(r"\b(?:indicated|recommended|should|with|present|detected|positive|has)\b", claim_text) and re.search(r"\b(?:contraindicated|avoid|should not|without|absent|absence|negative|no)\b", text))
            )
            semantic = float(semantic_support(claim, text)) if semantic_support is not None else 0.0
            if len(shared) >= 2 or (explicit and semantic >= 0.55):
                return True
        return False
    wrapped._functionality_contradiction_guard = True
    return wrapped


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED: return
        from rag_project.intelligence import med_evidence_pro
        try:
            from rag_project.intelligence import evidence_guard
        except Exception:
            evidence_guard = None
        original_retrieve=med_evidence_pro.MultiTierRetriever.retrieve
        if not getattr(original_retrieve,"_functionality_probe_guard",False):
            med_evidence_pro.MultiTierRetriever.retrieve=_wrap_retrieval_skip_health_probe(original_retrieve)
        current_retrieve=med_evidence_pro.MultiTierRetriever.retrieve
        if not getattr(current_retrieve,"_functionality_cache_fallthrough",False):
            med_evidence_pro.MultiTierRetriever.retrieve=_wrap_retrieval_cache_fallthrough(current_retrieve)
        original_citation=med_evidence_pro.AnswerCascade._citation_complete
        if not getattr(original_citation,"_functionality_citation_guard",False):
            med_evidence_pro.AnswerCascade._citation_complete=staticmethod(_citation_complete_without_shared_state(original_citation))
        original_generate=med_evidence_pro.AnswerCascade.generate
        if not getattr(original_generate,"_functionality_generate_guard",False):
            med_evidence_pro.AnswerCascade.generate=_wrap_generate(original_generate)
        original_verify=med_evidence_pro.ActiveVerifier.verify
        if not getattr(original_verify,"_functionality_numeric_guard",False):
            med_evidence_pro.ActiveVerifier.verify=_wrap_numeric_verifier(original_verify)
        original_route=med_evidence_pro.QueryRouter.route
        if not getattr(original_route,"_functionality_followup_guard",False):
            med_evidence_pro.QueryRouter.route=_wrap_route(original_route)
        original_sentences=med_evidence_pro._sentences
        if not getattr(original_sentences,"_functionality_short_sentence_guard",False):
            med_evidence_pro._sentences=_functionality_sentences; med_evidence_pro._sentences._functionality_short_sentence_guard=True
        original_contradiction=med_evidence_pro.EvidenceCompiler._detect_contradiction
        if not getattr(original_contradiction,"_functionality_unit_contradiction_guard",False):
            med_evidence_pro.EvidenceCompiler._detect_contradiction=staticmethod(_filter_false_numeric_contradictions(original_contradiction))
        current_contradiction=med_evidence_pro.EvidenceCompiler._detect_contradiction
        if not getattr(current_contradiction,"_functionality_contextual_numeric_guard",False):
            med_evidence_pro.EvidenceCompiler._detect_contradiction=staticmethod(_wrap_contextual_numeric_contradictions(current_contradiction))
        if evidence_guard is not None:
            original_guard_contradiction = evidence_guard.detect_contradiction
            if not getattr(original_guard_contradiction,"_functionality_contradiction_guard",False):
                evidence_guard.detect_contradiction = _wrap_contradiction_detection(original_guard_contradiction)
        _INSTALLED=True

__all__=["install","_wrap_retrieval_cache_fallthrough","_wrap_retrieval_skip_health_probe","_citation_complete_without_shared_state","_wrap_generate","_wrap_route","_functionality_sentences","_filter_false_numeric_contradictions","_wrap_numeric_verifier","_wrap_contradiction_detection","_wrap_contextual_numeric_contradictions"]
