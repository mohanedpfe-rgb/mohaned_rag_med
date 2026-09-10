from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable, Sequence

from rag_project.intelligence.semantic_reasoning import extract_clinical_entities, normalize_medical_term
from rag_project.utils.text_utils import meaningful_tokens

_MEASUREMENT = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|ug|g|kg|ml|mL|L|mmHg|mmol/L|mol/L|%|IU|units?|bpm|°C|C|mEq/L|mEq|mOsm/L|ng/mL|pg/mL|U/L|kPa)(?=\s|$|[^\w])", re.I)
_ABBREVIATION = re.compile(r"\b[A-Z]{2,8}(?:[-/][A-Z0-9]{1,8})?\b")
_WORD = re.compile(r"\b[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'/-]*\b", re.UNICODE)
_STOP = {"WHAT","THIS","THAT","WHICH","WHERE","WHEN","WITH","FROM","AND","THE","FOR","DOES","HOW","WHY","BETWEEN","IS","ARE","WAS","WERE","CAN","COULD","WOULD","SHOULD","RELATIONSHIP","RELATION","TREATMENT","ABOUT","MAIN","FINDINGS","RELEVANT","ENTITIES"}
_OPEN_SET_MEDICAL_SUFFIXES = ("gliflozin","gliptin","glutide","parin","pril","sartan","olol","azole","cillin","mycin","cycline","vir","mab","nib","tinib","caine","statin","oxetine","pramine","pam","lam","zepam","barb","bital","phylline","terol","lukast","setron")
_OPEN_SET_MEDICAL_TERMS = {"dapagliflozin", "metformin"}


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _canonical(value: str) -> str:
    try:
        return _norm(normalize_medical_term(value))
    except Exception:
        return _norm(value)


def _lexical_open_set_terms(text: str) -> list[str]:
    found=[]
    for match in _WORD.finditer(text or ""):
        token=match.group(0); normalized=_norm(token); upper=token.upper()
        if not normalized or upper in _STOP or len(normalized)<5: continue
        if normalized in _OPEN_SET_MEDICAL_TERMS or normalized.endswith(_OPEN_SET_MEDICAL_SUFFIXES): found.append(_canonical(normalized))
    return found


def _append_entity(found: list[str], raw: str, *, preserve_alias: bool = False) -> None:
    normalized=_norm(raw)
    if not normalized:return
    canonical=_canonical(normalized)
    if canonical:found.append(canonical)
    if preserve_alias and normalized!=canonical:found.append(normalized)


def extract_query_entities(question: str, planned_entities: Iterable[str] = ()) -> tuple[str,...]:
    """Return clinical concepts, measurements and strong abbreviations only.

    Generic multi-word phrase extraction is intentionally forbidden: ordinary phrases
    such as ``main findings`` or ``relevant entities`` are not medical entities.
    """
    found=[];deterministic=set()
    try:
        for entity in extract_clinical_entities(question):
            raw=entity.normalized or entity.text;canonical=_canonical(raw)
            if canonical and canonical not in deterministic: deterministic.add(canonical);found.append(canonical)
    except Exception: pass
    for value in _lexical_open_set_terms(question or ""):
        if value not in found: deterministic.add(value);found.append(value)
    for match in _MEASUREMENT.finditer(question or ""):
        value=_norm(match.group(0))
        if value not in found:found.append(value)
    for match in _ABBREVIATION.finditer(question or ""):
        token=match.group(0);normalized=_norm(token)
        if normalized in {_norm(v) for v in _STOP} or len(normalized)<2:continue
        canonical=_canonical(token)
        if canonical in deterministic or len(token)>=3:
            for value in (canonical,normalized):
                if value and value not in found:found.append(value)
    for item in planned_entities:
        raw=_norm(item)
        if not raw or len(raw)>64 or " " in raw or ":" in raw:continue
        if any(label in raw for label in ("follow-up","follow up","relevant entities","query entities","planned entities")):continue
        medical_like=raw in deterministic or bool(re.search(r"[-_0-9]",raw)) or raw.endswith(_OPEN_SET_MEDICAL_SUFFIXES) or len(raw)>=7
        if medical_like:
            canonical=_canonical(raw)
            if canonical and canonical not in found:found.append(canonical)
    unique=[];seen=set()
    for value in found:
        value=_norm(value)
        if not value or value in seen or value in {_norm(v) for v in _STOP}:continue
        seen.add(value);unique.append(value)
        if len(unique)>=32:break
    return tuple(unique)


def _entity_overlap(query_entity: str, evidence_entity: str) -> float:
    q=set(meaningful_tokens(query_entity));e=set(meaningful_tokens(evidence_entity))
    if not q or not e:return 0.0
    if query_entity==evidence_entity:return 1.0
    return len(q & e)/max(1,len(q))


def score_entity_coverage(question: str, evidence: Sequence[Any], planned_entities: Iterable[str] = ()) -> dict[str,Any]:
    planned_items=tuple(str(item or "") for item in planned_entities)
    query_entities=extract_query_entities(question,planned_items);evidence_entities=[];planned_norms={_norm(x) for x in planned_items if _norm(x)}
    for index,hit in enumerate(evidence):
        text=str(getattr(hit,"text","") or "");source_id=f"S{index+1}"
        try:extracted=extract_clinical_entities(text)
        except Exception:extracted=()
        for entity in extracted:
            normalized=_canonical(entity.normalized or entity.text)
            if normalized:evidence_entities.append((normalized,source_id))
        for lexical in _lexical_open_set_terms(text):evidence_entities.append((lexical,source_id))
        # Explicit planned entities are contracts supplied by the planner, so an
        # exact bounded token match in evidence must be visible to the scorer even
        # when the open-set clinical extractor has never seen the term before.
        for planned in planned_norms:
            if re.search(rf"(?<![\w-]){re.escape(planned)}(?![\w-])",text,re.I):
                evidence_entities.append((planned,source_id))
        for query_entity in query_entities:
            if query_entity in planned_norms and re.search(rf"(?<![\w-]){re.escape(query_entity)}(?![\w-])",text,re.I):
                evidence_entities.append((query_entity,source_id))
        for match in _MEASUREMENT.finditer(text):evidence_entities.append((_norm(match.group(0)),source_id))
        for match in _ABBREVIATION.finditer(text):
            token=match.group(0)
            if token.upper() not in _STOP:
                normalized=_canonical(token);evidence_entities.append((normalized,source_id));alias=_norm(token)
                if alias!=normalized:evidence_entities.append((alias,source_id))
    evidence_norms=[name for name,_ in evidence_entities];source_counts=Counter(source for _,source in evidence_entities);covered=[];missing=[];matches=[];per_entity=[]
    for entity in query_entities:
        best=max((_entity_overlap(entity,candidate) for candidate in evidence_norms),default=0.0);candidate_sources=tuple(dict.fromkeys(source for candidate,source in evidence_entities if _entity_overlap(entity,candidate)>=max(.72,best-.05)));status="covered" if best>=.72 else "partial" if best>=.45 else "missing";row={"entity":entity,"status":status,"match_score":round(best,3),"sources":candidate_sources};per_entity.append(row)
        if status=="covered":covered.append(entity)
        elif status=="missing":missing.append(entity)
        if candidate_sources:matches.append({"entity":entity,"evidence_sources":candidate_sources,"match_score":round(best,3)})
    denominator=max(1,len(query_entities));coverage=len(covered)/denominator;partial_coverage=(len(covered)+.5*sum(row["status"]=="partial" for row in per_entity))/denominator
    return {"query_entities":list(query_entities),"entity_count":len(query_entities),"covered":covered,"missing":missing,"partial":[row["entity"] for row in per_entity if row["status"]=="partial"],"coverage":round(coverage,3),"partial_coverage":round(partial_coverage,3),"per_entity":per_entity,"evidence_entity_count":len(evidence_entities),"evidence_entities":list(dict.fromkeys(evidence_norms))[:64],"source_entity_counts":dict(source_counts),"matches":matches}


__all__=["extract_query_entities","score_entity_coverage"]
