"""Production integrity fixes for query contamination and answer-gate failures."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable, Sequence

from rag_project.utils.text_utils import meaningful_tokens
from rag_project.intelligence.semantic_reasoning import extract_clinical_entities

_INTERNAL_LABELS={"follow-up","follow up","relevant entities","query entities","planned entities","intent","planner","planner confidence","conversation context","semantic understanding","query analysis","retrieval state","evidence confidence","final gate","claim matrix"}
_STOPWORDS={"the","and","with","from","this","that","what","main","findings","relevant","entities"}
_WORD=re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_-]{3,}")
_MEASUREMENT=re.compile(r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|ug|g|kg|mL|ml|L|mmHg|mmol/L|mol/L|%|IU|units?|bpm|°C|mEq/L|mEq|mOsm/L|ng/mL|pg/mL|U/L|kPa)(?=\s|$|[^\w])",re.I)
_ABBREVIATION=re.compile(r"\b[A-Z](?:[A-Z0-9]){1,7}(?:[-/][A-Z0-9]{1,6})?\b")
_ABSTENTION_PREFIXES=("the indexed evidence was insufficient","the evidence was retrieved, but","i could not verify","insufficient evidence","unable to verify","unsupported content was withheld")
_DRUG_SUFFIXES=("pril","olol","sartan","statin","azole","cillin","mycin","vir","mab","nib","prazole","tidine","caine","cycline","floxacin","lukast","setron","gliptin","gliflozin","tide","parin","dipine","xaban","oxetine","triptan","cept","formin")
_CONDITION_SUFFIXES=("itis","osis","emia","pathy","carcinoma","oma","algia","penia","iasis","megaly","cytosis","trophy","sclerosis","stenosis","ectasia")

def _normalize(value:str)->str:return re.sub(r"\s+"," ",str(value or "")).strip().casefold()
def _canonical(value:str)->str:
    try:
        from rag_project.intelligence.semantic_reasoning import normalize_medical_term
        return _normalize(normalize_medical_term(value))
    except Exception:return _normalize(value)
def _contains_internal_label(text:str)->bool:
    normalized=_normalize(text);return bool(normalized) and any(label in normalized for label in _INTERNAL_LABELS)
def _open_set_medical_terms(text:str)->list[str]:
    found=[]
    for match in _WORD.finditer(text or ""):
        token=match.group(0);normalized=_normalize(token)
        if len(normalized)<5 or normalized in _STOPWORDS:continue
        if normalized.endswith(_DRUG_SUFFIXES) or normalized.endswith(_CONDITION_SUFFIXES):
            canonical=_canonical(normalized)
            if canonical:found.append(canonical)
    return found

def safe_extract_query_entities(question:str,planned_entities:Iterable[str]=())->tuple[str,...]:
    found=[];deterministic=set()
    try:
        for entity in extract_clinical_entities(question or ""):
            canonical=_canonical(entity.normalized or entity.text)
            if canonical and canonical not in deterministic:deterministic.add(canonical);found.append(canonical)
    except Exception:pass
    for term in _open_set_medical_terms(question or ""):
        if term not in found:deterministic.add(term);found.append(term)
    for match in _MEASUREMENT.finditer(question or ""):
        value=_normalize(match.group(0))
        if value not in found:found.append(value)
    for match in _ABBREVIATION.finditer(question or ""):
        token=match.group(0);normalized=_normalize(token)
        if normalized in _STOPWORDS or len(normalized)<2:continue
        canonical=_canonical(token)
        if canonical in deterministic or len(token)>=3:
            for value in (canonical,normalized):
                if value and value not in found:found.append(value)
    for item in planned_entities:
        raw=_normalize(item)
        if not raw or len(raw)>64 or " " in raw or ":" in raw or _contains_internal_label(raw):continue
        medical_like=raw in deterministic or bool(re.search(r"[-_0-9]",raw)) or raw.endswith(_DRUG_SUFFIXES) or raw.endswith(_CONDITION_SUFFIXES) or len(raw)>=7
        if medical_like:
            canonical=_canonical(raw)
            if canonical and canonical not in found:found.append(canonical)
    unique=[];seen=set()
    for value in found:
        value=_normalize(value)
        if not value or value in seen or value in _STOPWORDS or _contains_internal_label(value):continue
        seen.add(value);unique.append(value)
        if len(unique)>=32:break
    return tuple(unique)

def _entity_overlap(query_entity:str,evidence_entity:str)->float:
    q=set(meaningful_tokens(query_entity));e=set(meaningful_tokens(evidence_entity))
    if not q or not e:return 0.0
    if query_entity==evidence_entity:return 1.0
    return len(q & e)/max(1,len(q))

def _evidence_entities(text:str)->list[str]:
    found=[]
    try:found.extend(_canonical(entity.normalized or entity.text) for entity in extract_clinical_entities(text or ""))
    except Exception:pass
    found.extend(_open_set_medical_terms(text or ""))
    for match in _MEASUREMENT.finditer(text or ""):found.append(_normalize(match.group(0)))
    for match in _ABBREVIATION.finditer(text or ""):
        token=match.group(0);canonical=_canonical(token)
        if canonical not in _STOPWORDS:found.extend((canonical,_normalize(token)))
    return [item for item in found if item]

def safe_score_entity_coverage(question:str,evidence:Sequence[Any],planned_entities:Iterable[str]=())->dict[str,Any]:
    planned_items=tuple(str(item or "") for item in planned_entities);query_entities=safe_extract_query_entities(question,planned_items);evidence_entities=[];planned_norms={_normalize(x) for x in planned_items if _normalize(x)}
    for index,hit in enumerate(evidence):
        text=str(getattr(hit,"text"," ") or "")
        for entity in _evidence_entities(text):evidence_entities.append((entity,f"S{index+1}"))
        for planned in planned_norms:
            if re.search(rf"(?<![\w-]){re.escape(planned)}(?![\w-])",text,re.I):evidence_entities.append((planned,f"S{index+1}"))
    evidence_norms=[name for name,_ in evidence_entities];source_counts=Counter(source for _,source in evidence_entities);covered=[];missing=[];partial=[];per_entity=[];matches=[]
    for entity in query_entities:
        best=max((_entity_overlap(entity,candidate) for candidate in evidence_norms),default=0.0);candidate_sources=tuple(dict.fromkeys(source for candidate,source in evidence_entities if _entity_overlap(entity,candidate)>=max(.72,best-.05)));status="covered" if best>=.72 else "partial" if best>=.45 else "missing";per_entity.append({"entity":entity,"status":status,"match_score":round(best,3),"sources":candidate_sources})
        (covered if status=="covered" else partial if status=="partial" else missing).append(entity)
        if candidate_sources:matches.append({"entity":entity,"evidence_sources":candidate_sources,"match_score":round(best,3)})
    denominator=max(1,len(query_entities));coverage=len(covered)/denominator;partial_coverage=(len(covered)+.5*len(partial))/denominator
    return {"query_entities":list(query_entities),"entity_count":len(query_entities),"covered":covered,"missing":missing,"partial":partial,"coverage":round(coverage,3),"partial_coverage":round(partial_coverage,3),"per_entity":per_entity,"evidence_entity_count":len(evidence_entities),"evidence_entities":list(dict.fromkeys(evidence_norms))[:64],"source_entity_counts":dict(source_counts),"matches":matches}

def _looks_like_followup(question:str,history:Sequence[tuple[str,str]])->bool:
    cleaned=_normalize(question)
    if not cleaned or not history:return False
    english_or_french=re.search(r"\b(and|also|then|it|this|that|they|them|those|these|what about|how about|the latter|the former|et|puis|ça|cela|celui|celle|et le|et la|et les)\b",cleaned,re.I|re.UNICODE)
    arabic=cleaned.startswith(("و","ثم","هذا","هذه","ذلك","تلك"))
    return bool(english_or_french or arabic)

def safe_rewrite_follow_up(question:str,history:Sequence[tuple[str,str]]|None=None)->str:
    cleaned=re.sub(r"\s+"," ",str(question or "")).strip()
    if not cleaned or not history or not _looks_like_followup(cleaned,history):return cleaned
    recent=list(history[-3:]);anchor_question=next((str(q).strip() for q,_ in reversed(recent) if str(q or "").strip()),"");anchor_answer=next((str(a).strip() for q,a in reversed(recent) if str(q or "").strip() and str(a or "").strip()),"")
    if not anchor_question:return cleaned
    context_terms=[]
    try:
        for entity in extract_clinical_entities(anchor_answer):
            term=_canonical(entity.normalized or entity.text)
            if term and term not in context_terms and not _contains_internal_label(term):context_terms.append(term)
    except Exception:pass
    for term in _open_set_medical_terms(anchor_answer):
        if term not in context_terms and not _contains_internal_label(term):context_terms.append(term)
    context=" ".join(context_terms[:6]);candidate=" ".join(part for part in (anchor_question,context,cleaned) if part).strip()
    return re.sub(r"\s+"," ",candidate)[:3500]

def _safe_simple_extractive_answer(question:str,selected_hits:Sequence[Any],max_sentences:int=6)->str:
    question_terms=set(re.findall(r"[\wÀ-ÿ-]{3,}",str(question or "").casefold()));question_terms-={"what","are","the","main","findings","is","this","that","does","document","report","explain","define","list","show","about","principal","biais"};candidates=[]
    for index,hit in enumerate(selected_hits or ()):
        if hit is None:continue
        raw=str(getattr(hit,"text","") or "");raw=re.sub(r"\[(?:section|source|file|page|document|metadata|citation|reference)\s*:\s*.*?\]\s*"," ",raw,flags=re.I|re.S)
        for sentence in re.split(r"(?<=[.!?؟])\s+|\n+",raw):
            sentence=re.sub(r"\s+"," ",sentence).strip()
            if not sentence or len(sentence)<12:continue
            tokens=set(re.findall(r"[\wÀ-ÿ-]{3,}",sentence.casefold()));overlap=len(tokens&question_terms)/max(1,len(question_terms)) if question_terms else 0.;score=.60*float(getattr(hit,"score",0.0) or 0.0)+.40*overlap;candidates.append((score,f"{sentence} [S{index+1}]") )
    candidates.sort(key=lambda row:row[0],reverse=True);chosen=[];seen=set()
    for score,sentence in candidates:
        normalized=re.sub(r"\[S\d+\]", "", sentence).casefold()
        if normalized in seen or score<.18:continue
        seen.add(normalized);chosen.append(sentence)
        if len(chosen)>=max_sentences:break
    return "\n".join(f"- {sentence}" for sentence in chosen)

def is_control_message(text:str)->bool:
    normalized=_normalize(text).lstrip("-•* ")
    if not normalized:return False
    if re.search(r"\[s\d+\]",normalized,re.I) and not any(normalized.startswith(prefix) for prefix in _ABSTENTION_PREFIXES):return False
    return any(normalized.startswith(prefix) for prefix in _ABSTENTION_PREFIXES)

def safe_verify_final_answer(answer:str,hits:Sequence[Any],*,require_entailment:bool=False)->dict[str,Any]:
    if is_control_message(answer):return {"checked":False,"allow":False,"reason":"abstention_not_claim","claim_count":0,"blocked_claims":0,"supported_ratio":0.0,"matrix_claim_count":0,"matrix_all_entailed":False,"claim_checks":[],"evidence_claim_matrix":[]}
    from rag_project.intelligence.evidence_guard import verify_claims
    from rag_project.intelligence.evidence_entailment import build_claim_evidence_matrix
    evidence=[str(getattr(hit,"text","") or "") for hit in hits];source_ids=[f"S{i+1}" for i in range(len(evidence))];checks=verify_claims(answer,evidence,source_ids) if answer.strip() and evidence else [];matrix=build_claim_evidence_matrix([check.claim for check in checks],hits,source_ids) if checks and hits else ();blocked_checks=[check for check in checks if check.status in {"UNSUPPORTED","WEAK","NUMERIC_MISMATCH","CONTRADICTED"} or check.contradiction];blocked_matrix=[record for record in matrix if record.status!="ENTAILED"];supported=sum(check.status in {"SUPPORTED","PARTIAL"} and not check.contradiction for check in checks);support_ratio=supported/max(1,len(checks));matrix_strong=bool(matrix) and not blocked_matrix;allow=bool(checks) and not blocked_checks and support_ratio>=.60 and (matrix_strong if require_entailment else True)
    if not checks:reason="no_verifiable_claims"
    elif blocked_checks:reason="blocked_claims"
    elif support_ratio<.60:reason="support_ratio_below_threshold"
    elif require_entailment and not matrix_strong:reason="final_matrix_not_fully_entailed"
    else:reason="verified"
    return {"checked":bool(checks),"allow":allow,"reason":reason,"claim_count":len(checks),"blocked_claims":len(blocked_checks),"supported_ratio":round(support_ratio,4),"matrix_claim_count":len(matrix),"matrix_all_entailed":matrix_strong,"claim_checks":[check.to_dict() for check in checks],"evidence_claim_matrix":[record.to_dict() for record in matrix]}

def _install_legacy_extractive_guard()->None:
    try:
        from rag_project.intelligence import god_mode as legacy_god_mode
        legacy_god_mode._simple_extractive_answer=_safe_simple_extractive_answer
    except Exception:pass
_install_legacy_extractive_guard()

def install()->None:
    from rag_project.intelligence import top_level_pipeline,entity_coverage,final_answer_contract,god_mode_100
    top_level_pipeline.rewrite_follow_up=safe_rewrite_follow_up;top_level_pipeline._production_integrity_rewrite_installed=True
    entity_coverage.extract_query_entities=safe_extract_query_entities;entity_coverage.score_entity_coverage=safe_score_entity_coverage;entity_coverage._production_integrity_entities_installed=True
    god_mode_100.score_entity_coverage=safe_score_entity_coverage;god_mode_100.verify_final_answer=safe_verify_final_answer;god_mode_100._simple_extractive_answer=_safe_simple_extractive_answer;final_answer_contract.verify_final_answer=safe_verify_final_answer

__all__=["safe_extract_query_entities","safe_score_entity_coverage","safe_rewrite_follow_up","safe_verify_final_answer","is_control_message","install"]
