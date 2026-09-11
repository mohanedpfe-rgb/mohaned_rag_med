from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from rag_project.intelligence.query_intelligence import plan_query
from rag_project.intelligence.semantic_reasoning import understand_query, extract_clinical_entities
from rag_project.utils.text_utils import meaningful_tokens
from rag_project.intelligence.evidence_guard import verify_claims

_PLANNER_SYSTEM=("You are a constrained medical RAG query planner. Return ONLY one JSON object. Do not answer the medical question. Do not invent facts.")
_ALLOWED_INTENTS={"factual","definition","comparison","causal","multi_part","navigation","diagnosis","management","etiology","mechanism","prognosis","relationship","numeric","table_lookup","figure_lookup","other"}
_ALLOWED_AMBIGUITY={"low","medium","high"}

@dataclass(frozen=True)
class PhasePlan:
    intent:str; entities:tuple[str,...]; sub_questions:tuple[str,...]; rewritten_queries:tuple[str,...]; must_contain:tuple[str,...]; ambiguity:str; needs_table:bool; needs_numeric:bool; needs_figure:bool; needs_multi_hop:bool; planner_source:str; planner_confidence:float
    def to_dict(self)->dict[str,Any]:return asdict(self)

def _clean_list(value:Any,limit:int=6)->tuple[str,...]:
    if not isinstance(value,list):return ()
    output=[]
    for item in value:
        text=re.sub(r"\s+"," ",str(item or "")).strip()
        if text and text not in output:output.append(text[:500])
        if len(output)>=limit:break
    return tuple(output)

def _parse_json(raw:str)->dict[str,Any]|None:
    text=str(raw or "").strip()
    if not text:return None
    fenced=re.search(r"```(?:json)?\s*(\{.*?\})\s*```",text,re.S|re.I);candidate=fenced.group(1) if fenced else None
    if candidate is None:
        start,end=text.find("{"),text.rfind("}")
        if start>=0 and end>start:candidate=text[start:end+1]
    if not candidate:return None
    try:
        value=json.loads(candidate);return value if isinstance(value,dict) else None
    except (TypeError,ValueError,json.JSONDecodeError):return None

def _hard_query(plan:Any,understanding:Any,question:str)->bool:
    tokens=meaningful_tokens(question)
    return bool(len(plan.subqueries)>1 or plan.needs_multi_hop or plan.needs_numeric or plan.needs_table or plan.needs_figure or understanding.primary_intent in {"comparison","etiology","mechanism","diagnosis","management","prognosis"} or len(tokens)>=16 or any(term in str(question).casefold() for term in ("why","how does","how do","contraindication","contraindications","dose","dosage","versus","compare","pourquoi","comment","مقارنة","سبب","جرعة","علاج","تشخيص")))

def deterministic_phase1(question:str,conversation_context:str="")->PhasePlan:
    understanding=understand_query(question,conversation_context=conversation_context);plan=plan_query(question,conversation_context=conversation_context);entities=tuple(dict.fromkeys([e.normalized for e in understanding.entities]+list(plan.entities)))[:16];rewritten=tuple(dict.fromkeys([plan.normalized,*plan.variants]))[:8];must=tuple(dict.fromkeys(entities[:8]));generic_factual=(plan.intent in {"factual","definition"} and not entities and not understanding.relations and not plan.needs_numeric and not plan.needs_table and not plan.needs_figure and not plan.needs_multi_hop);ambiguity="low" if generic_factual else "high" if understanding.confidence<.60 else "medium" if understanding.confidence<.82 else "low"
    if not question.strip():ambiguity="high"
    return PhasePlan(plan.intent if plan.intent in _ALLOWED_INTENTS else "other",entities,tuple(plan.subqueries[:6]),rewritten,must,ambiguity,bool(plan.needs_table),bool(plan.needs_numeric),bool(plan.needs_figure),bool(plan.needs_multi_hop),"deterministic",float(understanding.confidence))

def llm_phase1(llm:Any,question:str,deterministic:PhasePlan,conversation_context:str="")->PhasePlan:
    analysis=understand_query(question,conversation_context=conversation_context);core=plan_query(question,conversation_context=conversation_context)
    if llm is None or not _hard_query(core,analysis,question):return deterministic
    prompt=("Return JSON with exactly these keys: intent, entities, sub_questions, rewritten_queries, must_contain, ambiguity, needs_table, needs_numeric. Keep arrays <= 6 items.\n\n" f"Conversation context: {conversation_context[-1600:]}\nQuestion: {question[:2500]}\nDeterministic analysis: {json.dumps(deterministic.to_dict(),ensure_ascii=False)}")
    try:
        generator=getattr(llm,"generate_json",None);raw=generator(prompt=prompt,system_prompt=_PLANNER_SYSTEM,temperature=0.0,max_tokens=200) if callable(generator) else llm.generate(prompt=prompt,system_prompt=_PLANNER_SYSTEM,temperature=0.0);data=_parse_json(raw)
        if not data:return deterministic
        intent=str(data.get("intent") or deterministic.intent).strip().casefold();intent=intent if intent in _ALLOWED_INTENTS else deterministic.intent;ambiguity=str(data.get("ambiguity") or deterministic.ambiguity).strip().casefold();ambiguity=ambiguity if ambiguity in _ALLOWED_AMBIGUITY else deterministic.ambiguity;rewritten=_clean_list(data.get("rewritten_queries"),8) or deterministic.rewritten_queries;sub=_clean_list(data.get("sub_questions"),8) or deterministic.sub_questions;entities=_clean_list(data.get("entities"),16) or deterministic.entities;must=_clean_list(data.get("must_contain"),12) or deterministic.must_contain
        return PhasePlan(intent,entities,sub,rewritten,must,ambiguity,bool(data.get("needs_table",deterministic.needs_table)),bool(data.get("needs_numeric",deterministic.needs_numeric)),deterministic.needs_figure,deterministic.needs_multi_hop or len(sub)>1 or intent in {"causal","comparison","etiology","mechanism"},"small_llm",min(.97,deterministic.planner_confidence+.10))
    except Exception:return deterministic

def _clean_follow_up(question:str,history:Sequence[tuple[str,str]]|None=None)->str:
    cleaned=re.sub(r"\s+"," ",str(question or "")).strip()
    if not cleaned or not history:return cleaned
    explicit=bool(re.search(r"\b(what about|how about|it|this|that|they|them|those|these)\b",cleaned,re.I) or re.match(r"^(and|also|then|et|puis|و|ثم)\b",cleaned,re.I|re.UNICODE) or cleaned.startswith(("و","ثم","هذا","هذه","ذلك","تلك")))
    if not explicit:return cleaned
    recent=list(history)[-3:]
    anchor_question=next((str(q or "").strip() for q,_ in reversed(recent) if str(q or "").strip()),"")
    anchor_answer=next((str(a or "").strip() for q,a in reversed(recent) if str(q or "").strip() and str(a or "").strip()),"")
    if not anchor_question:return cleaned
    terms=[]
    try:
        for entity in extract_clinical_entities(anchor_answer):
            value=str(entity.normalized or entity.text or "").strip()
            if value and value.casefold() not in {x.casefold() for x in terms}:terms.append(value)
    except Exception:pass
    for token in re.findall(r"\b[a-zA-Z][a-zA-Z-]{5,}\b",anchor_answer):
        if token.casefold() not in {x.casefold() for x in terms}:terms.append(token)
        if len(terms)>=4:break
    return " ".join(x for x in (anchor_question," ".join(terms[:4]),cleaned) if x).strip()[:3500]

def rewrite_follow_up(question:str,history:Sequence[tuple[str,str]]|None=None)->str:
    return _clean_follow_up(question,history)

def medical_term_layer(question:str,evidence:Sequence[Any]=())->dict[str,Any]:
    text=" ".join([str(question or "")]+[str(getattr(hit,"text","") or "") for hit in evidence[:16]])
    units=re.findall(r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|ug|g|kg|mL|ml|L|mmHg|mmol/L|mol/L|%|IU|units?|bpm|°C|C|mEq/L|mEq|mOsm/L|ng/mL|pg/mL|U/L|kPa)(?=\s|$|[^\w])",text,re.I)
    abbreviations=re.findall(r"\b[A-Z](?:[A-Z0-9]){1,7}(?:[-/][A-Z0-9]{1,6})?\b",question or "")
    mixed_case_measurement_abbreviations=re.findall(r"\b(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Z])[A-Za-z][A-Za-z0-9]{1,7}\b",question or "")
    abbreviations.extend(mixed_case_measurement_abbreviations);abbreviations=[x for x in abbreviations if x.casefold() not in {"what","this","that","which","where","when","with","from","and","the"}]
    drug_names={"metformin","dapagliflozin","lisinopril","enalapril"};suffixes=("pril","olol","sartan","statin","azole","cillin","mycin","vir","mab","nib","prazole","tidine","caine","cycline","floxacin","lukast","setron","gliptin","gliflozin","tide","parin","dipine","xaban","oxetine","triptan","cept","formin")
    word_candidates=re.findall(r"(?<!\w)[A-Za-z][A-Za-z0-9-]{3,}(?!\w)",question or "",re.I);drug_like=[x for x in word_candidates if x.casefold() in drug_names or x.casefold().endswith(suffixes)];conditions=re.findall(r"\b[a-zà-ÿ][a-zà-ÿ-]{5,}(?:itis|osis|emia|pathy|carcinoma|oma|algia|penia|iasis|megaly|cytosis|trophy|sclerosis|stenosis|ectasia)\b",question or "",re.I)
    try:clinical=[e.normalized for e in extract_clinical_entities(question)]
    except Exception:clinical=[]
    terms=list(dict.fromkeys([*clinical,*abbreviations,*drug_like,*conditions]))[:48];return {"terms":terms,"units":list(dict.fromkeys(units))[:20],"abbreviations":list(dict.fromkeys(abbreviations))[:16],"drug_like":list(dict.fromkeys(drug_like))[:16],"condition_like":list(dict.fromkeys(conditions))[:16]}

def precision_filter(hits:Sequence[Any],phase:PhasePlan,limit:int=16)->list[Any]:
    required={re.sub(r"[^\w]+"," ",x.casefold()).strip() for x in phase.must_contain if x.strip()};query_tokens=set(meaningful_tokens(" ".join(phase.rewritten_queries)));scored=[]
    for hit in hits:
        text=str(getattr(hit,"text","") or "");norm=re.sub(r"[^\w]+"," ",text.casefold());overlap=sum(1 for term in required if term and term in norm);token_score=len(set(meaningful_tokens(text))&query_tokens)/max(1,len(query_tokens));numeric_bonus=.10 if phase.needs_numeric and re.search(r"\d",text) else 0.;base=float(getattr(hit,"score",0.) or 0.);total=.46*base+.34*token_score+.15*min(1.,overlap/max(1,len(required)))+.05+numeric_bonus;scored.append((total,-len(text),hit))
    scored.sort(key=lambda r:(r[0],r[1]),reverse=True);out=[];seen=set()
    for _,_,hit in scored:
        meta=getattr(hit,"metadata",{}) or {};key=(str(meta.get("document_id") or ""),str(meta.get("chunk_id") or getattr(hit,"text","")))
        if key in seen:continue
        seen.add(key);out.append(hit)
        if len(out)>=limit:break
    return out

def compress_context(question:str,hits:Sequence[Any],max_chars:int=6500)->tuple[str,dict[str,Any]]:
    query_tokens=set(meaningful_tokens(question));candidates=[]
    for hit in hits:
        for sentence in re.split(r"(?<=[.!?؟])\s+|\n+",str(getattr(hit,"text","") or "")):
            sentence=re.sub(r"\s+"," ",sentence).strip()
            if not sentence:continue
            tokens=set(meaningful_tokens(sentence));overlap=len(tokens&query_tokens)/max(1,len(query_tokens));number_bonus=.08 if re.search(r"\d",sentence) else 0.;length_penalty=.08 if len(sentence)>420 else 0.;candidates.append((overlap+number_bonus-length_penalty,sentence))
    selected=[];seen=set()
    for _,sentence in sorted(candidates,reverse=True):
        key=sentence.casefold()
        if key in seen:continue
        candidate=" ".join([*selected,sentence])
        if len(candidate)>max_chars:continue
        seen.add(key);selected.append(sentence)
    total=sum(len(str(getattr(h,"text","") or "")) for h in hits);return "\n".join(selected),{"input_sentences":len(candidates),"selected_sentences":len(selected),"compression_ratio":round(len(" ".join(selected))/max(1,total),3)}

def adaptive_retrieve(system:Any,question:str,phase:PhasePlan,initial_hits:Sequence[Any],metadata_filter:dict[str,Any]|None=None)->tuple[list[Any],dict[str,Any]]:
    settings=getattr(system,"settings",None);top_k=max(1,int(getattr(settings,"top_k",8))) if settings is not None else 8;budget={"stage":1,"queries":1,"candidate_k":max(12,top_k*3),"escalated":False,"reasons":[]};merged=list(initial_hits);hard=phase.ambiguity!='low' or phase.needs_multi_hop or phase.needs_numeric or phase.needs_table or len(phase.rewritten_queries)>1
    if system is None:
        budget.update({"stage":0,"queries":0,"candidate_k":0,"reasons":["no_system"]});return list(initial_hits),budget
    if hard:
        budget.update({"stage":2,"queries":min(8,max(2,len(phase.rewritten_queries)+len(phase.sub_questions))),"candidate_k":max(24,top_k*5),"escalated":True});queries=list(dict.fromkeys([*phase.rewritten_queries,*phase.sub_questions]))[:8]
        for query in queries:
            try:merged.extend(system.retriever.retrieve(query,top_k=budget["candidate_k"],where=metadata_filter) or ())
            except Exception as exc:budget["reasons"].append(f"retrieval_branch_failed:{type(exc).__name__}")
        if phase.needs_multi_hop:budget["stage"]=3;budget["reasons"].append("multi_hop")
        if phase.needs_numeric:budget["reasons"].append("numeric_precision")
        if phase.needs_table:budget["reasons"].append("table_lookup")
        if phase.needs_figure:budget["reasons"].append("figure_lookup")
    selected=precision_filter(merged,phase,limit=max(8,top_k*2));budget["final_hits"]=len(selected);return selected,budget

def dynamic_temperature(phase:PhasePlan,default:float=.2)->float:
    return 0.0 if (phase.needs_numeric or phase.needs_multi_hop or phase.needs_table or phase.needs_figure or phase.intent in {"diagnosis","management","etiology","mechanism","prognosis"} or phase.ambiguity!='low') else min(.2,max(0.,float(default)))

def extractive_draft(question:str,hits:Sequence[Any],phase:PhasePlan,max_sentences:int=8)->tuple[str,dict[str,Any]]:
    query_tokens=set(meaningful_tokens(" ".join(phase.rewritten_queries)));ranked=[]
    for index,hit in enumerate(hits):
        marker=f"[S{index+1}]";hit_score=max(0.,min(1.,float(getattr(hit,"score",0.) or 0.)))
        for sentence in re.split(r"(?<=[.!?؟])\s+|\n+",str(getattr(hit,"text","") or "")):
            sentence=re.sub(r"\s+"," ",sentence).strip()
            if not sentence:continue
            overlap=len(set(meaningful_tokens(sentence))&query_tokens)/max(1,len(query_tokens));bonus=.15 if re.search(r"\d",sentence) and phase.needs_numeric else 0.;retrieval_bonus=.22*hit_score;ranked.append((overlap+bonus+retrieval_bonus,f"{sentence} {marker}"))
    ranked.sort(key=lambda x:x[0],reverse=True);chosen=[];seen=set()
    for score,sentence in ranked:
        normalized=re.sub(r"\[S\d+\]","",sentence).casefold()
        if normalized in seen or score<=.06:continue
        seen.add(normalized);chosen.append(sentence)
        if len(chosen)>=max_sentences:break
    return "\n".join(f"- {s}" for s in chosen),{"sentence_count":len(chosen),"supported":bool(chosen)}

_SYNTHESIS_SYSTEM=("You are the final synthesis stage of a medical document RAG system. Use ONLY the supplied extractive facts. Never add a fact, recommendation, diagnosis, causal link, number, unit, population, timing, severity, frequency, or qualifier that is absent from the facts. Every material sentence MUST contain a supplied [S#] citation. Preserve negation exactly. For causal or multi-hop questions, reproduce only relationships explicitly present across supplied facts. If synthesis cannot be done without adding information, return an empty response.")
def synthesize_answer(llm:Any,question:str,draft:str,phase:PhasePlan,temperature:float=0.)->str|None:
    if llm is None or not draft.strip():return None
    prompt=f"Question: {question[:2200]}\n\nIntent: {phase.intent}\nExtractive facts (authoritative and complete):\n{draft[:9000]}"
    try:
        value=llm.generate(prompt=prompt,system_prompt=_SYNTHESIS_SYSTEM,temperature=temperature);return str(value or "").strip()[:10000] or None
    except Exception:return None

def complete_phases(system:Any,question:str,base_result:dict[str,Any],metadata_filter:dict[str,Any]|None=None)->dict[str,Any]:
    memory=getattr(system,"conversation_memory",None);history=getattr(memory,"history",[]) or [];context=getattr(memory,"prompt_context",lambda:" ")() if memory is not None else "";rewritten_question=rewrite_follow_up(question,history);deterministic=deterministic_phase1(rewritten_question,conversation_context=context);phase=llm_phase1(getattr(system,"llm",None),rewritten_question,deterministic,conversation_context=context);initial_hits=list(base_result.get("hits") or []);selected,retrieval_state=adaptive_retrieve(system,rewritten_question,phase,initial_hits,metadata_filter);terms=medical_term_layer(rewritten_question,selected);compact,compression=compress_context(rewritten_question,selected,max_chars=max(4000,int(getattr(getattr(system,"settings",None),"context_token_budget",3200))*3));extractive,extractive_state=extractive_draft(rewritten_question,selected,phase);temperature=dynamic_temperature(phase,getattr(getattr(system,"settings",None),"temperature",.2));hard_query=_hard_query(plan_query(rewritten_question,conversation_context=context),understand_query(rewritten_question,conversation_context=context),rewritten_question);enhanced=dict(base_result);enhanced.update({"phase_plan":phase.to_dict(),"rewritten_question":rewritten_question,"medical_term_layer":terms,"adaptive_retrieval":retrieval_state,"compressed_context":compact,"context_compression":compression,"extractive_stage":{"draft":extractive,**extractive_state},"generation_temperature":temperature,"two_stage_policy":{"required":hard_query,"reason":"hard_medical_query" if hard_query else "standard_answerable_query"},"phases":{"phase_1_query_understanding":"complete","phase_2_retrieval_precision":"complete" if retrieval_state.get("stage",0)>0 or selected else "abstain","phase_3_two_stage_generation":"required" if hard_query else "complete","phase_4_verification":"complete","phase_5_intelligence_visibility":"complete"}})
    llm=getattr(system,"llm",None) if system is not None else None
    if not extractive_state["supported"]:
        enhanced["two_stage_synthesis"]={"used":False,"attempted":False,"required":hard_query,"temperature":temperature,"fallback":True,"reason":"no_extractable_evidence"};enhanced["generation_path"]="required_two_stage_abstention" if hard_query else "certified_primary_fallback"
        enhanced["status"]="GENERATION_ABSTAIN" if hard_query else "ANSWER_UNAVAILABLE"
        if hard_query:enhanced["answer"]="The indexed evidence was insufficient to safely perform the required clinical synthesis."
        return enhanced
    if system is None and str(base_result.get("answer","")).strip():
        base_answer=str(base_result.get("answer","")).strip();evidence_texts=[str(getattr(h,"text","") or "") for h in selected];base_checks=verify_claims(base_answer,evidence_texts,[f"S{i+1}" for i in range(len(selected))]) if evidence_texts else [];base_blocked=any(c.status in {'UNSUPPORTED','WEAK','NUMERIC_MISMATCH','CONTRADICTED'} or c.contradiction for c in base_checks)
        if base_checks and not base_blocked:
            enhanced["two_stage_synthesis"]={"used":False,"attempted":False,"required":hard_query,"temperature":temperature,"fallback":False,"reason":"no_runtime_llm_preserved_verified_base"};enhanced["generation_path"]="verified_base_preserved_no_runtime_llm";enhanced["verification"]={"checked":True,"blocked":False,"claim_count":len(base_checks)};enhanced.setdefault("status","ANSWER_READY");return enhanced
    synthesized=synthesize_answer(llm,rewritten_question,extractive,phase,temperature=temperature);verified_synthesis=[]
    if synthesized:
        verified_synthesis=verify_claims(synthesized,[str(getattr(h,"text","") or "") for h in selected],[f"S{i+1}" for i in range(len(selected))]);blocked=any(c.status in {'UNSUPPORTED','WEAK','NUMERIC_MISMATCH','CONTRADICTED'} or c.contradiction for c in verified_synthesis)
        if blocked:synthesized=None
    enhanced["two_stage_synthesis"]={"used":bool(synthesized),"attempted":True,"required":hard_query,"temperature":temperature,"fallback":not bool(synthesized),"verification":{"checked":bool(verified_synthesis),"blocked":any(c.status in {'UNSUPPORTED','NUMERIC_MISMATCH','CONTRADICTED'} or c.contradiction for c in verified_synthesis),"claim_count":len(verified_synthesis)}}
    if synthesized:enhanced["answer"]=synthesized;enhanced["generation_path"]="two_stage_extract_synthesize";enhanced["status"]="ANSWER_READY"
    elif hard_query:enhanced["status"]="GENERATION_ABSTAIN";enhanced["answer"]="The evidence was retrieved, but the required synthesis could not be verified without adding unsupported clinical content.";enhanced["generation_path"]="required_two_stage_abstention"
    else:enhanced["generation_path"]="extractive_verified_fallback";enhanced["answer"]=extractive;enhanced["status"]="ANSWER_READY"
    return enhanced

__all__=["PhasePlan","deterministic_phase1","llm_phase1","rewrite_follow_up","medical_term_layer","precision_filter","compress_context","adaptive_retrieve","dynamic_temperature","extractive_draft","synthesize_answer","complete_phases"]