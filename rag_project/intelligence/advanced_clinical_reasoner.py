from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence
from rag_project.intelligence.semantic_reasoning import ClinicalEntity, QueryUnderstanding, extract_clinical_entities

RELATIONS=(('causes',('causes','cause','caused by','due to','leads to','results in','responsible for','provoque','entraine','entraîne','سبب','يؤدي')),('association',('associated with','associated','related to','linked to','association','lié à','lie a','associé à','مرتبط','علاقة')),('diagnoses',('diagnosed by','diagnostic criteria','criteria','supports the diagnosis','diagnostic','critères diagnostiques','معايير التشخيص')),('treated_with',('treated with','managed with','management includes','treated using','indicated for','traité par','prise en charge','يعالج بـ')),('contraindicated',('contraindicated','should not','avoid','not recommended','contre-indiqué','contre indique','éviter','ممنوع','تجنب')),('precedes',('before','prior to','precedes','avant','قبل')),('follows',('after','subsequently','then','followed by','après','ensuite','puis','بعد','ثم')))
_NEG=re.compile(r'\b(no|not|never|without|cannot|does not|doesn\'t|non|ne pas|sans|aucun|ليس|لا|دون)\b',re.I|re.UNICODE)

@dataclass(frozen=True)
class ClinicalFact:
    subject:str; predicate:str; object:str; polarity:int; confidence:float; node_id:str; sentence:str; document_id:str
@dataclass(frozen=True)
class ReasoningPath:
    nodes:tuple[str,...]; relations:tuple[str,...]; fact_ids:tuple[str,...]; support:float; explicit:bool; cross_document:bool
@dataclass(frozen=True)
class ClinicalReasoningAssessment:
    allow_generation:bool; mode:str; depth:int; entity_coverage:float; direct_support:float; path_support:float; source_agreement:float; contradiction:float; safety_conflict:float; confidence:float; supported_paths:tuple[ReasoningPath,...]; blocked_reasons:tuple[str,...]
    def to_dict(self)->dict[str,Any]:return asdict(self)

def _relation_hits(sentence:str):
    low=sentence.casefold();out=[]
    for rel,cues in RELATIONS:
        pos=[low.find(c.casefold()) for c in cues if c.casefold() in low]
        if pos:out.append((rel,min(pos)))
    return sorted(out,key=lambda x:x[1])

def _is_negated(sentence:str,pos:int,relation:str=''):
    return bool(_NEG.search(sentence[max(0,pos-90):pos]))

def _ordered_entities(sentence:str):
    low=sentence.casefold();candidates=[]
    known=(('diabetes mellitus',('diabetes mellitus','diabetes','diabetic')),('diabetic nephropathy',('diabetic nephropathy',)),('albuminuria',('albuminuria','albuminurie')),('hypertension',('hypertension','high blood pressure')),('hypokalemia',('hypokalemia','hypokaliemia','hypokaliémie')),('hyperaldosteronism',('hyperaldosteronism','hyperaldosteronisme','hyperaldostéronisme')))
    for normalized,cues in known:
        for cue in cues:
            start=low.find(cue.casefold())
            if start>=0:candidates.append((start,start+len(cue),normalized,ClinicalEntity(normalized,normalized,'condition',.92,False)))
    candidates.sort(key=lambda row:(-(row[1]-row[0]),row[0]))
    selected=[];occupied=[];seen=set()
    for start,end,normalized,entity in candidates:
        if normalized in seen or any(start<e and end>s for s,e in occupied):continue
        selected.append((start,entity));occupied.append((start,end));seen.add(normalized)
    for entity in extract_clinical_entities(sentence):
        if entity.normalized in seen:continue
        pos=low.find(entity.text.casefold())
        if pos>=0:selected.append((pos,entity));seen.add(entity.normalized)
    selected.sort(key=lambda x:x[0]);return [e for _,e in selected]

def _entity_spans(sentence:str,entities:Sequence[ClinicalEntity]):
    low=sentence.casefold();out=[]
    for entity in entities:
        pos=low.find(entity.text.casefold())
        if pos>=0:out.append((pos,pos+len(entity.text),entity))
    return sorted(out,key=lambda x:x[0])

def _relation_entity_pairs(sentence:str,relation_positions:Sequence[tuple[str,int]]):
    entities=extract_clinical_entities(sentence);spans=_entity_spans(sentence,entities)
    if len(spans)<2:return []
    pairs=[]
    for relation,pos in relation_positions[:3]:
        left=[row for row in spans if row[1]<=pos];right=[row for row in spans if row[0]>=pos]
        if not left or not right:continue
        subject=max(left,key=lambda row:row[1])[2];obj=min(right,key=lambda row:row[0])[2]
        if subject.normalized!=obj.normalized:pairs.append((relation,pos,subject,obj))
    return pairs

def extract_clinical_facts(text:str,*,node_id:str='',document_id:str='')->tuple[ClinicalFact,...]:
    facts=[]
    for sentence in re.split(r'(?<=[.!?。！？])\s+|\n+',str(text or '').strip()):
        if not sentence:continue
        relations=_relation_hits(sentence)
        if not relations:continue
        pairs=_relation_entity_pairs(sentence,relations)
        if not pairs:
            ents=_ordered_entities(sentence)
            if len(ents)>=2:
                ordered=sorted(ents,key=lambda e:sentence.casefold().find(e.text.casefold()))
                for relation,pos in relations[:3]:pairs.append((relation,pos,ordered[0],ordered[1]))
        for ri,(rel,pos,left,right) in enumerate(pairs[:3]):
            neg=_is_negated(sentence,pos,rel) or left.negated or right.negated
            facts.append(ClinicalFact(left.normalized,rel,right.normalized,-1 if neg else 1,min(.97,.68+.07*min(len(pairs)+1,3)+.03*ri),node_id or f'N{len(facts)+1}',sentence,document_id))
    return tuple(facts[:32])

def _find_paths(facts:Sequence[ClinicalFact],u:QueryUnderstanding,max_depth:int=3)->tuple[ReasoningPath,...]:
    q=[]
    for e in u.entities:
        if e.normalized not in q:q.append(e.normalized)
    if len(q)<2:return ()
    start=q[0];targets=set(q[1:]);adj={}
    for f in facts:
        if f.polarity>0:adj.setdefault(f.subject,[]).append(f)
    results=[];stack=[(start,(start,),(),(),1.,set())]
    while stack:
        node,nodes,rels,ids,score,docs=stack.pop()
        if len(rels)>max_depth:continue
        if node in targets and rels:
            results.append(ReasoningPath(nodes,rels,ids,round(score,3),True,len(docs)>1));continue
        for f in adj.get(node,()):
            if f.object in nodes:continue
            nd=set(docs)
            if f.document_id:nd.add(f.document_id)
            stack.append((f.object,nodes+(f.object,),rels+(f.predicate,),ids+(f'{f.node_id}:{len(ids)}',),score*(.72+.28*f.confidence),nd))
    unique={p.nodes+p.relations:p for p in sorted(results,key=lambda p:p.support,reverse=True)}
    return tuple(unique.values())[:8]

def _conflict(facts):
    groups={}
    for f in facts:groups.setdefault((f.subject,f.predicate,f.object),set()).add(f.polarity)
    return min(1.,sum(len(v)>1 for v in groups.values())/max(1,len(groups)))

def _safety_conflict(u:QueryUnderstanding,facts:Sequence[ClinicalFact],texts:Sequence[str]=())->float:
    if 'safety' not in u.constraints:return 0.
    contra=sum(1 for f in facts if f.predicate=='contraindicated' and f.polarity>0);treat=sum(1 for f in facts if f.predicate=='treated_with' and f.polarity>0);combined=' '.join(texts).casefold()
    contra=max(contra,1 if re.search(r'\b(contraindicated|contre-indiqué|contre indique|ممنوع|تجنب)\b',combined,re.I|re.UNICODE) else 0);treat=max(treat,1 if re.search(r'\b(treated with|managed with|traité par|يعالج بـ)\b',combined,re.I|re.UNICODE) else 0)
    return 1. if contra and treat else .5 if contra else 0.

def assess_clinical_reasoning(understanding:QueryUnderstanding,hits:Sequence[Any],*,max_depth:int=3)->ClinicalReasoningAssessment:
    facts=[];coverage=[];texts=[]
    for i,h in enumerate(hits):
        meta=getattr(h,'metadata',{}) or {};text=str(getattr(h,'text','') or '');texts.append(text)
        facts.extend(extract_clinical_facts(text,node_id=str(meta.get('chunk_id') or f'N{i+1}'),document_id=str(meta.get('document_id') or getattr(h,'doc_id',''))))
        qe={e.normalized for e in understanding.entities};ee={e.normalized for e in _ordered_entities(text)};coverage.append(len(qe&ee)/max(1,len(qe)))
    qset={e.normalized for e in understanding.entities};has_entities=bool(qset)
    paths=_find_paths(facts,understanding,max_depth)
    relation_needed=bool(understanding.relations or understanding.primary_intent in {'etiology','mechanism','association','comparison'})
    direct=[f for f in facts if f.polarity>0 and f.subject in qset and f.object in qset]
    cov=max(coverage,default=0.) if has_entities else (1.0 if hits else 0.0)
    retrieval_support=max((max(0.,min(1.,float(getattr(h,'score',0.0) or 0.0))) for h in hits),default=0.0)
    direct_support=max((f.confidence for f in direct),default=0.0)
    path_support=max((p.support for p in paths),default=0.0)
    docs={f.document_id for f in facts if f.document_id}
    source=min(1.,.5+.25*len(docs)) if paths else 0.0
    contradiction=_conflict(facts);safety=_safety_conflict(understanding,facts,texts)
    if direct:mode,depth='DIRECT',1
    elif paths and relation_needed:mode,depth=('MULTI_HOP' if len(paths[0].relations)>1 else 'ONE_HOP'),len(paths[0].relations)
    elif not relation_needed and has_entities and cov>=.5:mode,depth='DIRECT',1
    elif not relation_needed and not has_entities and hits:mode,depth='DIRECT_SUMMARY',1
    else:mode,depth='INSUFFICIENT',0
    if mode=='DIRECT' and not relation_needed:
        direct_support=max(direct_support,.70*cov,retrieval_support*.90)
        source=max(source,retrieval_support)
    if mode=='DIRECT_SUMMARY':
        direct_support=max(direct_support,retrieval_support*.90)
        source=max(source,retrieval_support)
    blocked=[]
    if has_entities and cov<.5:blocked.append('insufficient_entity_coverage')
    if relation_needed and not has_entities:blocked.append('missing_reasoning_entities')
    if relation_needed and not direct and not paths:blocked.append('no_explicit_reasoning_path')
    if contradiction>=.5:blocked.append('conflicting_evidence')
    if safety>=1.:blocked.append('safety_conflict')
    conf=max(0.,min(1.,.35*direct_support+.35*path_support+.15*cov+.15*source-.3*contradiction-.35*safety))
    minimum=.30 if mode in {'DIRECT','DIRECT_SUMMARY'} else .50
    allow=not blocked and conf>=minimum
    return ClinicalReasoningAssessment(allow,mode,depth,round(cov,3),round(direct_support,3),round(path_support,3),round(source,3),round(contradiction,3),round(safety,3),round(conf,3),paths,tuple(blocked))

def build_reasoning_instruction(understanding:QueryUnderstanding,assessment:ClinicalReasoningAssessment)->str:
    if assessment.mode=='MULTI_HOP':return 'Use only the explicit evidence path across the retrieved facts. Preserve uncertainty, population, timing, safety qualifiers, numbers, and units. Do not invent a missing link.'
    if assessment.mode=='ONE_HOP':return 'Use only the explicit evidence path and relationship supported by the retrieved facts. Do not invent a missing link.'
    if assessment.mode=='INSUFFICIENT' and (understanding.relations or understanding.primary_intent in {'etiology','mechanism','association','comparison'}):return 'Use only an explicit evidence path if one is present. Do not invent a missing link.'
    if assessment.mode=='DIRECT_SUMMARY':return 'Summarize only what is directly stated or clearly enumerated in the retrieved evidence. Preserve qualifiers, uncertainty, numbers, units, and negations. Do not infer relationships that are not stated.'
    if 'safety' in understanding.constraints:return 'Prioritize contraindications and safety qualifiers. Never convert a contraindication into a recommendation.'
    return 'Answer only from directly supported evidence; preserve qualifiers and uncertainty.'
