from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence
from rag_project.utils.text_utils import meaningful_tokens

@dataclass(frozen=True)
class ClinicalEntity:
    text: str
    normalized: str
    kind: str
    confidence: float
    negated: bool = False

@dataclass(frozen=True)
class QueryUnderstanding:
    normalized: str
    intents: tuple[str,...]
    primary_intent: str
    entities: tuple[ClinicalEntity,...]
    relations: tuple[str,...]
    constraints: tuple[str,...]
    answer_shape: str
    semantic_terms: tuple[str,...]
    confidence: float
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class EvidenceNode:
    node_id: str
    text: str
    score: float
    document_id: str
    chunk_id: str
    page_numbers: tuple[Any,...]

@dataclass(frozen=True)
class ReasoningEdge:
    source: str
    target: str
    relation: str
    confidence: float
    evidence: tuple[str,...]

ALIASES={
'diabetes mellitus':('diabetes mellitus','diabetes','diabète','diabete','dm','type 2 diabetes','داء السكري','السكري'),
'diabetic ketoacidosis':('diabetic ketoacidosis','dka','acidocétose diabétique','acidocetose diabetique','الحماض الكيتوني السكري'),
'diabetic nephropathy':('diabetic nephropathy','nephropathie diabetique','néphropathie diabétique','diabetic kidney disease','dkd','maladie rénale diabétique','اعتلال الكلية السكري'),
'albuminuria':('albuminuria','albuminurie','microalbuminuria','urinary albumin','albumin in urine','بيلة الألبومين','زلال البول'),
'hypertension':('hypertension','hta','high blood pressure','hypertension artérielle','hypertension arterielle','ارتفاع ضغط الدم'),
'hypokalemia':('hypokalemia','hypokaliémie','hypokaliemia','low potassium','نقص بوتاسيوم الدم'),
'hyperaldosteronism':('hyperaldosteronism','hyperaldostéronisme','hyperaldosteronisme','primary aldosteronism','فرط الألدوستيرونية'),
'thyroid cancer':('thyroid cancer','cancer thyroïde','cancer de la thyroïde','thyroid carcinoma','سرطان الغدة الدرقية'),
'insulin resistance':('insulin resistance','insulinorésistance','insulin resistance syndrome','مقاومة الأنسولين'),
'albumin':('albumin','albumine','الألبومين'),'potassium':('potassium','kaliémie','kalemie','k+','البوتاسيوم'),'insulin':('insulin','insuline','الأنسولين','الإنسولين')}
KIND={'diabetes mellitus':'disease','diabetic ketoacidosis':'disease','diabetic nephropathy':'complication','albuminuria':'finding','hypertension':'disease','hypokalemia':'finding','hyperaldosteronism':'disease','thyroid cancer':'disease','insulin resistance':'pathophysiology','albumin':'biomarker','potassium':'analyte','insulin':'hormone'}
INTENTS={
'definition':('what is','define','definition','meaning',"qu'est-ce",'définition','ما هو','ما هي','تعريف'),
'comparison':('compare','comparison','difference','differences','versus','vs','between','différence','مقارنة','فرق','بين'),
'diagnosis':('diagnosis','diagnostic','criteria','diagnostic criteria','diagnostiquer','تشخيص'),
'management':('treatment','treated','management','therapy','treat','prise en charge','traitement','علاج','التدبير'),
'etiology':('cause','causes','etiology','aetiology','why','risk factor','facteur','étiologie','سبب','أسباب'),
'mechanism':('mechanism','mechanisms','physiopathology','pathophysiology','how does','mécanisme','آلية'),
'prognosis':('prognosis','outcome','survival','prognostic','pronostic','مآل','التكهن'),
'numeric':('dose','dosage','mg','ml','mmhg','percentage','how many','how much','range','threshold','قيمة','جرعة','نسبة'),
'association':('related','relationship','associated','association','linked','lien','relation','علاقة','مرتبط'),
'navigation':('which page','page number','section','where','source','citation','quelle page','où','أين','أي صفحة'),
'table_lookup':('table','row','column','tableau','جدول','صف','عمود'),'figure_lookup':('figure','diagram','chart','graph','image','schéma','رسم','شكل')}
RELATIONS=(('causality',r'\b(cause|causes|caused by|due to|leads to|responsible for|provoque|entra[iî]ne|سبب|يؤدي)\b'),('association',r'\b(associated with|associated|related to|linked to|association|lié à|associé à|مرتبط|علاقة)\b'),('comparison',r'\b(compare|versus|vs|difference|différence|مقارنة|فرق)\b'),('sequence',r'\b(then|after|before|subsequently|ensuite|après|avant|ثم|بعد|قبل)\b'))
NEG=re.compile(r'\b(no|not|without|never|none|cannot|does not|doesn\'t|contraindicated|avoid|aucun|sans|jamais|ne pas|ممنوع|منع|لا|ليس|دون)\b',re.I)

def _norm(s): return re.sub(r'\s+',' ',str(s or '')).strip().casefold()
def _match(t,p): return bool(re.search(rf'(?<!\w){re.escape(p.casefold())}(?!\w)',t,re.I|re.UNICODE))
def normalize_medical_term(text):
    v=_norm(text)
    for c,a in ALIASES.items():
        if v==c or any(_match(v,x) for x in a): return c
    return v

def extract_clinical_entities(text):
    v=_norm(text); found={}
    for c,a in ALIASES.items():
        alias=next((x for x in a if _match(v,x)),None)
        if alias:
            pos=v.find(alias.casefold()); found[c]=ClinicalEntity(alias,c,KIND.get(c,'concept'),.96 if len(alias.split())>1 else .88,bool(NEG.search(v[max(0,pos-90):pos])))
    for tok in meaningful_tokens(v):
        if tok not in found and tok.endswith(('itis','osis','emia','pathy','carcinoma')): found[tok]=ClinicalEntity(tok,tok,'medical_concept',.72,False)
    return tuple(found.values())[:24]

def _intent_hits(t): return sorted([(k,sum(_match(t,c) for c in cues)) for k,cues in INTENTS.items() if any(_match(t,c) for c in cues)],key=lambda x:(-x[1],x[0]))
def understand_query(query,*,conversation_context=''):
    n=_norm(query); current=_intent_hits(n); ctx=_intent_hits(conversation_context[-1200:]) if conversation_context else []; combined=(n+' '+conversation_context[-1200:]).strip() if conversation_context else n; names={k for k,_ in current}; primary=next((k for k in ('comparison','management','diagnosis','numeric','mechanism','etiology','prognosis','association','navigation','table_lookup','figure_lookup') if k in names),current[0][0] if current else (ctx[0][0] if ctx else 'factual')); intents=[k for k,_ in _intent_hits(combined)] or ['factual'];
    if primary not in intents:intents.insert(0,primary)
    rel=tuple(k for k,p in RELATIONS if re.search(p,combined,re.I|re.UNICODE)); cons=[]
    if re.search(r'\b(exact|precise|exactly|strictly|exacte|précis)\b',combined,re.I): cons.append('exactness')
    if re.search(r'\b(adult|child|children|pediatric|pregnan|grossesse|enfant|enfants|adulte|pédiatrique)\b',combined,re.I): cons.append('population')
    if re.search(r'\b(first[- ]line|second[- ]line|initial|maintenance|acute|chronic|aigu|chronique|initiale|entretien)\b',combined,re.I): cons.append('clinical_phase')
    if re.search(r'\b(contraindication|contraindications|contraindicated|avoid|contre-indication|ممنوع|تجنب)\b',combined,re.I): cons.append('safety')
    ents=extract_clinical_entities(combined); shape='comparison' if primary=='comparison' else 'list' if primary in {'diagnosis','management','etiology','prognosis','numeric'} else 'definition' if primary=='definition' else 'explanation'; terms=tuple(dict.fromkeys([e.normalized for e in ents]+meaningful_tokens(n)))[:32]; conf=min(1.,.40+.12*len(intents)+.04*len(ents)+(.10 if len(meaningful_tokens(n))>=5 else 0))
    return QueryUnderstanding(n,tuple(intents),primary,ents,rel,tuple(cons),shape,terms,round(conf,3))

def build_evidence_graph(hits,understanding):
    nodes=[]; ent={}; rel={}
    for i,h in enumerate(hits):
        m=getattr(h,'metadata',{}) or {}; nid=str(m.get('chunk_id') or f'N{i+1}'); text=str(getattr(h,'text','') or ''); nodes.append(EvidenceNode(nid,text,max(0,min(1,float(getattr(h,'score',0)))),str(m.get('document_id') or getattr(h,'doc_id','')),nid,tuple(m.get('page_numbers') or ()))); ent[nid]={e.normalized for e in extract_clinical_entities(text)}; rel[nid]=tuple(k for k,p in RELATIONS if re.search(p,text,re.I|re.UNICODE))
    wanted={e.normalized for e in understanding.entities}; edges=[]
    for n in nodes:
        if len(ent[n.node_id]&wanted)>=2 and rel[n.node_id]:
            ordered=[e.normalized for e in extract_clinical_entities(n.text)]; r=next((x for x in rel[n.node_id] if x in understanding.relations),rel[n.node_id][0]);
            if len(ordered)>=2: edges.append(ReasoningEdge(ordered[0],ordered[1],r,.86,(n.node_id,)))
    for left,right in zip(nodes,nodes[1:]):
        shared=ent[left.node_id]&ent[right.node_id]; covered=(ent[left.node_id]|ent[right.node_id])&wanted
        if shared and len(covered)>=2: edges.append(ReasoningEdge(left.node_id,right.node_id,(rel[left.node_id] or rel[right.node_id] or understanding.relations or ('association',))[0],.74,(left.node_id,right.node_id)))
    unique={(e.source,e.target,e.relation):e for e in edges}; return tuple(nodes),tuple(unique.values())

def clinical_reasoning_ready(understanding,nodes,edges):
    cross=any(e.source!=e.target for e in edges); multi=cross and (understanding.primary_intent in {'etiology','mechanism','association','comparison'} or bool(understanding.relations)); cov=sum(1 for e in understanding.entities if any(e.normalized in _norm(n.text) for n in nodes))/max(1,len(understanding.entities)); return {'direct_evidence':bool(nodes),'multi_hop':multi,'node_count':len(nodes),'edge_count':len(edges),'entity_coverage':round(cov,3),'reasoning_depth':2 if multi else 1}

def _token_overlap(a,b):
    x=set(meaningful_tokens(a)); y=set(meaningful_tokens(b)); return len(x&y)/max(1,len(x))
def semantic_evidence_alignment(question,hits,*,conversation_context=''):
    u=understand_query(question,conversation_context=conversation_context); non=[h for h in hits if str(getattr(h,'text','') or '').strip()]
    if not non:return {'score':0.,'entity_coverage':0.,'semantic_overlap':0.,'relation_coverage':0.,'best_hit_score':0.,'decision':'NOT_SUPPORTED','reason':'No non-empty evidence was available for semantic alignment.','understanding':u.to_dict()}
    qe={e.normalized for e in u.entities}; qt=set(meaningful_tokens(u.normalized)); rows=[]
    for h in non:
        t=str(getattr(h,'text','') or ''); ee={e.normalized for e in extract_clinical_entities(t)}; es=len(qe&ee)/max(1,len(qe)); ls=len(qt&set(meaningful_tokens(t)))/max(1,len(qt)); er={k for k,p in RELATIONS if re.search(p,t,re.I|re.UNICODE)}; rs=len(set(u.relations)&er)/max(1,len(u.relations)) if u.relations else 0.; rb=max(0,min(1,float(getattr(h,'score',0)))); rows.append((min(1,.55*es+.25*ls+.10*rs+.10*rb),es,ls,rs))
    best=max(rows); score=best[0]; decision='DIRECTLY_SUPPORTED' if score>=.55 else 'PARTIALLY_SUPPORTED' if score>=.25 else 'RELATED_BUT_NOT_ANSWERING' if score>.05 else 'NOT_SUPPORTED'; return {'score':round(score,3),'entity_coverage':round(best[1],3),'semantic_overlap':round(best[2],3),'relation_coverage':round(best[3],3),'best_hit_score':round(score,3),'decision':decision,'reason':'Semantic entity, concept, and relation alignment was computed across retrieved evidence.','understanding':u.to_dict()}
