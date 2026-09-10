from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence
from rag_project.utils.text_utils import meaningful_tokens

@dataclass(frozen=True)
class ClinicalEntity:
    text:str; normalized:str; kind:str; confidence:float; negated:bool=False
@dataclass(frozen=True)
class QueryUnderstanding:
    normalized:str; intents:tuple[str,...]; primary_intent:str; entities:tuple[ClinicalEntity,...]; relations:tuple[str,...]; constraints:tuple[str,...]; answer_shape:str; semantic_terms:tuple[str,...]; confidence:float
    def to_dict(self):return asdict(self)
@dataclass(frozen=True)
class EvidenceNode:
    node_id:str; text:str; score:float; document_id:str; chunk_id:str; page_numbers:tuple[Any,...]
@dataclass(frozen=True)
class ReasoningEdge:
    source:str; target:str; relation:str; confidence:float; evidence:tuple[str,...]

# Retrieval vocabulary only: it normalizes query concepts, but never supplies medical facts.
ALIASES={
'diabetes mellitus':('diabetes mellitus','diabetes','diabète','diabete','dm','type 1 diabetes','type 2 diabetes','t1dm','t2dm','داء السكري','السكري'),
'diabetic ketoacidosis':('diabetic ketoacidosis','dka','acidocétose diabétique','acidocetose diabetique','الحماض الكيتوني السكري'),
'diabetic nephropathy':('diabetic nephropathy','nephropathie diabetique','néphropathie diabétique','diabetic kidney disease','dkd','maladie rénale diabétique','اعتلال الكلية السكري'),
'albuminuria':('albuminuria','albuminurie','microalbuminuria','urinary albumin','albumin in urine','uacr','بيلة الألبومين','زلال البول'),
'hypertension':('hypertension','hta','high blood pressure','hypertension artérielle','hypertension arterielle','blood pressure','bp','ارتفاع ضغط الدم'),
'hypotension':('hypotension','low blood pressure','hypotension artérielle','ضغط الدم المنخفض'),
'hypokalemia':('hypokalemia','hypokaliémie','hypokaliemia','low potassium','نقص بوتاسيوم الدم'),
'hyperkalemia':('hyperkalemia','hyperkaliémie','high potassium','فرط بوتاسيوم الدم'),
'hyperaldosteronism':('hyperaldosteronism','hyperaldostéronisme','hyperaldosteronisme','primary aldosteronism','hyperaldosteronisme primaire','فرط الألدوستيرونية'),
'thyroid cancer':('thyroid cancer','cancer thyroïde','cancer de la thyroïde','thyroid carcinoma','سرطان الغدة الدرقية'),
'hypothyroidism':('hypothyroidism','hypothyroïdie','hypothyroid','قصور الغدة الدرقية'),
'hyperthyroidism':('hyperthyroidism','hyperthyroïdie','hyperthyroid','فرط نشاط الغدة الدرقية'),
'insulin resistance':('insulin resistance','insulinorésistance','metabolic syndrome','مقاومة الأنسولين','متلازمة الأيض'),
'heart failure':('heart failure','cardiac failure','insuffisance cardiaque','congestive heart failure','chf','قصور القلب','فشل القلب'),
'coronary artery disease':('coronary artery disease','coronary disease','ischemic heart disease','cad','maladie coronarienne','maladie ischémique','مرض الشريان التاجي'),
'myocardial infarction':('myocardial infarction','heart attack','mi','infarctus du myocarde','stemi','nstemi','احتشاء عضلة القلب'),
'atrial fibrillation':('atrial fibrillation','afib','af','fibrillation atriale','رجفان أذيني'),
'stroke':('stroke','cva','ischemic stroke','hemorrhagic stroke','avc','accident vasculaire cérébral','السكتة الدماغية'),
'pulmonary embolism':('pulmonary embolism','pe','embolie pulmonaire','الانصمام الرئوي'),
'deep vein thrombosis':('deep vein thrombosis','dvt','thrombose veineuse profonde','tvp','الخثار الوريدي العميق'),
'chronic kidney disease':('chronic kidney disease','ckd','chronic renal disease','maladie rénale chronique','insuffisance rénale chronique','مرض الكلى المزمن'),
'acute kidney injury':('acute kidney injury','aki','acute renal failure','insuffisance rénale aiguë','ira','إصابة الكلى الحادة'),
'nephrotic syndrome':('nephrotic syndrome','syndrome néphrotique','syndrome nephrotique','المتلازمة النفروزية'),
'nephritic syndrome':('nephritic syndrome','syndrome néphritique','المتلازمة النفريتية'),
'asthma':('asthma','asthme','الربو'),'copd':('copd','chronic obstructive pulmonary disease','bronchopneumopathie chronique obstructive','bpco','داء الانسداد الرئوي المزمن'),
'pneumonia':('pneumonia','pneumonie','community acquired pneumonia','cap','الالتهاب الرئوي'),'pulmonary hypertension':('pulmonary hypertension','hta pulmonaire','pulmonary arterial hypertension','pah','ارتفاع ضغط الدم الرئوي'),
'bronchitis':('bronchitis','bronchite','التهاب الشعب الهوائية'),'fever':('fever','pyrexia','fièvre','حمى','الحمى'),
'anemia':('anemia','anaemia','anémie','فقر الدم','الأنيميا'),'iron deficiency anemia':('iron deficiency anemia','iron-deficiency anaemia','anémie ferriprive','ida','فقر الدم بعوز الحديد'),
'leukemia':('leukemia','leukaemia','leucémie','ابيضاض الدم'),'lymphoma':('lymphoma','lymphome','سرطان الغدد اللمفاوية'),
'neutropenia':('neutropenia','neutropénie','نقص العدلات'),'thrombocytopenia':('thrombocytopenia','thrombopénie','نقص الصفائح الدموية'),
'sepsis':('sepsis','septicemia','septicémie','الإنتان','تعفن الدم'),'meningitis':('meningitis','méningite','التهاب السحايا'),
'hepatitis':('hepatitis','hépatite','التهاب الكبد'),'cirrhosis':('cirrhosis','cirrhose','تليف الكبد'),'pancreatitis':('pancreatitis','pancréatite','التهاب البنكرياس'),'gastritis':('gastritis','gastrite','التهاب المعدة'),
'peptic ulcer disease':('peptic ulcer disease','peptic ulcer','ulcère gastroduodénal','pud','القرحة الهضمية'),'crohn disease':('crohn disease','crohn\'s disease','maladie de crohn','داء كرون'),'ulcerative colitis':('ulcerative colitis','colite ulcéreuse','rectocolite hémorragique','rchu','التهاب القولون التقرحي'),
'epilepsy':('epilepsy','épilepsie','seizure disorder','الصرع'),'migraine':('migraine','migraine headache','الصداع النصفي'),'parkinson disease':('parkinson disease','parkinson\'s disease','maladie de parkinson','مرض باركنسون'),'alzheimer disease':('alzheimer disease','alzheimer\'s disease','maladie d\'alzheimer','داء ألزهايمر'),
'multiple sclerosis':('multiple sclerosis','sclérose en plaques','ms','التصلب المتعدد'),'rheumatoid arthritis':('rheumatoid arthritis','polyarthrite rhumatoïde','ra','التهاب المفاصل الروماتويدي','الالتهاب المفصلي الروماتويدي'),'systemic lupus erythematosus':('systemic lupus erythematosus','sle','lupus','lupus érythémateux systémique','الذئبة الحمامية الجهازية'),
'osteoarthritis':('osteoarthritis','arthrose','oa','الفصال العظمي'),'gout':('gout','gouty arthritis','goutte','داء النقرس'),
'breast cancer':('breast cancer','cancer du sein','carcinome mammaire','سرطان الثدي'),'lung cancer':('lung cancer','cancer du poumon','carcinome pulmonaire','سرطان الرئة'),'prostate cancer':('prostate cancer','cancer de la prostate','سرطان البروستاتا'),'colorectal cancer':('colorectal cancer','colon cancer','rectal cancer','cancer colorectal','سرطان القولون والمستقيم'),
'albumin':('albumin','albumine','الألبومين'),'creatinine':('creatinine','créatinine','الكرياتينين'),'egfr':('egfr','estimated glomerular filtration rate','dfg estimé','débit de filtration glomérulaire','معدل الترشيح الكبيبي'),'hba1c':('hba1c','glycated hemoglobin','glycosylated hemoglobin','hémoglobine glyquée','الهيموغلوبين السكري'),
'potassium':('potassium','kaliémie','k+','البوتاسيوم'),'sodium':('sodium','natremia','natrémie','na+','الصوديوم'),'calcium':('calcium','calcémie','ca2+','الكالسيوم'),'magnesium':('magnesium','magnésémie','mg2+','المغنيسيوم'),'glucose':('glucose','blood glucose','glycemia','glycémie','سكر الدم'),
'hemoglobin':('hemoglobin','haemoglobin','hgb','hb','hémoglobine','الهيموغلوبين'),'platelets':('platelets','platelet count','thrombocytes','plaquettes','plt','الصفائح'),'white blood cells':('white blood cells','wbc','leukocytes','globules blancs','gb','كريات الدم البيضاء'),
'c-reactive protein':('c-reactive protein','crp','protéine c-réactive','البروتين المتفاعل c'),'erythrocyte sedimentation rate':('erythrocyte sedimentation rate','esr','vitesse de sédimentation','سرعة الترسيب'),'troponin':('troponin','troponine','hs-troponin','التروبونين'),'inr':('inr','international normalized ratio','ratio normalisé international','النسبة المعيارية الدولية'),
'ast':('ast','aspartate aminotransferase','tgo','asat','aspartate transaminase','ناقلة أمين الأسبارتات'),'alt':('alt','alanine aminotransferase','tgp','alat','alanine transaminase','ناقلة أمين الألانين'),'bilirubin':('bilirubin','bilirubine','bilirubinemia','البيليروبين'),'insulin':('insulin','insuline','الأنسولين','الإنسولين'),
'metformin':('metformin','metformine','الميتفورمين'),'dapagliflozin':('dapagliflozin','dapagliflozine','داباغليفلوزين'),'lisinopril':('lisinopril',),'enalapril':('enalapril',)}

LABS={'albumin','creatinine','egfr','hba1c','potassium','sodium','calcium','magnesium','glucose','hemoglobin','platelets','white blood cells','c-reactive protein','erythrocyte sedimentation rate','troponin','inr','ast','alt','bilirubin'}
FINDINGS={'albuminuria','hypokalemia','hyperkalemia','hypotension','fever','neutropenia','thrombocytopenia'}
COMPLICATIONS={'diabetic nephropathy'}
HORMONES={'insulin'}
KIND={k:('lab' if k in LABS else 'finding' if k in FINDINGS else 'complication' if k in COMPLICATIONS else 'hormone' if k in HORMONES else 'drug' if k in {'metformin','dapagliflozin','lisinopril','enalapril'} else 'disease') for k in ALIASES}

INTENTS={'definition':('what is','define','definition','meaning',"qu'est-ce",'définition','ما هو','ما هي','تعريف'),'comparison':('compare','comparison','difference','differences','versus','vs','between','différence','مقارنة','فرق','بين'),'diagnosis':('diagnosis','diagnostic','criteria','diagnostic criteria','diagnostiquer','تشخيص'),'management':('treatment','treated','management','therapy','treat','prise en charge','traitement','علاج','التدبير'),'etiology':('cause','causes','etiology','aetiology','why','risk factor','facteur','étiologie','سبب','أسباب'),'mechanism':('mechanism','mechanisms','physiopathology','pathophysiology','how does','mécanisme','آلية'),'prognosis':('prognosis','outcome','survival','prognostic','pronostic','مآل','التكهن'),'numeric':('dose','dosage','mg','ml','mmhg','percentage','how many','how much','range','threshold','value','قيمة','جرعة','نسبة'),'association':('related','relationship','associated','association','linked','lien','relation','علاقة','مرتبط'),'navigation':('which page','page number','section','where','source','citation','quelle page','où','أين','أي صفحة'),'table_lookup':('table','row','column','tableau','جدول','صف','عمود'),'figure_lookup':('figure','diagram','chart','graph','image','schéma','رسم','شكل')}
RELATIONS=(('causality',r'\b(cause|causes|caused by|due to|leads to|responsible for|provoque|entra[iî]ne|سبب|يؤدي)\b'),('association',r'\b(associated with|associated|related to|linked to|association|lié à|associé à|مرتبط|علاقة)\b'),('comparison',r'\b(compare|versus|vs|difference|différence|مقارنة|فرق)\b'),('sequence',r'\b(then|after|before|subsequently|ensuite|après|avant|ثم|بعد|قبل)\b'))
NEG=re.compile(r'\b(no|not|without|never|none|cannot|does not|doesn\'t|contraindicated|avoid|aucun|sans|jamais|ne pas|ممنوع|منع|لا|ليس|دون)\b',re.I|re.UNICODE)
_DRUG_SUFFIX=re.compile(r'^[a-z][a-z0-9-]{4,}(?:pril|olol|sartan|statin|azole|cillin|mycin|vir|mab|nib|prazole|tidine|caine|cycline|floxacin|lukast|setron|gliptin|gliflozin|tide|parin|dipine|xaban|oxetine|triptan|cept|navir|vudine|formin)$',re.I)
_CONDITION_SUFFIX=re.compile(r'^[a-z][a-z-]{4,}(?:itis|osis|emia|pathy|carcinoma|oma|algia|penia|iasis|megaly|cytosis|trophy|sclerosis|stenosis|ectasia)$',re.I)
_ABBREVIATION=re.compile(r'\b[A-Z](?:[A-Z0-9]){1,7}(?:[-/][A-Z0-9]{1,8})?\b')
_MEASUREMENT=re.compile(r'(?<!\w)\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|ug|g|kg|ml|mL|L|mmHg|mmol/L|mol/L|%|IU|units?|bpm|°C|C|mEq/L|mEq|mOsm/L|ng/mL|pg/mL|U/L|kPa)(?=\s|$|[^\w])',re.I)

def _norm(s):return re.sub(r'\s+',' ',str(s or '')).strip().casefold()
def _match(t,p):return bool(re.search(rf'(?<!\w){re.escape(p.casefold())}(?!\w)',t,re.I|re.UNICODE))
def normalize_medical_term(text):
    v=_norm(text)
    for canonical,aliases in ALIASES.items():
        if v==canonical or any(_match(v,a) for a in aliases):return canonical
    return v

def _open_set_entities(v:str):
    found=[]
    for m in _MEASUREMENT.finditer(v):found.append((m.start(),ClinicalEntity(m.group(0),m.group(0).casefold(),'measurement',.94,False)))
    for m in _ABBREVIATION.finditer(v):
        tok=m.group(0);canonical=normalize_medical_term(tok)
        if tok.casefold() not in {'what','this','that','which','where','when','with','from','and','the'}:found.append((m.start(),ClinicalEntity(tok,canonical,'abbreviation',.82,False)))
    for m in re.finditer(r'(?<!\w)[a-zà-ÿ][a-zà-ÿ0-9-]{3,}(?!\w)',v,re.I|re.UNICODE):
        tok=m.group(0);canonical=normalize_medical_term(tok)
        if canonical in {'metformin','dapagliflozin','lisinopril','enalapril'}:
            found.append((m.start(),ClinicalEntity(tok,canonical,'drug',.90,bool(NEG.search(v[max(0,m.start()-90):m.start()])))))
        elif _DRUG_SUFFIX.fullmatch(tok):found.append((m.start(),ClinicalEntity(tok,canonical,'drug',.78,bool(NEG.search(v[max(0,m.start()-90):m.start()])))))
        elif _CONDITION_SUFFIX.fullmatch(tok):found.append((m.start(),ClinicalEntity(tok,canonical,'medical_concept',.78,bool(NEG.search(v[max(0,m.start()-90):m.start()])))))
    return found

def extract_clinical_entities(text):
    v=_norm(text);found={}
    aliases_sorted=sorted(ALIASES.items(),key=lambda item:max(len(a) for a in item[1]),reverse=True)
    for canonical,aliases in aliases_sorted:
        alias=next((a for a in aliases if _match(v,a)),None)
        if alias:
            pos=v.find(alias.casefold());found.setdefault(canonical,ClinicalEntity(alias,canonical,KIND.get(canonical,'concept'),.96 if len(alias.split())>1 else .90,bool(NEG.search(v[max(0,pos-90):pos]))))
    for _,entity in _open_set_entities(v):found.setdefault(entity.normalized,entity)
    return tuple(found.values())[:48]

def _intent_hits(t):return sorted([(k,sum(_match(t,c) for c in cues)) for k,cues in INTENTS.items() if any(_match(t,c) for c in cues)],key=lambda x:(-x[1],x[0]))
def understand_query(query,*,conversation_context=''):
    n=_norm(query);ctx=conversation_context[-1600:] if conversation_context else '';current=_intent_hits(n);context_intents=_intent_hits(ctx) if ctx else [];combined=(n+' '+ctx).strip() if ctx else n;names={k for k,_ in current};primary=next((k for k in ('comparison','management','diagnosis','numeric','mechanism','etiology','prognosis','association','navigation','table_lookup','figure_lookup','definition') if k in names),current[0][0] if current else (context_intents[0][0] if context_intents else 'factual'));intents=[k for k,_ in _intent_hits(combined)] or ['factual']
    if primary not in intents:intents.insert(0,primary)
    relations=tuple(k for k,p in RELATIONS if re.search(p,combined,re.I|re.UNICODE));constraints=[]
    if re.search(r'\b(exact|precise|exactly|strictly|exacte|précis)\b',combined,re.I):constraints.append('exactness')
    if re.search(r'\b(adult|child|children|pediatric|pregnan|grossesse|enfant|enfants|adulte|pédiatrique|neonate|newborn)\b',combined,re.I):constraints.append('population')
    if re.search(r'\b(first[- ]line|second[- ]line|initial|maintenance|acute|chronic|aigu|chronique|initiale|entretien)\b',combined,re.I):constraints.append('clinical_phase')
    if re.search(r'\b(contraindication|contraindications|contraindicated|avoid|contre-indication|ممنوع|تجنب)\b',combined,re.I):constraints.append('safety')
    if re.search(r'\b(efficacy|effectiveness|benefit|benefits|advantage|disadvantage|فعالية|نجاعة)\b',combined,re.I):constraints.append('outcome')
    entities=extract_clinical_entities(n+' '+ctx if ctx else n);shape='comparison' if primary=='comparison' else 'list' if primary in {'diagnosis','management','etiology','prognosis','numeric'} else 'definition' if primary=='definition' else 'explanation';terms=tuple(dict.fromkeys([e.normalized for e in entities]+meaningful_tokens(n)))[:64];confidence=min(1.,.38+.10*len(intents)+.025*len(entities)+(.12 if len(meaningful_tokens(n))>=5 else 0)+(.08 if relations else 0))
    return QueryUnderstanding(n,tuple(intents),primary,entities,relations,tuple(constraints),shape,terms,round(confidence,3))

def build_evidence_graph(hits,understanding):
    nodes=[];entity_map={};relation_map={}
    for i,h in enumerate(hits):
        m=getattr(h,'metadata',{}) or {};nid=str(m.get('chunk_id') or f'N{i+1}');text=str(getattr(h,'text','') or '');nodes.append(EvidenceNode(nid,text,max(0,min(1,float(getattr(h,'score',0)))),str(m.get('document_id') or getattr(h,'doc_id','')),nid,tuple(m.get('page_numbers') or ())));entity_map[nid]={e.normalized for e in extract_clinical_entities(text)};relation_map[nid]=tuple(k for k,p in RELATIONS if re.search(p,text,re.I|re.UNICODE))
    wanted={e.normalized for e in understanding.entities};edges=[]
    for n in nodes:
        ordered=[e.normalized for e in extract_clinical_entities(n.text)]
        if len(ordered)>=2 and relation_map[n.node_id]:edges.append(ReasoningEdge(ordered[0],ordered[1],next((r for r in relation_map[n.node_id] if r in understanding.relations),relation_map[n.node_id][0]),.86,(n.node_id,)))
    for left,right in zip(nodes,nodes[1:]):
        shared=entity_map[left.node_id]&entity_map[right.node_id];covered=(entity_map[left.node_id]|entity_map[right.node_id])&wanted
        if shared and covered:edges.append(ReasoningEdge(left.node_id,right.node_id,(relation_map[left.node_id] or relation_map[right.node_id] or understanding.relations or ('association',))[0],.74,(left.node_id,right.node_id)))
    unique={(e.source,e.target,e.relation):e for e in edges};return tuple(nodes),tuple(unique.values())

def clinical_reasoning_ready(understanding,nodes,edges):
    cross=any(e.source!=e.target for e in edges);multi=cross and (understanding.primary_intent in {'etiology','mechanism','association','comparison'} or bool(understanding.relations));coverage=sum(1 for e in understanding.entities if any(e.normalized in _norm(n.text) for n in nodes))/max(1,len(understanding.entities));return {'direct_evidence':bool(nodes),'multi_hop':multi,'node_count':len(nodes),'edge_count':len(edges),'entity_coverage':round(coverage,3),'reasoning_depth':2 if multi else 1}

def semantic_evidence_alignment(question,hits,*,conversation_context=''):
    u=understand_query(question,conversation_context=conversation_context);non=[h for h in hits if str(getattr(h,'text','') or '').strip()]
    if not non:return {'score':0.,'entity_coverage':0.,'semantic_overlap':0.,'relation_coverage':0.,'best_hit_score':0.,'decision':'NOT_SUPPORTED','reason':'No non-empty evidence was available for semantic alignment.','understanding':u.to_dict()}
    qe={e.normalized for e in u.entities};qt=set(meaningful_tokens(u.normalized));rows=[]
    for h in non:
        t=str(getattr(h,'text','') or '');ee={e.normalized for e in extract_clinical_entities(t)};es=len(qe&ee)/max(1,len(qe));ls=len(qt&set(meaningful_tokens(t)))/max(1,len(qt));er={k for k,p in RELATIONS if re.search(p,t,re.I|re.UNICODE)};rs=len(set(u.relations)&er)/max(1,len(u.relations)) if u.relations else 0.;rb=max(0,min(1,float(getattr(h,'score',0))));rows.append((min(1,.55*es+.25*ls+.10*rs+.10*rb),es,ls,rs))
    score,es,ls,rs=max(rows);decision='DIRECTLY_SUPPORTED' if score>=.55 else 'PARTIALLY_SUPPORTED' if score>=.25 else 'RELATED_BUT_NOT_ANSWERING' if score>.05 else 'NOT_SUPPORTED';return {'score':round(score,3),'entity_coverage':round(es,3),'semantic_overlap':round(ls,3),'relation_coverage':round(rs,3),'best_hit_score':round(score,3),'decision':decision,'reason':'Semantic entity, concept, relation, abbreviation, and measurement alignment was computed across retrieved evidence.','understanding':u.to_dict()}
