from __future__ import annotations
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence
from rag_project.utils.text_utils import keyword_overlap_score, meaningful_tokens

@dataclass(frozen=True)
class ClaimCheck:
    claim: str; support: float; status: str; sources: tuple[str, ...]; numeric_mismatch: bool = False; contradiction: bool = False; reason: str = ''
    def __post_init__(self):
        if isinstance(self.support, str) and isinstance(self.status, (int, float)) and isinstance(self.sources, str):
            legacy_source=self.support; legacy_support=float(self.status); legacy_status=self.sources
            object.__setattr__(self,'support',legacy_support); object.__setattr__(self,'status',legacy_status); object.__setattr__(self,'sources',(legacy_source,))
        elif isinstance(self.sources,str): object.__setattr__(self,'sources',(self.sources,))
    def to_dict(self): return asdict(self)

SENT=re.compile(r'(?<=[.!?。！？])\s+|\n+')
MEASURE=re.compile(r'(?P<value>[-+]?\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+(?:[.,]\d+)?)?)\s*(?P<unit>mg|g|kg|mcg|µg|ug|ml|l|mmhg|cmh2o|mmol/l|mol/l|iu|units?|%|bpm|°c|c|mm|cm|m|hz|khz|m/s|h|min|s|day|days|week|weeks|month|months|year|years)(?=\s|$|[^\w])',re.I)
NEG=re.compile(r'\b(no|not|without|never|none|cannot|does not|doesn\'t|non|aucun|sans|jamais|ne pas|ممنوع|منع|لا|ليس|دون|absent|absence|inexistent|absent[e]?|غير موجود|غياب)\b',re.I|re.UNICODE)
OPPOSITES=((r'\bcontraindicated\b',r'\bindicated\b'),(r'\bshould not\b',r'\bshould\b'),(r'\bavoid\b',r'\brecommended\b'),(r'\bno\b',r'\bhas\b|\bwith\b'),(r'\bwithout\b',r'\bwith\b'),(r'\babsent\b|\babsence\b',r'\bpresent\b|\bdetected\b'),(r'\bnegative\b',r'\bpositive\b'))
SCALE={'ug':('mass',1e-6),'mcg':('mass',1e-6),'mg':('mass',1e-3),'g':('mass',1),'kg':('mass',1000),'ml':('volume',1),'l':('volume',1000),'mmhg':('pressure',1),'cmh2o':('pressure',.735559),'mmol/l':('amount_concentration',1),'mol/l':('amount_concentration',1000),'iu':('activity',1),'%':('percent',1),'bpm':('rate',1),'c':('temperature',1),'°c':('temperature',1),'mm':('length',1),'cm':('length',10),'m':('length',1000),'hz':('frequency',1),'khz':('frequency',1000),'m/s':('velocity',1),'s':('time',1),'min':('time',60),'h':('time',3600),'day':('time',86400),'days':('time',86400),'week':('time',604800),'weeks':('time',604800),'month':('time',2592000),'months':('time',2592000),'year':('time',31536000),'years':('time',31536000)}
_TINY={'yes','no','ok','okay','thanks','thank','maybe','sure'}
_METADATA_BLOCK=re.compile(r'\[(?:section|source|file|page|document|metadata|citation|reference)\s*:\s*[^\]]*\]\s*',re.I)
_METADATA_LABEL=re.compile(r'^\s*(?:sources?|citations?|references?)\s*:',re.I)
_CONCEPT_SYNONYMS=((r'\bhyperglyc(?:emia|émie)\b|\bhyperglycemia\b','hyperglycemia'),(r'\bcétose\b|\bketosis\b','ketosis'),(r'\bacidose métabolique\b|\bmetabolic acidosis\b','metabolic acidosis'),(r'\bhypoglyc(?:emia|émie)\b|\bhypoglycemia\b','hypoglycemia'),(r'\bhypokali(?:emia|émie)\b|\bhypokalemia\b','hypokalemia'),(r'\bacidocétose diabétique\b|\bdiabetic ketoacidosis\b','diabetic ketoacidosis'),(r'\bnéphropathie diabétique\b|\bnephropathie diabetique\b|\bdiabetic nephropathy\b','diabetic nephropathy'),(r'\bdiabète\b|\bdiabete\b|\bdiabetes mellitus\b','diabetes'),(r'\bcomplication microvasculaire\b|\bmicrovascular complication\b','microvascular complication'),(r'\bchronique\b|\bchronic\b','chronic'),(r'\bdu diabète\b|\bof diabetes\b','of diabetes'))

def _normalize_semantic_text(text:str)->str:
    value=str(text or '').casefold()
    for pattern,replacement in _CONCEPT_SYNONYMS: value=re.sub(pattern,replacement,value,flags=re.I|re.UNICODE)
    try:
        from rag_project.intelligence.semantic_reasoning import ALIASES
        for canonical,aliases in sorted(ALIASES.items(),key=lambda item:max(map(len,item[1])),reverse=True):
            for alias in sorted(aliases,key=len,reverse=True):
                escaped=re.escape(str(alias).casefold().strip())
                if escaped: value=re.sub(rf'(?<!\w){escaped}(?!\w)',canonical.casefold(),value,flags=re.I|re.UNICODE)
    except Exception:
        pass
    return re.sub(r'\s+',' ',value).strip()

def split_claims(answer:str)->list[str]:
    raw=str(answer or '').strip()
    if not raw:return []
    raw=re.sub(r'(?<=[.!?。！？])\s+(?=\[S\d+\])',' ',raw);raw=_METADATA_BLOCK.sub('',raw);out=[]
    for sentence in SENT.split(raw):
        sentence=re.sub(r'^\s*(?:[-*•]|\d+[.)])\s*','',sentence.strip())
        if _METADATA_LABEL.match(sentence):continue
        if re.fullmatch(r'(?:\[S\d+\]\s*)+',sentence,re.I):
            if out:out[-1]=f'{out[-1]} {sentence}'.strip()
            continue
        toks=[t.casefold() for t in meaningful_tokens(sentence)]
        if not toks or (len(toks)<=2 and set(toks).issubset(_TINY)):continue
        out.append(sentence)
        if len(out)>=40:break
    return out

def _norm_unit(u:str)->str:return {'µg':'ug','mcg':'ug','°c':'c','mmol/l':'mmol/l','mol/l':'mol/l'}.get(u.casefold(),u.casefold())
def _num(v:str)->float|None:
    try:return float(v.replace(',','.').replace(' ',''))
    except:return None

def extract_measurements(text:str)->list[tuple[str,str]]:
    out=[]
    for m in MEASURE.finditer(text or ''):
        x=(m.group('value').replace(',','.').replace(' ',''),_norm_unit(m.group('unit')))
        if x not in out:out.append(x)
    return out

def _compatible(a,b):
    av,au=a;bv,bu=b
    if '-' in av or '-' in bv:return av==bv and au==bu
    af,bf=_num(av),_num(bv)
    if af is None or bf is None:return au==bu and av==bv
    sa,sb=SCALE.get(au),SCALE.get(bu)
    if not sa or not sb or sa[0]!=sb[0]:return au==bu and math.isclose(af,bf,abs_tol=1e-9)
    return math.isclose(af*sa[1]/sb[1],bf,rel_tol=0,abs_tol=1e-6)

def _measurement_compatible(a,b):return _compatible(a,b)
def numeric_consistency(claim,evidence):
    cv,ev=extract_measurements(claim),extract_measurements(evidence)
    bad=[x for x in cv if not any(_compatible(x,y) for y in ev)] if cv else []
    return {'checked':bool(cv),'mismatch':bool(bad),'claim_values':[f'{v} {u}' for v,u in cv],'evidence_values':[f'{v} {u}' for v,u in ev],'unsupported_numeric':[f'{v} {u}' for v,u in bad]}

def _polarity(t):return -1 if NEG.search(t or '') else 1
def _score_text(claim:str)->str:return re.sub(r'\[S\d+\]','',claim or '').strip()
def _remove_measurements(text:str)->str:return MEASURE.sub(' ',text or '')

def semantic_support(claim,evidence):
    claim=_score_text(claim);evidence=str(evidence or '').strip()
    if not claim or not evidence:return 0.
    nclaim=re.sub(r'\s+',' ',_normalize_semantic_text(claim)).strip();nevidence=re.sub(r'\s+',' ',_normalize_semantic_text(evidence)).strip()
    if nclaim==nevidence:return 1.0
    ct=set(meaningful_tokens(nclaim));et=set(meaningful_tokens(nevidence))
    if not ct or not et:return 0.
    framing={'the','a','an','main','findings','finding','include','includes','included','reported','reports','observed','shows','show','identified','described','key','primary','principales','conséquences','biologiques','sont','les','des'}
    ct={t for t in ct if t not in framing} or ct
    shared=ct&et
    if len(shared)==1 and (ct-shared) and (et-shared):
        # A single shared concept is not enough to prove two distinct medical predicates.
        # Example: “Diabetes causes pneumonia” must not be supported by “Diabetes is chronic”.
        return 0.0
    coverage=len(shared)/len(ct)
    if coverage < .50:return 0.0
    char=keyword_overlap_score(nclaim,nevidence)
    jac=len(shared)/max(1,len(ct|et));polarity_penalty=.35 if _polarity(nclaim)!=_polarity(nevidence) else 0
    return max(0.,min(1.,.50*coverage+.25*jac+.25*char-polarity_penalty))

def detect_contradiction(claim,evidence_blocks:Sequence[str])->bool:
    cl=_score_text(claim).casefold();blocks=[evidence_blocks] if isinstance(evidence_blocks,str) else list(evidence_blocks or ());cl_tokens=set(meaningful_tokens(cl));generic={'patient','the','is','has','with','present','presence','absent','absence','not','no'}
    for ev in blocks:
        el=str(ev or '').casefold();shared=(cl_tokens&set(meaningful_tokens(el)))-generic;explicit=any(re.search(a,cl,re.I) and re.search(b,el,re.I) for a,b in OPPOSITES);polarity=bool(NEG.search(cl))!=bool(NEG.search(el)) and bool(shared)
        if (explicit or polarity) and (semantic_support(cl,el)>=.08 or len(shared)>=1):return True
    return False

def _best_support(claim,blocks,ids):
    rows=sorted(((semantic_support(claim,b),ids[i] if i<len(ids) else f'S{i+1}') for i,b in enumerate(blocks)),reverse=True);rows=[r for r in rows if r[0]>.05]
    return (rows[0][0],tuple(x[1] for x in rows[:3])) if rows else (0.,())

def verify_claims(answer,evidence_blocks:Sequence[str],source_ids:Sequence[str])->list[ClaimCheck]:
    checks=[];joined='\n'.join(evidence_blocks)
    for claim in split_claims(answer):
        best,sources=_best_support(claim,evidence_blocks,source_ids);num=numeric_consistency(claim,joined);contra=detect_contradiction(claim,evidence_blocks);numeric_bridge=max((semantic_support(_remove_measurements(claim),_remove_measurements(block)) for block in evidence_blocks),default=0.0) if num['checked'] and not num['mismatch'] else 0.0
        if contra:status,reason='CONTRADICTED','A source conflicts with the claim polarity or safety meaning.'
        elif num['mismatch']:status,reason='NUMERIC_MISMATCH','The stated measurement is not supported by a compatible evidence value.'
        elif best>=.62 or numeric_bridge>=.35:status,reason='SUPPORTED','Strong evidence support.'
        elif best>=.38:status,reason='PARTIAL','Partial evidence support.'
        elif best>.05:status,reason='WEAK','Weak evidence overlap.'
        else:status,reason='UNSUPPORTED','No meaningful evidence support.'
        checks.append(ClaimCheck(claim,round(max(best,numeric_bridge),4),status,sources,bool(num['mismatch']),contra,reason))
    return checks

def evidence_confidence(*,retrieval:float,rerank:float,entailment:float,quality:float,contradiction:float=0.,ocr_penalty:float=0.)->float:return round(max(0.,min(1.,.24*retrieval+.26*rerank+.30*entailment+.20*quality-.40*contradiction-.20*ocr_penalty)),4)
def contradiction_report(claims):
    bad=[c for c in claims if c.contradiction or c.status=='CONTRADICTED'];return {'has_contradiction':bool(bad),'count':len(bad),'claims':[c.to_dict() for c in bad]}
def citation_firewall(answer,claim_checks:Iterable[ClaimCheck]):
    checks=list(claim_checks);bad=[c for c in checks if c.status in {'UNSUPPORTED','NUMERIC_MISMATCH','CONTRADICTED'} or c.contradiction]
    if not bad:return answer,False
    safe=[c for c in checks if c.status in {'SUPPORTED','PARTIAL','ENTAILED'} and not c.contradiction];lines=['Verified findings:'] if safe else [];lines += [f'- {c.claim} {" ".join(f"[{s}]" for s in c.sources)}'.strip() for c in safe];lines.append('Some generated details were withheld because they could not be verified against the indexed evidence.')
    return '\n'.join(lines),True
def grounding_decision(claims,*,min_supported_ratio=.60):
    if not claims:return {'allow':False,'reason':'No claims were extracted from the generated answer.','supported_ratio':0.}
    supported_statuses={'SUPPORTED','PARTIAL','ENTAILED'};safe=sum(c.status in supported_statuses and not c.contradiction for c in claims);blocked=sum(c.status in {'UNSUPPORTED','WEAK','NUMERIC_MISMATCH','CONTRADICTED'} or c.contradiction for c in claims);ratio=safe/len(claims)
    return {'allow':ratio>=min_supported_ratio and blocked==0,'reason':'Grounding threshold passed.' if ratio>=min_supported_ratio and blocked==0 else 'Grounding threshold failed.','supported_ratio':round(ratio,4),'blocked_claims':blocked}