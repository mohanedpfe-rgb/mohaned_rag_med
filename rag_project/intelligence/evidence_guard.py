from __future__ import annotations

import math
import re
from dataclasses import asdict,dataclass
from typing import Any,Iterable,Sequence
from rag_project.utils.text_utils import keyword_overlap_score,meaningful_tokens

@dataclass(frozen=True)
class ClaimCheck:
    claim:str; support:float; status:str; sources:tuple[str,...]; numeric_mismatch:bool=False; contradiction:bool=False; reason:str=''
    def to_dict(self): return asdict(self)

SENT=re.compile(r'(?<=[.!?。！？])\s+|\n+')
MEASURE=re.compile(r'(?P<value>[-+]?\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+(?:[.,]\d+)?)?)\s*(?P<unit>mg|g|kg|mcg|µg|ug|ml|l|mmhg|cmh2o|%|bpm|°c|c|mm|cm|m|hz|khz|m/s|h|min|s|day|days|week|weeks|month|months|year|years)\b',re.I)
NEG=re.compile(r'\b(no|not|without|never|none|cannot|does not|doesn\'t|non|aucun|sans|jamais|ne\s+pas|ممنوع|منع|لا|ليس|دون)\b',re.I|re.UNICODE)
OPPOSITES=((r'\bcontraindicated\b',r'\bindicated\b'),(r'\bshould not\b',r'\bshould\b'),(r'\bavoid\b',r'\brecommended\b'),(r'\bno\b',r'\bhas\b|\bwith\b'),(r'\bwithout\b',r'\bwith\b'))
SCALE={'ug':('mass',1e-6),'mcg':('mass',1e-6),'mg':('mass',1e-3),'g':('mass',1),'kg':('mass',1000),'ml':('volume',1),'l':('volume',1000),'mmhg':('pressure',1),'cmh2o':('pressure',.735559),'%':('percent',1),'bpm':('rate',1),'c':('temperature',1),'°c':('temperature',1),'mm':('length',1),'cm':('length',10),'m':('length',1000),'hz':('frequency',1),'khz':('frequency',1000),'m/s':('velocity',1),'s':('time',1),'min':('time',60),'h':('time',3600),'day':('time',86400),'days':('time',86400),'week':('time',604800),'weeks':('time',604800),'month':('time',2592000),'months':('time',2592000),'year':('time',31536000),'years':('time',31536000)}
_TINY={'yes','no','ok','okay','thanks','thank','maybe','sure'}

def split_claims(answer:str)->list[str]:
    raw=str(answer or '').strip()
    if not raw:return []
    lines=[]
    bullet_mode=False
    for line in raw.splitlines():
        if not line.strip():continue
        bullet_mode = bullet_mode or bool(re.match(r'^\s*(?:[-*•]|\d+[.)])\s+', line))
        value=re.sub(r'^\s*(?:[-*•]|\d+[.)])\s*','',line).strip()
        if value:lines.append(value)
    if bullet_mode:
        return lines[:40]
    out=[]
    for sentence in SENT.split(raw):
        sentence=re.sub(r'^\s*(?:[-*•]|\d+[.)])\s*','',sentence.strip())
        toks=[t.casefold() for t in meaningful_tokens(sentence)]
        if not toks:continue
        if len(toks)<=2 and set(toks).issubset(_TINY):continue
        out.append(sentence)
        if len(out)>=40:break
    return out

def _norm_unit(u:str)->str:return {'µg':'ug','mcg':'ug','°c':'c'}.get(u.casefold(),u.casefold())
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
    av,au=a; bv,bu=b
    if '-' in av or '-' in bv:return av==bv and au==bu
    af,bf=_num(av),_num(bv)
    if af is None or bf is None:return au==bu and av==bv
    sa,sb=SCALE.get(au),SCALE.get(bu)
    if not sa or not sb or sa[0]!=sb[0]:return au==bu and math.isclose(af,bf,abs_tol=1e-9)
    return math.isclose(af*sa[1]/sb[1],bf,rel_tol=0,abs_tol=1e-6)

def numeric_consistency(claim,evidence):
    cv,ev=extract_measurements(claim),extract_measurements(evidence)
    bad=[x for x in cv if not any(_compatible(x,y) for y in ev)] if cv else []
    return {'checked':bool(cv),'mismatch':bool(bad),'claim_values':[f'{v} {u}' for v,u in cv],'evidence_values':[f'{v} {u}' for v,u in ev],'unsupported_numeric':[f'{v} {u}' for v,u in bad]}

def _polarity(t):return -1 if NEG.search(t or '') else 1

def _score_text(claim:str)->str:
    return re.sub(r'\[S\d+\]','',claim or '').strip()

def semantic_support(claim,evidence):
    claim=_score_text(claim); evidence=str(evidence or '')
    ct=set(meaningful_tokens(claim)); et=set(meaningful_tokens(evidence))
    if not ct or not et:return 0.
    overlap=len(ct&et)/len(ct); jac=len(ct&et)/max(1,len(ct|et)); char=keyword_overlap_score(claim,evidence)
    polarity_penalty=.35 if _polarity(claim)!=_polarity(evidence) else 0
    return max(0.,min(1.,.50*overlap+.25*jac+.25*char-polarity_penalty))

def detect_contradiction(claim,evidence_blocks:Sequence[str])->bool:
    cl=_score_text(claim).casefold()
    for ev in evidence_blocks:
        el=ev.casefold()
        opposite=any(re.search(a,cl,re.I) and re.search(b,el,re.I) for a,b in OPPOSITES) or (_polarity(cl)!=_polarity(el) and set(meaningful_tokens(cl))&set(meaningful_tokens(el)))
        if opposite and semantic_support(cl,el)>=.18:return True
    return False

def _best_support(claim,blocks,ids):
    rows=sorted(((semantic_support(claim,b),ids[i] if i<len(ids) else f'S{i+1}') for i,b in enumerate(blocks)),reverse=True)
    rows=[r for r in rows if r[0]>0]
    return (rows[0][0],tuple(x[1] for x in rows[:3])) if rows else (0.,())

def verify_claims(answer,evidence_blocks:Sequence[str],source_ids:Sequence[str])->list[ClaimCheck]:
    checks=[]; joined='\n'.join(evidence_blocks)
    for claim in split_claims(answer):
        best,sources=_best_support(claim,evidence_blocks,source_ids); num=numeric_consistency(claim,joined); contra=detect_contradiction(claim,evidence_blocks)
        if contra: status,reason='CONTRADICTED','A source conflicts with the claim polarity or safety meaning.'
        elif num['mismatch']: status,reason='NUMERIC_MISMATCH','The stated measurement is not supported by a compatible evidence value.'
        elif best>=.62: status,reason='SUPPORTED','Strong evidence support.'
        elif best>=.38: status,reason='PARTIAL','Partial evidence support.'
        elif best>0: status,reason='WEAK','Weak evidence overlap.'
        else: status,reason='UNSUPPORTED','No meaningful evidence support.'
        checks.append(ClaimCheck(claim,round(best,4),status,sources,bool(num['mismatch']),contra,reason))
    return checks

def evidence_confidence(*,retrieval:float,rerank:float,entailment:float,quality:float,contradiction:float=0.,ocr_penalty:float=0.)->float:
    return round(max(0.,min(1.,.24*retrieval+.26*rerank+.30*entailment+.20*quality-.40*contradiction-.20*ocr_penalty)),4)

def contradiction_report(claims):
    bad=[c for c in claims if c.contradiction or c.status=='CONTRADICTED']; return {'has_contradiction':bool(bad),'count':len(bad),'claims':[c.to_dict() for c in bad]}

def citation_firewall(answer,claim_checks:Iterable[ClaimCheck]):
    checks=list(claim_checks); bad=[c for c in checks if c.status in {'UNSUPPORTED','NUMERIC_MISMATCH','CONTRADICTED'} or c.contradiction]
    if not bad:return answer,False
    safe=[c for c in checks if c.status in {'SUPPORTED','PARTIAL'} and not c.contradiction]; lines=['Verified findings:'] if safe else []
    lines += [f'- {c.claim} {" ".join(f"[{s}]" for s in c.sources)}'.strip() for c in safe]; lines.append('Some generated details were withheld because they could not be verified against the indexed evidence.')
    return '\n'.join(lines),True

def grounding_decision(claims,*,min_supported_ratio=.60):
    if not claims:return {'allow':False,'reason':'No claims were extracted from the generated answer.','supported_ratio':0.}
    safe=sum(c.status in {'SUPPORTED','PARTIAL'} and not c.contradiction for c in claims); blocked=sum(c.status in {'UNSUPPORTED','NUMERIC_MISMATCH','CONTRADICTED'} or c.contradiction for c in claims); ratio=safe/len(claims)
    return {'allow':ratio>=min_supported_ratio and blocked==0,'reason':'Grounding threshold passed.' if ratio>=min_supported_ratio and blocked==0 else 'Grounding threshold failed.','supported_ratio':round(ratio,4),'blocked_claims':blocked}
