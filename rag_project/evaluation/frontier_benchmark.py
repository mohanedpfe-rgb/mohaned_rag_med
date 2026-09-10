from __future__ import annotations
import json
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import Any,Callable,Iterable
@dataclass(frozen=True)
class BenchmarkCase:
    case_id:str; category:str; question:str; expected_entities:tuple[str,...]=(); expected_intent:str=''; gold_evidence_ids:tuple[str,...]=(); answerable:bool=True
@dataclass(frozen=True)
class BenchmarkResult:
    case_id:str; category:str; entity_hit:bool; intent_hit:bool; evidence_hit:bool; grounded:bool; abstained_correctly:bool; citation_precision:float; support_ratio:float
    def to_dict(self):return asdict(self)
@dataclass(frozen=True)
class BenchmarkReport:
    total:int; entity_recall:float; intent_accuracy:float; evidence_recall:float; grounding_rate:float; abstention_accuracy:float; mean_citation_precision:float; mean_support_ratio:float; by_category:dict[str,dict[str,float]]
    def to_dict(self):return asdict(self)
def load_cases(path:str|Path)->tuple[BenchmarkCase,...]:
    p=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(p,list):raise ValueError('Benchmark file must contain a JSON array.')
    return tuple(BenchmarkCase(str(x.get('case_id','')),str(x.get('category','general')),str(x.get('question','')),tuple(map(str,x.get('expected_entities',()) or ())),str(x.get('expected_intent','')),tuple(map(str,x.get('gold_evidence_ids',()) or ())),bool(x.get('answerable',True))) for x in p if isinstance(x,dict))
def _abstained(r):return str(r.get('status','')).upper() in {'NOT_SUPPORTED','REASONING_ABSTAIN','LOW_QUALITY_QUERY','ABSTAINED','INTERNAL_ERROR','SYSTEM_NOT_READY'}
def evaluate_case(c:BenchmarkCase,r:dict[str,Any])->BenchmarkResult:
    a=r.get('query_analysis') or {}; ents={str(x).casefold() for x in a.get('entities',())}; expected={str(x).casefold() for x in c.expected_entities}; entity_hit=bool(expected and expected&ents)
    intent_hit=not c.expected_intent or str(a.get('intent',''))==c.expected_intent
    hit_ids={str((getattr(h,'metadata',{}) or {}).get('chunk_id') or getattr(h,'doc_id','')) for h in (r.get('hits') or [])}; evidence_hit=bool(c.gold_evidence_ids) and bool(hit_ids&set(c.gold_evidence_ids))
    if not c.gold_evidence_ids: evidence_hit=False
    g=r.get('grounding') or {}; grounded=bool(g.get('allow')) or str(r.get('status','')).upper()=='SUCCESS'; abst=_abstained(r); abstention_correct=(not c.answerable and abst) or (c.answerable and not abst)
    claims=r.get('claims') or []; sources={str(s) for x in claims if isinstance(x,dict) for s in x.get('sources',()) if s}; supports=[float(x.get('support',0.)) for x in claims if isinstance(x,dict)]; cp=sum(s.startswith('S') for s in sources)/max(1,len(sources)) if sources else 0.; sr=sum(supports)/max(1,len(supports)); return BenchmarkResult(c.case_id,c.category,entity_hit,intent_hit,evidence_hit,grounded,abstention_correct,round(cp,4),round(sr,4))
def run_benchmark(cases:Iterable[BenchmarkCase],runner:Callable[[str],dict[str,Any]])->BenchmarkReport:
    results=[evaluate_case(c,runner(c.question)) for c in cases]; total=len(results)
    if not total:return BenchmarkReport(0,0,0,0,0,0,0,0,{})
    cats={}
    for r in results:cats.setdefault(r.category,[]).append(r)
    def avg(items,attr):return sum(getattr(x,attr) for x in items)/len(items)
    by={k:{'count':float(len(v)),'entity_recall':avg(v,'entity_hit'),'intent_accuracy':avg(v,'intent_hit'),'evidence_recall':avg(v,'evidence_hit'),'grounding_rate':avg(v,'grounded'),'abstention_accuracy':avg(v,'abstained_correctly'),'citation_precision':avg(v,'citation_precision'),'support_ratio':avg(v,'support_ratio')} for k,v in cats.items()}
    return BenchmarkReport(total,avg(results,'entity_hit'),avg(results,'intent_hit'),avg(results,'evidence_hit'),avg(results,'grounded'),avg(results,'abstained_correctly'),avg(results,'citation_precision'),avg(results,'support_ratio'),by)
