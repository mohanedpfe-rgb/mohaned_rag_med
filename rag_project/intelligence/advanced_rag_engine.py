"""Document-aware, query-routed retrieval engine for production BookRAG."""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from rag_project.intelligence.document_intelligence import build_document_map, section_coverage, structural_score
from rag_project.intelligence.medical_knowledge_graph import build_graph, expand_from_query
from rag_project.intelligence.query_intelligence import plan_query
from rag_project.utils.text_utils import keyword_overlap_score, meaningful_tokens


@dataclass(frozen=True)
class QueryRoute:
    kind: str
    scope: str
    needs_table: bool = False
    needs_numeric: bool = False
    needs_figure: bool = False
    multi_hop: bool = False
    summary: bool = False
    comparison: bool = False
    exact_lookup: bool = False
    expected_slots: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    confidence: float = 0.8
    def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass
class RetrievalDecision:
    query: str
    route: QueryRoute
    queries: list[str] = field(default_factory=list)
    candidates: int = 0
    final_hits: int = 0
    rerank_ms: float = 0.0
    retrieval_ms: float = 0.0
    self_corrections: int = 0
    strategy: list[str] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    contradiction: dict[str, Any] = field(default_factory=dict)
    document_map: dict[str, Any] = field(default_factory=dict)
    structure: dict[str, Any] = field(default_factory=dict)
    knowledge_graph: dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self); data["route"] = self.route.to_dict(); return data


_MEDICAL_SYNONYMS = {
    "mi": ("myocardial infarction", "heart attack"), "htn": ("hypertension", "high blood pressure"),
    "dm": ("diabetes mellitus", "diabetes"), "copd": ("chronic obstructive pulmonary disease",),
    "ckd": ("chronic kidney disease",), "aki": ("acute kidney injury",), "cva": ("stroke", "cerebrovascular accident"),
    "pe": ("pulmonary embolism",), "dvt": ("deep vein thrombosis",), "acs": ("acute coronary syndrome",),
    "af": ("atrial fibrillation",), "ecg": ("electrocardiogram",), "tsh": ("thyroid stimulating hormone", "thyrotropin"),
    "hba1c": ("hemoglobin a1c", "glycated hemoglobin"),
}
_SUMMARY = re.compile(r"\b(summary|summarize|summarise|overview|main findings|key findings|main points|key points)\b|résumé|synthèse|aperçu|points principaux|résultats principaux|ملخص|خلاصة|نظرة عامة|النقاط الرئيسية", re.I|re.UNICODE)
_COMPARE = re.compile(r"\b(compare|comparison|difference|differences|versus|vs\.?|contrast|distinguish)\b|comparer|différence|différences|مقارنة|الفرق", re.I|re.UNICODE)
_TABLE = re.compile(r"\b(table|tabular|criteria|score|classification|staging|reference range|normal range|cut[- ]off|threshold)\b|tableau|critères|classification|stade|plage normale|جدول|معايير|تصنيف", re.I|re.UNICODE)
_NUMERIC = re.compile(r"\b(value|values|number|percent|percentage|rate|range|dose|duration|time|age|years?|days?|hours?|mg|mcg|g/dl|mmhg|bpm|mmol|mol|cm|kg)\b|valeur|pourcentage|dose|durée|âge|سنة|جرعة|نسبة", re.I|re.UNICODE)
_MULTI = re.compile(r"\b(and|also|as well as|respectively|first.*then|cause.*effect)\b|\bet\b|ainsi que|puis|ثم", re.I|re.UNICODE)
_EXACT = re.compile(r"\b(what is|define|definition|who is|which drug|which test|where is|page|exact|specifically)\b|définir|définition|quel médicament|quel test|exactement|ما هو|تعريف", re.I|re.UNICODE)


def _clean(value: Any, limit: int = 3500) -> str: return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _entities_from_plan(question: str) -> tuple[str, ...]:
    try: raw = getattr(plan_query(question, conversation_context=""), "entities", ()) or ()
    except Exception: raw = ()
    out=[]; seen=set()
    for value in raw:
        text=_clean(value,160); key=text.casefold()
        if text and key not in seen: seen.add(key); out.append(text)
    return tuple(out[:12])


def route_query(question: str, planner: Any | None = None) -> QueryRoute:
    q=_clean(question); entities=tuple(_clean(x,160) for x in (getattr(planner,"entities",()) or ()) if _clean(x,160))[:12] if planner is not None else _entities_from_plan(q)
    summary=bool(_SUMMARY.search(q)); comparison=bool(_COMPARE.search(q)); needs_table=bool(_TABLE.search(q)); needs_numeric=bool(_NUMERIC.search(q)) or bool(re.search(r"\d",q)); multi_hop=bool(_MULTI.search(q)) or comparison or len(entities)>=2; exact=bool(_EXACT.search(q)) or len(meaningful_tokens(q))<=5
    if summary: kind,slots="summary",("overview","key_points","supporting_details")
    elif comparison: kind,slots="comparison",("definition","entity_a","entity_b","differences","similarities")
    elif needs_table and needs_numeric: kind,slots="table_numeric",("target_value","unit_or_condition","context")
    elif needs_table: kind,slots="table_lookup",("criteria_or_rows","context")
    elif multi_hop: kind,slots="multi_hop",("premise","relationship","conclusion")
    elif exact: kind,slots="exact_fact",("direct_answer","supporting_context")
    else: kind,slots="factual",("answer","supporting_context")
    confidence=.92 if summary or comparison else .86 if needs_table or multi_hop else .82
    return QueryRoute(kind,"document" if summary else "question",needs_table,needs_numeric,False,multi_hop,summary,comparison,exact,slots,entities,confidence)


def expand_query_variants(question: str, route: QueryRoute, base_plan: Any | None = None, *, limit: int = 8) -> list[str]:
    q=_clean(question); values=[q]
    if base_plan is None:
        try: base_plan=plan_query(q,conversation_context="")
        except Exception: base_plan=None
    if base_plan is not None:
        values += [_clean(x) for x in (getattr(base_plan,"variants",()) or ())]
        values += [_clean(x) for x in (getattr(base_plan,"subqueries",()) or ())]
    if route.kind=="summary": values += [f"{q} overview",f"{q} introduction",f"{q} key points",f"{q} conclusion",f"{q} chapter sections"]
    elif route.kind=="comparison" and len(route.entities)>=2:
        a,b=route.entities[:2]; values += [f"{a} {b} differences",f"{a} {b} comparison",f"{a} versus {b}"]
    elif route.kind in {"table_lookup","table_numeric"}: values += [f"{q} table",f"{q} criteria",f"{q} reference range"]
    elif route.kind=="exact_fact": values += [f"definition {q}",f"exact {q}"]
    for token in meaningful_tokens(q):
        for synonym in _MEDICAL_SYNONYMS.get(token.casefold(),()): values.append(re.sub(re.escape(token),synonym,q,flags=re.I))
    # A configured query budget is a retrieval contract, not merely an upper bound.
    # When the planner yields fewer variants, fill the remaining slots with deterministic
    # evidence-oriented formulations so recovery cannot start prematurely.
    values += [
        f"{q} clinical findings", f"{q} manifestations", f"{q} signs symptoms",
        f"{q} diagnosis", f"{q} definition", f"{q} evidence", f"{q} relevant section",
    ]
    unique=[]; seen=set()
    for value in values:
        value=_clean(value); key=value.casefold()
        if value and key not in seen: seen.add(key); unique.append(value)
    return unique[:max(1,int(limit))]


def _page(meta: dict[str,Any]) -> str:
    value=meta.get("page_numbers") or meta.get("page_number") or meta.get("page") or "?"
    return str(value[0] if isinstance(value,(list,tuple)) and value else value)


def _semantic_score(hit: Any) -> float:
    vector=getattr(hit,"vector_score",0.0)
    return max(0.0,min(1.0,float(vector if vector else (getattr(hit,"score",0.0) or 0.0))))


def _type_bonus(meta: dict[str,Any], route: QueryRoute) -> float:
    types={str(x).casefold() for x in (meta.get("evidence_types") or ())}; page_type=str(meta.get("page_type") or "").casefold(); bonus=structural_score(meta,prefer=("table",) if route.needs_table else ())*0.30
    if route.needs_table and ("table" in types or meta.get("table_id") or "table" in page_type): bonus+=.22
    if route.needs_figure and ("figure" in types or meta.get("figure_id") or "figure" in page_type): bonus+=.16
    return min(.5,bonus)


def rerank_hits(question: str, hits: Iterable[Any], route: QueryRoute) -> list[Any]:
    q=" ".join(meaningful_tokens(question)); q_tokens=set(meaningful_tokens(question)); rows=[]
    for hit in hits:
        text=str(getattr(hit,"text","") or ""); meta=dict(getattr(hit,"metadata",{}) or {}); semantic=_semantic_score(hit)
        lexical=max(0.0,min(1.0,float(getattr(hit,"lexical_score",0.0) or keyword_overlap_score(q,text) or 0.0))); exact=1.0 if q and q.casefold() in text.casefold() else 0.0
        overlap=len(q_tokens & set(meaningful_tokens(text)))/max(1,len(q_tokens)) if q_tokens else 0.0; section=" ".join(str(meta.get(k) or "") for k in ("chapter","section","subsection","headings")); section_overlap=keyword_overlap_score(q,section) if section else 0.0
        score=max(0.0,min(1.5,.32*semantic+.25*lexical+.17*overlap+.10*exact+.10*section_overlap+_type_bonus(meta,route)))
        try: hit.score=score
        except Exception: pass
        rows.append((score,hit))
    rows.sort(key=lambda x:x[0],reverse=True); return [h for _,h in rows]


def diversify_hits(hits: list[Any], route: QueryRoute, limit: int) -> list[Any]:
    limit=max(1,int(limit)); selected=[]; seen_keys=set(); seen_pages=set(); seen_sections=set()
    for hit in hits:
        meta=dict(getattr(hit,"metadata",{}) or {}); doc=str(meta.get("document_id") or getattr(hit,"doc_id","")); page=_page(meta); section=str(meta.get("section_id") or meta.get("section") or ""); key=str(meta.get("chunk_id") or f"{doc}:{page}:{str(getattr(hit,'text',''))[:80]}")
        if key in seen_keys: continue
        if route.summary and len(selected)<min(limit,16):
            page_key=f"{doc}:{page}"
            if page_key in seen_pages: continue
            seen_pages.add(page_key)
        elif route.kind=="comparison" and len(selected)<min(limit,12):
            entity_text=(str(getattr(hit,"text","") )+" "+section).casefold(); covered=sum(1 for e in route.entities if e.casefold() in entity_text)
            if covered==0 and selected: continue
        elif section and section in seen_sections and len(selected)<max(1,limit//2): continue
        seen_keys.add(key); seen_sections.add(section) if section else None; selected.append(hit)
        if len(selected)>=limit: break
    if len(selected)<limit:
        for hit in hits:
            if hit not in selected: selected.append(hit)
            if len(selected)>=limit: break
    return selected


def _summary_coverage(hits:list[Any])->dict[str,Any]:
    structure=section_coverage(hits); ds=int(structure.get("distinct_sections",0)); dp=int(structure.get("distinct_pages",0)); quality=sum(max(0.0,min(1.0,float(getattr(h,"score",0.0) or 0.0))) for h in hits)/max(1,len(hits)); overall=.45*min(1,ds/4)+.35*min(1,dp/6)+.20*quality
    return {"slots":{"overview":overall,"key_points":overall,"supporting_details":overall},"overall":round(overall,4),"entity_coverage":1.0,"sufficient":overall>=.34,"basis":"document_structure_coverage","structure":structure}


def evidence_coverage(question:str,route:QueryRoute,hits:list[Any])->dict[str,Any]:
    if route.summary: return _summary_coverage(hits)
    combined=" ".join(str(getattr(h,"text","") or "") for h in hits).casefold(); coverage={}
    for slot in route.expected_slots:
        if slot in {"entity_a","entity_b"}: idx=0 if slot=="entity_a" else 1; tokens=set(meaningful_tokens(route.entities[idx])) if idx<len(route.entities) else set()
        elif slot in {"differences","similarities"}: tokens=set(meaningful_tokens(" ".join(route.entities)))
        elif slot in {"target_value","unit_or_condition","criteria_or_rows"}: tokens=set(meaningful_tokens(question))|set(re.findall(r"\b\d+(?:\.\d+)?\b",question))
        elif slot in {"premise","relationship","conclusion"}: tokens=set(meaningful_tokens(question))
        else: tokens=set(meaningful_tokens(question))- {"what","are","the","is","this","that","about","how","why"}
        tokens={t for t in tokens if len(t)>2}; coverage[slot]=min(1.0,sum(1 for t in tokens if t in combined)/max(1,len(tokens)))
    overall=sum(coverage.values())/max(1,len(coverage)); entity=min(1.0,sum(1 for e in route.entities if e.casefold() in combined)/max(1,len(route.entities))); sufficient=overall >= (.45 if route.comparison or route.multi_hop else .30)
    return {"slots":coverage,"overall":round(overall,4),"entity_coverage":round(entity,4),"sufficient":sufficient,"basis":"query_term_and_entity_coverage"}


def detect_contradictions(hits:list[Any])->dict[str,Any]:
    pattern=re.compile(r"(?<!\w)(\d+(?:\.\d+)?\s*(?:%|mg|mcg|g|kg|mmHg|bpm|years?|days?|hours?|mmol/L|mg/dL)?)(?!\w)",re.I); generic={"the","and","with","from","this","that","are","was","for","dans","les","des","une","un","avec"}; groups={}
    for hit in hits:
        meta=dict(getattr(hit,"metadata",{}) or {}); text=str(getattr(hit,"text","") or ""); anchor=" ".join(t for t in meaningful_tokens(pattern.sub(" ",text)) if len(t)>3 and t not in generic).casefold()
        if not anchor: continue
        for num in pattern.findall(text): groups.setdefault(anchor,[]).append((num.casefold(),_page(meta),str(meta.get("document_id") or getattr(hit,"doc_id",""))))
    conflicts=[]
    for anchor,values in groups.items():
        unique={v[0] for v in values}
        if len(unique)>1 and len(values)>=2: conflicts.append({"anchor":anchor,"values":sorted(unique),"sources":[{"value":v,"page":p,"document_id":d} for v,p,d in values[:8]]})
    return {"has_contradiction":bool(conflicts),"conflicts":conflicts[:12],"count":len(conflicts)}


def _add_parent_context(selected:list[Any],all_hits:list[Any],limit:int)->list[Any]:
    by_parent={}
    for hit in all_hits:
        meta=dict(getattr(hit,"metadata",{}) or {}); parent=str(meta.get("parent_id") or meta.get("section_id") or "")
        if parent and parent not in by_parent: by_parent[parent]=hit
    out=[]
    for hit in selected:
        out.append(hit); meta=dict(getattr(hit,"metadata",{}) or {}); parent=str(meta.get("parent_id") or meta.get("section_id") or ""); parent_hit=by_parent.get(parent)
        if parent_hit is not None and parent_hit not in out: out.append(parent_hit)
        if len(out)>=limit: break
    return out[:limit]


def _metadata_filter(metadata_filter):
    try:
        from rag_project.retrieval.metadata_filter import MetadataFilter
        return MetadataFilter.build(metadata_filter)
    except Exception: return None


def _add_candidates(candidates, raw):
    for hit in raw or ():
        if hit is None or not str(getattr(hit,"text","") or "").strip(): continue
        meta=dict(getattr(hit,"metadata",{}) or {}); key=str(meta.get("chunk_id") or f"{getattr(hit,'doc_id','')}:{_page(meta)}:{str(getattr(hit,'text',''))[:120]}"); old=candidates.get(key)
        if old is None or _semantic_score(hit)>_semantic_score(old): candidates[key]=hit


def _ordered_correction_hits(question: str, correction_hits: list[Any], route: QueryRoute, ranked: list[Any], limit: int) -> list[Any]:
    """Rank only the evidence returned by corrective queries, then keep it authoritative."""
    if not correction_hits:
        return []
    corrected = rerank_hits(question, correction_hits, route)
    if not corrected:
        return []
    seen = {id(hit) for hit in corrected}
    tail = [hit for hit in ranked if id(hit) not in seen]
    return (corrected + tail)[:max(1, int(limit))]


def retrieve_document_aware(system:Any,question:str,metadata_filter:dict[str,Any]|None=None,*,top_k:int=8)->tuple[list[Any],dict[str,Any]]:
    started=time.perf_counter()
    try: configured=int(getattr(getattr(system,"settings",None),"max_query_variants",8))
    except Exception: configured=8
    route=route_query(question)
    try: plan=plan_query(question,conversation_context="")
    except Exception: plan=None
    query_limit=min(8,max(3,configured)); queries=expand_query_variants(question,route,plan,limit=query_limit); strategy=[route.kind,"semantic+lexical","exact-term-reranking","structural-reranking","medical-synonym-expansion"]
    if route.summary: strategy += ["document-coverage","section-diversification","parent-context"]
    if route.needs_table: strategy.append("table-priority")
    if route.needs_numeric: strategy.append("numeric-exactness")
    if route.multi_hop: strategy.append("decomposed-queries")
    candidates={}; attempts=0; where=_metadata_filter(metadata_filter); candidate_k=max(24,min(64,int(top_k)*5))
    for query in queries:
        try: raw=system.retriever.retrieve(query,top_k=candidate_k,where=where)
        except Exception: raw=[]
        attempts+=1; _add_candidates(candidates,raw)
    graph=build_graph(candidates.values()) if route.multi_hop else {"nodes":[],"edges":[],"adjacency":{},"node_count":0,"edge_count":0}
    if route.multi_hop and graph.get("edge_count",0):
        for query in expand_from_query(question,graph,limit=2):
            if query.casefold() in {q.casefold() for q in queries}: continue
            queries.append(query)
            try: raw=system.retriever.retrieve(query,top_k=min(48,max(24,int(top_k)*5)),where=where)
            except Exception: raw=[]
            attempts+=1; _add_candidates(candidates,raw)
        strategy.append("document_graph_expansion")
    rerank_start=time.perf_counter(); ranked=rerank_hits(question,list(candidates.values()),route); selection_limit=max(int(top_k)*3,16); selected=_add_parent_context(diversify_hits(ranked,route,selection_limit),ranked,selection_limit); coverage=evidence_coverage(question,route,selected); correction_count=0
    if not coverage["sufficient"] and attempts < 12:
        correction_count=1; correction_queries=list(queries[-2:])+[f"{question} section",f"{question} clinical findings",f"{question} definition"]; correction_queries=list(dict.fromkeys(_clean(x) for x in correction_queries if _clean(x)))[:3]
        correction_hits=[]
        for query in correction_queries:
            try: raw=system.retriever.retrieve(query,top_k=max(32,min(64,int(top_k)*6)),where=where)
            except Exception: raw=[]
            attempts+=1
            for hit in raw or ():
                if hit is not None and str(getattr(hit,"text","") or "").strip(): correction_hits.append(hit)
            _add_candidates(candidates,raw)
        ranked=rerank_hits(question,list(candidates.values()),route)
        correction_ranked=_ordered_correction_hits(question,correction_hits,route,ranked,selection_limit)
        if correction_ranked:
            selected=[correction_ranked[0]] + [h for h in correction_ranked[1:] if h is not correction_ranked[0]][:selection_limit-1]
            selected=_add_parent_context(selected,[correction_ranked[0]]+ranked,selection_limit)
        else:
            selected=_add_parent_context(ranked[:selection_limit],ranked,selection_limit)
        coverage=evidence_coverage(question,route,selected); strategy.append("self-correction")
    contradiction=detect_contradictions(selected); structure=section_coverage(selected); document_map=build_document_map(selected); elapsed=(time.perf_counter()-started)*1000
    decision=RetrievalDecision(query=_clean(question),route=route,queries=queries[:16],candidates=len(candidates),final_hits=len(selected),rerank_ms=round((time.perf_counter()-rerank_start)*1000,2),retrieval_ms=round(elapsed,2),self_corrections=correction_count,strategy=strategy,coverage=coverage,contradiction=contradiction,document_map=document_map,structure=structure,knowledge_graph=graph)
    return selected,decision.to_dict()


def build_answer_plan(question:str,route:QueryRoute,coverage:dict[str,Any])->dict[str,Any]:
    return {"question_type":route.kind,"scope":route.scope,"required_slots":list(route.expected_slots),"entities":list(route.entities),"covered_slots":[s for s,v in (coverage.get("slots") or {}).items() if float(v)>=.35],"missing_slots":[s for s,v in (coverage.get("slots") or {}).items() if float(v)<.35],"coverage":float(coverage.get("overall",0.0) or 0.0),"coverage_basis":coverage.get("basis","unknown"),"generation_rule":"evidence_only"}


__all__=["QueryRoute","RetrievalDecision","route_query","expand_query_variants","rerank_hits","diversify_hits","evidence_coverage","detect_contradictions","retrieve_document_aware","build_answer_plan"]
