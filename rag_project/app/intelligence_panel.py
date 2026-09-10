from __future__ import annotations
from typing import Any
import streamlit as st

def _pct(value: Any) -> str:
    try:return f'{float(value)*100:.0f}%'
    except:return '—'

def _show_list(label: str, values: Any):
    vals=[str(x) for x in (values or ()) if str(x).strip()]
    if vals: st.write(f'**{label}:** ' + ', '.join(vals[:12]))

def render_intelligence_panel(result: dict[str,Any]):
    with st.expander('Intelligence pipeline · 5 phases', expanded=False):
        plan=result.get('phase_plan') or result.get('query_analysis') or {}
        phase_state=result.get('phases') or {}
        cols=st.columns(5)
        labels=(('P1','Query'),('P2','Retrieval'),('P3','Generation'),('P4','Verification'),('P5','UI'))
        for col,(code,label) in zip(cols,labels):
            with col:
                key={'P1':'phase_1_query_understanding','P2':'phase_2_retrieval_precision','P3':'phase_3_two_stage_generation','P4':'phase_4_verification','P5':'phase_5_intelligence_visibility'}[code]
                value=phase_state.get(key,'complete')
                st.metric(f'{code} {label}', str(value).upper())
        st.divider()
        _show_list('Intent', [plan.get('intent','')])
        _show_list('Entities', plan.get('entities'))
        _show_list('Sub-questions', plan.get('sub_questions'))
        _show_list('Rewritten queries', plan.get('rewritten_queries'))
        _show_list('Must contain', plan.get('must_contain'))
        st.write(f"**Ambiguity:** {plan.get('ambiguity','—')} · **Planner:** {plan.get('planner_source','—')} · **Planner confidence:** {_pct(plan.get('planner_confidence'))}")
        retrieval=result.get('adaptive_retrieval') or result.get('adaptive_retrieval_budget') or {}
        st.write(f"**Retrieval:** stage {retrieval.get('stage','—')} · queries {retrieval.get('queries','—')} · candidate K {retrieval.get('candidate_k','—')} · final hits {retrieval.get('final_hits','—')}")
        terms=result.get('medical_term_layer') or {}
        _show_list('Detected medical terms', terms.get('terms'))
        _show_list('Units', terms.get('units'))
        extractive=result.get('extractive_stage') or {}
        st.write(f"**Extractive stage:** {extractive.get('sentence_count',0)} evidence sentences selected · supported={bool(extractive.get('supported'))}")
        calibration=result.get('confidence_calibration') or result.get('confidence') or {}
        st.write(f"**Calibrated confidence:** {_pct(calibration.get('calibrated', calibration.get('evidence_confidence',0)))} · level {calibration.get('level','—')}")
        matrix=result.get('evidence_claim_matrix') or []
        if matrix:
            st.write(f'**Claim → evidence matrix:** {len(matrix)} claim record(s)')
            for row in matrix[:12]:
                claim=row.get('claim',''); support=row.get('support',row.get('entailment',0)); spans=row.get('spans') or row.get('evidence_spans') or []
                st.write(f'• {claim} · support {_pct(support)} · spans {len(spans)}')
        cert=result.get('certification') or {}
        if cert: st.write(f"**Certification:** fail-closed={cert.get('fail_closed',{}).get('allow') is False} · firewall={cert.get('firewall_used',False)}")
        reasons=(result.get('advanced_reasoning') or {}).get('blocked_reasons') or []
        if result.get('abstained') or str(result.get('status','')).upper() in {'NOT_SUPPORTED','REASONING_ABSTAIN','GENERATION_ABSTAIN'}:
            st.warning('The system withheld unsupported content.' + (f" Reason: {', '.join(map(str,reasons))}" if reasons else ''))
