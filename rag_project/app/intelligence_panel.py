from __future__ import annotations
from typing import Any
import streamlit as st

def _pct(value:Any)->str:
    try:return f'{float(value)*100:.0f}%'
    except:return '—'
def _show_list(label:str,values:Any):
    vals=[str(x) for x in (values or ()) if str(x).strip()]
    if vals:st.write(f'**{label}:** '+', '.join(vals[:16]))
def _status_icon(status:str)->str:return {'SUPPORTED':'✅','PARTIAL':'🟡','PARTIALLY_ENTAILED':'🟡','WEAK':'🟠','UNSUPPORTED':'❌','CONTRADICTED':'❌','NUMERIC_MISMATCH':'❌','ENTAILED':'✅','NOT_ENTAILED':'❌'}.get(str(status).upper(),'•')

def render_intelligence_panel(result:dict[str,Any]):
    with st.expander('Intelligence pipeline · 5 phases',expanded=True):
        authority=result.get('pipeline_authority') or (result.get('phase_implementation') or {}).get('authority') or '—';st.caption(f'Authoritative pipeline: `{authority}`')
        plan=result.get('phase_plan') or result.get('query_analysis') or {};phase_state=result.get('phases') or {};implementation=result.get('phase_implementation') or {}
        cols=st.columns(5);labels=(('P1','Query'),('P2','Retrieval'),('P3','Generation'),('P4','Verification'),('P5','UI'));keys={'P1':'phase_1_query_understanding','P2':'phase_2_retrieval_precision','P3':'phase_3_two_stage_generation','P4':'phase_4_verification','P5':'phase_5_intelligence_visibility'}
        for col,(code,label) in zip(cols,labels):
            with col:
                raw=implementation.get(keys[code]);value=(raw or {}).get('status') if isinstance(raw,dict) else raw;value=value or phase_state.get(keys[code]) or 'unknown';st.metric(f'{code} {label}',str(value).upper())
        st.divider();_show_list('Intent',[plan.get('intent','')]);_show_list('Entities',plan.get('entities'));_show_list('Sub-questions',plan.get('sub_questions'));_show_list('Rewritten queries',plan.get('rewritten_queries'));_show_list('Must contain',plan.get('must_contain'));st.write(f"**Ambiguity:** {plan.get('ambiguity','—')} · **Planner:** {plan.get('planner_source','—')} · **Planner confidence:** {_pct(plan.get('planner_confidence'))}");st.write(f"**Answer shape:** {plan.get('answer_shape','—')} · **Numeric:** {bool(plan.get('needs_numeric'))} · **Table:** {bool(plan.get('needs_table'))} · **Figure:** {bool(plan.get('needs_figure'))} · **Multi-hop:** {bool(plan.get('needs_multi_hop'))}")
        retrieval=result.get('adaptive_retrieval') or result.get('adaptive_retrieval_budget') or {};st.write(f"**Adaptive retrieval:** stage {retrieval.get('stage','—')} · queries {retrieval.get('queries','—')} · candidate K {retrieval.get('candidate_k',retrieval.get('budget','—'))} · final hits {retrieval.get('final_hits','—')} · escalated={bool(retrieval.get('escalated'))}");_show_list('Escalation reasons',retrieval.get('reasons'))
        terms=result.get('medical_term_layer') or {};_show_list('Detected medical terms',terms.get('terms'));_show_list('Drugs',terms.get('drug_like'));_show_list('Condition-like terms',terms.get('condition_like'));_show_list('Abbreviations',terms.get('abbreviations'));_show_list('Units',terms.get('units'))
        extractive=result.get('extractive_stage') or {};two_stage=result.get('two_stage_synthesis') or {};st.write(f"**Extractive stage:** {extractive.get('sentence_count',0)} evidence sentences · supported={bool(extractive.get('supported'))}");st.write(f"**Synthesis stage:** required={bool(two_stage.get('required'))} · attempted={bool(two_stage.get('attempted'))} · used={bool(two_stage.get('used'))} · fallback={bool(two_stage.get('fallback'))} · temperature={two_stage.get('temperature',result.get('generation_temperature','—'))}")
        calibration=result.get('confidence_calibration') or {};confidence=result.get('confidence') or {};calibrated=calibration.get('calibrated',confidence.get('evidence_confidence',0));st.write(f"**Calibrated confidence:** {_pct(calibrated)} · **level:** {calibration.get('level',confidence.get('level','—'))}");_show_list('Confidence reasons',calibration.get('reasons'))
        matrix=result.get('evidence_claim_matrix') or [];checks=result.get('claim_checks') or result.get('claims') or []
        if matrix:
            st.markdown('**Claim → evidence matrix**')
            for row in matrix[:16]:
                row=row if isinstance(row,dict) else (row.to_dict() if hasattr(row,'to_dict') else {});status=row.get('status','');st.write(f"{_status_icon(status)} **{status or 'UNKNOWN'}** · {row.get('claim','')} · support {_pct(row.get('support',0))} · evidence spans {len(row.get('evidence') or ())}")
        elif checks:
            st.markdown('**Claim support**')
            for row in checks[:16]:
                if hasattr(row,'to_dict'):row=row.to_dict()
                st.write(f"{_status_icon(row.get('status',''))} **{row.get('status','UNKNOWN')}** · {row.get('claim','')} · support {_pct(row.get('support',0))} · sources {', '.join(map(str,row.get('sources',()))) if row.get('sources') else '—'}")
        reasoning=result.get('advanced_reasoning') or {};_show_list('Reasoning blocked reasons',reasoning.get('blocked_reasons'));st.write(f"**Reasoning:** mode={reasoning.get('mode','—')} · depth={reasoning.get('depth','—')} · path support={_pct(reasoning.get('path_support',0))} · source agreement={_pct(reasoning.get('source_agreement',0))} · safety conflict={_pct(reasoning.get('safety_conflict',0))}")
        reasons=reasoning.get('blocked_reasons') or result.get('abstention_reasons') or [];status=str(result.get('status','')).upper();contract=result.get('production_contract') or {};st.write(f"**Pipeline contract:** features={contract.get('feature_count','—')} · all resolved={contract.get('all_features_resolved','—')}")
        if result.get('abstained') or status in {'NOT_SUPPORTED','REASONING_ABSTAIN','GENERATION_ABSTAIN'}:st.warning('The system withheld unsupported content.'+(f" Reasons: {', '.join(map(str,reasons))}" if reasons else ''))
