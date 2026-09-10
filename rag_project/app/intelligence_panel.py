from __future__ import annotations
from typing import Any
import streamlit as st

def _pct(value: Any) -> str:
    try:return f'{float(value)*100:.0f}%'
    except:return '—'

def _show_list(label: str, values: Any):
    vals=[str(x) for x in (values or ()) if str(x).strip()]
    if vals: st.write(f'**{label}:** ' + ', '.join(vals[:16]))

def _status_icon(status: str) -> str:
    return {'SUPPORTED':'✅','PARTIAL':'🟡','WEAK':'🟠','UNSUPPORTED':'❌','CONTRADICTED':'❌','NUMERIC_MISMATCH':'❌'}.get(str(status).upper(),'•')

def render_intelligence_panel(result: dict[str,Any]):
    with st.expander('Intelligence pipeline · 5 phases', expanded=False):
        plan=result.get('phase_plan') or result.get('query_analysis') or {};phase_state=result.get('phases') or {}
        cols=st.columns(5);labels=(('P1','Query'),('P2','Retrieval'),('P3','Generation'),('P4','Verification'),('P5','UI'))
        for col,(code,label) in zip(cols,labels):
            with col:
                key={'P1':'phase_1_query_understanding','P2':'phase_2_retrieval_precision','P3':'phase_3_two_stage_generation','P4':'phase_4_verification','P5':'phase_5_intelligence_visibility'}[code];value=phase_state.get(key,'complete');st.metric(f'{code} {label}',str(value).upper())
        st.divider()
        _show_list('Intent',[plan.get('intent','')]);_show_list('Entities',plan.get('entities'));_show_list('Sub-questions',plan.get('sub_questions'));_show_list('Rewritten queries',plan.get('rewritten_queries'));_show_list('Must contain',plan.get('must_contain'))
        st.write(f"**Ambiguity:** {plan.get('ambiguity','—')} · **Planner:** {plan.get('planner_source','—')} · **Planner confidence:** {_pct(plan.get('planner_confidence'))}")
        st.write(f"**Answer shape:** {plan.get('answer_shape','—')} · **Numeric:** {bool(plan.get('needs_numeric'))} · **Table:** {bool(plan.get('needs_table'))} · **Multi-hop:** {bool(plan.get('needs_multi_hop'))}")
        retrieval=result.get('adaptive_retrieval') or result.get('adaptive_retrieval_budget') or {};st.write(f"**Adaptive retrieval:** stage {retrieval.get('stage','—')} · queries {retrieval.get('queries','—')} · candidate K {retrieval.get('candidate_k','—')} · final hits {retrieval.get('final_hits','—')} · escalated={bool(retrieval.get('escalated'))}")
        _show_list('Escalation reasons',retrieval.get('reasons'))
        terms=result.get('medical_term_layer') or {};_show_list('Detected medical terms',terms.get('terms'));_show_list('Drugs',terms.get('drug_like'));_show_list('Condition-like terms',terms.get('condition_like'));_show_list('Abbreviations',terms.get('abbreviations'));_show_list('Units',terms.get('units'))
        extractive=result.get('extractive_stage') or {};two_stage=result.get('two_stage_synthesis') or {};st.write(f"**Extractive stage:** {extractive.get('sentence_count',0)} evidence sentences · supported={bool(extractive.get('supported'))}");st.write(f"**Synthesis stage:** attempted={bool(two_stage.get('attempted'))} · used={bool(two_stage.get('used'))} · fallback={bool(two_stage.get('fallback'))} · temperature={two_stage.get('temperature',result.get('generation_temperature','—'))}")
        calibration=result.get('confidence_calibration') or {};confidence=result.get('confidence') or {};calibrated=calibration.get('calibrated',confidence.get('evidence_confidence',0));st.write(f"**Calibrated confidence:** {_pct(calibrated)} · **level:** {calibration.get('level',confidence.get('level','—'))}")
        _show_list('Confidence reasons',calibration.get('reasons'))
        matrix=result.get('evidence_claim_matrix') or [];checks=result.get('claim_checks') or result.get('claims') or []
        if matrix:st.write(f'**Claim → evidence matrix:** {len(matrix)} claim record(s)')
        if checks:
            st.markdown('**Claim support**')
            for row in checks[:16]:
                if hasattr(row,'to_dict'):row=row.to_dict()
                claim=row.get('claim','');status=row.get('status', '');support=row.get('support',row.get('entailment',0));sources=row.get('sources',())
                st.write(f"{_status_icon(status)} **{status or 'UNKNOWN'}** · {claim} · support {_pct(support)} · sources {', '.join(map(str,sources)) if sources else '—'}")
        reasoning=result.get('advanced_reasoning') or {};_show_list('Reasoning blocked reasons',reasoning.get('blocked_reasons'));st.write(f"**Reasoning:** mode={reasoning.get('mode','—')} · depth={reasoning.get('depth','—')} · path support={_pct(reasoning.get('path_support',0))} · source agreement={_pct(reasoning.get('source_agreement',0))} · safety conflict={_pct(reasoning.get('safety_conflict',0))}")
        status=str(result.get('status','')).upper();reasons=reasoning.get('blocked_reasons') or result.get('abstention_reasons') or []
        if result.get('abstained') or status in {'NOT_SUPPORTED','REASONING_ABSTAIN','GENERATION_ABSTAIN'}:st.warning('The system withheld unsupported content.'+(f" Reasons: {', '.join(map(str,reasons))}" if reasons else ''))
