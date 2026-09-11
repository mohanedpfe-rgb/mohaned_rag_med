from __future__ import annotations
import re
import time
from pathlib import Path
from typing import Any,Dict
from rag_project.app import rag_system as rag_system_module
from rag_project.app.resilient_rag import ResilientRAGSystem
from rag_project.ingestion import robust_ingestor
from rag_project.intelligence.god_mode import audit_god_mode_index
from rag_project.intelligence.god_mode_100 import enhanced_god_answer
from rag_project.intelligence.medical_safety import apply_medical_safety_policy
from rag_project.intelligence.production_contract import sanitize_trace,validate_feature_contract
from rag_project.application import ANSWER_PIPELINE_AUTHORITY
from rag_project.generation.latency_budget import request_budget, exhausted, elapsed

_FOLLOWUP_PATTERN=re.compile(r"\b(it|this|that|they|them|those|these|the latter|the former|what about|how about)\b|^(and|also|then|et|puis|و|ثم)\b|^و(?=\S)|\b(ça|cela|celui|celle|et le|et la)\b",re.I|re.UNICODE)

def _is_explicit_followup(question:str)->bool:
    text=str(question or '').strip()
    return bool(_FOLLOWUP_PATTERN.search(text))


def _safe_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except (TypeError, ValueError):
        return []


class ProductionRAGSystem(ResilientRAGSystem):
    _certified_god_answer=enhanced_god_answer
    def __init__(self,settings=None):super().__init__(settings);self._production_feature_contract=validate_feature_contract()
    def _new_cancel_flag(self,document_id):
        flag=rag_system_module._IngestCancelFlag()
        with rag_system_module._INGEST_LOCK:rag_system_module._INGEST_CANCEL_FLAGS[document_id]=flag
        return flag
    def _remove_cancel_flag(self,document_id):
        with rag_system_module._INGEST_LOCK:rag_system_module._INGEST_CANCEL_FLAGS.pop(document_id,None)
    def _archive_duplicate_upload(self,pdf_path,document_id,result):
        source=Path(pdf_path);incoming_dir=getattr(self.settings,'incoming_dir',None)
        if incoming_dir is None:return result
        try:source.resolve().relative_to(Path(incoming_dir).resolve())
        except ValueError:return result
        if not source.is_file():return result
        try:
            digest=self._hash_file(source);archive=Path(self.settings.archive_dir);archive.mkdir(parents=True,exist_ok=True);target=archive/source.name
            if target.exists():target=archive/f'{source.stem}-duplicate-{digest[:12]}{source.suffix}'
            if target.exists():target=archive/f'{source.stem}-duplicate-{digest[:12]}-{document_id[:8]}{source.suffix}'
            source.replace(target);out=dict(result);out['archived_duplicate']=str(target);return out
        except OSError as exc:out=dict(result);out['archive_warning']=f'Duplicate was skipped but could not be archived: {type(exc).__name__}';return out
    def ingest_file(self,pdf_path):
        result=robust_ingestor.robust_ingest_file(self,pdf_path)
        if str((result or {}).get('status') or '').lower()=='skipped':result=self._archive_duplicate_upload(pdf_path,str((result or {}).get('document_id') or 'unknown'),result)
        return result
    def ingest_directory(self,directory=None):
        source=Path(directory) if directory else self.settings.incoming_dir;source.mkdir(parents=True,exist_ok=True);return [self.ingest_file(p) for p in sorted(source.glob('*.pdf'))]
    def cancel_all_ingests(self):
        with rag_system_module._INGEST_LOCK:
            n=0
            for flag in rag_system_module._INGEST_CANCEL_FLAGS.values():
                if not flag.cancelled:flag.cancel();n+=1
            return n

    def _recovery_answer(self,question:str,metadata_filter:Dict[str,Any]|None,exc:Exception)->dict[str,Any]:
        """Return a grounded extractive answer when the certified pipeline hits an unexpected runtime error."""
        try:
            from rag_project.retrieval.metadata_filter import MetadataFilter
            where=MetadataFilter.build(metadata_filter)
        except Exception:
            where=None
        try:
            hits=_safe_list(self.retriever.retrieve(str(question or '').strip(),top_k=max(1,min(int(getattr(self.settings,'top_k',8)),12)),where=where))
            hits=[hit for hit in hits if hit is not None]
        except Exception as retrieve_exc:
            self.logger.exception("Recovery retrieval failed")
            return {
                'status':'ANSWER_UNAVAILABLE',
                'answer':'I could not safely produce an answer from the indexed evidence right now.',
                'citations':[],
                'hits':[],
                'confidence':{'level':'none','evidence_confidence':0.0},
                'recovery':{'attempted':True,'retrieval_failed':type(retrieve_exc).__name__,'pipeline_error':type(exc).__name__},
                'pipeline_authority':ANSWER_PIPELINE_AUTHORITY,
            }
        if not hits:
            return {
                'status':'NOT_SUPPORTED',
                'answer':'I could not find sufficient evidence in the indexed documents to answer this question.',
                'citations':[],
                'hits':[],
                'confidence':{'level':'none','evidence_confidence':0.0},
                'recovery':{'attempted':True,'pipeline_error':type(exc).__name__},
                'pipeline_authority':ANSWER_PIPELINE_AUTHORITY,
            }
        try:
            from rag_project.intelligence.god_mode import _simple_extractive_answer
            answer=str(_simple_extractive_answer(str(question or ''),hits,max_sentences=6) or '').strip()
        except Exception as answer_exc:
            self.logger.exception("Recovery extractive answer failed")
            answer=''
            answer_error=type(answer_exc).__name__
        else:
            answer_error=None
        if not answer:
            return {
                'status':'ANSWER_UNAVAILABLE',
                'answer':'The indexed evidence was retrieved, but it could not be safely converted into a grounded answer.',
                'citations':[],
                'hits':hits,
                'confidence':{'level':'low','evidence_confidence':0.0},
                'recovery':{'attempted':True,'pipeline_error':type(exc).__name__,'extractive_failed':answer_error},
                'pipeline_authority':ANSWER_PIPELINE_AUTHORITY,
            }
        try:
            built=self.citation_manager.build(hits) or []
            citations=self.citation_manager.validate(built,hits) or []
        except Exception:
            citations=[]
        try:
            from rag_project.intelligence.evidence_guard import verify_claims, grounding_decision
            blocks=[str(getattr(hit,'text','') or '') for hit in hits]
            checks=_safe_list(verify_claims(answer,blocks,[f'S{i+1}' for i in range(len(hits))]))
            ground=grounding_decision(checks,min_supported_ratio=0.60) if checks else {'allow':False,'supported_ratio':0.0}
        except Exception:
            checks=[];ground={'allow':False,'supported_ratio':0.0}
        if not checks or not ground.get('allow',False):
            return {
                'status':'ANSWER_UNAVAILABLE',
                'answer':'The indexed evidence was retrieved, but the answer could not pass the grounding check safely.',
                'citations':[],
                'hits':hits,
                'confidence':{'level':'low','evidence_confidence':float(ground.get('supported_ratio',0.0) or 0.0)},
                'recovery':{'attempted':True,'pipeline_error':type(exc).__name__,'grounding_failed':True},
                'pipeline_authority':ANSWER_PIPELINE_AUTHORITY,
            }
        return {
            'status':'SUCCESS_WITH_WARNINGS',
            'answer':answer,
            'citations':citations,
            'hits':hits,
            'confidence':{'level':'medium','evidence_confidence':float(ground.get('supported_ratio',0.0) or 0.0)},
            'grounding':ground,
            'claims':[getattr(check,'to_dict',lambda: {'claim':str(getattr(check,'claim',''))})() for check in checks],
            'recovery':{'attempted':True,'pipeline_error':type(exc).__name__,'grounded_extractive_fallback':True},
            'pipeline_authority':ANSWER_PIPELINE_AUTHORITY,
        }

    def answer(self,question:str,metadata_filter:Dict[str,Any]|None=None)->dict[str,Any]:
        request_started=time.perf_counter()
        if not self._production_feature_contract['all_resolved']:
            return {'status':'SYSTEM_NOT_READY','answer':'The production feature contract is incomplete; a grounded answer is disabled.','citations':[],'hits':[],'confidence':{'level':'none','evidence_confidence':0.0},'production_contract':self._production_feature_contract}
        memory=getattr(self,'conversation_memory',None)
        isolated=memory is not None and not _is_explicit_followup(question)
        saved_history=list(getattr(memory,'history',[]) or []) if isolated else []
        result=None
        with request_budget(self.settings) as budget:
            if isolated:
                memory.history=[]
            try:
                try:
                    result=self._certified_god_answer(question,metadata_filter)
                except Exception as exc:
                    self.logger.exception("Certified answer pipeline failed; entering grounded recovery path")
                    result=self._recovery_answer(question,metadata_filter,exc)
                result=apply_medical_safety_policy(question,result,self.settings)
                result.setdefault('pipeline_authority',ANSWER_PIPELINE_AUTHORITY)
                if 'query_trace' in result:result['query_trace']=sanitize_trace(result['query_trace'])
                result['latency_budget_seconds']=budget
                result['latency_elapsed_seconds']=round(elapsed(),3)
                result['latency_budget_exhausted']=exhausted()
                result['production_contract']={'feature_count':44,'all_features_resolved':True}
                return result
            finally:
                if isolated:
                    memory.history=saved_history
                    if isinstance(result,dict) and str(result.get('answer') or '').strip():
                        add=getattr(memory,'add',None)
                        if callable(add):
                            add(question,str(result.get('answer') or ''))
                        else:
                            memory.history.append((question,str(result.get('answer') or '')))
    def audit_god_mode_index(self):return audit_god_mode_index(self)
    def health_report(self):
        checks={}
        try:self._ensure_embedding_dimension();identity=self.embedding_service.identity;err=getattr(self,'embedding_startup_error',None);checks['embedding']={'ok':err is None and identity is not None,'identity':identity,'error':err}
        except Exception as exc:checks['embedding']={'ok':False,'error':type(exc).__name__}
        try:checks['index']=self.vector_store.compatibility_report(self.embedding_service.identity)
        except Exception as exc:checks['index']={'status':'UNAVAILABLE','error':type(exc).__name__}
        try:checks['audit']=self.audit_god_mode_index()
        except Exception as exc:checks['audit']={'ok':False,'error':type(exc).__name__}
        checks['feature_contract']=self._production_feature_contract;checks['models']={'embedding_model':self.settings.embedding_model,'generation_model':self.settings.generation_model};checks['pipeline']={'explicit_composition':True,'authority':ANSWER_PIPELINE_AUTHORITY,'medical_safety_policy':True,'privacy_safe_trace':True,'feature_count':44,'incremental_ingestion':True,'canonical_ingestion':'robust_ingestor','claim_evidence_matrix':True,'hierarchical_evidence':True,'adaptive_retrieval':True,'confidence_calibration':True,'critic_repair_reverification':True,'shared_latency_budget':True,'latency_hard_cap_seconds':45.0};status=str(checks.get('index',{}).get('status','READY')).upper();checks['ready']=bool(checks['embedding']['ok'] and status in {'READY','OK'} and self._production_feature_contract['all_features_resolved']);return checks
__all__=['ProductionRAGSystem']
