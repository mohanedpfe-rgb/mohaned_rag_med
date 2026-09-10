from __future__ import annotations
from pathlib import Path
from typing import Any,Dict,List
from rag_project.app import rag_system as rag_system_module
from rag_project.app.resilient_rag import ResilientRAGSystem
from rag_project.ingestion import robust_ingestor
from rag_project.intelligence.god_mode import audit_god_mode_index
from rag_project.intelligence.god_mode_100 import enhanced_god_answer
from rag_project.intelligence.final_44 import _wrap_answer
from rag_project.intelligence.medical_safety import apply_medical_safety_policy
from rag_project.intelligence.production_contract import sanitize_trace,validate_feature_contract
class ProductionRAGSystem(ResilientRAGSystem):
    _certified_god_answer=_wrap_answer(enhanced_god_answer)
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
    def answer(self,question:str,metadata_filter:Dict[str,Any]|None=None)->dict[str,Any]:
        if not self._production_feature_contract['all_resolved']:return {'status':'SYSTEM_NOT_READY','answer':'The production feature contract is incomplete; a grounded answer is disabled.','citations':[],'hits':[],'confidence':{'level':'none','evidence_confidence':0.0},'production_contract':self._production_feature_contract}
        result=self._certified_god_answer(question,metadata_filter);result=apply_medical_safety_policy(question,result,self.settings)
        if 'query_trace' in result:result['query_trace']=sanitize_trace(result['query_trace'])
        result['production_contract']={'feature_count':44,'all_features_resolved':True}
        return result
    def audit_god_mode_index(self):return audit_god_mode_index(self)
    def health_report(self):
        checks={}
        try:
            self._ensure_embedding_dimension();identity=self.embedding_service.identity;err=getattr(self,'embedding_startup_error',None);checks['embedding']={'ok':err is None and identity is not None,'identity':identity,'error':err}
        except Exception as exc:checks['embedding']={'ok':False,'error':type(exc).__name__}
        try:checks['index']=self.vector_store.compatibility_report(self.embedding_service.identity)
        except Exception as exc:checks['index']={'status':'UNAVAILABLE','error':type(exc).__name__}
        try:checks['audit']=self.audit_god_mode_index()
        except Exception as exc:checks['audit']={'ok':False,'error':type(exc).__name__}
        checks['feature_contract']=self._production_feature_contract;checks['models']={'embedding_model':self.settings.embedding_model,'generation_model':self.settings.generation_model};checks['pipeline']={'explicit_composition':True,'medical_safety_policy':True,'privacy_safe_trace':True,'feature_count':44,'incremental_ingestion':True,'canonical_ingestion':'robust_ingestor','god_mode_100':True,'claim_evidence_matrix':True,'hierarchical_evidence':True,'adaptive_retrieval':True,'confidence_calibration':True,'critic_repair_reverification':True,'phase_1_query_understanding':True,'phase_2_retrieval_precision':True,'phase_3_two_stage_generation':True,'phase_4_verification':True,'phase_5_intelligence_visibility':True}
        status=str(checks.get('index',{}).get('status','READY')).upper();checks['ready']=bool(checks['embedding']['ok'] and status in {'READY','OK'} and self._production_feature_contract['all_resolved']);return checks
__all__=['ProductionRAGSystem']
