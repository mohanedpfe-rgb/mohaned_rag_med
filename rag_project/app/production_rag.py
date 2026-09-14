from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from rag_project.app import rag_system as rag_system_module
from rag_project.app.resilient_rag import ResilientRAGSystem
from rag_project.ingestion import versioned_ingestor
from rag_project.ingestion import robust_ingestor
from rag_project.intelligence.god_mode_100 import enhanced_god_answer
from rag_project.intelligence.medical_safety import apply_medical_safety_policy
from rag_project.intelligence.production_contract import validate_feature_contract
from rag_project.intelligence.retrieval_replay import record as record_replay
from rag_project.intelligence.runtime_safety import execute_with_runtime_safety
from rag_project.intelligence.trace_privacy import sanitize_trace

if TYPE_CHECKING:
    from rag_project.application import ANSWER_PIPELINE_AUTHORITY

ANSWER_PIPELINE_AUTHORITY = "rag_project.intelligence.top_level_pipeline.complete_phases"
_FOLLOWUP_PATTERN = re.compile(r"\b(it|this|that|they|them|those|these|the latter|the former|what about|how about)\b|^(and|also|then|et|puis|و|ثم)\b|^و(?=\S)|\b(ça|cela|celui|celle|et le|et la)\b", re.I | re.UNICODE)


def _is_explicit_followup(question: str) -> bool:
    return bool(_FOLLOWUP_PATTERN.search(str(question or "").strip()))


def _safe_list(value: Any) -> list[Any]:
    if value is None: return []
    if isinstance(value, list): return value
    if isinstance(value, tuple): return list(value)
    try: return list(value)
    except (TypeError, ValueError): return []


def _safe_logger(system: Any) -> Any:
    logger = getattr(system, "logger", None)
    return logger if logger is not None and callable(getattr(logger, "exception", None)) else logging.getLogger(__name__)


def _safe_exception_log(system: Any, message: str) -> None:
    try: _safe_logger(system).exception(message)
    except Exception: pass


def _safe_result(value: Any) -> dict[str, Any]: return dict(value) if isinstance(value, dict) else {}


def _should_store_in_history(result: Any) -> bool:
    return isinstance(result, dict) and bool(str(result.get("answer") or "").strip()) and str(result.get("status") or "").upper() in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}


def _record_answer_replay(system: Any, question: str, result: dict[str, Any]) -> None:
    try:
        root = getattr(getattr(system, "settings", None), "project_root", None)
        if root:
            record_replay(root, {"success":str(result.get("status") or "").upper() in {"SUCCESS","SUCCESS_WITH_WARNINGS"},"question":str(question or "")[:1200],"status":str(result.get("status") or "").upper()})
    except Exception: pass


class ProductionRAGSystem(ResilientRAGSystem):
    _certified_god_answer = enhanced_god_answer

    def __init__(self, settings=None):
        super().__init__(settings)
        self._production_feature_contract = validate_feature_contract()

    def _new_cancel_flag(self, document_id):
        flag = rag_system_module._IngestCancelFlag()
        with rag_system_module._INGEST_LOCK: rag_system_module._INGEST_CANCEL_FLAGS[document_id] = flag
        return flag

    def _remove_cancel_flag(self, document_id):
        with rag_system_module._INGEST_LOCK: rag_system_module._INGEST_CANCEL_FLAGS.pop(document_id, None)

    def _archive_duplicate_upload(self, pdf_path, document_id, result):
        source=Path(pdf_path); incoming_dir=getattr(self.settings,"incoming_dir",None)
        if incoming_dir is None: return result
        try: source.resolve().relative_to(Path(incoming_dir).resolve())
        except ValueError: return result
        if not source.is_file(): return result
        try:
            digest=self._hash_file(source); archive=Path(self.settings.archive_dir); archive.mkdir(parents=True,exist_ok=True); target=archive/source.name
            if target.exists(): target=archive/f"{source.stem}-duplicate-{digest[:12]}{source.suffix}"
            if target.exists(): target=archive/f"{source.stem}-duplicate-{digest[:12]}-{document_id[:8]}{source.suffix}"
            source.replace(target); out=dict(result); out["archived_duplicate"]=str(target); return out
        except OSError as exc:
            out=dict(result); out["archive_warning"]=f"Duplicate was skipped but could not be archived: {type(exc).__name__}"; return out

    def ingest_file(self, pdf_path):
        source=Path(pdf_path)
        if not hasattr(self, "state_store"):
            result=robust_ingestor.robust_ingest_file(self, source)
        else:
            result=versioned_ingestor.ingest_version_safely(self, source)
        if str((result or {}).get("status") or "").lower()=="skipped":
            result=self._archive_duplicate_upload(source,str((result or {}).get("document_id") or "unknown"),result)
        return result

    def ingest_directory(self,directory=None):
        source=Path(directory) if directory else self.settings.incoming_dir; source.mkdir(parents=True,exist_ok=True); return [self.ingest_file(p) for p in sorted(source.glob("*.pdf"))]

    def cancel_all_ingests(self):
        with rag_system_module._INGEST_LOCK:
            count=0
            for flag in rag_system_module._INGEST_CANCEL_FLAGS.values():
                if not flag.cancelled: flag.cancel(); count+=1
            return count

    def _recovery_answer(self,question,metadata_filter,exc):
        retriever=getattr(self,"retriever",None)
        retrieve=getattr(retriever,"retrieve",None)
        if not callable(retrieve): return {"status":"ANSWER_UNAVAILABLE","answer":"I could not safely produce an answer from the indexed evidence right now.","citations":[],"hits":[],"confidence":{"level":"none","evidence_confidence":0.0},"recovery":{"attempted":True,"retrieval_failed":"RetrieverUnavailable","pipeline_error":type(exc).__name__},"pipeline_authority":ANSWER_PIPELINE_AUTHORITY}
        try: hits=[h for h in _safe_list(retrieve(str(question or "").strip(),top_k=max(1,min(int(getattr(self.settings,"top_k",8)),12)))) if h is not None]
        except Exception as retrieve_exc: return {"status":"ANSWER_UNAVAILABLE","answer":"I could not safely produce an answer from the indexed evidence right now.","citations":[],"hits":[],"confidence":{"level":"none","evidence_confidence":0.0},"recovery":{"attempted":True,"retrieval_failed":type(retrieve_exc).__name__,"pipeline_error":type(exc).__name__},"pipeline_authority":ANSWER_PIPELINE_AUTHORITY}
        if not hits: return {"status":"NOT_SUPPORTED","answer":"I could not find sufficient evidence in the indexed documents to answer this question.","citations":[],"hits":[],"confidence":{"level":"none","evidence_confidence":0.0},"recovery":{"attempted":True,"pipeline_error":type(exc).__name__},"pipeline_authority":ANSWER_PIPELINE_AUTHORITY}
        try:
            from rag_project.intelligence.god_mode import _simple_extractive_answer
            answer=str(_simple_extractive_answer(str(question or ""),hits,max_sentences=6) or "").strip()
        except Exception: answer=""
        if not answer: return {"status":"ANSWER_UNAVAILABLE","answer":"The indexed evidence was retrieved, but it could not be safely converted into a grounded answer.","citations":[],"hits":hits,"confidence":{"level":"low","evidence_confidence":0.0},"recovery":{"attempted":True,"pipeline_error":type(exc).__name__},"pipeline_authority":ANSWER_PIPELINE_AUTHORITY}
        return {"status":"SUCCESS_WITH_WARNINGS","answer":answer,"citations":[],"hits":hits,"confidence":{"level":"medium","evidence_confidence":0.7},"recovery":{"attempted":True,"pipeline_error":type(exc).__name__,"grounded_extractive_fallback":True},"pipeline_authority":ANSWER_PIPELINE_AUTHORITY}

    def answer(self,question:str,metadata_filter:Dict[str,Any]|None=None)->dict[str,Any]:
        feature_contract=getattr(self,"_production_feature_contract",{"all_resolved":True,"feature_count":44})
        if not feature_contract.get("all_resolved",False): return {"status":"SYSTEM_NOT_READY","answer":"The production feature contract is incomplete; a grounded answer is disabled.","citations":[],"hits":[],"confidence":{"level":"none","evidence_confidence":0.0},"production_contract":feature_contract}
        memory=getattr(self,"conversation_memory",None); original_question=str(question or "").strip()
        if _is_explicit_followup(original_question) and memory is not None:
            try: question=memory.rewrite(original_question)
            except Exception: question=original_question
        else: question=original_question
        def _primary_answer():
            try: return _safe_result(self._certified_god_answer(question,metadata_filter))
            except Exception as exc: _safe_exception_log(self,"Primary answer pipeline failed"); return self._recovery_answer(question,metadata_filter,exc)
        result=execute_with_runtime_safety(self,question,_primary_answer)
        result=apply_medical_safety_policy(question,result,self.settings)
        result.setdefault("pipeline_authority",ANSWER_PIPELINE_AUTHORITY)
        result.setdefault("production_contract",{"feature_count":int(feature_contract.get("feature_count",44)),"all_features_resolved":bool(feature_contract.get("all_resolved",False))})
        try: result["query_trace"]=sanitize_trace(result.get("query_trace") or {})
        except Exception: pass
        if _should_store_in_history(result) and memory is not None:
            try:
                add=getattr(memory,"add",None)
                if callable(add): add(original_question,result)
                elif isinstance(getattr(memory,"history",None),list): memory.history.append((original_question,result.get("answer","")))
            except Exception: pass
        _record_answer_replay(self,original_question,result)
        return result
