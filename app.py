from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import streamlit as st
from rag_project.app import bookrag_ui
from rag_project.app.intelligence_panel import render_intelligence_panel
from rag_project.ingestion.responsive_supervisor import start as start_supervisor
from rag_project.security import register_session_upload,validate_ollama_url,validate_pdf_payload,validate_storage_path

def _load_local_env():
    f=Path(__file__).resolve().parent/'.env'
    if not f.is_file():return
    try:
        for raw in f.read_text(encoding='utf-8').splitlines():
            line=raw.strip()
            if not line or line.startswith('#') or '=' not in line:continue
            k,v=line.split('=',1);k=k.strip();v=v.strip()
            if not k or k in os.environ:continue
            if len(v)>=2 and v[0]==v[-1] and v[0] in {'"',"'"}:v=v[1:-1]
            os.environ[k]=v
    except OSError:pass

def _clamp_local_embedding_profile():
    try:b=int(os.getenv('EMBEDDING_BATCH_SIZE','16'))
    except (TypeError,ValueError):b=16
    try:r=int(os.getenv('EMBEDDING_RETRIES','2'))
    except (TypeError,ValueError):r=2
    try:t=float(os.getenv('EMBEDDING_TIMEOUT_SECONDS','180'))
    except (TypeError,ValueError):t=180.
    os.environ['EMBEDDING_BATCH_SIZE']=str(max(16,min(b,32)));os.environ['EMBEDDING_RETRIES']=str(max(1,min(r,3)));os.environ['EMBEDDING_TIMEOUT_SECONDS']=str(max(30.,min(t,300.)))

def _install_ui_guards():
    if getattr(bookrag_ui,'_bookrag_ui_guards_installed',False):return
    save=bookrag_ui.save_pdf; start=bookrag_ui.start_ingestion; health=bookrag_ui.ollama_health; ask=bookrag_ui.ask_page
    def secure_save(incoming:Any,name:str,content:bytes)->str:
        s=bookrag_ui.get_system(); safe=validate_storage_path(s.settings.project_root,incoming,'incoming folder');validate_pdf_payload(name,content);register_session_upload(len(content));return save(safe,name,content)
    def secure_start(system:Any,source_dir:str,*,trigger='manual')->str:
        safe=validate_storage_path(system.settings.project_root,source_dir,'incoming folder');return start(system,str(safe),trigger=trigger)
    def secure_health(url:str):return health(validate_ollama_url(url))
    def enhanced_ask(system:Any):
        ask(system); result=st.session_state.get('answer_result')
        if isinstance(result,dict):render_intelligence_panel(result)
    bookrag_ui.save_pdf=secure_save;bookrag_ui.start_ingestion=secure_start;bookrag_ui.ollama_health=secure_health;bookrag_ui.ask_page=enhanced_ask;bookrag_ui._bookrag_ui_guards_installed=True

def main():
    _load_local_env();_clamp_local_embedding_profile();_install_ui_guards();system=bookrag_ui.get_system();start_supervisor(system,interval_seconds=1.0);bookrag_ui.main()
if __name__=='__main__':main()
