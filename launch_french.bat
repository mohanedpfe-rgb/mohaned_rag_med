@echo off
set LANGUAGE=fr
set LANG=fr_FR.UTF-8
echo Starting BookRAG Medical in French mode...
python -m streamlit run rag_project/app/bookrag_ui.py --server.port 8503
pause