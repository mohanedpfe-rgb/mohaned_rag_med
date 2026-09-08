@echo off
setlocal
cd /d "%~dp0"
rem Always run the files from this project folder, so code changes are used on every launch.
set "STREAMLIT_ARGS=run app.py --server.headless true --server.port 8501 --server.fileWatcherType auto --server.runOnSave true --browser.gatherUsageStats false"
if exist ".venv-1\Scripts\python.exe" (
    start "BookRAG Studio" /min cmd /c ""%CD%\.venv-1\Scripts\python.exe" -m streamlit %STREAMLIT_ARGS%"
) else if exist ".venv\Scripts\python.exe" (
    start "BookRAG Studio" /min cmd /c ""%CD%\.venv\Scripts\python.exe" -m streamlit %STREAMLIT_ARGS%"
) else (
    start "BookRAG Studio" /min cmd /c "py -3 -m streamlit %STREAMLIT_ARGS%"
)
timeout /t 3 /nobreak >nul
start "" "http://localhost:8501"
endlocal
