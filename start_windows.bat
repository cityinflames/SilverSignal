@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PY=py -3"
) else (
    set "PY=python"
)
if not exist ".venv\Scripts\python.exe" (
    echo Setting up SilverSignal Python environment...
    %PY% -m venv .venv
    if errorlevel 1 goto :failed
)
call ".venv\Scripts\activate.bat"
python -m pip install -r requirements.txt
if errorlevel 1 goto :failed
echo Launching SilverSignal at http://localhost:8501
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
if errorlevel 1 goto :failed
exit /b 0
:failed
echo.
echo Could not start. Make sure Python 3.10+ is installed and the internet is available.
pause
exit /b 1
