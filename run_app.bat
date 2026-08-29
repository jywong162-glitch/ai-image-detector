@echo off
REM ===================================================================
REM  Double-click this file to launch the AI-image detector app.
REM  No need to activate the venv or deal with "python not found".
REM
REM  Uses model_v2.pth. To use a different model, change the line below
REM  (or delete the SET line to use the default model.pth).
REM ===================================================================
cd /d "%~dp0"
set MODEL_PATH=model_v2.pth
".venv\Scripts\python.exe" app.py
pause
