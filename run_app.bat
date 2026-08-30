@echo off
REM ===================================================================
REM  Double-click this file to launch the AI-image detector app.
REM  No need to activate the venv or deal with "python not found".
REM
REM  Uses the default model (model_v3.pth). To use a different model,
REM  add:  set MODEL_PATH=some_model.pth  before the launch line.
REM ===================================================================
cd /d "%~dp0"
".venv\Scripts\python.exe" app.py
pause
