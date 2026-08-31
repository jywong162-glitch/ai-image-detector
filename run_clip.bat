@echo off
REM Double-click to launch the app with the CLIP model (model_clip.pth).
REM (Shows the CLIP attention heatmap.)
cd /d "%~dp0"
set MODEL_ARCH=clip
set MODEL_PATH=model_clip.pth
".venv\Scripts\python.exe" app.py
pause
