@echo off
title Antigravity LM Studio - Desktop GUI
echo Starting Antigravity Language Model Desktop GUI...
cd /d "%~dp0"
.\.venv\Scripts\python scripts\gui.py
pause
