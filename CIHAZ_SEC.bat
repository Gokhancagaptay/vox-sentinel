@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo VoxSentinel - Mikrofon Test (float32)
echo.
python scripts\_mic_test_basit.py
echo.
pause
