@echo off
title RAGFlow Insurance Policy Assistant

echo.
echo ╔══════════════════════════════════════════════════════════════════╗
echo ║                  RAGFlow Insurance Policy Assistant             ║
echo ║                        Windows Launcher                         ║
echo ╚══════════════════════════════════════════════════════════════════╝
echo.
echo 🏥 Starting RAGFlow Web Interface...
echo 🌐 Powered by Streamlit and Local Llama 3
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python and try again.
    pause
    exit /b 1
)

REM Launch the application
echo ✅ Python found. Starting application...
echo.
python app_launcher.py

if errorlevel 1 (
    echo.
    echo ❌ Application failed to start.
    echo 💡 Try running: python setup_llama3.py
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ RAGFlow session completed.
pause 