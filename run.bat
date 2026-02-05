@echo off
echo Starting CrisisMap System...

REM Check if venv exists
if not exist venv (
    echo Virtual environment not found. Please run setup.bat first!
    pause
    exit /b
)

REM Activate venv globally for this script
call venv\Scripts\activate.bat

echo ---------------------------------------------------
echo 1. Launching Backend API (in new window)...
start "CrisisMap Backend" cmd /k "venv\Scripts\activate && python backend/complete_main.py"

echo Waiting 5 seconds for backend to initialize...
timeout /t 5 /nobreak >nul

echo ---------------------------------------------------
echo 2. Launching Frontend Dashboard (in new window)...
start "CrisisMap Frontend" cmd /k "venv\Scripts\activate && streamlit run frontend/modern_ui.py"

echo ---------------------------------------------------
echo Done! The app should open in your browser shortly.
echo If windows don't appear, check your taskbar.
