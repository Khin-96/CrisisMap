@echo off
REM CrisisMap Development Setup Script (Windows)

echo Setting up CrisisMap Development Environment...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python 3 is required but not installed.
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo Creating data directories...
mkdir data\raw 2>nul
mkdir data\processed 2>nul
mkdir data\exports 2>nul

REM Create environment file
echo Creating environment configuration...
if not exist .env (
    echo # CrisisMap Configuration > .env
    echo ACLED_API_KEY=your_acled_api_key_here >> .env
    echo ACLED_EMAIL=your_email_here >> .env
    echo DATABASE_URL=sqlite:///crisismap.db >> .env
    echo DEBUG=True >> .env
)

echo Setup complete!
echo.
echo To start development:
echo 1. Activate virtual environment: venv\Scripts\activate.bat
echo 2. Start backend server: python backend/complete_main.py
echo 3. Start frontend dashboard: streamlit run frontend/modern_ui.py
echo.
echo Note:
echo - Add your API keys to .env file
echo - Update the database configuration as needed