@echo off
echo Starting CrisisMap Backend (Local Development)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Start the backend server
cd backend_v2
python main.py