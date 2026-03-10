@echo off
echo Starting CrisisMap v2.0 - Local Development Mode

REM Start backend in new window
start "CrisisMap Backend" cmd /k "call venv\Scripts\activate.bat && cd backend_v2 && python main.py"

REM Wait a moment for backend to start
timeout /t 5 /nobreak >nul

REM Start frontend in new window
start "CrisisMap Frontend" cmd /k "cd next-frontend && npm run dev"

echo.
echo CrisisMap v2.0 is starting...
echo.
echo Backend will be available at: http://localhost:8000
echo Frontend will be available at: http://localhost:3000
echo.
echo Press any key to close this window...
pause >nul