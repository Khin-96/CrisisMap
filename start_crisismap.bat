@echo off
echo Starting CrisisMap v2.0 - Complete System

echo.
echo Checking MongoDB...
REM Check if MongoDB is running
tasklist /FI "IMAGENAME eq mongod.exe" 2>NUL | find /I /N "mongod.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo ✅ MongoDB is already running
) else (
    echo ⚠️  MongoDB not detected. Please start MongoDB first.
    echo    You can start it with: net start MongoDB
    echo    Or run mongod.exe manually
    pause
    exit /b 1
)

echo.
echo Starting Backend Server...
start "CrisisMap Backend" cmd /k "call venv\Scripts\activate.bat && cd backend_v2 && python main.py"

echo.
echo Waiting for backend to start...
timeout /t 5 /nobreak >nul

echo.
echo Starting Frontend...
start "CrisisMap Frontend" cmd /k "cd next-frontend && npm run dev"

echo.
echo ✅ CrisisMap v2.0 is starting up!
echo.
echo 🌐 Frontend: http://localhost:3000
echo 🔧 Backend API: http://localhost:8000
echo 📚 API Docs: http://localhost:8000/docs
echo.
echo Press any key to close this window (services will continue running)...
pause >nul