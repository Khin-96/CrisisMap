@echo off
echo Setting up CrisisMap v2.0 - Local Development Setup (No Docker)

REM Check if Python is installed
py --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Node.js is not installed. Please install Node.js 18+ first.
    pause
    exit /b 1
)

REM Create necessary directories
echo Creating project directories...
if not exist uploads mkdir uploads
if not exist models mkdir models
if not exist data\exports mkdir data\exports
if not exist data\processed mkdir data\processed
if not exist data\raw mkdir data\raw

REM Create Python virtual environment
echo Creating Python virtual environment...
if not exist venv (
    python -m venv venv
)

REM Activate virtual environment and install dependencies
echo Installing Python dependencies...
call venv\Scripts\activate.bat
pip install -r requirements_v2.txt

REM Install frontend dependencies
echo Installing frontend dependencies...
cd next-frontend
call npm install

REM Install additional UI dependencies
echo Installing UI components...
call npm install tailwindcss-animate @radix-ui/react-slot
cd ..

REM Create environment file for local development
echo Creating environment configuration...
(
echo # Local Development Configuration
echo MONGODB_URL=mongodb://localhost:27017/Crisis
echo.
echo # API Configuration
echo NEXT_PUBLIC_API_URL=http://localhost:8000
echo NEXT_PUBLIC_WS_URL=ws://localhost:8000
echo.
echo # Security
echo JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production
echo ENCRYPTION_KEY=your-32-character-encryption-key
echo.
echo # External APIs
echo ACLED_API_KEY=your-acled-api-key-here
echo HDX_API_KEY=your-hdx-api-key-here
echo.
echo # Environment
echo ENVIRONMENT=development
echo DEBUG=true
) > .env

echo.
echo CrisisMap v2.0 local setup completed successfully!
echo.
echo To start the application:
echo    1. Backend: run_backend_local.bat
echo    2. Frontend: run_frontend_local.bat
echo.
echo Or use the combined launcher: run_local.bat
echo.
pause