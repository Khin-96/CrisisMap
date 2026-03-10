@echo off
echo Setting up CrisisMap v2.0 - Silicon Valley Grade Platform

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker Compose is not installed. Please install Docker Desktop first.
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
if not exist ssl mkdir ssl

REM Install frontend dependencies
echo Installing frontend dependencies...
cd next-frontend
call npm install

REM Install additional UI dependencies
echo Installing UI components...
call npm install tailwindcss-animate @radix-ui/react-slot
cd ..

REM Create environment file
echo Creating environment configuration...
(
echo # Database Configuration
echo MONGODB_URL=mongodb://admin:crisismap2024@localhost:27017/crisismap?authSource=admin
echo MONGODB_DATABASE=crisismap
echo.
echo # Redis Configuration
echo REDIS_URL=redis://localhost:6379
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

REM Create MongoDB initialization script
echo Setting up MongoDB initialization...
if not exist mongo-init mkdir mongo-init
(
echo db = db.getSiblingDB('crisismap'^);
echo.
echo // Create collections
echo db.createCollection('events'^);
echo db.createCollection('models'^);
echo db.createCollection('predictions'^);
echo db.createCollection('uploads'^);
echo.
echo // Create indexes
echo db.events.createIndex({ "event_date": -1 }^);
echo db.events.createIndex({ "country": 1 }^);
echo db.events.createIndex({ "location": "2d" }^);
echo db.events.createIndex({ "latitude": 1, "longitude": 1 }^);
echo.
echo db.models.createIndex({ "model_id": 1 }, { unique: true }^);
echo db.models.createIndex({ "created_at": -1 }^);
echo.
echo db.predictions.createIndex({ "model_id": 1 }^);
echo db.predictions.createIndex({ "prediction_date": -1 }^);
echo.
echo print("CrisisMap database initialized successfully"^);
) > mongo-init\init.js

REM Build and start services
echo Building and starting Docker services...
docker-compose up -d --build

REM Wait for services to be ready
echo Waiting for services to start...
timeout /t 30 /nobreak >nul

REM Check service health
echo Checking service health...
curl -f http://localhost:8000/ >nul 2>&1
if %errorlevel% equ 0 (
    echo Backend API is running
) else (
    echo Backend API is not responding
)

curl -f http://localhost:3000/ >nul 2>&1
if %errorlevel% equ 0 (
    echo Frontend is running
) else (
    echo Frontend is not responding
)

REM Display success message
echo.
echo CrisisMap v2.0 setup completed successfully!
echo.
echo Access the application:
echo    Frontend: http://localhost:3000
echo    Backend API: http://localhost:8000
echo    API Documentation: http://localhost:8000/docs
echo.
echo Database access:
echo    MongoDB: mongodb://admin:crisismap2024@localhost:27017/crisismap
echo    Redis: redis://localhost:6379
echo.
echo Next steps:
echo    1. Upload CSV data via the web interface
echo    2. Train ML models on your data
echo    3. Monitor real-time conflict events
echo    4. Explore interactive visualizations
echo.
echo Development commands:
echo    - View logs: docker-compose logs -f
echo    - Stop services: docker-compose down
echo    - Restart services: docker-compose restart
echo.
pause