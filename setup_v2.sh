#!/bin/bash

echo "Setting up CrisisMap v2.0 - Silicon Valley Grade Platform"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Create necessary directories
echo "Creating project directories..."
mkdir -p uploads models data/exports data/processed data/raw ssl

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd next-frontend
npm install

# Install additional UI dependencies
echo "Installing UI components..."
npm install tailwindcss-animate @radix-ui/react-slot
cd ..

# Create environment file
echo "Creating environment configuration..."
cat > .env << EOF
# Database Configuration
MONGODB_URL=mongodb://admin:crisismap2024@localhost:27017/crisismap?authSource=admin
MONGODB_DATABASE=crisismap

# Redis Configuration
REDIS_URL=redis://localhost:6379

# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Security
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production
ENCRYPTION_KEY=your-32-character-encryption-key

# External APIs
ACLED_API_KEY=your-acled-api-key-here
HDX_API_KEY=your-hdx-api-key-here

# Environment
ENVIRONMENT=development
DEBUG=true
EOF

# Create MongoDB initialization script
echo "Setting up MongoDB initialization..."
mkdir -p mongo-init
cat > mongo-init/init.js << EOF
db = db.getSiblingDB('crisismap');

// Create collections
db.createCollection('events');
db.createCollection('models');
db.createCollection('predictions');
db.createCollection('uploads');

// Create indexes
db.events.createIndex({ "event_date": -1 });
db.events.createIndex({ "country": 1 });
db.events.createIndex({ "location": "2d" });
db.events.createIndex({ "latitude": 1, "longitude": 1 });

db.models.createIndex({ "model_id": 1 }, { unique: true });
db.models.createIndex({ "created_at": -1 });

db.predictions.createIndex({ "model_id": 1 });
db.predictions.createIndex({ "prediction_date": -1 });

print("CrisisMap database initialized successfully");
EOF

# Create Nginx configuration
echo "Setting up Nginx configuration..."
cat > nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server backend:8000;
    }

    upstream frontend {
        server frontend:3000;
    }

    server {
        listen 80;
        server_name localhost;

        # Frontend routes
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }

        # API routes
        location /api/ {
            proxy_pass http://backend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }

        # WebSocket routes
        location /ws {
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
    }
}
EOF

# Build and start services
echo "Building and starting Docker services..."
docker-compose up -d --build

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# Check service health
echo "Checking service health..."
if curl -f http://localhost:8000/ > /dev/null 2>&1; then
    echo "Backend API is running"
else
    echo "Backend API is not responding"
fi

if curl -f http://localhost:3000/ > /dev/null 2>&1; then
    echo "Frontend is running"
else
    echo "Frontend is not responding"
fi

# Display success message
echo ""
echo "CrisisMap v2.0 setup completed successfully!"
echo ""
echo "Access the application:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   API Documentation: http://localhost:8000/docs"
echo ""
echo "Database access:"
echo "   MongoDB: mongodb://admin:crisismap2024@localhost:27017/crisismap"
echo "   Redis: redis://localhost:6379"
echo ""
echo "Next steps:"
echo "   1. Upload CSV data via the web interface"
echo "   2. Train ML models on your data"
echo "   3. Monitor real-time conflict events"
echo "   4. Explore interactive visualizations"
echo ""
echo "Development commands:"
echo "   - View logs: docker-compose logs -f"
echo "   - Stop services: docker-compose down"
echo "   - Restart services: docker-compose restart"
echo ""
EOF

# Make setup script executable
chmod +x crisismap/setup_v2.sh