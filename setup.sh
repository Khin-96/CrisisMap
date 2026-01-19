#!/bin/bash

# CrisisMap Development Setup Script

echo "Setting up CrisisMap Development Environment..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating data directories..."
mkdir -p data/raw data/processed data/exports

# Create environment file
echo "Creating environment configuration..."
if [ ! -f .env ]; then
cat > .env << EOF
# CrisisMap Configuration
ACLED_API_KEY=your_acled_api_key_here
ACLED_EMAIL=your_email_here
DATABASE_URL=sqlite:///crisismap.db
DEBUG=True
EOF
fi

echo "Setup complete!"
echo ""
echo "To start development:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Start backend server: cd backend && python complete_main.py"
echo "3. Start frontend dashboard: cd frontend && streamlit run modern_ui.py"
echo ""
echo "Note:"
echo "- Add your API keys to .env file"
echo "- Update the database configuration as needed"