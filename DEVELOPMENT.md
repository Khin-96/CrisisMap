# CrisisMap Development Guide

## Quick Start

1. **Setup Environment:**
   ```bash
   # Run setup script
   .\setup.bat  # Windows
   # or
   ./setup.sh   # Unix/Linux
   
   # Activate virtual environment
   .\venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate   # Unix/Linux
   ```

2. **Start Backend:**
   ```bash
   python backend/complete_main.py
   ```

3. **Start Frontend:**
   ```bash
   streamlit run frontend/modern_ui.py
   ```

## Architecture Overview

### Backend (FastAPI)
- **`complete_main.py`**: Main API server with all endpoints
- **`data_processing.py`**: Data ingestion and processing
- **`statistical_analysis.py`**: Advanced statistical analysis
- **`report_generator.py`**: Export and report generation
- **`hdx_ingestion.py`**: Humanitarian data from OCHA/IOM

### Frontend (Streamlit)
- **`modern_ui.py`**: Main modern dashboard
- **`app.py`**: Original dashboard scaffolding

## Key Features Implemented

### Data Processing
- ACLED data ingestion with live API support
- HDX integration for displacement data (OCHA/IOM)
- Local SQLite database persistence
- CSV import functionality for manual ingestion

### Statistical Analysis
- Temporal trend analysis with Mann-Kendall test
- Spatial hotspot detection using DBSCAN clustering
- Actor network analysis
- Violence pattern classification
- Early warning indicators with risk scoring

### Interactive Dashboard
- Overview with key metrics
- Trends visualization with Plotly
- Geographic hotspots with Folium maps
- Actor analysis charts
- Advanced statistical analysis pages
- Early warning system with risk assessment
- Humanitarian crisis monitoring (HDX)

### Export Functionality
- CSV export for raw data
- JSON export for reports
- Excel export with multiple sheets
- Policy brief generation

## API Endpoints

### Data Endpoints
- `GET /api/events` - Conflict events with filtering
- `GET /api/trends` - Trend analysis
- `GET /api/hotspots` - Geographic hotspots
- `POST /api/sync/regional` - Trigger HDX sync
- `GET /api/analysis/humanitarian` - Humanitarian data

### Analysis Endpoints
- `GET /api/statistical-analysis` - Advanced statistical analysis
- `GET /api/early-warning` - Early warning indicators
- `GET /api/predictions/fatalities` - ML fatality forecast
- `GET /api/predictions/hotspots` - ML hotspot predictions

### Report Endpoints
- `GET /api/reports/policy-brief` - Policy brief
- `GET /api/export/csv` - CSV export
- `POST /api/import/csv` - Manual CSV import

## Configuration

Update `.env` file with:
- `ACLED_API_KEY`: Your ACLED API key
- `ACLED_EMAIL`: Your ACLED account email
- `DATABASE_URL`: Database connection string
- `DEBUG`: Debug mode flag