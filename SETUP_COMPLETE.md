# CrisisMap v2.0 - Setup Complete! 🎉

## What's Been Implemented

### ✅ Adaptive CSV Processing
- **Smart Column Detection**: Automatically detects and maps common column variations
- **Flexible Data Formats**: Handles different CSV structures and naming conventions
- **Manual Mapping Interface**: Web UI for custom column mapping when auto-detection fails
- **Data Validation**: Comprehensive validation and quality checks

### ✅ Real Map Integration
- **Leaflet Maps**: Real interactive maps instead of 3D mesh
- **Event Markers**: Color-coded markers based on fatality levels
- **Heatmap Mode**: Toggle between individual markers and heatmap view
- **Interactive Popups**: Detailed event information on click
- **Geographic Filtering**: Filter by country, date range, event type

### ✅ MongoDB Integration
- **Local MongoDB**: Uses `mongodb://localhost:27017/Crisis`
- **Flexible Schema**: Adapts to different data structures
- **Efficient Indexing**: Optimized for geographic and temporal queries
- **Real-time Updates**: WebSocket support for live data updates

### ✅ No Docker Required
- **Local Development**: Runs entirely on local machine
- **Simple Setup**: Just Python virtual environment and Node.js
- **Easy Startup**: Single batch file to start everything

## Your CSV Files Analysis

### 📊 Africa_aggregated_data_up_to-2026-02-28.xlsx
- **Status**: ✅ **Ready to use automatically**
- **Records**: 266,828 events
- **Columns**: Perfectly mapped to CrisisMap format
- **Contains**: Individual conflict events with coordinates, dates, fatalities

### 📊 Other Files
- **Status**: ⚠️ Need manual mapping (summary data, not individual events)
- **Use**: Good for aggregate statistics and trends

## How to Start

### 1. Quick Start
```bash
# Start everything at once
start_crisismap.bat
```

### 2. Manual Start
```bash
# Backend only
run_backend_local.bat

# Frontend only (in separate terminal)
run_frontend_local.bat
```

### 3. Verify Setup
```bash
# Check if everything is working
venv\Scripts\activate
python verify_setup.py
```

## Access Points

- **🌐 Main Application**: http://localhost:3000
- **🔧 Backend API**: http://localhost:8000
- **📚 API Documentation**: http://localhost:8000/docs
- **🗄️ MongoDB**: mongodb://localhost:27017/Crisis

## Using the System

### 1. Upload Data
1. Go to http://localhost:3000
2. Click "Upload Data" tab
3. Select your CSV/Excel file
4. System will analyze and suggest mappings
5. Adjust mappings if needed
6. Upload and process

### 2. View Results
1. Switch to "Dashboard" tab
2. See events on the real map
3. Filter by country, date, event type
4. Toggle heatmap mode
5. Click markers for details

### 3. Data Requirements

#### Required Columns (must be present or mapped):
- **event_date**: Date in YYYY-MM-DD format
- **latitude**: Decimal degrees (-90 to 90)
- **longitude**: Decimal degrees (-180 to 180)
- **event_type**: Type of conflict event
- **fatalities**: Number of casualties (numeric)

#### Optional Columns (enhance the data):
- **location**: Place name
- **country**: Country name
- **actor1/actor2**: Conflict parties
- **notes**: Additional details

## Features

### 🗺️ Interactive Map
- Real Leaflet map with OpenStreetMap tiles
- Color-coded markers by fatality level
- Heatmap visualization option
- Zoom and pan controls
- Event details on click

### 📊 Smart CSV Processing
- Automatic column detection
- Handles variations like:
  - `event_date`, `date`, `week`, `occurrence_date`
  - `latitude`, `lat`, `centroid_latitude`
  - `longitude`, `lng`, `centroid_longitude`
  - `fatalities`, `deaths`, `casualties`

### 🔍 Data Quality
- Coordinate validation
- Date format standardization
- Missing data handling
- Duplicate detection
- Quality reports

### 📈 Analytics Ready
- Temporal trend analysis
- Geographic hotspot detection
- Anomaly detection
- Risk assessment
- Predictive modeling foundation

## Next Steps

1. **Upload your main data file** (Africa_aggregated_data_up_to-2026-02-28.xlsx)
2. **Explore the interactive map** with your real data
3. **Test different filters** and visualizations
4. **Add more data sources** as needed
5. **Customize analytics** for your specific needs

## Troubleshooting

### Backend won't start:
- Check if MongoDB is running: `net start MongoDB`
- Verify Python packages: `pip install -r requirements_v2.txt`

### Frontend won't start:
- Install dependencies: `cd next-frontend && npm install`
- Check Node.js version: `node --version` (should be 18+)

### CSV upload fails:
- Check file format (CSV, XLSX, XLS only)
- Verify required columns are present
- Use manual mapping interface if auto-detection fails

## Success! 🚀

Your CrisisMap v2.0 is now ready with:
- ✅ Real interactive maps
- ✅ Adaptive CSV processing
- ✅ MongoDB integration
- ✅ No Docker dependency
- ✅ Your actual conflict data ready to load

Start with `start_crisismap.bat` and visit http://localhost:3000 to begin!