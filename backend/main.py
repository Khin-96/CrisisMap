from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import uvicorn
from data_processing import DataIngestion, DataProcessor
from statistical_analysis import StatisticalAnalyzer

app = FastAPI(title="CrisisMap API", description="Conflict Early Warning & Trend Analysis")

# CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data processing components
data_ingestion = DataIngestion()
data_processor = DataProcessor()
statistical_analyzer = StatisticalAnalyzer()

class ConflictEvent(BaseModel):
    event_id: str
    event_date: str
    location: str
    latitude: float
    longitude: float
    event_type: str
    actor1: str
    fatalities: int
    country: str

class TrendAnalysis(BaseModel):
    period: str
    total_events: int
    total_fatalities: int
    hotspot_locations: List[str]
    trend_direction: str

@app.get("/")
async def root():
    return {"message": "CrisisMap API - Conflict Early Warning System"}

@app.get("/api/events", response_model=List[ConflictEvent])
async def get_events(
    country: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100
):
    """Get conflict events with optional filtering"""
    try:
        # Fetch data from ACLED
        df = data_ingestion.fetch_acled_data(
            country=country or "DR Congo",
            start_date=start_date or None,
            end_date=end_date or None
        )
        
        # Validate and process data
        df = data_ingestion.validate_data(df)
        data_processor.load_data(df)
        
        # Convert to response format
        events = []
        for _, row in df.head(limit).iterrows():
            events.append(ConflictEvent(
                event_id=row.get("event_id", ""),
                event_date=row["event_date"].strftime("%Y-%m-%d"),
                location=row["location"],
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                event_type=row["event_type"],
                actor1=row["actor1"],
                fatalities=int(row["fatalities"]),
                country=row["country"]
            ))
        
        return events
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trends", response_model=TrendAnalysis)
async def get_trends(
    country: Optional[str] = None,
    period: str = "monthly"
):
    """Get trend analysis for conflict events"""
    try:
        # Fetch and process data
        df = data_ingestion.fetch_acled_data(country=country or "DR Congo")
        df = data_ingestion.validate_data(df)
        data_processor.load_data(df)
        
        # Calculate trends
        trends_data = data_processor.calculate_trends(period)
        
        # Get hotspots
        hotspots = data_processor.identify_hotspots()
        hotspot_locations = [h["location"] for h in hotspots[:5]]
        
        # Calculate totals
        total_events = len(df)
        total_fatalities = int(df["fatalities"].sum())
        
        # Determine trend direction (simplified)
        if len(trends_data) >= 2:
            recent_events = trends_data[-1]["total_events"]
            previous_events = trends_data[-2]["total_events"]
            if recent_events > previous_events * 1.1:
                trend_direction = "increasing"
            elif recent_events < previous_events * 0.9:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "stable"
        
        return TrendAnalysis(
            period=period,
            total_events=total_events,
            total_fatalities=total_fatalities,
            hotspot_locations=hotspot_locations,
            trend_direction=trend_direction
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/hotspots")
async def get_hotspots(
    country: Optional[str] = None,
    threshold: int = 10
):
    """Get geographic hotspots of conflict"""
    try:
        # Fetch and process data
        df = data_ingestion.fetch_acled_data(country=country or "DR Congo")
        df = data_ingestion.validate_data(df)
        data_processor.load_data(df)
        statistical_analyzer.load_data(df)
        
        # Advanced hotspot analysis
        hotspot_analysis = statistical_analyzer.spatial_hotspot_analysis()
        
        return hotspot_analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)