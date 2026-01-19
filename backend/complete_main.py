from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import uvicorn
from datetime import datetime, timedelta

from data_processing import DataIngestion, DataProcessor
from statistical_analysis import StatisticalAnalyzer
from report_generator import ReportGenerator
from ml_predictor import ConflictPredictor
from anomaly_detector import ConflictAnomalyDetector
from conflict_driver_analyzer import ConflictDriverAnalyzer

app = FastAPI(title="CrisisMap API", description="Conflict Early Warning & Trend Analysis")

# CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
data_ingestion = DataIngestion()
data_processor = DataProcessor()
statistical_analyzer = StatisticalAnalyzer()
report_generator = ReportGenerator()
ml_predictor = ConflictPredictor()
anomaly_detector = ConflictAnomalyDetector()
driver_analyzer = ConflictDriverAnalyzer()

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

@app.get("/")
async def root():
    return {"message": "CrisisMap API - Conflict Early Warning System Operational"}

@app.get("/api/events", response_model=List[ConflictEvent])
async def get_events(
    country: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100,
    force_refresh: bool = False
):
    try:
        df = data_ingestion.fetch_acled_data(
            country=country,
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh
        )
        
        events = []
        for _, row in df.head(limit).iterrows():
            events.append(ConflictEvent(
                event_id=str(row.get("event_id", "")),
                event_date=str(row["event_date"]),
                location=str(row["location"]),
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                event_type=str(row["event_type"]),
                actor1=str(row["actor1"]),
                fatalities=int(row["fatalities"]),
                country=str(row["country"])
            ))
        return events
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trends")
async def get_trends(country: Optional[str] = None, period: str = "monthly"):
    try:
        df = data_ingestion.fetch_acled_data(country=country)
        if df is None or df.empty:
             return {
                "period": period,
                "total_events": 0,
                "total_fatalities": 0,
                "temporal_data": [],
                "hotspot_locations": [],
                "trend_direction": "stable"
            }
            
        data_processor.load_data(df)
        trends_data = data_processor.calculate_trends(period)
        hotspots = data_processor.identify_hotspots()
        
        return {
            "period": period,
            "total_events": len(df),
            "total_fatalities": int(df.get("fatalities", pd.Series([0])).sum()),
            "temporal_data": trends_data,
            "hotspot_locations": hotspots,
            "trend_direction": "increasing" if len(trends_data) > 1 and trends_data[-1]["total_events"] > trends_data[-2]["total_events"] else "stable"
        }
    except Exception as e:
        print(f"ERROR in get_trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/hotspots")
async def get_hotspots(country: Optional[str] = None):
    try:
        df = data_ingestion.fetch_acled_data(country=country)
        if df is None or df.empty:
            return {"hotspots": []}
            
        data_processor.load_data(df)
        hotspots = data_processor.identify_hotspots()
        return {"hotspots": hotspots}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions/fatalities")
async def get_fatality_predictions(country: Optional[str] = None, days: int = 14):
    try:
        df = data_ingestion.fetch_acled_data(country=country)
        ml_predictor.train_fatalities_model(df)
        predictions = ml_predictor.predict_fatalities(df, days_ahead=days)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions/hotspots")
async def get_hotspot_predictions(country: Optional[str] = None):
    try:
        df = data_ingestion.fetch_acled_data(country=country)
        hotspots = ml_predictor.predict_hotspots(df)
        return hotspots
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/alerts/anomalies")
async def get_anomalies(country: Optional[str] = None):
    try:
        df = data_ingestion.fetch_acled_data(country=country)
        anomaly_detector.load_data(df)
        report = anomaly_detector.generate_anomaly_report()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/drivers")
async def get_drivers(country: Optional[str] = None):
    try:
        df = data_ingestion.fetch_acled_data(country=country)
        driver_analyzer.load_data(df)
        drivers = driver_analyzer.identify_conflict_drivers()
        return drivers
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/network")
async def get_actor_network(country: Optional[str] = None):
    try:
        df = data_ingestion.fetch_acled_data(country=country)
        statistical_analyzer.load_data(df)
        network = statistical_analyzer.actor_network_analysis()
        return network
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reports/policy-brief")
async def generate_brief(country: Optional[str] = None):
    try:
        df = data_ingestion.fetch_acled_data(country=country)
        report_generator.load_data(df)
        brief = report_generator.generate_policy_brief(country or "Region")
        return brief
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/import/csv")
async def import_csv(file_path: str):
    """Import an ACLED CSV file from the local filesystem"""
    try:
        success = data_ingestion.import_from_csv(file_path)
        if success:
            return {"status": "success", "message": f"Successfully imported data from {file_path}"}
        else:
            raise HTTPException(status_code=400, detail="Failed to import CSV. Check file path and format.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sync/regional")
async def sync_regional():
    """Trigger HDX synchronization for humanitarian indicators"""
    try:
        from hdx_ingestion import run_regional_sync
        run_regional_sync()
        return {"status": "success", "message": "Regional data synchronization completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/humanitarian")
async def get_humanitarian_data(indicator_type: str = "displacement"):
    """Fetch humanitarian indicators from DB"""
    try:
        from models import SessionLocal, HumanitarianIndicator
        session = SessionLocal()
        indicators = session.query(HumanitarianIndicator).filter(HumanitarianIndicator.indicator_type == indicator_type).all()
        result = []
        for ind in indicators:
            result.append({
                "region": ind.region,
                "admin1": ind.admin1,
                "admin2": ind.admin2,
                "value": ind.value,
                "unit": ind.unit,
                "source": ind.source,
                "date": str(ind.date)
            })
        session.close()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)