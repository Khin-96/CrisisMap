from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import uvicorn
import asyncio
from datetime import datetime, timedelta
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
import logging
from pathlib import Path
import json

# Enhanced imports
from enhanced_data_processing import EnhancedDataProcessor
from ml_model_manager import MLModelManager
from csv_processor import CSVProcessor
from websocket_manager import WebSocketManager

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CrisisMap Enhanced API",
    description="Advanced Conflict Early Warning & Predictive Analytics Platform",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "crisismap")

class DatabaseManager:
    def __init__(self):
        self.client = None
        self.db = None
    
    async def connect(self):
        self.client = AsyncIOMotorClient(MONGODB_URL)
        self.db = self.client[DATABASE_NAME]
        logger.info("Connected to MongoDB")
    
    async def disconnect(self):
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")

# Global instances
db_manager = DatabaseManager()
data_processor = EnhancedDataProcessor()
ml_manager = MLModelManager()
csv_processor = CSVProcessor()
ws_manager = WebSocketManager()

# Pydantic models
class ConflictEvent(BaseModel):
    event_id: str
    event_date: datetime
    location: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    event_type: str
    actor1: Optional[str] = None
    actor2: Optional[str] = None
    fatalities: int = Field(..., ge=0)
    country: str
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class TrendAnalysis(BaseModel):
    period: str
    total_events: int
    total_fatalities: int
    hotspot_locations: List[str]
    trend_direction: str
    temporal_data: Optional[List[Dict[str, Any]]] = None
    confidence_score: Optional[float] = None

class UploadResponse(BaseModel):
    success: bool
    message: str
    processed_rows: int
    validation_errors: Optional[List[str]] = None
    dataset_id: Optional[str] = None

class ModelTrainingRequest(BaseModel):
    dataset_id: str
    model_type: str = "random_forest"
    hyperparameters: Optional[Dict[str, Any]] = None

class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
    date_range: int = 30  # days ahead
    features: Optional[Dict[str, Any]] = None

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    await db_manager.connect()
    await data_processor.initialize(db_manager.db)
    await ml_manager.initialize(db_manager.db)
    logger.info("CrisisMap Enhanced API started")

@app.on_event("shutdown")
async def shutdown_event():
    await db_manager.disconnect()
    logger.info("CrisisMap Enhanced API shutdown")

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "2.0.0",
        "database": "connected" if db_manager.client else "disconnected"
    }

# Enhanced events endpoint
@app.get("/api/events", response_model=List[ConflictEvent])
async def get_events(
    country: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    event_type: Optional[str] = None,
    min_fatalities: Optional[int] = None,
    limit: int = 100,
    offset: int = 0
):
    """Get conflict events with advanced filtering"""
    try:
        # Build MongoDB query
        query = {}
        
        if country:
            query["country"] = {"$regex": country, "$options": "i"}
        
        if start_date or end_date:
            date_query = {}
            if start_date:
                date_query["$gte"] = datetime.fromisoformat(start_date)
            if end_date:
                date_query["$lte"] = datetime.fromisoformat(end_date)
            query["event_date"] = date_query
        
        if event_type:
            query["event_type"] = {"$regex": event_type, "$options": "i"}
        
        if min_fatalities is not None:
            query["fatalities"] = {"$gte": min_fatalities}
        
        # Execute query
        cursor = db_manager.db.events.find(query).skip(offset).limit(limit)
        events = await cursor.to_list(length=limit)
        
        # Convert to response format
        return [ConflictEvent(**event) for event in events]
        
    except Exception as e:
        logger.error(f"Error fetching events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced trends endpoint
@app.get("/api/trends", response_model=TrendAnalysis)
async def get_trends(
    country: Optional[str] = None,
    period: str = "monthly",
    lookback_months: int = 12
):
    """Get enhanced trend analysis with ML predictions"""
    try:
        trends_data = await data_processor.calculate_enhanced_trends(
            country=country,
            period=period,
            lookback_months=lookback_months
        )
        
        return TrendAnalysis(**trends_data)
        
    except Exception as e:
        logger.error(f"Error calculating trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# CSV Upload endpoint
@app.post("/api/upload/csv", response_model=UploadResponse)
async def upload_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process CSV/Excel files for model training"""
    try:
        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(
                status_code=400,
                detail="Only CSV and Excel files are supported"
            )
        
        # Read file content
        content = await file.read()
        
        # Process file
        result = await csv_processor.process_file(
            content, 
            file.filename,
            db_manager.db
        )
        
        if result["success"]:
            # Schedule background model retraining
            background_tasks.add_task(
                ml_manager.retrain_models,
                result["dataset_id"]
            )
            
            # Notify via WebSocket
            await ws_manager.broadcast({
                "type": "data_uploaded",
                "message": f"New dataset uploaded: {result['processed_rows']} rows",
                "dataset_id": result["dataset_id"]
            })
        
        return UploadResponse(**result)
        
    except Exception as e:
        logger.error(f"Error uploading CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model training endpoint
@app.post("/api/ml/train")
async def train_model(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks
):
    """Train ML models on uploaded data"""
    try:
        # Start training in background
        training_id = await ml_manager.start_training(
            dataset_id=request.dataset_id,
            model_type=request.model_type,
            hyperparameters=request.hyperparameters
        )
        
        background_tasks.add_task(
            ml_manager.execute_training,
            training_id
        )
        
        return {
            "success": True,
            "training_id": training_id,
            "message": "Model training started"
        }
        
    except Exception as e:
        logger.error(f"Error starting model training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Prediction endpoint
@app.post("/api/ml/predict")
async def predict_conflict_risk(request: PredictionRequest):
    """Generate conflict risk predictions for a location"""
    try:
        prediction = await ml_manager.predict_risk(
            latitude=request.latitude,
            longitude=request.longitude,
            date_range=request.date_range,
            features=request.features
        )
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model metrics endpoint
@app.get("/api/ml/models/{model_id}/metrics")
async def get_model_metrics(model_id: str):
    """Get performance metrics for a trained model"""
    try:
        metrics = await ml_manager.get_model_metrics(model_id)
        return metrics
        
    except Exception as e:
        logger.error(f"Error fetching model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Real-time hotspots with ML enhancement
@app.get("/api/hotspots/enhanced")
async def get_enhanced_hotspots(
    country: Optional[str] = None,
    prediction_days: int = 14
):
    """Get current hotspots with ML-based future risk predictions"""
    try:
        hotspots = await data_processor.calculate_enhanced_hotspots(
            country=country,
            prediction_days=prediction_days
        )
        
        return hotspots
        
    except Exception as e:
        logger.error(f"Error calculating enhanced hotspots: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Echo back for now, can be enhanced for specific commands
            await websocket.send_text(f"Echo: {data}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        ws_manager.disconnect(websocket)

# Data export endpoint
@app.get("/api/export/{format}")
async def export_data(
    format: str,
    country: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Export data in various formats (csv, json, geojson)"""
    try:
        if format not in ["csv", "json", "geojson"]:
            raise HTTPException(status_code=400, detail="Unsupported format")
        
        exported_data = await data_processor.export_data(
            format=format,
            country=country,
            start_date=start_date,
            end_date=end_date
        )
        
        return exported_data
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "enhanced_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )