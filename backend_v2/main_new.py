import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import uvicorn
import asyncio
from datetime import datetime, timedelta
import uuid
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import pipeline components
from database_v2 import DatabaseManager
from ml_pipeline import MLPipeline
from data_processor_v2 import DataProcessor
from websocket_manager import WebSocketManager
from file_processor import FileProcessor
from analysis_engine import AnalysisEngine
from event_classifier import EventClassifier

# Data Models
class ConflictEvent(BaseModel):
    event_id: str
    event_date: str
    location: str
    latitude: float
    longitude: float
    event_type: str
    actor1: Optional[str] = None
    actor2: Optional[str] = None
    fatalities: int
    country: str
    confidence_score: Optional[float] = None

class TrendAnalysis(BaseModel):
    period: str
    total_events: int
    total_fatalities: int
    hotspot_locations: List[str]
    trend_direction: str
    temporal_data: Optional[List[Dict[str, Any]]] = None
    predictions: Optional[Dict[str, Any]] = None

class UploadResponse(BaseModel):
    upload_id: str
    status: str
    message: str
    records_processed: Optional[int] = None

class ModelTrainingRequest(BaseModel):
    dataset_id: str
    model_type: str = "random_forest"
    hyperparameters: Optional[Dict[str, Any]] = None

# Initialize components
db_manager = DatabaseManager()
ml_pipeline = MLPipeline()
data_processor = DataProcessor()
ws_manager = WebSocketManager()

# Global state
upload_status = {}

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup, cleanup on shutdown"""
    logger.info("Initializing CrisisMap API v2.0...")
    try:
        await db_manager.initialize()
        await ml_pipeline.load_models()
        logger.info("CrisisMap API v2.0 initialized successfully")
    except Exception as e:
        logger.error(f"Initialization error: {e}")
    
    yield
    
    logger.info("Shutting down CrisisMap API v2.0...")
    try:
        await db_manager.close()
    except:
        pass

# Create FastAPI app
app = FastAPI(
    title="CrisisMap API v2.0",
    description="Advanced Conflict Early Warning & Predictive Analytics Platform",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health & Status Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CrisisMap API v2.0 - Advanced Conflict Early Warning System",
        "version": "2.0.0",
        "status": "online",
        "features": [
            "MongoDB integration",
            "ML model training pipeline",
            "Real-time WebSocket updates",
            "CSV upload and processing",
            "Advanced analytics with full pipeline"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        await db_manager.client.admin.command('ping')
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Event Endpoints
@app.get("/api/events", response_model=List[ConflictEvent])
async def get_events(
    country: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100,
    event_type: Optional[str] = None,
    source: Optional[str] = None
):
    """Get conflict events with filtering (source: acled, cast, or both)"""
    try:
        filters = {}
        if country:
            filters["country"] = country
        if start_date:
            filters["event_date"] = {"$gte": start_date}
        if end_date:
            if "event_date" in filters:
                filters["event_date"]["$lte"] = end_date
            else:
                filters["event_date"] = {"$lte": end_date}
        if event_type:
            filters["event_type"] = event_type
        if source and source in ["acled", "cast"]:
            filters["data_source"] = source

        events = await db_manager.get_events(filters, limit)
        return events
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trends", response_model=TrendAnalysis)
async def get_trends(
    country: Optional[str] = None,
    period: str = "monthly",
    include_predictions: bool = True,
    source: Optional[str] = None
):
    """Get comprehensive trend analysis (source: acled, cast, or both)"""
    try:
        trends_data = await data_processor.calculate_trends(country=country, period=period, source=source)
        hotspots = await data_processor.identify_hotspots(country=country, source=source)
        
        predictions = None
        if include_predictions:
            predictions = await ml_pipeline.generate_predictions(country=country, horizon_days=14)
        
        return TrendAnalysis(
            period=period,
            total_events=trends_data["total_events"],
            total_fatalities=trends_data["total_fatalities"],
            hotspot_locations=hotspots[:10],
            trend_direction=trends_data["trend_direction"],
            temporal_data=trends_data.get("temporal_series"),
            predictions=predictions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/hotspots")
async def get_hotspots(country: Optional[str] = None, threshold: int = 5, source: Optional[str] = None):
    """Get geographic hotspots (source: acled, cast, or both)"""
    try:
        hotspots = await data_processor.identify_hotspots(country=country, threshold=threshold, source=source)
        return {"hotspots": hotspots}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/anomalies")
async def detect_anomalies(country: Optional[str] = None, days_back: int = 30, source: Optional[str] = None):
    """Detect anomalies in recent patterns (source: acled, cast, or both)"""
    try:
        anomalies = await data_processor.detect_anomalies(country=country, days_back=days_back, source=source)
        return {"anomalies": anomalies}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Upload Endpoints
@app.post("/api/upload/csv", response_model=UploadResponse)
async def upload_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: Optional[str] = None
):
    """Upload and process CSV files with full analysis pipeline"""
    try:
        upload_id = str(uuid.uuid4())
        
        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")
        
        # Save file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / f"{upload_id}_{file.filename}"
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Initialize status
        upload_status[upload_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Processing uploaded file...",
            "filename": file.filename,
            "created_at": datetime.utcnow()
        }
        
        # Process in background
        background_tasks.add_task(process_uploaded_file, upload_id, file_path, metadata)
        
        return UploadResponse(
            upload_id=upload_id,
            status="processing",
            message="File uploaded successfully. Processing in background."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload/acled", response_model=UploadResponse)
async def upload_acled(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: Optional[str] = None
):
    """Upload and process ACLED event data"""
    try:
        upload_id = str(uuid.uuid4())
        
        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")
        
        # Save file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / f"{upload_id}_{file.filename}"
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Initialize status
        upload_status[upload_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Processing ACLED data...",
            "filename": file.filename,
            "source": "acled",
            "created_at": datetime.utcnow()
        }
        
        # Process in background
        background_tasks.add_task(process_uploaded_file, upload_id, file_path, metadata, source="acled")
        
        return UploadResponse(
            upload_id=upload_id,
            status="processing",
            message="ACLED file uploaded successfully. Processing in background."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload/cast", response_model=UploadResponse)
async def upload_cast(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: Optional[str] = None
):
    """Upload and process CAST forecast data"""
    try:
        upload_id = str(uuid.uuid4())
        
        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")
        
        # Save file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / f"{upload_id}_{file.filename}"
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Initialize status
        upload_status[upload_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Processing CAST forecast data...",
            "filename": file.filename,
            "source": "cast",
            "created_at": datetime.utcnow()
        }
        
        # Process in background
        background_tasks.add_task(process_uploaded_file, upload_id, file_path, metadata, source="cast")
        
        return UploadResponse(
            upload_id=upload_id,
            status="processing",
            message="CAST file uploaded successfully. Processing in background."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/upload/status/{upload_id}")
async def get_upload_status(upload_id: str):
    """Get upload processing status"""
    if upload_id not in upload_status:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    status = upload_status[upload_id].copy()
    if isinstance(status.get('created_at'), datetime):
        status['created_at'] = status['created_at'].isoformat()
    return status

# ML Training Endpoints
@app.post("/api/ml/train")
async def train_model(
    background_tasks: BackgroundTasks,
    request: ModelTrainingRequest
):
    """Train ML model on dataset"""
    try:
        training_id = str(uuid.uuid4())
        background_tasks.add_task(
            ml_pipeline.train_model,
            training_id,
            request.dataset_id,
            request.model_type,
            request.hyperparameters or {}
        )
        
        return {
            "training_id": training_id,
            "status": "started",
            "message": "Model training started in background"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ml/models/{model_id}/metrics")
async def get_model_metrics(model_id: str):
    """Get model performance metrics"""
    try:
        metrics = await ml_pipeline.get_model_metrics(model_id)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await ws_manager.connect(websocket)

# Background Tasks
async def process_uploaded_file(upload_id: str, file_path: Path, metadata: Optional[str], source: Optional[str] = None):
    """Background task: Process uploaded CSV with full analysis pipeline"""
    try:
        # Step 1: Read file
        upload_status[upload_id]["progress"] = 15
        upload_status[upload_id]["message"] = "Reading file..."
        logger.info(f"Upload {upload_id}: Reading file...")
        
        df = FileProcessor.read_file(str(file_path))
        
        # Step 2: Classify events if source not specified
        upload_status[upload_id]["progress"] = 30
        if source is None:
            upload_status[upload_id]["message"] = "Classifying event type..."
            logger.info(f"Upload {upload_id}: Classifying events...")
            event_type, classification_info = EventClassifier.classify_dataframe(df)
            source = event_type if event_type in ["acled", "cast"] else None
            logger.info(f"Upload {upload_id}: Detected source: {source}")
        else:
            logger.info(f"Upload {upload_id}: Using specified source: {source}")
        
        # Step 3: Separate and normalize events
        upload_status[upload_id]["progress"] = 45
        upload_status[upload_id]["message"] = "Processing and normalizing data..."
        logger.info(f"Upload {upload_id}: Processing {len(df)} rows...")
        
        acled_df, cast_df = EventClassifier.separate_events(df)
        
        # Process ACLED events if present
        acled_count = 0
        if len(acled_df) > 0:
            upload_status[upload_id]["message"] = "Processing ACLED events..."
            logger.info(f"Upload {upload_id}: Processing {len(acled_df)} ACLED events...")
            acled_count = await data_processor.process_csv_data(acled_df, upload_id, source="acled")
        
        # Process CAST forecasts if present
        cast_count = 0
        if len(cast_df) > 0:
            upload_status[upload_id]["message"] = "Processing CAST forecasts..."
            logger.info(f"Upload {upload_id}: Processing {len(cast_df)} CAST forecasts...")
            cast_count = await data_processor.process_csv_data(cast_df, upload_id, source="cast")
        
        processed_records = acled_count + cast_count
        
        # Step 4: Generate analysis
        upload_status[upload_id]["progress"] = 75
        upload_status[upload_id]["message"] = "Generating analysis report..."
        logger.info(f"Upload {upload_id}: Generating analysis...")
        
        report = await data_processor.get_analysis_report(upload_id)
        
        # Step 5: Complete
        upload_status[upload_id]["status"] = "completed"
        upload_status[upload_id]["progress"] = 100
        upload_status[upload_id]["message"] = f"Successfully processed {processed_records} records"
        upload_status[upload_id]["records_processed"] = processed_records
        upload_status[upload_id]["acled_records"] = acled_count
        upload_status[upload_id]["cast_records"] = cast_count
        
        if report:
            upload_status[upload_id]["analysis"] = {
                "hotspots": len(report['report'].get('hotspots', [])),
                "anomalies": len(report['report'].get('anomalies', [])),
                "trends": len(report['report'].get('trends', [])),
                "actors": len(report['report'].get('actors', {}).get('top_actors', {}))
            }
        
        logger.info(f"Upload {upload_id}: Completed successfully (ACLED: {acled_count}, CAST: {cast_count})")
        
        # Notify clients
        await ws_manager.broadcast({
            "type": "upload_completed",
            "upload_id": upload_id,
            "records": processed_records,
            "acled_records": acled_count,
            "cast_records": cast_count,
            "analysis": upload_status[upload_id].get("analysis")
        })
        
    except Exception as e:
        upload_status[upload_id]["status"] = "error"
        upload_status[upload_id]["message"] = f"Processing failed: {str(e)}"
        logger.error(f"Upload {upload_id} failed: {e}")
        
        await ws_manager.broadcast({
            "type": "upload_failed",
            "upload_id": upload_id,
            "error": str(e)
        })
    
    finally:
        # Cleanup
        if file_path.exists():
            try:
                file_path.unlink()
            except:
                pass

# Run
if __name__ == "__main__":
    uvicorn.run(
        "main_new:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
