import warnings
warnings.filterwarnings('ignore')
import warnings as warnings_lib
warnings_lib.simplefilter("ignore")

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
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

# Define base directory
BASE_DIR = Path(__file__).resolve().parent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import new modules
from database_sqlite import DatabaseManager
from ml_pipeline import MLPipeline
from data_processor import DataProcessor
from websocket_manager import WebSocketManager

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database, ML models, and external API tokens on startup"""
    await db_manager.initialize()
    await ml_pipeline.load_models()

    # Always get a fresh ACLED token on startup so it is valid for 24 h
    from acled_adapter import ACLEDAdapter
    _acled = ACLEDAdapter()
    auth_ok = _acled.ensure_authenticated()
    if auth_ok:
        print("ACLED OAuth: authenticated successfully")
    else:
        print("ACLED OAuth: authentication failed - check ACLED_EMAIL / ACLED_PASSWORD in .env")

    # Auto-sync: pull recent data from ACLED API if DB is sparse
    import asyncio
    asyncio.ensure_future(_startup_acled_sync())

    print("CrisisMap API v2.0 initialized successfully")
    yield
    # Cleanup code would go here if needed


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

@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {
        "message": "CrisisMap API v2.0 is running",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "events": "/api/events",
            "ai_insights": "/api/ai/insights",
            "ml_models": "/api/ml/models",
            "ml_train": "/api/ml/train-auto"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        db = DatabaseManager()
        await db.initialize()
        
        # Get a sample event to test DB
        events = await db.get_events({}, limit=1)
        db_status = "connected" if events else "no_data"
        
        await db.close()
        
        return {
            "status": "healthy",
            "database": db_status,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.0.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Initialize components
db_manager = DatabaseManager()
ml_pipeline = MLPipeline()
data_processor = DataProcessor()
ws_manager = WebSocketManager()

# Data models
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

class ModelTrainingRequest(BaseModel):
    dataset_id: str
    model_type: str = "random_forest"
    hyperparameters: Optional[Dict[str, Any]] = None

class UploadResponse(BaseModel):
    upload_id: str
    status: str
    message: str
    records_processed: Optional[int] = None

class ModelTrainingRequest(BaseModel):
    dataset_id: str
    model_type: str = "random_forest"
    hyperparameters: Optional[Dict[str, Any]] = None

# Global state for uploads
upload_status = {}

@app.get("/")
async def root():
    return {
        "message": "CrisisMap API v2.0 - Advanced Conflict Early Warning System",
        "version": "2.0.0",
        "features": [
            "MongoDB integration",
            "ML model training pipeline", 
            "Real-time WebSocket updates",
            "CSV upload and processing",
            "Advanced analytics"
        ]
    }

@app.get("/api/")
async def api_root():
    return {
        "message": "CrisisMap API v2.0 - Advanced Conflict Early Warning System",
        "version": "2.0.0",
        "status": "online",
        "endpoints": [
            "/api/events",
            "/api/trends", 
            "/api/hotspots",
            "/api/upload/csv",
            "/api/system/status"
        ]
    }

@app.get("/")
async def root():
    return {
        "message": "CrisisMap API v2.0 - Advanced Conflict Early Warning System",
        "version": "2.0.0",
        "features": [
            "MongoDB integration",
            "ML model training pipeline",
            "Real-time WebSocket updates",
            "CSV upload and processing",
            "Advanced analytics"
        ]
    }

@app.get("/api/events", response_model=List[ConflictEvent])
async def get_events(
    country: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100,
    event_type: Optional[str] = None
):
    """Get conflict events with advanced filtering"""
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

        events = await db_manager.get_events(filters, limit)
        return events
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trends", response_model=TrendAnalysis)
async def get_trends(
    country: Optional[str] = None,
    period: str = "monthly",
    include_predictions: bool = True
):
    """Get comprehensive trend analysis with ML predictions"""
    try:
        # Get historical trends
        trends_data = await data_processor.calculate_trends(
            country=country,
            period=period
        )
        
        # Get hotspots
        hotspots = await data_processor.identify_hotspots(country=country)
        
        # Generate predictions if requested
        predictions = None
        if include_predictions:
            predictions = await ml_pipeline.generate_predictions(
                country=country,
                horizon_days=14
            )
        
        return TrendAnalysis(
            period=period,
            total_events=trends_data["total_events"],
            total_fatalities=trends_data["total_fatalities"],
            hotspot_locations=hotspots[:10],
            trend_direction=trends_data["trend_direction"],
            temporal_data=trends_data["temporal_series"],
            predictions=predictions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload/csv", response_model=UploadResponse)
async def upload_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    custom_mappings: Optional[str] = None
):
    """Upload and process CSV files with adaptive column mapping"""
    try:
        upload_id = str(uuid.uuid4())
        
        # Save uploaded file
        upload_dir = BASE_DIR / "uploads"
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / f"{upload_id}_{file.filename}"
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Initialize upload status
        upload_status[upload_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Processing uploaded file...",
            "filename": file.filename,
            "created_at": datetime.utcnow()
        }
        
        # Parse custom mappings if provided
        mappings_dict = None
        if custom_mappings:
            try:
                import json
                mappings_dict = json.loads(custom_mappings)
            except:
                pass
        
        # Process file in background
        background_tasks.add_task(
            process_uploaded_file,
            upload_id,
            file_path,
            mappings_dict
        )
        
        return UploadResponse(
            upload_id=upload_id,
            status="processing",
            message="File uploaded successfully. Processing in background."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# In-memory status store for CAST CSV uploads (separate from ACLED event uploads)
cast_csv_upload_status: Dict[str, Any] = {}


@app.post("/api/upload/cast-csv")
async def upload_cast_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    custom_mappings: Optional[str] = None,
):
    """
    Upload a CAST-format CSV file.
    Accepted columns: id, level, country, admin1, outcome, period,
                      expected_forecast, low_forecast, high_forecast
    Records are stored in cast_predictions and used to boost ML hotspot scoring.
    """
    fetch_id = str(uuid.uuid4())

    upload_dir = BASE_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / f"{fetch_id}_{file.filename}"

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    mappings_dict = None
    if custom_mappings:
        try:
            import json
            mappings_dict = json.loads(custom_mappings)
        except Exception:
            pass

    cast_csv_upload_status[fetch_id] = {
        "fetch_id": fetch_id,
        "status": "queued",
        "progress": 0,
        "records_fetched": 0,
        "records_stored": 0,
        "filename": file.filename,
        "started_at": datetime.utcnow().isoformat(),
        "message": "Queued for processing",
    }

    background_tasks.add_task(
        _process_cast_csv_file,
        fetch_id,
        file_path,
        mappings_dict,
    )

    return {
        "fetch_id": fetch_id,
        "status": "queued",
        "message": "CAST CSV upload received. Processing in background.",
    }


async def _process_cast_csv_file(
    fetch_id: str,
    file_path: Path,
    custom_mappings: Optional[Dict[str, str]],
):
    """Background task: parse CAST CSV and store predictions."""
    try:
        import pandas as pd

        cast_csv_upload_status[fetch_id]["status"] = "processing"
        cast_csv_upload_status[fetch_id]["progress"] = 20
        cast_csv_upload_status[fetch_id]["message"] = "Reading CAST CSV file..."

        # Read file using robust adapter logic
        from csv_adapter import CSVAdapter
        adapter = CSVAdapter()
        df = adapter._read_file_robust(str(file_path))


        cast_csv_upload_status[fetch_id]["records_fetched"] = len(df)
        cast_csv_upload_status[fetch_id]["progress"] = 40
        cast_csv_upload_status[fetch_id]["message"] = f"Read {len(df)} rows. Normalising columns..."

        # Apply custom mappings if provided
        if custom_mappings:
            df = df.rename(columns={v: k for k, v in custom_mappings.items()})

        # Normalise expected CAST columns
        df.columns = [c.strip().lower() for c in df.columns]

        # Enforce types
        for col in ["expected_forecast", "low_forecast", "high_forecast"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        if "period" in df.columns:
            df["period"] = pd.to_datetime(df["period"], errors="coerce").dt.strftime("%Y-%m-%d")

        # Build records list
        cast_csv_upload_status[fetch_id]["progress"] = 60
        cast_csv_upload_status[fetch_id]["message"] = "Storing CAST predictions in database..."

        records = []
        for _, row in df.iterrows():
            # Handle NaN values for string columns
            country = str(row.get("country", "")) if pd.notna(row.get("country")) else "Global"
            admin1 = str(row.get("admin1", "")) if pd.notna(row.get("admin1")) else ""
            period_val = row.get("period")
            
            # Format period to string YYYY-MM-DD
            month_str = ""
            year_val = None
            if pd.notna(period_val):
                try:
                    dt = pd.to_datetime(period_val)
                    month_str = dt.strftime("%Y-%m-%d")
                    year_val = int(dt.year)
                except:
                    month_str = str(period_val)
            
            records.append({
                "country": country,
                "admin1": admin1,
                "month": month_str,
                "year": year_val,
                "total_forecast": int(row.get("expected_forecast", 0)) if pd.notna(row.get("expected_forecast")) else 0,
                "battles_forecast": 0,
                "erv_forecast": int(row.get("low_forecast", 0)) if pd.notna(row.get("low_forecast")) else 0,
                "vac_forecast": int(row.get("high_forecast", 0)) if pd.notna(row.get("high_forecast")) else 0,
                "total_observed": 0,
                "battles_observed": 0,
                "erv_observed": 0,
                "vac_observed": 0,
                "timestamp": None,
                "upload_id": fetch_id,
                "data_source": "cast_csv_upload",
                "processed_at": datetime.utcnow().isoformat(),
            })

        inserted = await db_manager.insert_cast_predictions(records)
        
        # MongoDB dual-write
        try:
            from database import DatabaseManager as MongoManager
            mongo_db = MongoManager()
            await mongo_db.initialize()
            await mongo_db.insert_cast_predictions(records)
            logger.info("Successfully replicated CAST predictions to MongoDB Atlas")
        except Exception as e:
            logger.error(f"Failed to dual-write CAST predictions to MongoDB Atlas: {e}")

        cast_csv_upload_status[fetch_id]["status"] = "completed"
        cast_csv_upload_status[fetch_id]["progress"] = 100
        cast_csv_upload_status[fetch_id]["records_stored"] = inserted
        cast_csv_upload_status[fetch_id]["completed_at"] = datetime.utcnow().isoformat()
        cast_csv_upload_status[fetch_id]["message"] = (
            f"Stored {inserted} CAST predictions. ML hotspot scoring will use this data."
        )

        await ws_manager.broadcast({
            "type": "cast_csv_upload_completed",
            "fetch_id": fetch_id,
            "records_stored": inserted,
        })

    except Exception as exc:
        cast_csv_upload_status[fetch_id]["status"] = "error"
        cast_csv_upload_status[fetch_id]["message"] = f"Processing failed: {str(exc)}"
        import traceback
        traceback.print_exc()
    finally:
        if file_path.exists():
            try:
                # Give the OS a tiny moment to release handles if needed
                import gc
                import os
                gc.collect()
                file_path.unlink()
            except Exception as e:
                logger.warning(f"Could not delete temp file {file_path}: {e}")



@app.get("/api/cast/fetch/{fetch_id}")
async def _cast_csv_status_compat(fetch_id: str):
    """
    Unified CAST fetch status — checks both API fetch jobs and CSV upload jobs.
    """
    if fetch_id in cast_fetch_status:
        return cast_fetch_status[fetch_id]
    if fetch_id in cast_csv_upload_status:
        return cast_csv_upload_status[fetch_id]
    raise HTTPException(status_code=404, detail="CAST job not found")


@app.post("/api/upload/analyze")
async def analyze_csv_structure(
    file: UploadFile = File(...),
    data_type: str = "acled_events"
):
    """Analyze CSV structure and suggest column mappings"""
    try:
        # Save temporary file
        temp_dir = BASE_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / f"temp_{file.filename}"
        
        content = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(content)
        
        # Analyze file structure
        analysis = await data_processor.analyze_csv_file(str(temp_file_path), data_type)
        
        # Clean up temp file
        try:
            import gc
            gc.collect()
            temp_file_path.unlink()
        except:
            pass

        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/upload/status/{upload_id}")
async def get_upload_status(upload_id: str):
    """Get status of file upload and processing"""
    if upload_id not in upload_status:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    return upload_status[upload_id]

@app.post("/api/ml/train-auto")
async def train_model_auto(background_tasks: BackgroundTasks):
    """Automatically train ML model on current database data"""
    try:
        training_id = str(uuid.uuid4())
        
        # Start training in background with current data
        background_tasks.add_task(
            train_model_on_current_data,
            training_id
        )
        
        return {
            "training_id": training_id,
            "status": "started",
            "message": "Model training started on current database data",
            "estimated_time": "2-5 minutes"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def train_model_on_current_data(training_id: str):
    """Background task to train model on current database data"""
    try:
        # Get all available events for training
        all_events = await db_manager.get_events({}, limit=50000)
        
        if len(all_events) < 100:
            print(f"Training {training_id}: Insufficient data ({len(all_events)} events)")
            return
        
        print(f"Training {training_id}: Starting with {len(all_events)} events")
        
        # Train multiple models for comparison
        models_to_train = [
            {
                "type": "random_forest",
                "params": {"n_estimators": 100, "max_depth": 15, "min_samples_split": 5}
            },
            {
                "type": "gradient_boosting", 
                "params": {"n_estimators": 100, "max_depth": 8, "learning_rate": 0.1}
            },
            {
                "type": "random_forest",
                "params": {"n_estimators": 200, "max_depth": 20, "min_samples_split": 3}
            }
        ]
        
        best_model_id = None
        best_r2_score = -1
        
        for i, model_config in enumerate(models_to_train):
            try:
                print(f"Training {training_id}: Starting model {i+1}/{len(models_to_train)} ({model_config['type']})")
                
                model_id = await ml_pipeline.train_model(
                    training_id=f"{training_id}_{i}",
                    dataset_id="current_database",
                    model_type=model_config["type"],
                    hyperparameters=model_config["params"]
                )
                
                # Check if this is the best model so far
                if model_id in ml_pipeline.model_metadata:
                    r2_score = ml_pipeline.model_metadata[model_id].get("metrics", {}).get("r2", -1)
                    if r2_score > best_r2_score:
                        best_r2_score = r2_score
                        best_model_id = model_id
                
                print(f"Training {training_id}: Completed model {i+1}/{len(models_to_train)} - R² = {r2_score:.3f}")
                
            except Exception as e:
                print(f"Training {training_id}: Failed model {i+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Training {training_id}: Completed. Best model: {best_model_id} (R² = {best_r2_score:.3f})")
        
        # Generate initial predictions with best model
        if best_model_id:
            try:
                predictions = await ml_pipeline.generate_predictions(
                    model_id=best_model_id,
                    horizon_days=14
                )
                print(f"Training {training_id}: Generated {len(predictions.get('predictions', []))} predictions")
            except Exception as e:
                print(f"Training {training_id}: Failed to generate predictions: {str(e)}")
        
    except Exception as e:
        print(f"Training {training_id}: Failed - {str(e)}")
        import traceback
        traceback.print_exc()

@app.get("/api/ml/train-status/{training_id}")
async def get_training_status(training_id: str):
    """Get status of model training"""
    try:
        # Check if any models with this training_id exist
        matching_models = [
            model_id for model_id in ml_pipeline.model_metadata.keys()
            if training_id in model_id
        ]
        
        if not matching_models:
            return {
                "training_id": training_id,
                "status": "in_progress",
                "message": "Training in progress..."
            }
        
        # Get best model from this training session
        best_model = None
        best_r2 = -1
        
        for model_id in matching_models:
            metadata = ml_pipeline.model_metadata[model_id]
            r2_score = metadata.get("metrics", {}).get("r2", -1)
            if r2_score > best_r2:
                best_r2 = r2_score
                best_model = metadata
        
        return {
            "training_id": training_id,
            "status": "completed",
            "models_trained": len(matching_models),
            "best_model": {
                "model_id": best_model.get("model_id"),
                "r2_score": best_model.get("metrics", {}).get("r2"),
                "rmse": best_model.get("metrics", {}).get("rmse"),
                "training_samples": best_model.get("training_samples")
            } if best_model else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ml/models/{model_id}/metrics")
async def get_model_metrics(model_id: str):
    """Get performance metrics for a trained model"""
    try:
        metrics = await ml_pipeline.get_model_metrics(model_id)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/anomalies")
async def detect_anomalies(
    country: Optional[str] = None,
    days_back: int = 30
):
    """Detect anomalies in recent conflict patterns"""
    try:
        anomalies = await data_processor.detect_anomalies(
            country=country,
            days_back=days_back
        )
        return anomalies
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/hotspots")
async def get_hotspots(
    country: Optional[str] = None,
    threshold: int = 5
):
    """Get geographic hotspots"""
    try:
        hotspots = await data_processor.identify_hotspots(
            country=country,
            threshold=threshold
        )
        return {"hotspots": hotspots}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/status")
async def get_system_status():
    """Get system health status"""
    try:
        db = await data_processor._get_db_manager()
        
        # Test database connection
        try:
            await db.client.admin.command('ping')
            db_status = "connected"
        except:
            db_status = "disconnected"
        
        # Get basic stats
        total_events = len(await db.get_events({}, limit=1000000))
        
        return {
            "backend": "online",
            "database": db_status,
            "lastUpdate": datetime.utcnow().isoformat(),
            "dataStats": {
                "totalEvents": total_events,
                "totalUploads": len(upload_status),
                "lastUpload": max([status.get("created_at") for status in upload_status.values()], default=None),
                "dataQuality": "good" if total_events > 0 else "warning"
            }
        }
    except Exception as e:
        return {
            "backend": "error",
            "database": "error", 
            "lastUpdate": datetime.utcnow().isoformat(),
            "dataStats": {
                "totalEvents": 0,
                "totalUploads": 0,
                "lastUpload": None,
                "dataQuality": "error"
            },
            "error": str(e)
        }

@app.websocket("/ws")
async def websocket_endpoint(websocket):
    """WebSocket endpoint for real-time updates"""
    await ws_manager.connect(websocket)

async def process_uploaded_file(upload_id: str, file_path: Path, custom_mappings: Optional[Dict[str, str]]):
    """Background task to process uploaded CSV files with adaptive mapping"""
    try:
        # Update status
        upload_status[upload_id]["progress"] = 25
        upload_status[upload_id]["message"] = "Analyzing file structure..."
        
        # Analyze file first
        analysis = await data_processor.analyze_csv_file(str(file_path))
        
        upload_status[upload_id]["progress"] = 50
        upload_status[upload_id]["message"] = "Processing data with adaptive mapping..."
        
        # Check if required columns can be mapped
        missing_required = analysis.get('missing_required', [])
        if missing_required and not custom_mappings:
            upload_status[upload_id]["status"] = "needs_mapping"
            upload_status[upload_id]["message"] = f"Manual column mapping required. Missing: {missing_required}"
            upload_status[upload_id]["analysis"] = analysis
            return
        
        upload_status[upload_id]["progress"] = 75
        upload_status[upload_id]["message"] = "Storing in database..."
        
        # Process and store data
        processed_records = await data_processor.process_csv_data(
            str(file_path), 
            custom_mappings, 
            upload_id
        )
        
        upload_status[upload_id]["status"] = "completed"
        upload_status[upload_id]["progress"] = 100
        upload_status[upload_id]["message"] = f"Successfully processed {processed_records} records"
        upload_status[upload_id]["records_processed"] = processed_records
        
        # Notify connected clients via WebSocket
        await ws_manager.broadcast({
            "type": "upload_completed",
            "upload_id": upload_id,
            "records": processed_records
        })
        
    except Exception as e:
        upload_status[upload_id]["status"] = "error"
        upload_status[upload_id]["message"] = f"Processing failed: {str(e)}"
    
    finally:
        # Clean up uploaded file
        if file_path.exists():
            file_path.unlink()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# ML Model Endpoints
@app.get("/api/ml/predictions")
async def get_predictions(
    country: Optional[str] = None,
    horizon_days: int = 14,
    model_id: Optional[str] = None
):
    """Generate ML predictions for future conflict events"""
    try:
        predictions = await ml_pipeline.generate_predictions(
            country=country,
            horizon_days=horizon_days,
            model_id=model_id
        )
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ml/models")
async def list_models():
    """List all trained ML models"""
    try:
        models = []
        for model_id, metadata in ml_pipeline.model_metadata.items():
            models.append({
                "model_id": model_id,
                "model_type": metadata.get("model_type"),
                "status": metadata.get("status"),
                "metrics": metadata.get("metrics", {}),
                "training_samples": metadata.get("training_samples"),
                "created_at": metadata.get("created_at")
            })
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ml/drift-detection/{model_id}")
async def detect_model_drift(model_id: str):
    """Detect model drift for a specific model"""
    try:
        # Get recent data for drift detection
        recent_events = await db_manager.get_events(
            filters={
                "event_date": {
                    "$gte": (datetime.utcnow() - timedelta(days=30)).isoformat()
                }
            },
            limit=1000
        )
        
        if not recent_events:
            raise HTTPException(status_code=400, detail="No recent data available for drift detection")
        
        recent_df = pd.DataFrame(recent_events)
        drift_analysis = await ml_pipeline.detect_model_drift(model_id, recent_df)
        
        return drift_analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Groq AI Integration with Database Access
@app.post("/api/ai/insights")
async def get_ai_insights(request: Dict[str, Any]):
    """Get AI-powered insights and guidance using Groq with database context"""
    try:
        from groq import Groq
        
        # Initialize Groq client with API key from environment
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(status_code=500, detail="Groq API key not configured")
        
        client = Groq(api_key=groq_api_key)
        
        # Get query and context
        query = request.get("query", "Provide insights on the current conflict situation")
        provided_context = request.get("context", {})
        
        # Get comprehensive database context
        db_context = await get_comprehensive_context()
        
        # Merge provided context with database context
        full_context = {**db_context, **provided_context}
        
        # Prepare enhanced context for AI
        system_prompt = """You are Dr. Sarah Chen, a senior conflict analyst with 15+ years of experience in early warning systems, 
        peacekeeping operations, and humanitarian response. You specialize in:
        - Conflict pattern analysis and trend identification
        - Risk assessment and early warning indicators
        - Strategic intervention recommendations
        - Humanitarian impact evaluation
        - Peacekeeping and conflict resolution strategies
        
        FORMATTING INSTRUCTIONS:
        - Use **bold text** for section headers and important terms
        - Use bullet points with • instead of asterisks
        - Structure responses with clear sections
        - Avoid using * for emphasis, use **bold** instead
        - Use proper paragraph breaks for readability
        
        Provide expert analysis that is:
        1. Data-driven and evidence-based
        2. Actionable with specific recommendations
        3. Risk-focused with confidence levels
        4. Strategic with short and long-term perspectives
        5. Humanitarian-centered considering civilian protection
        
        Always structure responses with:
        **Executive Summary** (2-3 sentences)
        
        **Key Findings**
        • Point 1
        • Point 2
        • Point 3
        
        **Risk Assessment**
        Level: High/Medium/Low (confidence percentage)
        
        **Immediate Actions**
        • Action 1
        • Action 2
        • Action 3
        
        **Strategic Actions**
        • Strategy 1
        • Strategy 2
        • Strategy 3
        
        **Monitoring Indicators**
        • Indicator 1
        • Indicator 2
        • Indicator 3"""
        
        user_prompt = f"""
        CONFLICT DATA ANALYSIS REQUEST:
        Query: {query}
        
        CURRENT SITUATION OVERVIEW:
        - Total Events: {full_context.get('total_events', 0):,}
        - Total Fatalities: {full_context.get('total_fatalities', 0):,}
        - Active Countries: {full_context.get('active_countries', 0)}
        - Date Range: {full_context.get('date_range', 'Unknown')}
        - Trend Direction: {full_context.get('trend_direction', 'Unknown')}
        
        TOP AFFECTED COUNTRIES:
        {chr(10).join([f"- {country}: {stats['events']} events, {stats['fatalities']} fatalities" 
                      for country, stats in full_context.get('top_countries', [])[:5]])}
        
        DOMINANT EVENT TYPES:
        {chr(10).join([f"- {event_type}: {stats['events']} events" 
                      for event_type, stats in full_context.get('top_event_types', [])[:5]])}
        
        RECENT HIGH-IMPACT EVENTS:
        {chr(10).join([f"- {event.get('location', 'Unknown')}, {event.get('country', 'Unknown')}: {event.get('fatalities', 0)} fatalities on {event.get('event_date', 'Unknown')[:10]}" 
                      for event in full_context.get('recent_high_impact', [])[:3]])}
        
        GEOGRAPHIC HOTSPOTS:
        {chr(10).join([f"- {location}: {stats['events']} events, {stats['fatalities']} fatalities" 
                      for location, stats in full_context.get('hotspots', [])[:5]])}
        
        ML PREDICTIONS (if available):
        {full_context.get('ml_predictions_summary', 'No ML predictions available')}
        
        Please provide your expert analysis and recommendations based on this comprehensive conflict data.
        """
        
        # Get AI response
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=1500
        )
        
        ai_response = chat_completion.choices[0].message.content
        
        # Store AI interaction in database for learning
        interaction_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "response": ai_response,
            "context_summary": {
                "total_events": full_context.get('total_events', 0),
                "total_fatalities": full_context.get('total_fatalities', 0),
                "countries_analyzed": len(full_context.get('top_countries', [])),
                "data_quality": "high" if full_context.get('total_events', 0) > 1000 else "medium"
            },
            "model_used": "llama-3.1-8b-instant",
            "response_length": len(ai_response)
        }
        
        # Store in database (optional - for analytics)
        try:
            await db_manager.store_ai_interaction(interaction_record)
        except:
            pass  # Don't fail if storage fails
        
        return {
            "query": query,
            "response": ai_response,
            "model": "llama-3.1-8b-instant",
            "timestamp": datetime.utcnow().isoformat(),
            "context_provided": True,
            "data_quality": interaction_record["context_summary"]["data_quality"],
            "analysis_scope": {
                "events_analyzed": full_context.get('total_events', 0),
                "countries_covered": len(full_context.get('top_countries', [])),
                "time_period": full_context.get('date_range', 'Unknown')
            }
        }
        
    except Exception as e:
        print(f"AI service error: {str(e)}")  # Log for debugging
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

async def get_comprehensive_context():
    """Get comprehensive context from database for AI analysis using optimized SQL queries"""
    try:
        # Get temporal trends (Last 6 months by default for AI context)
        trends = await db_manager.get_temporal_trends(period="monthly")
        
        # Get hotspots
        hotspots_data = await db_manager.get_hotspots(threshold=5)
        
        # Get latest predictions
        ml_predictions_summary = "No ML predictions available"
        try:
            if ml_pipeline.models:
                predictions = await ml_pipeline.generate_predictions(horizon_days=7)
                if predictions and 'predictions' in predictions:
                    high_risk_count = len([p for p in predictions['predictions'] 
                                         if p.get('risk_level') == 'high'])
                    ml_predictions_summary = f"ML Model predicts {high_risk_count} high-risk areas in next 7 days"
        except:
            pass

        # Get CAST predictions summary
        cast_summary = "No CAST prediction data available"
        try:
            cast_preds = await db_manager.get_cast_predictions(limit=5)
            if cast_preds:
                cast_summary = "Latest ACLED CAST Forecasts:\n"
                for cp in cast_preds:
                    cast_summary += f"- {cp.get('country')}, {cp.get('admin1') or ''}: {cp.get('total_forecast', 0)} expected events in {cp.get('month', 'future')}\n"
        except:
            pass

        # Get comprehensive stats
        stats = await db_manager.get_comprehensive_stats()
        
        # Basic Stats
        total_events = sum(t['total_events'] for t in trends) if trends else 0
        total_fatalities = sum(t['total_fatalities'] for t in trends) if trends else 0
        
        # Trend Direction
        trend_direction = "stable"
        if len(trends) >= 2:
            if trends[-1]['total_events'] > trends[-2]['total_events'] * 1.15:
                trend_direction = "increasing"
            elif trends[-1]['total_events'] < trends[-2]['total_events'] * 0.85:
                trend_direction = "decreasing"

        # Recent high-impact events (already in records format)
        recent_high_impact_raw = await db_manager.get_events(
            filters={"fatalities": {"$gte": 10}},
            limit=5,
            sort_by="event_date",
            sort_order=-1
        )

        return {
            "total_events": total_events,
            "total_fatalities": int(total_fatalities),
            "active_countries": stats['active_countries_count'],
            "date_range": stats['date_range'],
            "trend_direction": trend_direction,
            "top_countries": stats['top_countries'],
            "top_event_types": stats['top_event_types'],
            "hotspots": [(h['location'], {"events": h['event_count'], "fatalities": h['total_fatalities']}) for h in hotspots_data[:10]],
            "recent_high_impact": recent_high_impact_raw,
            "ml_predictions_summary": ml_predictions_summary,
            "cast_predictions_summary": cast_summary,
            "data_freshness": "Last 6 months of historical data + latest forecasts",
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Context generation error: {str(e)}")
        return {"error": f"Failed to generate context: {str(e)}"}

@app.post("/api/ai/risk-assessment")
async def get_risk_assessment(request: Dict[str, Any]):
    """Get AI-powered risk assessment for specific regions or scenarios"""
    try:
        from groq import Groq
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(status_code=500, detail="Groq API key not configured")
        
        client = Groq(api_key=groq_api_key)
        
        location = request.get("location", "")
        timeframe = request.get("timeframe", "next 30 days")
        
        # Get location-specific data from database
        location_context = await get_location_context(location)
        
        system_prompt = """You are a senior conflict risk assessment specialist with expertise in:
        - Quantitative risk modeling and probability assessment
        - Early warning indicator identification
        - Scenario planning and contingency analysis
        - Humanitarian impact forecasting
        
        FORMATTING INSTRUCTIONS:
        - Use **bold text** for section headers and important terms
        - Use bullet points with • instead of asterisks
        - Structure responses clearly with proper sections
        - Avoid using * for emphasis, use **bold** instead
        - Use proper paragraph breaks for readability
        
        Provide structured risk assessments with:
        
        **Risk Level:** Critical/High/Medium/Low (**confidence percentage**)
        
        **Key Risk Factors** (ranked by impact and likelihood)
        • Factor 1
        • Factor 2
        • Factor 3
        
        **Trigger Events to Monitor**
        • Event 1
        • Event 2
        • Event 3
        
        **Scenario Analysis**
        • **Best Case:** Description
        • **Most Likely:** Description  
        • **Worst Case:** Description
        
        **Mitigation Strategies**
        • **Immediate:** Action 1
        • **Short-term:** Action 2
        • **Long-term:** Action 3"""
        
        user_prompt = f"""
        RISK ASSESSMENT REQUEST:
        Location: {location}
        Timeframe: {timeframe}
        
        LOCATION-SPECIFIC DATA:
        {location_context}
        
        Provide a comprehensive risk assessment with specific probability estimates and actionable recommendations.
        """
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=1200
        )
        
        ai_response = chat_completion.choices[0].message.content
        
        return {
            "location": location,
            "timeframe": timeframe,
            "assessment": ai_response,
            "model": "llama-3.1-8b-instant",
            "timestamp": datetime.utcnow().isoformat(),
            "data_sources": location_context.get("data_sources", []),
            "confidence": "high" if location_context.get("data_points", 0) > 50 else "medium"
        }
        
    except Exception as e:
        print(f"Risk assessment error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Risk assessment error: {str(e)}")

async def get_location_context(location: str):
    """Get location-specific context for risk assessment"""
    try:
        if not location:
            return {"error": "No location specified"}
        
        # Search for events in the specified location
        location_events = await db_manager.get_events(
            filters={
                "$or": [
                    {"location": {"$regex": location, "$options": "i"}},
                    {"country": {"$regex": location, "$options": "i"}}
                ],
                "event_date": {
                    "$gte": (datetime.utcnow() - timedelta(days=365)).isoformat()
                }
            },
            limit=1000
        )
        
        if not location_events:
            return {
                "message": f"No recent conflict data found for {location}",
                "data_points": 0,
                "recommendation": "Limited data available for assessment"
            }
        
        df = pd.DataFrame(location_events)
        
        # Calculate location statistics
        total_events = len(df)
        total_fatalities = df['fatalities'].sum()
        avg_fatalities = df['fatalities'].mean()
        
        # Recent trend (last 3 months vs previous 3 months)
        recent_events = df[df['event_date'] >= (datetime.utcnow() - timedelta(days=90)).isoformat()]
        previous_events = df[
            (df['event_date'] >= (datetime.utcnow() - timedelta(days=180)).isoformat()) &
            (df['event_date'] < (datetime.utcnow() - timedelta(days=90)).isoformat())
        ]
        
        trend = "stable"
        if len(recent_events) > len(previous_events) * 1.2:
            trend = "escalating"
        elif len(recent_events) < len(previous_events) * 0.8:
            trend = "de-escalating"
        
        # Event types in location
        event_types = df['event_type'].value_counts().head(5).to_dict()
        
        # Recent high-impact events
        high_impact = df[df['fatalities'] > 5].sort_values('event_date', ascending=False).head(3)
        
        context_summary = f"""
        LOCATION: {location}
        
        HISTORICAL DATA (Last 12 months):
        - Total Events: {total_events}
        - Total Fatalities: {int(total_fatalities)}
        - Average Fatalities per Event: {avg_fatalities:.1f}
        - Current Trend: {trend}
        
        DOMINANT EVENT TYPES:
        {chr(10).join([f"- {event_type}: {count} events" for event_type, count in event_types.items()])}
        
        RECENT HIGH-IMPACT INCIDENTS:
        {chr(10).join([f"- {row['event_date'][:10]}: {row['event_type']} - {row['fatalities']} fatalities" 
                      for _, row in high_impact.iterrows()])}
        
        TEMPORAL PATTERN:
        - Recent 3 months: {len(recent_events)} events
        - Previous 3 months: {len(previous_events)} events
        - Trend Direction: {trend}
        """
        
        return {
            "context": context_summary,
            "data_points": total_events,
            "trend": trend,
            "risk_indicators": {
                "event_frequency": len(recent_events),
                "fatality_rate": avg_fatalities,
                "escalation_factor": len(recent_events) / max(len(previous_events), 1)
            },
            "data_sources": ["conflict_events_database"],
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {"error": f"Failed to get location context: {str(e)}"}


# ---------------------------------------------------------------------------
# Flutter Mobile App Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/ml/predictions/map")
async def get_map_predictions(
    risk_level: Optional[str] = None,
    country: Optional[str] = None,
    limit: int = 200
):
    """
    Return geo-tagged ML predictions for Flutter map display.
    Each result contains latitude, longitude, risk_level, predicted_fatalities.
    """
    try:
        db = DatabaseManager()
        await db.initialize()
        results = await db.get_map_predictions(
            risk_level=risk_level,
            country=country,
            limit=limit
        )
        await db.close()
        return {
            "predictions": results,
            "total": len(results),
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/notifications/alerts")
async def get_notification_alerts(
    unread_only: bool = False,
    limit: int = 50
):
    """Return alerts for Flutter push notification feed."""
    try:
        db = DatabaseManager()
        await db.initialize()
        alerts = await db.get_alerts(unread_only=unread_only, limit=limit)
        await db.close()
        return {
            "alerts": alerts,
            "total": len(alerts),
            "unread_count": len([a for a in alerts if not a.get("is_read")]),
            "fetched_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/notifications/mark-read/{alert_id}")
async def mark_alert_read(alert_id: str):
    """Mark a notification alert as read."""
    try:
        db = DatabaseManager()
        await db.initialize()
        success = await db.mark_alert_read(alert_id)
        await db.close()
        if success:
            return {"alert_id": alert_id, "status": "marked_read"}
        raise HTTPException(status_code=404, detail="Alert not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/incident-response")
async def get_incident_response(request: Dict[str, Any]):
    """
    Generate an AI-powered response suggestion for a specific predicted incident.
    Used by Flutter app when a notification is tapped to show suggested actions.
    """
    try:
        from groq import Groq

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(status_code=500, detail="Groq API key not configured")

        client = Groq(api_key=groq_api_key)

        location = request.get("location", "Unknown location")
        country = request.get("country", "")
        risk_level = request.get("risk_level", "medium")
        predicted_fatalities = request.get("predicted_fatalities", 0)
        event_type = request.get("event_type", "conflict event")
        prediction_date = request.get("prediction_date", "")
        actor1 = request.get("actor1", "")
        actor2 = request.get("actor2", "")

        system_prompt = (
            "You are a senior humanitarian response coordinator specializing in conflict "
            "zone operations. Your role is to provide immediate, practical, and actionable "
            "response recommendations when a conflict event is predicted. "
            "Keep responses concise, structured, and field-ready. "
            "Do not use emojis. Use plain text with clear section headers separated by newlines."
        )

        user_prompt = f"""
PREDICTED INCIDENT REPORT:
Location: {location}, {country}
Risk Level: {risk_level.upper()}
Predicted Fatalities: {predicted_fatalities:.0f}
Event Type: {event_type}
Predicted Date: {prediction_date[:10] if prediction_date else 'Within 14 days'}
Involved Actor 1: {actor1 if actor1 else 'Unknown'}
Involved Actor 2: {actor2 if actor2 else 'Unknown'}

Based on this prediction, provide a structured response plan covering:

IMMEDIATE RESPONSE (first 24 hours)
List 3-4 immediate actions for field teams.

RESOURCE DEPLOYMENT
What resources and personnel to pre-position.

COMMUNICATION PROTOCOL
Who to notify and through what channels.

MONITORING TRIGGERS
Key indicators that confirm or disprove this prediction.

HUMANITARIAN CONSIDERATIONS
Civilian protection measures to activate.

Keep each section brief and operational.
"""

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.15,
            max_tokens=900
        )

        ai_response = chat_completion.choices[0].message.content

        # Optionally update the alert with the AI response
        alert_id = request.get("alert_id")
        if alert_id:
            try:
                db = DatabaseManager()
                await db.initialize()
                async with aiosqlite.connect(db.db_path) as conn:
                    await conn.execute(
                        "UPDATE alerts SET ai_response = ? WHERE alert_id = ?",
                        (ai_response, alert_id)
                    )
                    await conn.commit()
                await db.close()
            except Exception:
                pass

        return {
            "location": location,
            "country": country,
            "risk_level": risk_level,
            "predicted_fatalities": predicted_fatalities,
            "response_plan": ai_response,
            "model": "llama-3.1-8b-instant",
            "generated_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        print(f"Incident response error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")


@app.get("/api/dashboard/summary")
async def get_dashboard_summary():
    """
    Aggregated summary for the Flutter dashboard home screen.
    Returns total events, fatalities, alert counts, and trend direction.
    """
    try:
        db = DatabaseManager()
        await db.initialize()

        all_events = await db.get_events({}, limit=100000)
        alerts = await db.get_alerts(unread_only=False, limit=1000)
        map_preds = await db.get_map_predictions(limit=1000)

        total_events = len(all_events)
        total_fatalities = sum(e.get("fatalities", 0) or 0 for e in all_events)
        high_risk_count = len([p for p in map_preds if p.get("risk_level") == "high"])
        medium_risk_count = len([p for p in map_preds if p.get("risk_level") == "medium"])
        unread_alerts = len([a for a in alerts if not a.get("is_read")])

        # Simple trend: compare last 30 days vs previous 30 days
        now = datetime.utcnow()
        cutoff_recent = (now - timedelta(days=30)).isoformat()
        cutoff_older = (now - timedelta(days=60)).isoformat()
        recent_events = [
            e for e in all_events
            if e.get("event_date", "") >= cutoff_recent
        ]
        older_events = [
            e for e in all_events
            if cutoff_older <= e.get("event_date", "") < cutoff_recent
        ]
        trend = "stable"
        if len(recent_events) > len(older_events) * 1.15:
            trend = "increasing"
        elif len(recent_events) < len(older_events) * 0.85:
            trend = "decreasing"

        await db.close()

        return {
            "total_events": total_events,
            "total_fatalities": total_fatalities,
            "high_risk_predictions": high_risk_count,
            "medium_risk_predictions": medium_risk_count,
            "unread_alerts": unread_alerts,
            "trend_direction": trend,
            "recent_events_30d": len(recent_events),
            "summary_generated_at": now.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import aiosqlite  # ensure aiosqlite is available for inline use above

# ------------------------------------------------------------------ #
# ACLED API Integration Endpoints
# ------------------------------------------------------------------ #

# In-memory store for ongoing ACLED fetch jobs
acled_fetch_status: Dict[str, Any] = {}


class ACLEDFetchRequest(BaseModel):
    country: Optional[str] = None
    countries: Optional[List[str]] = None         # multiple countries at once
    start_date: Optional[str] = None              # YYYY-MM-DD
    end_date: Optional[str] = None                # YYYY-MM-DD
    event_type: Optional[str] = None
    max_records: int = 5000
    year: Optional[int] = None


@app.post("/api/acled/fetch")
async def fetch_acled_data(
    request: ACLEDFetchRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger a background fetch from the ACLED API and store results in the DB.
    Supports filtering by country, date range, event type, and year.
    """
    fetch_id = str(uuid.uuid4())

    # Build ACLED query filters
    acled_filters: Dict[str, Any] = {}

    if request.country:
        acled_filters["country"] = request.country
    elif request.countries:
        # ACLED supports OR syntax: country=X:OR:country=Y
        acled_filters["country"] = ":OR:country=".join(request.countries)

    if request.year:
        acled_filters["year"] = request.year

    if request.start_date and request.end_date:
        acled_filters["event_date"] = f"{request.start_date}|{request.end_date}"
        acled_filters["event_date_where"] = "BETWEEN"
    elif request.start_date:
        acled_filters["event_date"] = request.start_date
        acled_filters["event_date_where"] = ">="
    elif request.end_date:
        acled_filters["event_date"] = request.end_date
        acled_filters["event_date_where"] = "<="

    if request.event_type:
        acled_filters["event_type"] = request.event_type

    # Initialise status entry
    acled_fetch_status[fetch_id] = {
        "fetch_id": fetch_id,
        "status": "queued",
        "progress": 0,
        "records_fetched": 0,
        "records_stored": 0,
        "filters": acled_filters,
        "max_records": request.max_records,
        "started_at": datetime.utcnow().isoformat(),
        "message": "Queued for processing"
    }

    background_tasks.add_task(
        _run_acled_fetch,
        fetch_id,
        acled_filters,
        request.max_records
    )

    return {
        "fetch_id": fetch_id,
        "status": "queued",
        "message": "ACLED data fetch started in the background.",
        "filters_applied": acled_filters
    }


async def _run_acled_fetch(
    fetch_id: str,
    filters: Dict[str, Any],
    max_records: int
):
    """Background task: fetch from ACLED and insert into the DB."""
    try:
        acled_fetch_status[fetch_id]["status"] = "fetching"
        acled_fetch_status[fetch_id]["progress"] = 10
        acled_fetch_status[fetch_id]["message"] = "Connecting to ACLED API..."

        records = data_processor.acled_adapter.fetch_paginated_data(
            filters=filters,
            max_records=max_records
        )

        acled_fetch_status[fetch_id]["records_fetched"] = len(records)
        acled_fetch_status[fetch_id]["progress"] = 50
        acled_fetch_status[fetch_id]["message"] = f"Fetched {len(records)} records, storing in database..."

        if not records:
            acled_fetch_status[fetch_id]["status"] = "completed"
            acled_fetch_status[fetch_id]["progress"] = 100
            acled_fetch_status[fetch_id]["message"] = "Completed with 0 records (check filters or token)"
            return

        # Standardise and persist
        df = data_processor.acled_adapter.standardize_data(records)

        import uuid as _uuid
        fetch_label = f"acled_{_uuid.uuid4().hex[:8]}"
        df["upload_id"] = fetch_label
        df["data_source"] = "acled_api"
        df["processed_at"] = datetime.utcnow().isoformat()

        events = df.to_dict("records")
        inserted = await db_manager.insert_events(events)

        # MongoDB Atlas dual-write
        try:
            from database import DatabaseManager as MongoManager
            mongo_db = MongoManager()
            await mongo_db.initialize()
            await mongo_db.insert_events(events)
            logger.info(f"Dual-wrote {inserted} ACLED events to MongoDB Atlas")
        except Exception as mongo_exc:
            logger.error(f"MongoDB Atlas dual-write failed: {mongo_exc}")

        acled_fetch_status[fetch_id]["status"] = "completed"
        acled_fetch_status[fetch_id]["progress"] = 100
        acled_fetch_status[fetch_id]["records_stored"] = inserted
        acled_fetch_status[fetch_id]["completed_at"] = datetime.utcnow().isoformat()
        acled_fetch_status[fetch_id]["message"] = f"Successfully stored {inserted} events from ACLED"

        # Notify connected WebSocket clients
        await ws_manager.broadcast({
            "type": "acled_fetch_completed",
            "fetch_id": fetch_id,
            "records_stored": inserted
        })

    except Exception as exc:
        acled_fetch_status[fetch_id]["status"] = "error"
        acled_fetch_status[fetch_id]["message"] = f"Fetch failed: {str(exc)}"
        import traceback
        traceback.print_exc()


async def _startup_acled_sync():
    """
    Runs once at startup.
    If the local DB has fewer than 100 events, automatically pulls the last
    90 days of data from ACLED API so the dashboard has something to show.
    """
    import asyncio
    await asyncio.sleep(5)  # let the server finish starting up
    try:
        existing = await db_manager.get_events({}, limit=100)
        if len(existing) >= 100:
            logger.info(f"Startup sync skipped - DB already has {len(existing)}+ events")
            return

        logger.info("Startup auto-sync: DB is sparse, pulling last 90 days from ACLED API...")
        from datetime import timedelta
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
        start_date = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")

        filters = {
            "event_date": f"{start_date}|{end_date}",
            "event_date_where": "BETWEEN",
        }

        import uuid as _uuid
        fetch_id = f"startup_{_uuid.uuid4().hex[:8]}"
        acled_fetch_status[fetch_id] = {
            "fetch_id": fetch_id,
            "status": "queued",
            "progress": 0,
            "records_fetched": 0,
            "records_stored": 0,
            "filters": filters,
            "max_records": 10000,
            "started_at": datetime.utcnow().isoformat(),
            "message": "Startup auto-sync"
        }
        await _run_acled_fetch(fetch_id, filters, max_records=10000)
        logger.info(f"Startup auto-sync complete: {acled_fetch_status[fetch_id].get('records_stored', 0)} events stored")
    except Exception as exc:
        logger.error(f"Startup auto-sync failed: {exc}")


@app.get("/api/acled/fetch/{fetch_id}")
async def get_acled_fetch_status(fetch_id: str):
    """Poll the status of an ACLED fetch job."""
    if fetch_id not in acled_fetch_status:
        raise HTTPException(status_code=404, detail="Fetch job not found")
    return acled_fetch_status[fetch_id]


@app.get("/api/acled/fetch")
async def list_acled_fetches():
    """List all ACLED fetch jobs (most recent first)."""
    jobs = sorted(
        acled_fetch_status.values(),
        key=lambda j: j.get("started_at", ""),
        reverse=True
    )
    return {"jobs": jobs, "total": len(jobs)}


@app.get("/api/acled/token-status")
async def acled_token_status():
    """
    Check whether ACLED credentials are present and optionally verify
    the token by making a minimal test request to the API.
    """
    from acled_adapter import ACLEDAdapter
    adapter = ACLEDAdapter()

    has_token = bool(adapter.access_token)
    has_refresh = bool(adapter.refresh_token)
    email = adapter.email or "not configured"

    # Quick verification request (limit=1)
    verified = False
    error_msg = None
    if has_token:
        test_records = adapter.fetch_data({"limit": 1})
        verified = isinstance(test_records, list)
        if not verified:
            error_msg = "Token present but test request returned no data"

    return {
        "email": email,
        "has_access_token": has_token,
        "has_refresh_token": has_refresh,
        "token_verified": verified,
        "error": error_msg
    }


@app.post("/api/acled/refresh-token")
async def refresh_acled_token():
    """Manually trigger an ACLED OAuth token refresh."""
    from acled_adapter import ACLEDAdapter
    adapter = ACLEDAdapter()
    success = adapter.refresh_access_token()
    if success:
        return {
            "status": "refreshed",
            "message": "ACLED access token refreshed and persisted to .env"
        }
    raise HTTPException(
        status_code=502,
        detail="Failed to refresh ACLED token. Attempted password re-auth as fallback."
    )


@app.post("/api/acled/authenticate")
async def acled_authenticate():
    """
    Force a full password-based re-authentication with ACLED.
    Uses ACLED_EMAIL and ACLED_PASSWORD from .env.
    Call this whenever tokens are stale or lost.
    """
    from acled_adapter import ACLEDAdapter
    adapter = ACLEDAdapter()
    success = adapter.authenticate()
    if success:
        return {
            "status": "authenticated",
            "message": "Fresh ACLED tokens obtained and persisted to .env"
        }
    raise HTTPException(
        status_code=502,
        detail="ACLED authentication failed. Check ACLED_EMAIL / ACLED_PASSWORD in .env."
    )


# ------------------------------------------------------------------ #
# CAST (Conflict Alert System) Prediction Endpoints
# ------------------------------------------------------------------ #

cast_fetch_status: Dict[str, Any] = {}


class CASTFetchRequest(BaseModel):
    country: Optional[str] = None
    countries: Optional[List[str]] = None
    admin1: Optional[str] = None
    month: Optional[str] = None          # e.g. "March"
    year: Optional[int] = None
    max_records: int = 5000


@app.post("/api/cast/fetch")
async def fetch_cast_data(
    request: CASTFetchRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger a background fetch from the ACLED CAST prediction endpoint.
    Returns forecasted political violence counts by country and admin1.
    """
    fetch_id = str(uuid.uuid4())

    # Build CAST query filters
    cast_filters: Dict[str, Any] = {}

    if request.country:
        cast_filters["country"] = request.country
    elif request.countries:
        cast_filters["country"] = "|".join(request.countries)

    if request.admin1:
        cast_filters["admin1"] = request.admin1
    if request.month:
        cast_filters["month"] = request.month
    if request.year:
        cast_filters["year"] = request.year

    cast_fetch_status[fetch_id] = {
        "fetch_id": fetch_id,
        "status": "queued",
        "progress": 0,
        "records_fetched": 0,
        "records_stored": 0,
        "filters": cast_filters,
        "max_records": request.max_records,
        "started_at": datetime.utcnow().isoformat(),
        "message": "Queued for processing"
    }

    background_tasks.add_task(
        _run_cast_fetch,
        fetch_id,
        cast_filters,
        request.max_records
    )

    return {
        "fetch_id": fetch_id,
        "status": "queued",
        "message": "CAST data fetch started in the background.",
        "filters_applied": cast_filters
    }


async def _run_cast_fetch(
    fetch_id: str,
    filters: Dict[str, Any],
    max_records: int
):
    """Background task: fetch from CAST and store in DB."""
    try:
        cast_fetch_status[fetch_id]["status"] = "fetching"
        cast_fetch_status[fetch_id]["progress"] = 10
        cast_fetch_status[fetch_id]["message"] = "Connecting to ACLED CAST API..."

        records = data_processor.acled_adapter.fetch_cast_paginated(
            filters=filters,
            max_records=max_records
        )
        cast_fetch_status[fetch_id]["records_fetched"] = len(records)
        cast_fetch_status[fetch_id]["progress"] = 50
        cast_fetch_status[fetch_id]["message"] = (
            f"Fetched {len(records)} CAST records, storing in database..."
        )

        if not records:
            cast_fetch_status[fetch_id]["status"] = "completed"
            cast_fetch_status[fetch_id]["progress"] = 100
            cast_fetch_status[fetch_id]["message"] = "Completed with 0 records (check filters or token)"
            return

        df = data_processor.acled_adapter.standardize_cast_data(records)

        import uuid as _uuid
        fetch_label = f"cast_{_uuid.uuid4().hex[:8]}"
        df["upload_id"] = fetch_label
        df["data_source"] = "acled_cast"
        df["processed_at"] = datetime.utcnow().isoformat()

        # Store CAST predictions as their own collection / table
        events = df.to_dict("records")
        inserted = await db_manager.insert_cast_predictions(events)

        cast_fetch_status[fetch_id]["status"] = "completed"
        cast_fetch_status[fetch_id]["progress"] = 100
        cast_fetch_status[fetch_id]["records_stored"] = inserted
        cast_fetch_status[fetch_id]["completed_at"] = datetime.utcnow().isoformat()
        cast_fetch_status[fetch_id]["message"] = (
            f"Successfully stored {inserted} CAST predictions"
        )

        await ws_manager.broadcast({
            "type": "cast_fetch_completed",
            "fetch_id": fetch_id,
            "records_stored": inserted
        })

    except Exception as exc:
        cast_fetch_status[fetch_id]["status"] = "error"
        cast_fetch_status[fetch_id]["message"] = f"CAST fetch failed: {str(exc)}"
        import traceback
        traceback.print_exc()


@app.get("/api/cast/fetch/{fetch_id}")
async def get_cast_fetch_status(fetch_id: str):
    """Poll the status of a CAST fetch job."""
    if fetch_id not in cast_fetch_status:
        raise HTTPException(status_code=404, detail="CAST fetch job not found")
    return cast_fetch_status[fetch_id]


@app.get("/api/cast/fetch")
async def list_cast_fetches():
    """List all CAST fetch jobs (most recent first)."""
    jobs = sorted(
        cast_fetch_status.values(),
        key=lambda j: j.get("started_at", ""),
        reverse=True
    )
    return {"jobs": jobs, "total": len(jobs)}


@app.get("/api/cast/predictions")
async def get_cast_predictions(
    country: Optional[str] = None,
    admin1: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[str] = None,
    limit: int = 500
):
    """
    Query stored CAST predictions from the local database.
    Supports filtering by country, admin1, year, and month.
    """
    try:
        filters: Dict[str, Any] = {}
        if country:
            filters["country"] = country
        if admin1:
            filters["admin1"] = admin1
        if year:
            filters["year"] = year
        if month:
            filters["month"] = month

        predictions = await db_manager.get_cast_predictions(filters, limit)
        return {
            "count": len(predictions),
            "filters": filters,
            "predictions": predictions
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
