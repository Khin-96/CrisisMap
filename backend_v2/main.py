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
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import new modules
from database import DatabaseManager
from ml_pipeline import MLPipeline
from data_processor import DataProcessor
from websocket_manager import WebSocketManager

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and ML models on startup"""
    await db_manager.initialize()
    await ml_pipeline.load_models()
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
        
        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(
                status_code=400,
                detail="Only CSV and Excel files are supported"
            )
        
        # Save uploaded file
        upload_dir = Path("uploads")
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

@app.post("/api/upload/analyze")
async def analyze_csv_structure(file: UploadFile = File(...)):
    """Analyze CSV structure and suggest column mappings"""
    try:
        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(
                status_code=400,
                detail="Only CSV and Excel files are supported"
            )
        
        # Save temporary file
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / f"temp_{file.filename}"
        
        content = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(content)
        
        # Analyze file structure
        analysis = await data_processor.analyze_csv_file(str(temp_file_path))
        
        # Clean up temp file
        temp_file_path.unlink()
        
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
    """Get comprehensive context from database for AI analysis"""
    try:
        # Get recent events (last 6 months)
        recent_events = await db_manager.get_events(
            filters={
                "event_date": {
                    "$gte": (datetime.utcnow() - timedelta(days=180)).isoformat()
                }
            },
            limit=10000
        )
        
        if not recent_events:
            return {"error": "No recent data available"}
        
        df = pd.DataFrame(recent_events)
        
        # Calculate comprehensive statistics
        total_events = len(df)
        total_fatalities = df['fatalities'].sum()
        
        # Country analysis
        country_stats = df.groupby('country').agg({
            'fatalities': 'sum',
            'event_id': 'count'
        }).rename(columns={'event_id': 'events'}).to_dict('index')
        
        top_countries = sorted(country_stats.items(), 
                             key=lambda x: x[1]['fatalities'], reverse=True)[:10]
        
        # Event type analysis
        event_type_stats = df.groupby('event_type').agg({
            'fatalities': 'sum',
            'event_id': 'count'
        }).rename(columns={'event_id': 'events'}).to_dict('index')
        
        top_event_types = sorted(event_type_stats.items(), 
                               key=lambda x: x[1]['events'], reverse=True)[:10]
        
        # Geographic hotspots
        location_stats = df.groupby('location').agg({
            'fatalities': 'sum',
            'event_id': 'count'
        }).rename(columns={'event_id': 'events'}).to_dict('index')
        
        hotspots = sorted(location_stats.items(), 
                         key=lambda x: x[1]['fatalities'], reverse=True)[:10]
        
        # Recent high-impact events
        recent_high_impact = df[df['fatalities'] > 10].sort_values(
            'event_date', ascending=False
        ).head(5).to_dict('records')
        
        # Trend analysis
        df['event_date'] = pd.to_datetime(df['event_date'])
        monthly_trends = df.groupby(df['event_date'].dt.to_period('M')).agg({
            'fatalities': 'sum',
            'event_id': 'count'
        }).rename(columns={'event_id': 'events'})
        
        # Determine trend direction
        if len(monthly_trends) >= 2:
            recent_avg = monthly_trends.tail(2)['events'].mean()
            earlier_avg = monthly_trends.head(max(1, len(monthly_trends)-2))['events'].mean()
            trend_direction = "increasing" if recent_avg > earlier_avg else "decreasing"
        else:
            trend_direction = "stable"
        
        # Try to get ML predictions
        ml_predictions_summary = "No ML predictions available"
        try:
            # Get latest ML predictions if available
            ml_pipeline = MLPipeline()
            if ml_pipeline.models:
                predictions = await ml_pipeline.generate_predictions(horizon_days=7)
                if predictions and 'predictions' in predictions:
                    high_risk_count = len([p for p in predictions['predictions'] 
                                         if p.get('risk_level') == 'high'])
                    ml_predictions_summary = f"ML Model predicts {high_risk_count} high-risk areas in next 7 days"
        except:
            pass
        
        return {
            "total_events": total_events,
            "total_fatalities": int(total_fatalities),
            "active_countries": len(country_stats),
            "date_range": f"{df['event_date'].min().strftime('%Y-%m-%d')} to {df['event_date'].max().strftime('%Y-%m-%d')}",
            "trend_direction": trend_direction,
            "top_countries": top_countries,
            "top_event_types": top_event_types,
            "hotspots": hotspots,
            "recent_high_impact": recent_high_impact,
            "ml_predictions_summary": ml_predictions_summary,
            "data_freshness": "last_6_months",
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        print(f"Context generation error: {str(e)}")
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