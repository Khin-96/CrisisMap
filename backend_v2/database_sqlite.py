import sqlite3
import aiosqlite
from typing import Dict, List, Optional, Any
import os
from datetime import datetime
import logging
import json
import uuid

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        # SQLite database file
        self.db_path = os.getenv("DATABASE_PATH", "crisismap.db")
        self.db = None
        
    async def initialize(self):
        """Initialize SQLite database and create tables"""
        try:
            # Create database and tables
            await self._create_tables()
            logger.info("Connected to SQLite database successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
            raise
    
    async def _create_tables(self):
        """Create database tables"""
        async with aiosqlite.connect(self.db_path) as db:
            # Events table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE,
                    event_date TEXT NOT NULL,
                    location TEXT,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    actor1 TEXT,
                    actor2 TEXT,
                    fatalities INTEGER DEFAULT 0,
                    country TEXT,
                    confidence_score REAL,
                    upload_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            # Models table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT UNIQUE NOT NULL,
                    model_type TEXT NOT NULL,
                    status TEXT DEFAULT 'training',
                    metrics TEXT,
                    hyperparameters TEXT,
                    file_path TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            # Predictions table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    prediction_date TEXT NOT NULL,
                    country TEXT,
                    predicted_events INTEGER,
                    predicted_fatalities INTEGER,
                    confidence_score REAL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            ''')
            
            # Create indexes
            await db.execute('CREATE INDEX IF NOT EXISTS idx_events_date ON events(event_date)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_events_country ON events(country)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_events_location ON events(latitude, longitude)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_models_type ON models(model_type)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_id)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date)')
            
            await db.commit()
            logger.info("Database tables created successfully")
    
    async def insert_events(self, events: List[Dict[str, Any]]) -> int:
        """Insert multiple conflict events"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                current_time = datetime.utcnow().isoformat()
                
                for event in events:
                    # Generate event_id if not provided
                    if 'event_id' not in event:
                        event['event_id'] = str(uuid.uuid4())
                    
                    await db.execute('''
                        INSERT OR REPLACE INTO events 
                        (event_id, event_date, location, latitude, longitude, event_type, 
                         actor1, actor2, fatalities, country, confidence_score, upload_id, 
                         created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        event.get('event_id'),
                        event.get('event_date'),
                        event.get('location'),
                        event.get('latitude'),
                        event.get('longitude'),
                        event.get('event_type'),
                        event.get('actor1'),
                        event.get('actor2'),
                        event.get('fatalities', 0),
                        event.get('country'),
                        event.get('confidence_score'),
                        event.get('upload_id'),
                        current_time,
                        current_time
                    ))
                
                await db.commit()
                return len(events)
                
        except Exception as e:
            logger.error(f"Failed to insert events: {e}")
            raise
    
    async def get_events(
        self, 
        filters: Dict[str, Any] = None, 
        limit: int = 100,
        sort_by: str = "event_date",
        sort_order: int = -1
    ) -> List[Dict[str, Any]]:
        """Retrieve events with filtering and pagination"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                query = "SELECT * FROM events"
                params = []
                
                if filters:
                    conditions = []
                    for key, value in filters.items():
                        if key == "event_date" and isinstance(value, dict):
                            if "$gte" in value:
                                conditions.append("event_date >= ?")
                                params.append(value["$gte"])
                            if "$lte" in value:
                                conditions.append("event_date <= ?")
                                params.append(value["$lte"])
                        else:
                            conditions.append(f"{key} = ?")
                            params.append(value)
                    
                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)
                
                # Add sorting
                order = "DESC" if sort_order == -1 else "ASC"
                query += f" ORDER BY {sort_by} {order} LIMIT ?"
                params.append(limit)
                
                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()
                
                events = []
                for row in rows:
                    event = dict(row)
                    event["_id"] = str(event["id"])  # Compatibility with MongoDB format
                    events.append(event)
                
                return events
                
        except Exception as e:
            logger.error(f"Failed to retrieve events: {e}")
            raise
    
    async def store_model(self, model_data: Dict[str, Any]) -> str:
        """Store ML model metadata"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                current_time = datetime.utcnow().isoformat()
                model_id = model_data.get('model_id', str(uuid.uuid4()))
                
                await db.execute('''
                    INSERT OR REPLACE INTO models 
                    (model_id, model_type, status, metrics, hyperparameters, file_path, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model_id,
                    model_data.get('model_type'),
                    model_data.get('status', 'training'),
                    json.dumps(model_data.get('metrics', {})),
                    json.dumps(model_data.get('hyperparameters', {})),
                    model_data.get('file_path'),
                    current_time,
                    current_time
                ))
                
                await db.commit()
                return model_id
                
        except Exception as e:
            logger.error(f"Failed to store model: {e}")
            raise
    
    async def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve model by ID"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                cursor = await db.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
                row = await cursor.fetchone()
                
                if row:
                    model = dict(row)
                    model["_id"] = str(model["id"])
                    model["metrics"] = json.loads(model["metrics"]) if model["metrics"] else {}
                    model["hyperparameters"] = json.loads(model["hyperparameters"]) if model["hyperparameters"] else {}
                    return model
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve model: {e}")
            raise
    
    async def store_predictions(self, predictions: List[Dict[str, Any]]) -> int:
        """Store model predictions"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                current_time = datetime.utcnow().isoformat()
                
                for prediction in predictions:
                    await db.execute('''
                        INSERT INTO predictions 
                        (model_id, prediction_date, country, predicted_events, predicted_fatalities, 
                         confidence_score, metadata, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        prediction.get('model_id'),
                        prediction.get('prediction_date'),
                        prediction.get('country'),
                        prediction.get('predicted_events'),
                        prediction.get('predicted_fatalities'),
                        prediction.get('confidence_score'),
                        json.dumps(prediction.get('metadata', {})),
                        current_time
                    ))
                
                await db.commit()
                return len(predictions)
                
        except Exception as e:
            logger.error(f"Failed to store predictions: {e}")
            raise
    
    async def get_latest_predictions(
        self,
        model_id: str,
        country: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get latest predictions from a model"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                query = "SELECT * FROM predictions WHERE model_id = ?"
                params = [model_id]
                
                if country:
                    query += " AND country = ?"
                    params.append(country)
                
                query += " ORDER BY prediction_date DESC LIMIT 50"
                
                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()
                
                predictions = []
                for row in rows:
                    prediction = dict(row)
                    prediction["_id"] = str(prediction["id"])
                    prediction["metadata"] = json.loads(prediction["metadata"]) if prediction["metadata"] else {}
                    predictions.append(prediction)
                
                return predictions
                
        except Exception as e:
            logger.error(f"Failed to retrieve predictions: {e}")
            raise
    
    async def get_temporal_trends(
        self,
        country: Optional[str] = None,
        period: str = "monthly"
    ) -> List[Dict[str, Any]]:
        """Get temporal trends using SQL aggregation"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # Determine date format based on period
                if period == "daily":
                    date_format = "%Y-%m-%d"
                elif period == "weekly":
                    date_format = "%Y-%W"
                else:  # monthly
                    date_format = "%Y-%m"
                
                query = f'''
                    SELECT 
                        strftime('{date_format}', event_date) as period,
                        COUNT(*) as total_events,
                        SUM(fatalities) as total_fatalities,
                        AVG(fatalities) as avg_fatalities,
                        COUNT(DISTINCT event_type) as unique_event_types
                    FROM events
                '''
                
                params = []
                if country:
                    query += " WHERE country = ?"
                    params.append(country)
                
                query += " GROUP BY period ORDER BY period"
                
                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()
                
                trends = []
                for row in rows:
                    trend = dict(row)
                    trend["avg_fatalities"] = round(trend["avg_fatalities"], 2) if trend["avg_fatalities"] else 0
                    trends.append(trend)
                
                return trends
                
        except Exception as e:
            logger.error(f"Failed to get temporal trends: {e}")
            raise
    
    async def get_hotspots(
        self,
        country: Optional[str] = None,
        threshold: int = 5
    ) -> List[Dict[str, Any]]:
        """Identify geographic hotspots"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                query = '''
                    SELECT 
                        location,
                        ROUND(latitude, 2) as latitude,
                        ROUND(longitude, 2) as longitude,
                        COUNT(*) as event_count,
                        SUM(fatalities) as total_fatalities,
                        AVG(fatalities) as avg_fatalities,
                        (COUNT(*) * AVG(fatalities)) as intensity_score
                    FROM events
                '''
                
                params = []
                if country:
                    query += " WHERE country = ?"
                    params.append(country)
                
                query += '''
                    GROUP BY location, ROUND(latitude, 2), ROUND(longitude, 2)
                    HAVING event_count >= ?
                    ORDER BY total_fatalities DESC
                    LIMIT 20
                '''
                params.append(threshold)
                
                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()
                
                hotspots = []
                for row in rows:
                    hotspot = dict(row)
                    hotspot["avg_fatalities"] = round(hotspot["avg_fatalities"], 2) if hotspot["avg_fatalities"] else 0
                    hotspot["intensity_score"] = round(hotspot["intensity_score"], 2) if hotspot["intensity_score"] else 0
                    hotspots.append(hotspot)
                
                return hotspots
                
        except Exception as e:
            logger.error(f"Failed to get hotspots: {e}")
            raise
    
    async def get_events_aggregated(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compatibility method for MongoDB-style aggregation (simplified)"""
        # For SQLite, we'll implement basic aggregation using the existing methods
        # This is a simplified version - complex pipelines would need custom implementation
        return []
    
    async def close(self):
        """Close database connection (SQLite handles this automatically)"""
        logger.info("SQLite database connection closed")