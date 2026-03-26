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
        self.db_path = os.getenv("DATABASE_PATH", "crisismap.db")
        self.db = None

    async def initialize(self):
        """Initialize SQLite database and create tables"""
        try:
            await self._create_tables()
            logger.info("Connected to SQLite database successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
            raise

    async def _create_tables(self):
        """Create database tables"""
        async with aiosqlite.connect(self.db_path, timeout=60.0) as db:
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

            # Map predictions table - stores per-location forecasts for the Flutter map
            await db.execute('''
                CREATE TABLE IF NOT EXISTS map_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT UNIQUE NOT NULL,
                    model_id TEXT NOT NULL,
                    location_name TEXT,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    country TEXT,
                    event_type TEXT,
                    predicted_fatalities REAL NOT NULL,
                    predicted_events INTEGER DEFAULT 1,
                    risk_level TEXT NOT NULL,
                    risk_score REAL,
                    confidence REAL,
                    prediction_for_date TEXT NOT NULL,
                    horizon_days INTEGER,
                    actor1 TEXT,
                    actor2 TEXT,
                    ai_summary TEXT,
                    created_at TEXT NOT NULL
                )
            ''')

            # Alerts table - high-risk predictions flagged for push notifications
            await db.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    prediction_id TEXT,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    location_name TEXT,
                    country TEXT,
                    predicted_fatalities REAL,
                    event_type TEXT,
                    ai_response TEXT,
                    is_read INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            ''')

            await db.execute('CREATE INDEX IF NOT EXISTS idx_events_date ON events(event_date)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_events_country ON events(country)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_events_location ON events(latitude, longitude)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_models_type ON models(model_type)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_id)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_map_predictions_date ON map_predictions(prediction_for_date)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_map_predictions_risk ON map_predictions(risk_level)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_alerts_read ON alerts(is_read)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_alerts_risk ON alerts(risk_level)')

            # CAST predictions table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS cast_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cast_id TEXT UNIQUE,
                    country TEXT NOT NULL,
                    admin1 TEXT,
                    month TEXT,
                    year INTEGER,
                    total_forecast INTEGER DEFAULT 0,
                    battles_forecast INTEGER DEFAULT 0,
                    erv_forecast INTEGER DEFAULT 0,
                    vac_forecast INTEGER DEFAULT 0,
                    total_observed INTEGER DEFAULT 0,
                    battles_observed INTEGER DEFAULT 0,
                    erv_observed INTEGER DEFAULT 0,
                    vac_observed INTEGER DEFAULT 0,
                    acled_timestamp INTEGER,
                    upload_id TEXT,
                    data_source TEXT DEFAULT 'acled_cast',
                    processed_at TEXT,
                    created_at TEXT NOT NULL
                )
            ''')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_cast_country ON cast_predictions(country)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_cast_year ON cast_predictions(year)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_cast_month ON cast_predictions(month)')

            await db.commit()
            logger.info("Database tables created/verified successfully")

    async def insert_events(self, events: List[Dict[str, Any]]) -> int:
        """Insert multiple conflict events"""
        try:
            async with aiosqlite.connect(self.db_path, timeout=60.0) as db:
                current_time = datetime.utcnow().isoformat()
                for event in events:
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
            async with aiosqlite.connect(self.db_path, timeout=60.0) as db:
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
                order = "DESC" if sort_order == -1 else "ASC"
                query += f" ORDER BY {sort_by} {order} LIMIT ?"
                params.append(limit)
                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()
                events = []
                for row in rows:
                    event = dict(row)
                    event["_id"] = str(event["id"])
                    events.append(event)
                return events
        except Exception as e:
            logger.error(f"Failed to retrieve events: {e}")
            raise

    async def store_model(self, model_data: Dict[str, Any]) -> str:
        """Store ML model metadata"""
        try:
            async with aiosqlite.connect(self.db_path, timeout=60.0) as db:
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
            async with aiosqlite.connect(self.db_path, timeout=60.0) as db:
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
            async with aiosqlite.connect(self.db_path, timeout=60.0) as db:
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

    async def store_map_predictions(self, map_predictions: List[Dict[str, Any]]) -> int:
        """Store geo-tagged predictions for Flutter map display"""
        try:
            async with aiosqlite.connect(self.db_path, timeout=60.0) as db:
                current_time = datetime.utcnow().isoformat()
                stored = 0
                for pred in map_predictions:
                    prediction_id = pred.get('prediction_id', str(uuid.uuid4()))
                    await db.execute('''
                        INSERT OR REPLACE INTO map_predictions
                        (prediction_id, model_id, location_name, latitude, longitude, country,
                         event_type, predicted_fatalities, predicted_events, risk_level, risk_score,
                         confidence, prediction_for_date, horizon_days, actor1, actor2, ai_summary, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        prediction_id,
                        pred.get('model_id', ''),
                        pred.get('location_name', ''),
                        pred.get('latitude'),
                        pred.get('longitude'),
                        pred.get('country', ''),
                        pred.get('event_type', ''),
                        pred.get('predicted_fatalities', 0),
                        pred.get('predicted_events', 1),
                        pred.get('risk_level', 'low'),
                        pred.get('risk_score', 0.0),
                        pred.get('confidence', 0.5),
                        pred.get('prediction_for_date', ''),
                        pred.get('horizon_days', 14),
                        pred.get('actor1', ''),
                        pred.get('actor2', ''),
                        pred.get('ai_summary', ''),
                        current_time
                    ))
                    stored += 1
                await db.commit()
                return stored
        except Exception as e:
            logger.error(f"Failed to store map predictions: {e}")
            raise

    async def get_map_predictions(
        self,
        risk_level: Optional[str] = None,
        country: Optional[str] = None,
        limit: int = 200
    ) -> List[Dict[str, Any]]:
        """Get geo-tagged predictions for Flutter map display"""
        try:
            async with aiosqlite.connect(self.db_path, timeout=60.0) as db:
                db.row_factory = aiosqlite.Row
                query = "SELECT * FROM map_predictions"
                params = []
                conditions = []
                if risk_level:
                    conditions.append("risk_level = ?")
                    params.append(risk_level)
                if country:
                    conditions.append("country = ?")
                    params.append(country)
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                query += " ORDER BY risk_score DESC, created_at DESC LIMIT ?"
                params.append(limit)
                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to retrieve map predictions: {e}")
            raise

    async def store_alert(self, alert: Dict[str, Any]) -> str:
        """Store a push notification alert"""
        try:
            async with aiosqlite.connect(self.db_path, timeout=60.0) as db:
                current_time = datetime.utcnow().isoformat()
                alert_id = alert.get('alert_id', str(uuid.uuid4()))
                await db.execute('''
                    INSERT OR REPLACE INTO alerts
                    (alert_id, prediction_id, title, body, risk_level, latitude, longitude,
                     location_name, country, predicted_fatalities, event_type, ai_response,
                     is_read, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert_id,
                    alert.get('prediction_id', ''),
                    alert.get('title', ''),
                    alert.get('body', ''),
                    alert.get('risk_level', 'medium'),
                    alert.get('latitude'),
                    alert.get('longitude'),
                    alert.get('location_name', ''),
                    alert.get('country', ''),
                    alert.get('predicted_fatalities', 0),
                    alert.get('event_type', ''),
                    alert.get('ai_response', ''),
                    0,
                    current_time
                ))
                await db.commit()
                return alert_id
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
            raise

    async def get_alerts(
        self,
        unread_only: bool = False,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get alerts for Flutter notification feed"""
        try:
            async with aiosqlite.connect(self.db_path, timeout=60.0) as db:
                db.row_factory = aiosqlite.Row
                query = "SELECT * FROM alerts"
                params = []
                if unread_only:
                    query += " WHERE is_read = 0"
                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to retrieve alerts: {e}")
            raise

    async def mark_alert_read(self, alert_id: str) -> bool:
        """Mark a notification alert as read"""
        try:
            async with aiosqlite.connect(self.db_path, timeout=60.0) as db:
                await db.execute("UPDATE alerts SET is_read = 1 WHERE alert_id = ?", (alert_id,))
                await db.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to mark alert read: {e}")
            return False

    async def get_latest_predictions(
        self,
        model_id: str,
        country: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get latest predictions from a model"""
        try:
            async with aiosqlite.connect(self.db_path, timeout=60.0) as db:
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
            async with aiosqlite.connect(self.db_path, timeout=60.0) as db:
                db.row_factory = aiosqlite.Row
                if period == "daily":
                    date_format = "%Y-%m-%d"
                elif period == "weekly":
                    date_format = "%Y-%W"
                else:
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
            async with aiosqlite.connect(self.db_path, timeout=60.0) as db:
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
        """Compatibility method for MongoDB-style aggregation"""
        return []

    async def insert_cast_predictions(self, records: List[Dict[str, Any]]) -> int:
        """Insert CAST forecast records into cast_predictions table."""
        try:
            async with aiosqlite.connect(self.db_path, timeout=60.0) as db:
                current_time = datetime.utcnow().isoformat()
                
                # Prepare data for batch insert
                batch_data = []
                for rec in records:
                    batch_data.append((
                        str(uuid.uuid4()), # cast_id
                        rec.get('country', ''),
                        rec.get('admin1', ''),
                        rec.get('month', ''),
                        rec.get('year'),
                        rec.get('total_forecast', 0),
                        rec.get('battles_forecast', 0),
                        rec.get('erv_forecast', 0),
                        rec.get('vac_forecast', 0),
                        rec.get('total_observed', 0),
                        rec.get('battles_observed', 0),
                        rec.get('erv_observed', 0),
                        rec.get('vac_observed', 0),
                        rec.get('timestamp'),
                        rec.get('upload_id'),
                        rec.get('data_source', 'acled_cast'),
                        rec.get('processed_at', current_time),
                        current_time
                    ))
                
                if batch_data:
                    await db.executemany('''
                        INSERT OR REPLACE INTO cast_predictions
                        (cast_id, country, admin1, month, year,
                         total_forecast, battles_forecast, erv_forecast, vac_forecast,
                         total_observed, battles_observed, erv_observed, vac_observed,
                         acled_timestamp, upload_id, data_source, processed_at, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', batch_data)
                    await db.commit()
                    return len(batch_data)
                return 0
        except Exception as e:
            logger.error(f"Failed to insert CAST predictions: {e}")
            raise

    async def get_cast_predictions(
        self,
        filters: Dict[str, Any] = None,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Retrieve stored CAST predictions with optional filters."""
        try:
            async with aiosqlite.connect(self.db_path, timeout=60.0) as db:
                db.row_factory = aiosqlite.Row
                query = "SELECT * FROM cast_predictions"
                params = []
                conditions = []
                if filters:
                    for key, value in filters.items():
                        conditions.append(f"{key} = ?")
                        params.append(value)
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to retrieve CAST predictions: {e}")
            raise

    async def close(self):
        """Close database connection"""
        logger.info("SQLite database connection closed")