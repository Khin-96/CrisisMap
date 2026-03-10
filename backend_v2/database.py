import motor.motor_asyncio
from pymongo import IndexModel, ASCENDING, DESCENDING, GEO2D
from typing import Dict, List, Optional, Any
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        # MongoDB connection string - use environment variable or default
        self.connection_string = os.getenv(
            "MONGODB_URL", 
            "mongodb://localhost:27017/Crisis"
        )
        self.database_name = "Crisis"
        self.client = None
        self.db = None
        
    async def initialize(self):
        """Initialize MongoDB connection and create indexes"""
        try:
            self.client = motor.motor_asyncio.AsyncIOMotorClient(self.connection_string)
            self.db = self.client[self.database_name]
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info("Connected to MongoDB successfully")
            
            # Create collections and indexes
            await self._create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def _create_indexes(self):
        """Create database indexes for optimal performance"""
        
        # Events collection indexes
        events_indexes = [
            IndexModel([("event_date", DESCENDING)]),
            IndexModel([("country", ASCENDING)]),
            IndexModel([("event_type", ASCENDING)]),
            IndexModel([("location", GEO2D)]),  # For geospatial queries
            IndexModel([("latitude", ASCENDING), ("longitude", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("upload_id", ASCENDING)]),
        ]
        
        await self.db.events.create_indexes(events_indexes)
        
        # Models collection indexes
        models_indexes = [
            IndexModel([("model_id", ASCENDING)], unique=True),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("model_type", ASCENDING)]),
            IndexModel([("status", ASCENDING)]),
        ]
        
        await self.db.models.create_indexes(models_indexes)
        
        # Predictions collection indexes
        predictions_indexes = [
            IndexModel([("model_id", ASCENDING)]),
            IndexModel([("prediction_date", DESCENDING)]),
            IndexModel([("country", ASCENDING)]),
        ]
        
        await self.db.predictions.create_indexes(predictions_indexes)
        
        logger.info("Database indexes created successfully")
    
    async def insert_events(self, events: List[Dict[str, Any]]) -> int:
        """Insert multiple conflict events"""
        try:
            # Add metadata
            for event in events:
                event["created_at"] = datetime.utcnow()
                event["updated_at"] = datetime.utcnow()
            
            result = await self.db.events.insert_many(events)
            return len(result.inserted_ids)
            
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
            query = filters or {}
            
            cursor = self.db.events.find(query).sort(sort_by, sort_order).limit(limit)
            events = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string for JSON serialization
            for event in events:
                event["_id"] = str(event["_id"])
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to retrieve events: {e}")
            raise
    
    async def get_events_aggregated(
        self,
        pipeline: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute aggregation pipeline on events collection"""
        try:
            cursor = self.db.events.aggregate(pipeline)
            results = await cursor.to_list(length=None)
            return results
            
        except Exception as e:
            logger.error(f"Failed to execute aggregation: {e}")
            raise
    
    async def store_model(self, model_data: Dict[str, Any]) -> str:
        """Store ML model metadata and artifacts"""
        try:
            model_data["created_at"] = datetime.utcnow()
            model_data["updated_at"] = datetime.utcnow()
            
            result = await self.db.models.insert_one(model_data)
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to store model: {e}")
            raise
    
    async def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve model by ID"""
        try:
            model = await self.db.models.find_one({"model_id": model_id})
            if model:
                model["_id"] = str(model["_id"])
            return model
            
        except Exception as e:
            logger.error(f"Failed to retrieve model: {e}")
            raise
    
    async def store_predictions(self, predictions: List[Dict[str, Any]]) -> int:
        """Store model predictions"""
        try:
            for prediction in predictions:
                prediction["created_at"] = datetime.utcnow()
            
            result = await self.db.predictions.insert_many(predictions)
            return len(result.inserted_ids)
            
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
            query = {"model_id": model_id}
            if country:
                query["country"] = country
            
            cursor = self.db.predictions.find(query).sort("prediction_date", -1).limit(50)
            predictions = await cursor.to_list(length=50)
            
            for prediction in predictions:
                prediction["_id"] = str(prediction["_id"])
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to retrieve predictions: {e}")
            raise
    
    async def get_temporal_trends(
        self,
        country: Optional[str] = None,
        period: str = "monthly"
    ) -> List[Dict[str, Any]]:
        """Get temporal trends using aggregation pipeline"""
        try:
            # Build aggregation pipeline based on period
            if period == "daily":
                date_format = "%Y-%m-%d"
            elif period == "weekly":
                date_format = "%Y-%U"
            else:  # monthly
                date_format = "%Y-%m"
            
            pipeline = []
            
            # Match stage for country filter
            if country:
                pipeline.append({"$match": {"country": country}})
            
            # Group by time period
            pipeline.extend([
                {
                    "$group": {
                        "_id": {
                            "$dateToString": {
                                "format": date_format,
                                "date": {"$dateFromString": {"dateString": "$event_date"}}
                            }
                        },
                        "total_events": {"$sum": 1},
                        "total_fatalities": {"$sum": "$fatalities"},
                        "avg_fatalities": {"$avg": "$fatalities"},
                        "event_types": {"$addToSet": "$event_type"}
                    }
                },
                {
                    "$sort": {"_id": 1}
                },
                {
                    "$project": {
                        "period": "$_id",
                        "total_events": 1,
                        "total_fatalities": 1,
                        "avg_fatalities": {"$round": ["$avg_fatalities", 2]},
                        "unique_event_types": {"$size": "$event_types"},
                        "_id": 0
                    }
                }
            ])
            
            return await self.get_events_aggregated(pipeline)
            
        except Exception as e:
            logger.error(f"Failed to get temporal trends: {e}")
            raise
    
    async def get_hotspots(
        self,
        country: Optional[str] = None,
        threshold: int = 5
    ) -> List[Dict[str, Any]]:
        """Identify geographic hotspots using aggregation"""
        try:
            pipeline = []
            
            if country:
                pipeline.append({"$match": {"country": country}})
            
            pipeline.extend([
                {
                    "$group": {
                        "_id": {
                            "location": "$location",
                            "lat": {"$round": ["$latitude", 2]},
                            "lng": {"$round": ["$longitude", 2]}
                        },
                        "event_count": {"$sum": 1},
                        "total_fatalities": {"$sum": "$fatalities"},
                        "avg_fatalities": {"$avg": "$fatalities"},
                        "recent_events": {
                            "$push": {
                                "date": "$event_date",
                                "type": "$event_type",
                                "fatalities": "$fatalities"
                            }
                        }
                    }
                },
                {
                    "$match": {"event_count": {"$gte": threshold}}
                },
                {
                    "$sort": {"total_fatalities": -1}
                },
                {
                    "$limit": 20
                },
                {
                    "$project": {
                        "location": "$_id.location",
                        "latitude": "$_id.lat",
                        "longitude": "$_id.lng",
                        "event_count": 1,
                        "total_fatalities": 1,
                        "avg_fatalities": {"$round": ["$avg_fatalities", 2]},
                        "intensity_score": {
                            "$multiply": ["$event_count", "$avg_fatalities"]
                        },
                        "_id": 0
                    }
                }
            ])
            
            return await self.get_events_aggregated(pipeline)
            
        except Exception as e:
            logger.error(f"Failed to get hotspots: {e}")
            raise
    
    async def store_ai_interaction(self, interaction_data: Dict[str, Any]):
        """Store AI interaction for analytics and learning"""
        try:
            collection = self.db["ai_interactions"]
            interaction_data["_id"] = str(uuid.uuid4())
            interaction_data["created_at"] = datetime.utcnow()
            
            result = await collection.insert_one(interaction_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to store AI interaction: {e}")
            raise

    async def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")