import motor.motor_asyncio
from pymongo import IndexModel, ASCENDING, DESCENDING
from typing import Dict, List, Optional, Any
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.connection_string = os.getenv(
            "MONGODB_URL", 
            "mongodb://localhost:27017"
        )
        self.database_name = os.getenv("MONGODB_DATABASE", "crisismap")
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
        
        # ACLED Events collection
        acled_indexes = [
            IndexModel([("event_date", DESCENDING)]),
            IndexModel([("country", ASCENDING)]),
            IndexModel([("event_type", ASCENDING)]),
            IndexModel([("latitude", ASCENDING), ("longitude", ASCENDING)]),
            IndexModel([("data_source", ASCENDING)]),
            IndexModel([("upload_id", ASCENDING)]),
        ]
        await self.db.acled_events.create_indexes(acled_indexes)
        
        # CAST Forecasts collection
        cast_indexes = [
            IndexModel([("event_date", DESCENDING)]),
            IndexModel([("country", ASCENDING)]),
            IndexModel([("data_source", ASCENDING)]),
            IndexModel([("upload_id", ASCENDING)]),
        ]
        await self.db.cast_forecasts.create_indexes(cast_indexes)
        
        # Analysis reports
        reports_indexes = [
            IndexModel([("upload_id", ASCENDING)]),
            IndexModel([("data_source", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
        ]
        await self.db.analysis_reports.create_indexes(reports_indexes)
        
        logger.info("Database indexes created successfully")
    
    # ACLED Events Methods
    async def insert_acled_events(self, events: List[Dict[str, Any]]) -> int:
        """Insert ACLED events"""
        try:
            for event in events:
                event["created_at"] = datetime.utcnow()
                event["data_source"] = "acled"
            
            result = await self.db.acled_events.insert_many(events)
            return len(result.inserted_ids)
        except Exception as e:
            logger.error(f"Failed to insert ACLED events: {e}")
            raise
    
    async def get_acled_events(
        self, 
        filters: Dict[str, Any] = None, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get ACLED events"""
        try:
            query = filters or {}
            query["data_source"] = "acled"
            
            cursor = self.db.acled_events.find(query).sort("event_date", -1).limit(limit)
            events = await cursor.to_list(length=limit)
            
            for event in events:
                event["_id"] = str(event["_id"])
            
            return events
        except Exception as e:
            logger.error(f"Failed to get ACLED events: {e}")
            raise
    
    # CAST Forecasts Methods
    async def insert_cast_forecasts(self, forecasts: List[Dict[str, Any]]) -> int:
        """Insert CAST forecasts"""
        try:
            for forecast in forecasts:
                forecast["created_at"] = datetime.utcnow()
                forecast["data_source"] = "cast"
            
            result = await self.db.cast_forecasts.insert_many(forecasts)
            return len(result.inserted_ids)
        except Exception as e:
            logger.error(f"Failed to insert CAST forecasts: {e}")
            raise
    
    async def get_cast_forecasts(
        self, 
        filters: Dict[str, Any] = None, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get CAST forecasts"""
        try:
            query = filters or {}
            query["data_source"] = "cast"
            
            cursor = self.db.cast_forecasts.find(query).sort("event_date", -1).limit(limit)
            forecasts = await cursor.to_list(length=limit)
            
            for forecast in forecasts:
                forecast["_id"] = str(forecast["_id"])
            
            return forecasts
        except Exception as e:
            logger.error(f"Failed to get CAST forecasts: {e}")
            raise
    
    # Generic Methods (for backward compatibility)
    async def insert_events(self, events: List[Dict[str, Any]]) -> int:
        """Insert events (auto-detect source)"""
        if not events:
            return 0
        
        # Separate by source
        acled_events = [e for e in events if e.get("data_source") == "acled"]
        cast_events = [e for e in events if e.get("data_source") == "cast"]
        
        count = 0
        if acled_events:
            count += await self.insert_acled_events(acled_events)
        if cast_events:
            count += await self.insert_cast_forecasts(cast_events)
        
        return count
    
    async def get_events(
        self, 
        filters: Dict[str, Any] = None, 
        limit: int = 100,
        source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get events from specified source or both"""
        try:
            if source == "acled":
                return await self.get_acled_events(filters, limit)
            elif source == "cast":
                return await self.get_cast_forecasts(filters, limit)
            else:
                # Get from both
                acled = await self.get_acled_events(filters, limit // 2)
                cast = await self.get_cast_forecasts(filters, limit // 2)
                return acled + cast
        except Exception as e:
            logger.error(f"Failed to get events: {e}")
            raise
    
    # Analysis Reports
    async def store_analysis_report(
        self, 
        upload_id: str, 
        report: Dict[str, Any],
        data_source: str
    ) -> str:
        """Store analysis report"""
        try:
            result = await self.db.analysis_reports.insert_one({
                'upload_id': upload_id,
                'data_source': data_source,
                'report': report,
                'created_at': datetime.utcnow()
            })
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to store analysis report: {e}")
            raise
    
    async def get_analysis_report(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis report"""
        try:
            report = await self.db.analysis_reports.find_one({'upload_id': upload_id})
            if report:
                report['_id'] = str(report['_id'])
            return report
        except Exception as e:
            logger.error(f"Failed to get analysis report: {e}")
            raise
    
    # Aggregation Methods
    async def get_temporal_trends(
        self,
        country: Optional[str] = None,
        period: str = "monthly",
        source: str = "acled"
    ) -> List[Dict[str, Any]]:
        """Get temporal trends for specific source"""
        try:
            if period == "daily":
                date_format = "%Y-%m-%d"
            elif period == "weekly":
                date_format = "%Y-%U"
            else:
                date_format = "%Y-%m"
            
            pipeline = [
                {"$match": {"data_source": source}},
            ]
            
            if country:
                pipeline.append({"$match": {"country": country}})
            
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
                        "avg_fatalities": {"$avg": "$fatalities"}
                    }
                },
                {"$sort": {"_id": 1}},
                {
                    "$project": {
                        "period": "$_id",
                        "total_events": 1,
                        "total_fatalities": 1,
                        "avg_fatalities": {"$round": ["$avg_fatalities", 2]},
                        "_id": 0
                    }
                }
            ])
            
            collection = self.db.acled_events if source == "acled" else self.db.cast_forecasts
            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=None)
            return results
        except Exception as e:
            logger.error(f"Failed to get temporal trends: {e}")
            raise
    
    async def get_hotspots(
        self,
        country: Optional[str] = None,
        threshold: int = 5,
        source: str = "acled"
    ) -> List[Dict[str, Any]]:
        """Get geographic hotspots for specific source"""
        try:
            pipeline = [
                {"$match": {"data_source": source}},
            ]
            
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
                        "avg_fatalities": {"$avg": "$fatalities"}
                    }
                },
                {"$match": {"event_count": {"$gte": threshold}}},
                {"$sort": {"total_fatalities": -1}},
                {"$limit": 20},
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
            
            collection = self.db.acled_events if source == "acled" else self.db.cast_forecasts
            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=None)
            return results
        except Exception as e:
            logger.error(f"Failed to get hotspots: {e}")
            raise
    
    async def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
