import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from database_v2 import DatabaseManager
from file_processor import FileProcessor
from analysis_engine import AnalysisEngine
import asyncio

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.db_manager = None
    
    async def _get_db_manager(self):
        """Get database manager instance"""
        if not self.db_manager:
            self.db_manager = DatabaseManager()
            await self.db_manager.initialize()
        return self.db_manager
    
    async def process_csv_data(self, df: pd.DataFrame, upload_id: str, source: Optional[str] = None) -> int:
        """Process and validate CSV data with full analysis pipeline"""
        try:
            db = await self._get_db_manager()
            
            # Validate data using FileProcessor
            df_clean, validation_report = FileProcessor.validate_data(df)
            logger.info(f"Validation report: {validation_report}")
            
            # Enrich data with derived features
            df_clean = FileProcessor.enrich_data(df_clean)
            
            # Add metadata
            df_clean['upload_id'] = upload_id
            df_clean['data_source'] = source or 'csv_upload'
            df_clean['processed_at'] = datetime.utcnow().isoformat()
            
            # Generate event IDs if not present
            if 'event_id' not in df_clean.columns:
                df_clean['event_id'] = [f"{source or 'csv'}_{upload_id}_{i}" for i in range(len(df_clean))]
            
            # Convert to records for database insertion
            events = df_clean.to_dict('records')
            
            # Convert datetime objects to strings for MongoDB
            for event in events:
                if isinstance(event['event_date'], pd.Timestamp):
                    event['event_date'] = event['event_date'].isoformat()
            
            # Insert into appropriate collection based on source
            if source == 'acled':
                inserted_count = await db.insert_acled_events(events)
                logger.info(f"Inserted {inserted_count} ACLED events")
            elif source == 'cast':
                inserted_count = await db.insert_cast_forecasts(events)
                logger.info(f"Inserted {inserted_count} CAST forecasts")
            else:
                inserted_count = await db.insert_events(events)
                logger.info(f"Inserted {inserted_count} generic events")
            
            # Generate comprehensive analysis report
            logger.info("Generating comprehensive analysis report...")
            analysis_report = AnalysisEngine.generate_report(df_clean)
            
            # Store analysis report
            await db.db.analysis_reports.insert_one({
                'upload_id': upload_id,
                'source': source or 'unknown',
                'report': analysis_report,
                'created_at': datetime.utcnow()
            })
            
            logger.info(f"Processed and stored {inserted_count} events from upload {upload_id} (source: {source})")
            logger.info(f"Analysis complete: {len(analysis_report['hotspots'])} hotspots, {len(analysis_report['anomalies'])} anomalies detected")
            
            return inserted_count
            
        except Exception as e:
            logger.error(f"Failed to process CSV data: {e}")
            raise
    
    async def calculate_trends(
        self,
        country: Optional[str] = None,
        period: str = "monthly",
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate temporal trends and statistics"""
        try:
            db = await self._get_db_manager()
            
            # Get temporal trends from database
            temporal_data = await db.get_temporal_trends(country=country, period=period, source=source)
            
            if not temporal_data:
                return {
                    "total_events": 0,
                    "total_fatalities": 0,
                    "trend_direction": "stable",
                    "temporal_series": []
                }
            
            # Calculate overall statistics
            total_events = sum(item['total_events'] for item in temporal_data)
            total_fatalities = sum(item['total_fatalities'] for item in temporal_data)
            
            # Determine trend direction
            if len(temporal_data) >= 2:
                recent_events = temporal_data[-1]['total_events']
                previous_events = temporal_data[-2]['total_events']
                
                if recent_events > previous_events * 1.2:
                    trend_direction = "increasing"
                elif recent_events < previous_events * 0.8:
                    trend_direction = "decreasing"
                else:
                    trend_direction = "stable"
            else:
                trend_direction = "stable"
            
            return {
                "total_events": total_events,
                "total_fatalities": total_fatalities,
                "trend_direction": trend_direction,
                "temporal_series": temporal_data,
                "analysis_period": period,
                "data_points": len(temporal_data),
                "source": source or "all"
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate trends: {e}")
            raise
    
    async def identify_hotspots(
        self,
        country: Optional[str] = None,
        threshold: int = 5,
        source: Optional[str] = None
    ) -> List[str]:
        """Identify geographic hotspots"""
        try:
            db = await self._get_db_manager()
            
            hotspots_data = await db.get_hotspots(country=country, threshold=threshold, source=source)
            
            # Extract location names
            hotspot_locations = [hotspot['location'] for hotspot in hotspots_data]
            
            return hotspot_locations
            
        except Exception as e:
            logger.error(f"Failed to identify hotspots: {e}")
            raise
    
    async def detect_anomalies(
        self,
        country: Optional[str] = None,
        days_back: int = 30,
        source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in recent conflict patterns"""
        try:
            db = await self._get_db_manager()
            
            # Get recent events
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            filters = {
                "event_date": {
                    "$gte": start_date.isoformat(),
                    "$lte": end_date.isoformat()
                }
            }
            
            if country:
                filters["country"] = country
            
            if source and source in ["acled", "cast"]:
                filters["data_source"] = source
            
            events = await db.get_events(filters, limit=1000)
            
            if len(events) < 10:
                return []
            
            df = pd.DataFrame(events)
            df['event_date'] = pd.to_datetime(df['event_date'])
            
            anomalies = []
            
            # Detect fatality anomalies
            fatality_threshold = df['fatalities'].quantile(0.95)
            high_fatality_events = df[df['fatalities'] > fatality_threshold]
            
            for _, event in high_fatality_events.iterrows():
                anomalies.append({
                    "type": "high_fatalities",
                    "event_id": event['event_id'],
                    "location": event['location'],
                    "fatalities": int(event['fatalities']),
                    "date": event['event_date'].isoformat(),
                    "severity": "high" if event['fatalities'] > fatality_threshold * 1.5 else "medium",
                    "description": f"Unusually high fatalities: {event['fatalities']} (threshold: {fatality_threshold:.1f})"
                })
            
            # Detect geographic clustering anomalies
            location_counts = df.groupby(['location']).size()
            location_threshold = location_counts.quantile(0.9)
            
            for location, count in location_counts.items():
                if count > location_threshold:
                    recent_events = df[df['location'] == location]
                    total_fatalities = recent_events['fatalities'].sum()
                    
                    anomalies.append({
                        "type": "geographic_clustering",
                        "location": location,
                        "event_count": int(count),
                        "total_fatalities": int(total_fatalities),
                        "date_range": f"{recent_events['event_date'].min().date()} to {recent_events['event_date'].max().date()}",
                        "severity": "high" if count > location_threshold * 1.5 else "medium",
                        "description": f"Unusual concentration of events: {count} events in {location}"
                    })
            
            # Detect temporal anomalies (event spikes)
            daily_counts = df.groupby(df['event_date'].dt.date).size()
            daily_threshold = daily_counts.quantile(0.9)
            
            for date, count in daily_counts.items():
                if count > daily_threshold:
                    day_events = df[df['event_date'].dt.date == date]
                    total_fatalities = day_events['fatalities'].sum()
                    
                    anomalies.append({
                        "type": "temporal_spike",
                        "date": date.isoformat(),
                        "event_count": int(count),
                        "total_fatalities": int(total_fatalities),
                        "severity": "high" if count > daily_threshold * 1.5 else "medium",
                        "description": f"Unusual spike in daily events: {count} events on {date}"
                    })
            
            # Sort by severity and date
            anomalies.sort(key=lambda x: (x['severity'] == 'high', x.get('date', '')), reverse=True)
            
            return anomalies[:20]  # Return top 20 anomalies
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
            raise
    
    async def get_analysis_report(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """Get stored analysis report for an upload"""
        try:
            db = await self._get_db_manager()
            
            report = await db.db.analysis_reports.find_one({'upload_id': upload_id})
            
            if report:
                report['_id'] = str(report['_id'])
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to get analysis report: {e}")
            raise
