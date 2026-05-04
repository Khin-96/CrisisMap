import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from database_sqlite import DatabaseManager
from csv_adapter import CSVAdapter
from acled_adapter import ACLEDAdapter
import asyncio

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.db_manager = None
        self.csv_adapter = CSVAdapter()
        self.acled_adapter = ACLEDAdapter()
    
    async def _get_db_manager(self):
        """Get database manager instance"""
        if not self.db_manager:
            self.db_manager = DatabaseManager()
            await self.db_manager.initialize()
        return self.db_manager
    
    async def analyze_csv_file(self, file_path: str, data_type: str = 'acled_events') -> Dict[str, Any]:
        """Analyze CSV file structure and provide mapping suggestions"""
        try:
            analysis = self.csv_adapter.analyze_csv(file_path, data_type)
            return analysis
        except Exception as e:
            logger.error(f"Failed to analyze CSV file: {e}")
            raise
    
    async def process_csv_data(self, file_path: str, custom_mappings: Dict[str, str] = None, upload_id: str = None) -> int:
        """Process and validate CSV data with adaptive column mapping"""
        try:
            db = await self._get_db_manager()
            
            # Process CSV with adaptive mapping
            df_processed = self.csv_adapter.process_csv(file_path, custom_mappings)
            
            # Validate processed data
            validation_report = self.csv_adapter.validate_processed_data(df_processed)
            
            if not validation_report['is_valid']:
                raise ValueError(f"Data validation failed: {validation_report['errors']}")
            
            # Add metadata
            df_processed['upload_id'] = upload_id or f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            df_processed['data_source'] = 'csv_upload'
            df_processed['processed_at'] = datetime.now().isoformat()
            
            # Convert to records for database insertion
            events = df_processed.to_dict('records')
            
            # Insert into database
            inserted_count = await db.insert_events(events)
            
            # MongoDB dual-write
            try:
                from database import DatabaseManager as MongoManager
                mongo_db = MongoManager()
                await mongo_db.initialize()
                await mongo_db.insert_events(events)
                logger.info("Successfully replicated to MongoDB Atlas")
            except Exception as e:
                logger.error(f"Failed to dual-write to MongoDB Atlas: {e}")
            
            logger.info(f"Processed and stored {inserted_count} events from upload {upload_id}")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Failed to process CSV data: {e}")
            raise
    
    async def process_acled_data(self, filters: Dict[str, Any] = None, max_records: int = 5000) -> int:
        """Fetch and process data from ACLED API"""
        try:
            db = await self._get_db_manager()
            
            # Fetch data from ACLED
            records = self.acled_adapter.fetch_paginated_data(filters, max_records)
            
            if not records:
                logger.warning("No records returned from ACLED API")
                return 0
                
            # Convert and clean data
            df_processed = self.acled_adapter.standardize_data(records)
            
            # Reuse validation logic if needed
            validation_report = self.csv_adapter.validate_processed_data(df_processed)
            
            if not validation_report['is_valid']:
                logger.warning(f"ACLED data validation warnings: {validation_report['warnings']}")
                # We skip critical error if we trust ACLED enough, or handle accordingly
            
            # Add metadata
            import uuid
            fetch_id = f"acled_{uuid.uuid4().hex[:8]}"
            df_processed['upload_id'] = fetch_id
            df_processed['data_source'] = 'acled_api'
            df_processed['processed_at'] = datetime.now().isoformat()
            
            # Convert to records for database insertion
            events = df_processed.to_dict('records')
            
            # Insert into database
            inserted_count = await db.insert_events(events)
            
            # MongoDB dual-write
            try:
                from database import DatabaseManager as MongoManager
                mongo_db = MongoManager()
                await mongo_db.initialize()
                await mongo_db.insert_events(events)
                logger.info("Successfully replicated API data to MongoDB Atlas")
            except Exception as e:
                logger.error(f"Failed to dual-write API data to MongoDB Atlas: {e}")
            
            logger.info(f"Successfully processed and stored {inserted_count} ACLED records")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Failed to process ACLED data: {e}")
            raise
    
    async def calculate_trends(
        self,
        country: Optional[str] = None,
        period: str = "monthly"
    ) -> Dict[str, Any]:
        """Calculate temporal trends and statistics"""
        try:
            db = await self._get_db_manager()
            
            # Get temporal trends from database
            temporal_data = await db.get_temporal_trends(country=country, period=period)
            
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
                "data_points": len(temporal_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate trends: {e}")
            raise
    
    async def identify_hotspots(
        self,
        country: Optional[str] = None,
        threshold: int = 5
    ) -> List[str]:
        """Identify geographic hotspots"""
        try:
            db = await self._get_db_manager()
            
            hotspots_data = await db.get_hotspots(country=country, threshold=threshold)
            
            # Extract location names
            hotspot_locations = [hotspot['location'] for hotspot in hotspots_data]
            
            return hotspot_locations
            
        except Exception as e:
            logger.error(f"Failed to identify hotspots: {e}")
            raise
    
    async def detect_anomalies(
        self,
        country: Optional[str] = None,
        days_back: int = 30
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
    
    async def generate_summary_report(
        self,
        country: Optional[str] = None,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """Generate a comprehensive summary report"""
        try:
            # Get recent trends
            trends = await self.calculate_trends(country=country, period="daily")
            
            # Get hotspots
            hotspots = await self.identify_hotspots(country=country)
            
            # Get anomalies
            anomalies = await self.detect_anomalies(country=country, days_back=days_back)
            
            # Calculate risk assessment
            risk_factors = []
            risk_score = 0
            
            if trends['trend_direction'] == 'increasing':
                risk_factors.append("Increasing trend in conflict events")
                risk_score += 30
            
            if len(hotspots) > 5:
                risk_factors.append(f"Multiple active hotspots ({len(hotspots)})")
                risk_score += 20
            
            high_severity_anomalies = [a for a in anomalies if a['severity'] == 'high']
            if len(high_severity_anomalies) > 0:
                risk_factors.append(f"High-severity anomalies detected ({len(high_severity_anomalies)})")
                risk_score += 25
            
            # Determine overall risk level
            if risk_score >= 60:
                risk_level = "critical"
            elif risk_score >= 40:
                risk_level = "high"
            elif risk_score >= 20:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            return {
                "generated_at": datetime.now().isoformat(),
                "country": country or "All regions",
                "analysis_period_days": days_back,
                "summary": {
                    "total_events": trends['total_events'],
                    "total_fatalities": trends['total_fatalities'],
                    "trend_direction": trends['trend_direction'],
                    "active_hotspots": len(hotspots),
                    "anomalies_detected": len(anomalies)
                },
                "risk_assessment": {
                    "level": risk_level,
                    "score": risk_score,
                    "factors": risk_factors
                },
                "top_hotspots": hotspots[:5],
                "recent_anomalies": anomalies[:5],
                "recommendations": self._generate_recommendations(risk_level, trends, anomalies)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
            raise
    
    def _generate_recommendations(
        self,
        risk_level: str,
        trends: Dict[str, Any],
        anomalies: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        if risk_level in ["critical", "high"]:
            recommendations.append("Increase monitoring frequency and alert thresholds")
            recommendations.append("Deploy additional resources to identified hotspots")
        
        if trends['trend_direction'] == 'increasing':
            recommendations.append("Investigate underlying drivers of conflict escalation")
            recommendations.append("Consider preventive interventions in affected areas")
        
        high_fatality_anomalies = [a for a in anomalies if a['type'] == 'high_fatalities']
        if high_fatality_anomalies:
            recommendations.append("Investigate high-fatality incidents for potential war crimes")
        
        geographic_anomalies = [a for a in anomalies if a['type'] == 'geographic_clustering']
        if geographic_anomalies:
            recommendations.append("Focus humanitarian assistance on areas with event clustering")
        
        if not recommendations:
            recommendations.append("Continue routine monitoring and data collection")
            recommendations.append("Maintain current alert thresholds and response protocols")
        
        return recommendations