import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AnalysisEngine:
    @staticmethod
    def analyze_trends(df: pd.DataFrame, period: str = 'monthly') -> List[Dict[str, Any]]:
        """Analyze temporal trends"""
        df = df.copy()
        df['event_date'] = pd.to_datetime(df['event_date'])
        
        if period == 'daily':
            groupby_col = df['event_date'].dt.date
        elif period == 'weekly':
            groupby_col = df['event_date'].dt.isocalendar().week
        else:  # monthly
            groupby_col = df['event_date'].dt.to_period('M')
        
        trends = df.groupby(groupby_col).agg({
            'event_date': 'count',
            'fatalities': ['sum', 'mean'],
            'latitude': 'mean',
            'longitude': 'mean'
        }).reset_index()
        
        trends.columns = ['period', 'total_events', 'total_fatalities', 'avg_fatalities', 'avg_lat', 'avg_lng']
        trends['period'] = trends['period'].astype(str)
        
        return trends.to_dict('records')
    
    @staticmethod
    def identify_hotspots(df: pd.DataFrame, threshold: int = 5) -> List[Dict[str, Any]]:
        """Identify geographic hotspots"""
        df = df.copy()
        
        # Group by location
        hotspots = df.groupby('location').agg({
            'event_date': 'count',
            'fatalities': ['sum', 'mean'],
            'latitude': 'first',
            'longitude': 'first'
        }).reset_index()
        
        hotspots.columns = ['location', 'event_count', 'total_fatalities', 'avg_fatalities', 'latitude', 'longitude']
        
        # Filter by threshold
        hotspots = hotspots[hotspots['event_count'] >= threshold]
        
        # Calculate intensity score
        hotspots['intensity_score'] = hotspots['event_count'] * hotspots['avg_fatalities']
        
        # Sort by intensity
        hotspots = hotspots.sort_values('intensity_score', ascending=False)
        
        return hotspots.head(20).to_dict('records')
    
    @staticmethod
    def detect_anomalies(df: pd.DataFrame, days_back: int = 30) -> List[Dict[str, Any]]:
        """Detect anomalies in recent data"""
        df = df.copy()
        df['event_date'] = pd.to_datetime(df['event_date'])
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_df = df[df['event_date'] >= cutoff_date]
        
        anomalies = []
        
        # High fatality anomalies
        fatality_threshold = recent_df['fatalities'].quantile(0.95)
        high_fatality = recent_df[recent_df['fatalities'] > fatality_threshold]
        
        for _, row in high_fatality.iterrows():
            anomalies.append({
                'type': 'high_fatalities',
                'location': row['location'],
                'fatalities': int(row['fatalities']),
                'date': row['event_date'].isoformat(),
                'severity': 'high' if row['fatalities'] > fatality_threshold * 1.5 else 'medium',
                'description': f"Unusually high fatalities: {row['fatalities']}"
            })
        
        # Geographic clustering anomalies
        location_counts = recent_df.groupby('location').size()
        location_threshold = location_counts.quantile(0.9)
        
        for location, count in location_counts.items():
            if count > location_threshold:
                location_data = recent_df[recent_df['location'] == location]
                anomalies.append({
                    'type': 'geographic_clustering',
                    'location': location,
                    'event_count': int(count),
                    'total_fatalities': int(location_data['fatalities'].sum()),
                    'severity': 'high' if count > location_threshold * 1.5 else 'medium',
                    'description': f"Unusual concentration: {count} events in {location}"
                })
        
        # Temporal anomalies
        daily_counts = recent_df.groupby(recent_df['event_date'].dt.date).size()
        daily_threshold = daily_counts.quantile(0.9)
        
        for date, count in daily_counts.items():
            if count > daily_threshold:
                anomalies.append({
                    'type': 'temporal_spike',
                    'date': str(date),
                    'event_count': int(count),
                    'severity': 'high' if count > daily_threshold * 1.5 else 'medium',
                    'description': f"Spike in daily events: {count} on {date}"
                })
        
        return sorted(anomalies, key=lambda x: x['severity'] == 'high', reverse=True)[:20]
    
    @staticmethod
    def actor_analysis(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze conflict actors"""
        analysis = {}
        
        if 'actor1' in df.columns:
            actor_counts = df['actor1'].value_counts().head(10)
            analysis['top_actors'] = actor_counts.to_dict()
        
        if 'actor2' in df.columns:
            actor_pairs = df.groupby(['actor1', 'actor2']).size().nlargest(10)
            analysis['top_conflicts'] = actor_pairs.to_dict()
        
        return analysis
    
    @staticmethod
    def event_type_analysis(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze event types"""
        event_dist = df['event_type'].value_counts().to_dict()
        
        event_fatalities = df.groupby('event_type')['fatalities'].agg(['sum', 'mean', 'count'])
        
        return {
            'distribution': event_dist,
            'fatalities_by_type': event_fatalities.to_dict('index')
        }
    
    @staticmethod
    def generate_report(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        logger.info("Generating comprehensive analysis report")
        
        return {
            'summary': {
                'total_events': len(df),
                'total_fatalities': int(df['fatalities'].sum()),
                'avg_fatalities_per_event': float(df['fatalities'].mean()),
                'date_range': {
                    'start': df['event_date'].min().isoformat(),
                    'end': df['event_date'].max().isoformat()
                }
            },
            'trends': AnalysisEngine.analyze_trends(df),
            'hotspots': AnalysisEngine.identify_hotspots(df),
            'anomalies': AnalysisEngine.detect_anomalies(df),
            'actors': AnalysisEngine.actor_analysis(df),
            'event_types': AnalysisEngine.event_type_analysis(df),
            'generated_at': datetime.now().isoformat()
        }
